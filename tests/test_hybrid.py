"""Branch-isolated Qwen3.5 forward (:mod:`l2r4kie.model.hybrid`) against Hugging Face's own forward.

A tiny random Qwen3.5 (3 DeltaNet layers + 1 attention layer) with the real
Qwen3.5 processor. Needs CUDA, ``flash-linear-attention`` and a cached
``Qwen/Qwen3.5-0.8B`` processor; skipped otherwise.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip('transformers')
pytest.importorskip('fla')
if not torch.cuda.is_available() or not hasattr(transformers, 'Qwen3_5ForConditionalGeneration'):
    pytest.skip('needs CUDA and transformers with Qwen3.5', allow_module_level=True)

from PIL import Image

from l2r4kie.data.types import FieldRequest
from l2r4kie.model.decode import decode_prefix
from l2r4kie.model.extractor import Extractor, ExtractorConfig
from l2r4kie.model.format import EncodedBranch, KevFormat
from l2r4kie.model.hybrid import HybridCore
from l2r4kie.model.packing import pack, prefix_positions
from l2r4kie.train.losses import value_loss

DEVICE = 'cuda:0'


@pytest.fixture(scope='module')
def extractor(tmp_path_factory: pytest.TempPathFactory) -> Extractor:
    try:
        processor = transformers.AutoProcessor.from_pretrained('Qwen/Qwen3.5-0.8B', min_pixels=64 * 64,
                                                               max_pixels=160 * 160, local_files_only=True)
    except OSError:
        pytest.skip('Qwen/Qwen3.5-0.8B processor not cached')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    config = transformers.Qwen3_5Config(
        text_config={'vocab_size': 248_320, 'hidden_size': 64, 'intermediate_size': 128, 'num_hidden_layers': 4,
                     'layer_types': ['linear_attention'] * 3 + ['full_attention'], 'num_attention_heads': 4,
                     'num_key_value_heads': 2, 'head_dim': 32, 'linear_num_key_heads': 2,
                     'linear_num_value_heads': 4, 'linear_key_head_dim': 16, 'linear_value_head_dim': 16,
                     'rope_parameters': {'rope_type': 'default', 'rope_theta': 10_000_000, 'mrope_interleaved': True,
                                         'mrope_section': [2, 1, 1], 'partial_rotary_factor': 0.25}},
        vision_config={'depth': 1, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 2, 'out_hidden_size': 64,
                       'patch_size': 16, 'spatial_merge_size': 2, 'temporal_patch_size': 2},
        image_token_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>'),
        video_token_id=processor.tokenizer.convert_tokens_to_ids('<|video_pad|>'),
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids('<|vision_start|>'))
    config._attn_implementation = 'sdpa'
    model = transformers.Qwen3_5ForConditionalGeneration(config).to(DEVICE).eval()
    # Positive, large-ish recurrence inputs so that the prefix state matters.
    for layer in model.model.language_model.layers[:3]:
        layer.linear_attn.A_log.data.fill_(-2.0)
    page = tmp_path_factory.mktemp('page') / 'page.png'
    Image.effect_noise((128, 96), 64).convert('RGB').save(page)
    ex = Extractor(model, processor, KevFormat(processor.tokenizer),
                   ExtractorConfig(device=DEVICE, precision='float32', max_value_tokens=12, early_stop=False))
    ex.page = str(page)  # type: ignore[attr-defined]
    return ex


def branches(fmt: KevFormat) -> list[EncodedBranch]:
    fields = [('ho_ten', 'Họ và tên', 'NGUYỄN VĂN A'), ('ngay', 'Ngày', '01/02/2024'), ('so', 'Số', '12')]
    return [EncodedBranch(f, tuple(fmt.prompt_ids(f, d)), tuple(fmt.target_ids(v))) for f, d, v in fields]


def reference(extractor: Extractor, prefix: dict, tokens: list[int], **kwargs: object) -> object:
    """Hugging Face forward of [prefix + tokens] (one branch, positions continuing the prefix)."""
    ids = torch.cat([prefix['input_ids'], torch.tensor([tokens], device=DEVICE)], 1)
    types = torch.cat([prefix['mm_token_type_ids'], torch.zeros((1, len(tokens)), dtype=torch.long, device=DEVICE)], 1)
    return extractor.base.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                pixel_values=prefix['pixel_values'], image_grid_thw=prefix['image_grid_thw'],
                                mm_token_type_ids=types, **kwargs)


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


def test_core_is_hybrid(extractor: Extractor) -> None:
    assert extractor.hybrid and isinstance(extractor.core, HybridCore)


@torch.no_grad()
def test_packed_branches_equal_each_branch_run_alone(extractor: Extractor) -> None:
    prefix = extractor.encode_prefix([extractor.page])
    encoded = branches(extractor.format)
    packed = pack(prefix, prefix_positions(extractor.core, prefix), encoded, 100)
    hidden = extractor.core.packed_hidden(packed.inputs, packed.segments)
    shared = prefix['input_ids'].shape[1]
    start = shared
    for branch in encoded:
        expected = reference(extractor, prefix, [*branch.prompt, *branch.target]).last_hidden_state[0]
        assert relative(hidden[:shared], expected[:shared]) < 1e-3
        assert relative(hidden[start:start + branch.length], expected[shared:]) < 1e-3
        start += branch.length


@torch.no_grad()
def test_decode_equals_huggingface_greedy(extractor: Extractor) -> None:
    fmt = extractor.format
    prefix = extractor.encode_prefix([extractor.page])
    requests = [FieldRequest('ho_ten', 'Họ và tên'), FieldRequest('so', 'Số'), FieldRequest('ngay', 'Ngày')]
    results = decode_prefix(extractor, prefix, requests, trace=True, layers=(1, 3))
    banned = torch.tensor(fmt.banned_ids(extractor.vocab_size), device=DEVICE)
    base = int(prefix_positions(extractor.core, prefix).max()) + 1
    for request, result in zip(requests, results, strict=True):
        prompt = list(fmt.request(request.id, request.description).prompt)
        cache = transformers.DynamicCache(config=extractor.base.model.config)
        out = reference(extractor, prefix, prompt, past_key_values=cache, use_cache=True, output_hidden_states=True)
        assert relative(result.signals.key[None], out.last_hidden_state[0, -2:-1].cpu()) < 1e-3
        assert relative(result.signals.layers[1, 0][None], out.hidden_states[1][0, -1:].cpu()) < 1e-3
        state, tokens = out.last_hidden_state[:, -1], []
        length = prefix['input_ids'].shape[1] + len(prompt)
        for step in range(12):
            logits = extractor.lm_head(state)
            logits[:, banned] = torch.finfo(logits.dtype).min
            tokens.append(int(logits.argmax(-1)))
            out = extractor.base.model(
                input_ids=torch.tensor([[tokens[-1]]], device=DEVICE), past_key_values=cache, use_cache=True,
                attention_mask=torch.ones((1, length + step + 1), dtype=torch.long, device=DEVICE),
                position_ids=torch.full((3, 1, 1), base + len(prompt) + step, device=DEVICE))
            state = out.last_hidden_state[:, -1]
            assert relative(result.trace.hidden[step][None], state.cpu()) < 1e-3
            if tokens[-1] == fmt.markers.value_close:
                break
        assert list(result.trace.tokens) == tokens


@torch.no_grad()
def test_chunks_and_compaction_do_not_change_results(extractor: Extractor) -> None:
    prefix = extractor.encode_prefix([extractor.page])
    requests = [FieldRequest(f'f{i}', f'trường {i}') for i in range(5)]
    whole = decode_prefix(extractor, prefix, requests, trace=True)
    small = Extractor(extractor.model, extractor.processor, extractor.format,
                      ExtractorConfig(device=DEVICE, precision='float32', max_value_tokens=12, early_stop=False,
                                      max_branches=2))
    chunked = decode_prefix(small, prefix, requests, trace=True)
    for a, b in zip(whole, chunked, strict=True):
        assert a.trace.tokens == b.trace.tokens
        assert relative(a.trace.hidden, b.trace.hidden) < 1e-4


def test_loss_backpropagates_into_every_layer_type(extractor: Extractor) -> None:
    prefix = extractor.encode_prefix([extractor.page])
    with torch.no_grad():
        positions = prefix_positions(extractor.core, prefix)
    packed = pack(prefix, positions, branches(extractor.format), 100)
    extractor.model.zero_grad()
    out = value_loss(extractor.core, extractor.lm_head, packed)
    out.loss.backward()
    layers = extractor.base.model.language_model.layers
    for grad in (layers[0].linear_attn.in_proj_qkv.weight.grad, layers[0].linear_attn.A_log.grad,
                 layers[3].self_attn.q_proj.weight.grad):
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    extractor.model.zero_grad()
