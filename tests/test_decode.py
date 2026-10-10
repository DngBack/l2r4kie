"""Tests for ``l2r4kie.model.decode`` on a tiny random Qwen2-VL (CPU, FP32).

The tiny model keeps Qwen2-VL's full vocabulary, so the real tokenizer,
processor and :class:`~l2r4kie.model.format.KevFormat` are used unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tiny import HIDDEN, VOCAB, tiny_model

from l2r4kie.data.types import FieldRequest, FieldSpec
from l2r4kie.model.decode import TOKEN_STATS, DecodeResult, decode_prefix, extract
from l2r4kie.model.extractor import Extractor, ExtractorConfig
from l2r4kie.model.format import EncodedBranch, KevFormat
from l2r4kie.model.packing import Packed, pack, prefix_positions


@pytest.fixture(scope='module')
def processor(qwen_processor):  # noqa: ANN001, ANN201
    return qwen_processor


def make_extractor(processor, model: torch.nn.Module | None = None, **config) -> Extractor:  # noqa: ANN001
    return Extractor(model or tiny_model(), processor, KevFormat(processor.tokenizer),
                     ExtractorConfig(device='cpu', precision='float32', **config))


def text_prefix(extractor: Extractor) -> dict[str, torch.Tensor]:
    """A short text-only prefix (no image), enough for decode mechanics."""
    ids = extractor.format.tokenizer('Document text', add_special_tokens=False, return_tensors='pt')['input_ids']
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def requests() -> list[FieldRequest]:
    return [FieldRequest('/ho_ten', 'Họ và tên'), FieldRequest('/ngay_sinh'), FieldRequest('/dia_chi', 'Địa chỉ thường trú')]


def teacher_forced(extractor: Extractor, prefix: dict[str, torch.Tensor],
                   results: list[DecodeResult], layer: int | None = None) -> tuple[torch.Tensor, Packed]:
    """Pack the decoded values as targets and return the packed hidden states (last layer or ``layer``)."""
    fmt = extractor.format
    branches = []
    for r, result in zip(requests(), results, strict=True):
        tokens = result.trace.tokens if result.status != 'truncated' else (*result.trace.tokens, fmt.markers.value_close)
        branches.append(EncodedBranch(r.id, fmt.request(r.id, r.description).prompt, tokens))
    packed = pack(prefix, prefix_positions(extractor.core, prefix), branches, max_value_tokens=10_000)
    with torch.no_grad():
        output = extractor.core(**packed.inputs, output_hidden_states=layer is not None)
    return (output.last_hidden_state if layer is None else output.hidden_states[layer])[0], packed


# ------------------------------------------------------------------ mechanics

def test_never_generates_banned_tokens_even_when_they_dominate(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor)
    fmt = extractor.format
    banned = fmt.banned_ids(VOCAB)
    with torch.no_grad():  # make every banned row (im_end, vision, unused rows...) win the raw argmax
        extractor.lm_head.weight[banned] *= 1000
    prefix = text_prefix(extractor)
    results = decode_prefix(extractor, prefix, requests(), max_value_tokens=12, trace=True)
    generated = {t for r in results for t in r.trace.tokens}
    assert generated and not generated & set(banned)
    assert all(r.status == 'truncated' and r.signals.value is None and len(r.trace.tokens) == 12 for r in results)
    assert all(r.signals.layers is None for r in results)
    assert all(r.trace.stats.shape == (12, len(TOKEN_STATS)) and torch.isfinite(r.trace.stats).all()
               for r in results)


def test_stops_at_close_marker(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor)
    close = extractor.format.markers.value_close
    # A separate (untied) head whose bias makes the close marker always win: every value is empty.
    head = torch.nn.Linear(HIDDEN, VOCAB, bias=True)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    with torch.no_grad():
        head.bias[close] = 1
    extractor.base.lm_head = head
    results = decode_prefix(extractor, text_prefix(extractor), requests(), max_value_tokens=12, trace=True)
    assert [(r.status, r.text, r.value, r.trace.tokens) for r in results] == [('ok', '', '', (close,))] * 3
    assert all(r.signals.value is not None for r in results)
    # the close row: chosen id is the close marker, so both log-probabilities agree
    assert all(r.trace.stats[-1, 0] == r.trace.stats[-1, TOKEN_STATS.index('close_log_probability')]
               for r in results)


def test_layers_are_validated_and_nan_until_closed(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor)
    with pytest.raises(ValueError, match='layers'):
        decode_prefix(extractor, text_prefix(extractor), requests(), layers=(3,))
    results = decode_prefix(extractor, text_prefix(extractor), requests(), max_value_tokens=2, layers=(0,))
    assert all(r.status == 'truncated' and r.signals.layers.shape == (3, 1, HIDDEN) for r in results)
    assert all(r.signals.layers[2].isnan().all() and not r.signals.layers[:2].isnan().any() for r in results)


def test_duplicate_ids_are_rejected(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor)
    with pytest.raises(ValueError, match='Duplicate'):
        decode_prefix(extractor, text_prefix(extractor), [FieldRequest('/a'), FieldRequest('/a')])


# --------------------------------------------- equivalence with teacher forcing

def overfit(extractor: Extractor, prefix: dict[str, torch.Tensor], values: dict[str, str], steps: int = 150) -> None:
    """Train the tiny model to answer each request with ``values`` (packing + CE)."""
    fmt = extractor.format
    branches = [fmt.encode(FieldSpec(r.id, r.description, values[r.id], 'scalar')) for r in requests()]
    packed = pack(prefix, prefix_positions(extractor.core, prefix), branches, max_value_tokens=64)
    extractor.model.train()
    optimizer = torch.optim.Adam(extractor.model.parameters(), lr=3e-3)
    for _ in range(steps):
        hidden = extractor.core(**packed.inputs).last_hidden_state[0]
        loss = torch.nn.functional.cross_entropy(extractor.lm_head(hidden[packed.predict_positions]), packed.targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    extractor.model.eval()


def test_decode_matches_teacher_forced_packing(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor, max_branches=2)  # 3 fields: two chunks, one prefix encode
    prefix = text_prefix(extractor)
    values = {'/ho_ten': 'Nguyễn Văn A', '/ngay_sinh': '01/02/1990', '/dia_chi': 'Hà Nội'}
    overfit(extractor, prefix, values)
    results = decode_prefix(extractor, prefix, requests(), max_value_tokens=32, trace=True, layers=(1, 2))
    assert {r.field_id: (r.status, r.value) for r in results} == {k: ('ok', v) for k, v in values.items()}

    hidden, packed = teacher_forced(extractor, prefix, results)
    middle = teacher_forced(extractor, prefix, results, layer=1)[0]
    for i, result in enumerate(results):
        # intermediate layer 1, and layer 2 (the last) equal to the main signals
        positions = (packed.key_positions[i], packed.decide_positions[i], packed.value_positions[i])
        torch.testing.assert_close(result.signals.layers[:, 0], middle[torch.stack(positions)], atol=1e-5, rtol=0)
        torch.testing.assert_close(result.signals.layers[:, 1],
                                   torch.stack((result.signals.key, result.signals.decide, result.signals.value)),
                                   atol=1e-5, rtol=0)
        torch.testing.assert_close(result.signals.key, hidden[packed.key_positions[i]], atol=1e-5, rtol=0)
        torch.testing.assert_close(result.signals.decide, hidden[packed.decide_positions[i]], atol=1e-5, rtol=0)
        torch.testing.assert_close(result.signals.value, hidden[packed.value_positions[i]], atol=1e-5, rtol=0)
        torch.testing.assert_close(result.trace.hidden[-1], result.signals.value, atol=0, rtol=0)
        # every value token's state, not only the close marker's
        start = int(packed.decide_positions[i]) + 1
        torch.testing.assert_close(result.trace.hidden, hidden[start:start + len(result.trace.tokens)],
                                   atol=1e-5, rtol=0)


def test_chunking_does_not_change_results(processor) -> None:  # noqa: ANN001
    model = tiny_model(seed=5)
    one = make_extractor(processor, model, max_branches=64)
    many = make_extractor(processor, model, max_branches=1)
    prefix = text_prefix(one)
    a = decode_prefix(one, prefix, requests(), max_value_tokens=8)
    b = decode_prefix(many, prefix, requests(), max_value_tokens=8)
    assert [(r.text, r.status) for r in a] == [(r.text, r.status) for r in b]
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x.signals.decide, y.signals.decide, atol=1e-5, rtol=0)


# ------------------------------------------------------------ images, end to end

def test_extract_runs_the_vision_encoder_once(processor, tmp_path: Path) -> None:  # noqa: ANN001
    from PIL import Image

    pages = []
    for n, colour in enumerate(('white', 'gray')):
        pages.append(tmp_path / f'page{n}.png')
        Image.new('RGB', (120, 90), colour).save(pages[-1])
    extractor = make_extractor(processor, max_branches=2)
    calls = []
    extractor.core.visual.register_forward_hook(lambda *_: calls.append(1))
    results = extract(extractor, pages, requests(), max_value_tokens=4)
    assert len(calls) == 1
    assert [r.field_id for r in results] == [r.id for r in requests()]
    assert all(r.status == 'truncated' and r.signals.key.shape == (HIDDEN,) for r in results)
