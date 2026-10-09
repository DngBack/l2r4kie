"""Tests for ``l2r4kie.train``: config validation, loss, resume, accumulation, K6, monitor.

Training runs use the tiny random Qwen2-VL (``tests/tiny.py``) on CPU in FP32,
with the real processor and a two-page synthetic prepared directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file
from tiny import HIDDEN, VOCAB, tiny_model

from l2r4kie.data.types import Document, FieldSpec
from l2r4kie.eval.comparator import correct
from l2r4kie.eval.errors import is_coordinates
from l2r4kie.model.format import EncodedBranch
from l2r4kie.model.packing import pack, prefix_positions
from l2r4kie.train.config import TrainConfig
from l2r4kie.train.losses import position_weights, value_loss
from l2r4kie.train.schedule import lr_multiplier
from l2r4kie.train.trainer import Trainer, train
from l2r4kie.utils.io import read_json, read_jsonl, write_jsonl

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope='module')
def prepared(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Prepared directory with 3 train and 1 dev documents of small page images."""
    from PIL import Image

    root = tmp_path_factory.mktemp('prepared')
    values = [{'/ho_ten': 'Nguyễn Văn A', '/ngay_sinh': '01/02/1990', '/nam': True, '/con': ['B', 'C']},
              {'/ho_ten': 'Trần Thị B', '/ngay_sinh': '03/04/1985', '/nam': False, '/con': []},
              {'/ho_ten': 'Lê C', '/ngay_sinh': '', '/nam': True, '/con': ['D']},
              {'/ho_ten': 'Phạm D', '/ngay_sinh': '05/06/2000', '/nam': False, '/con': ['E', 'F']}]
    documents = []
    for n, fields in enumerate(values):
        page = root / f'doc{n}.png'
        Image.new('RGB', (100, 80), (40 * n, 90, 200)).save(page)
        specs = tuple(FieldSpec(k, k.strip('/'), v, 'array' if isinstance(v, list) else 'scalar')
                      for k, v in fields.items())
        documents.append(Document(f'form{n % 2}__{n}', f'form{n % 2}__{n}', f'form{n % 2}', (str(page),), specs,
                                  ('0' * 64,)))
    write_jsonl(root / 'train.jsonl', [d.to_json() for d in documents[:3]])
    write_jsonl(root / 'dev.jsonl', [documents[3].to_json()])
    return root


def make_config(prepared: Path, output: Path, **overrides: Any) -> TrainConfig:
    values: dict[str, Any] = {'output': str(output), 'selection': {'prepared': str(prepared)}, 'device': 'cpu',
                              'precision': 'float32', 'steps': 4, 'fields_per_document': 3, 'lr': 1e-2,
                              'format': {'max_value_tokens': 64}}
    for key, value in overrides.items():
        if isinstance(value, dict):
            values[key] = {**values.get(key, {}), **value}
        else:
            values[key] = value
    return TrainConfig.from_dict(values)


def run(config: TrainConfig, processor: Any, until: int | None = None) -> dict[str, Any] | None:
    return train(config, base=tiny_model(), processor=processor, until=until)


def adapter(output: Path) -> dict[str, torch.Tensor]:
    return load_file(output / 'adapter' / 'adapter_model.safetensors')


# -------------------------------------------------------------------- config


def test_config_rejects_unknown_keys_at_every_level(prepared: Path, tmp_path: Path) -> None:
    base = {'output': str(tmp_path), 'selection': {'prepared': str(prepared)}}
    with pytest.raises(ValueError, match="Unknown key.*gradient_acumulation"):
        TrainConfig.from_dict({**base, 'gradient_acumulation': 2})
    with pytest.raises(ValueError, match="Unknown key.*in format.*clos"):
        TrainConfig.from_dict({**base, 'format': {'clos': 'im_end'}})
    with pytest.raises(ValueError, match='output'):
        TrainConfig.from_dict({'selection': {'prepared': str(prepared)}})
    with pytest.raises(ValueError, match='warmup_updates'):
        TrainConfig.from_dict({**base, 'steps': 4, 'gradient_accumulation': 2, 'warmup_updates': 3})


def test_config_round_trips_and_resume_ignores_operational_keys(prepared: Path, tmp_path: Path) -> None:
    config = make_config(prepared, tmp_path, monitor={'every': 2, 'splits': ['train']})
    assert TrainConfig.from_dict(config.to_dict()) == config
    moved = make_config(prepared, tmp_path / 'elsewhere', device='cuda:3', log_every=1)
    assert moved.comparable() == make_config(prepared, tmp_path).comparable()
    assert make_config(prepared, tmp_path, lr=1e-3).comparable() != config.comparable()


def test_lr_schedule() -> None:
    assert [lr_multiplier(u, 10, warmup=2) for u in range(3)] == [0.5, 1.0, 1.0]
    assert lr_multiplier(10, 10, cosine=True) == pytest.approx(0.0)


# ---------------------------------------------------------------------- loss


def toy_packed(processor: Any, model: torch.nn.Module) -> Any:
    ids = processor.tokenizer('Document', return_tensors='pt')['input_ids']
    prefix = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    branches = [EncodedBranch('/a', (1, 2, 3), (10, 11, 12, 13, 14)), EncodedBranch('/b', (4, 5, 6), (20,))]
    return pack(prefix, prefix_positions(model.model, prefix), branches, max_value_tokens=8)


def test_loss_weighting(qwen_processor: Any) -> None:
    packed = toy_packed(qwen_processor, tiny_model())
    token, field = position_weights(packed, 'token'), position_weights(packed, 'field')
    assert token.tolist() == pytest.approx([1 / 6] * 6)
    assert field.tolist() == pytest.approx([0.1] * 5 + [0.5])  # each field sums to 1/2


def test_chunked_loss_equals_unchunked(qwen_processor: Any) -> None:
    model = tiny_model()
    packed = toy_packed(qwen_processor, model)
    grads = []
    for chunk in (1024, 2):
        model.zero_grad()
        out = value_loss(model.model, model.lm_head, packed, chunk=chunk)
        out.loss.backward()
        grads.append((out.loss.detach(), model.lm_head.weight.grad.clone(), out.token_accuracy, out.close_accuracy))
    torch.testing.assert_close(grads[0][0], grads[1][0])
    torch.testing.assert_close(grads[0][1], grads[1][1])
    assert grads[0][2:] == grads[1][2:]
    reference = torch.nn.functional.cross_entropy(
        model.lm_head(model.model(**packed.inputs).last_hidden_state[0, packed.predict_positions]), packed.targets)
    torch.testing.assert_close(grads[0][0], reference.detach())


# ------------------------------------------------------------------ training


def test_resume_matches_an_uninterrupted_run(prepared: Path, tmp_path: Path, qwen_processor: Any) -> None:
    straight = make_config(prepared, tmp_path / 'a', save_every=2)
    run(straight, qwen_processor)
    resumed = make_config(prepared, tmp_path / 'b', save_every=2)
    assert run(resumed, qwen_processor, until=3) is None  # snapshot at step 2, step 3 lost
    assert read_json(tmp_path / 'b' / 'latest_training.json')['step'] == 2
    summary = run(resumed, qwen_processor)
    assert summary is not None and summary['resumed_from_step'] == 2

    a, b = read_jsonl(tmp_path / 'a' / 'train.jsonl'), read_jsonl(tmp_path / 'b' / 'train.jsonl')
    assert [r['step'] for r in b] == [1, 2, 3, 4]
    assert [(r['document_id'], r['loss']) for r in a] == [(r['document_id'], r['loss']) for r in b]
    for name, weight in adapter(tmp_path / 'a').items():
        torch.testing.assert_close(weight, adapter(tmp_path / 'b')[name], atol=0, rtol=0)


def test_existing_run_without_snapshot_is_not_overwritten(prepared: Path, tmp_path: Path,
                                                          qwen_processor: Any) -> None:
    (tmp_path / 'train.jsonl').write_text('{"step": 1}\n')
    with pytest.raises(ValueError, match='without a resumable snapshot'):
        Trainer(make_config(prepared, tmp_path), tiny_model(), qwen_processor)


def test_resume_refuses_a_changed_config(prepared: Path, tmp_path: Path, qwen_processor: Any) -> None:
    run(make_config(prepared, tmp_path, steps=2, save_every=1), qwen_processor, until=1)
    with pytest.raises(ValueError, match=r"differs.*\['lr'\]"):
        Trainer(make_config(prepared, tmp_path, steps=2, save_every=1, lr=1e-3), tiny_model(), qwen_processor)


def test_partial_accumulation_is_a_mean_over_its_steps(prepared: Path, tmp_path: Path, qwen_processor: Any) -> None:
    # steps=1 with accumulation 2 ends on a partial update of one step: it must
    # equal a plain single-step update, not half of it.
    run(make_config(prepared, tmp_path / 'one', steps=1), qwen_processor)
    run(make_config(prepared, tmp_path / 'partial', steps=1, gradient_accumulation=2), qwen_processor)
    for name, weight in adapter(tmp_path / 'one').items():
        torch.testing.assert_close(weight, adapter(tmp_path / 'partial')[name], atol=1e-7, rtol=1e-6)
    run(make_config(prepared, tmp_path / 'odd', steps=3, gradient_accumulation=2), qwen_processor)
    updates = [r['step'] for r in read_jsonl(tmp_path / 'odd' / 'train.jsonl') if 'grad_norm' in r]
    assert updates == [2, 3]


def test_trainable_markers_change_only_the_marker_rows(prepared: Path, tmp_path: Path, qwen_processor: Any) -> None:
    # Qwen2-VL-2B ties lm_head to the input embedding, so the tiny model does too:
    # the marker rows must change in both, and nothing else. (PEFT 0.18.1 wraps
    # lm_head with the embedding's delta whether or not the model is tied, so K6
    # is only meaningful for tied models.)
    base = tiny_model()
    base.config.tie_word_embeddings = base.config.text_config.tie_word_embeddings = True
    base.tie_weights()
    trainer = Trainer(make_config(prepared, tmp_path, steps=2, trainable_markers=True), base, qwen_processor)
    m = trainer.extractor.format.markers
    markers = sorted([m.key_open, m.key_close, m.value_open, m.value_close])
    embeddings, head = trainer.model.get_input_embeddings(), trainer.extractor.lm_head
    probe = torch.eye(HIDDEN)

    def snapshot() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return embeddings(torch.arange(VOCAB)).clone(), head(probe).clone()

    rows_before, logits_before = snapshot()
    trainer.run()
    rows_after, logits_after = snapshot()
    assert (rows_before != rows_after).any(-1).nonzero().flatten().tolist() == markers
    assert (logits_before != logits_after).any(0).nonzero().flatten().tolist() == markers
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad and 'lora_' not in n]
    assert trainable == ['base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta.default']


def test_monitor_logs_decoding_metrics(prepared: Path, tmp_path: Path, qwen_processor: Any) -> None:
    config = make_config(prepared, tmp_path, steps=2, monitor={'every': 2, 'splits': ['train', 'dev'],
                                                              'documents': 1, 'fields': 2, 'max_value_tokens': 4})
    summary = run(config, qwen_processor)
    records = read_jsonl(tmp_path / 'monitor.jsonl')
    assert [(r['step'], r['split']) for r in records] == [(0, 'train'), (0, 'dev'), (2, 'train'), (2, 'dev')]
    assert all(r['fields'] == 2 and 0 <= r['coordinate_rate'] <= 1 for r in records)
    assert summary is not None and len(summary['monitor_last']) == 2
    assert (tmp_path / 'adapter' / 'adapter_config.json').exists()
    assert TrainConfig.from_dict(read_json(tmp_path / 'config.json')) == config


# ------------------------------------------------------------------ helpers


def test_comparator_and_coordinates() -> None:
    assert correct('Nguyễn Văn A ', 'Nguyễn Văn A') and not correct('nguyễn văn a', 'Nguyễn Văn A')
    assert correct(['a', ' b'], ['a', 'b']) and not correct('true', True)
    assert is_coordinates('(295,434),(526,492)') and not is_coordinates('(295,434)') and not is_coordinates('12/03')
