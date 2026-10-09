"""Tests for ``l2r4kie.utils``: atomic IO, fingerprints, config overrides, seeding."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from l2r4kie.cli import main
from l2r4kie.utils.config import apply_overrides, load_config, parse_override
from l2r4kie.utils.fingerprint import checkpoint_fingerprint, hash_bucket, sha256_file
from l2r4kie.utils.io import (
    append_jsonl, atomic_path, read_json, read_jsonl, write_json, write_jsonl, write_text,
)

#: Winner checkpoint of the old repository and its fingerprint as recorded in
#: ``artifacts/token-review-v2/selected/frozen_selection.json``.
R4_CHECKPOINT = Path('/home/jovyan/bachdx2/l2r4kie/artifacts/training-optimization/r4_2m_12f')
R4_FINGERPRINT = '596bba7c8c85ec86c31c14fe907fd0888d88841f3bffef85537ce9e08c5c62e3'


# --------------------------------------------------------------------------- io

def test_json_round_trip_keeps_unicode(tmp_path: Path) -> None:
    value = {'họ_tên': 'NGUYỄN VĂN A', 'n': [1, 2]}
    path = write_json(tmp_path / 'nested' / 'x.json', value)
    assert read_json(path) == value
    assert 'NGUYỄN' in path.read_text(encoding='utf-8')  # not \u-escaped
    assert path.read_text(encoding='utf-8').endswith('\n')


def test_jsonl_round_trip_and_blank_lines(tmp_path: Path) -> None:
    rows = [{'id': 1}, {'id': 2, 'v': 'a\nb'}]
    path = write_jsonl(tmp_path / 'x.jsonl', rows)
    path.write_text(path.read_text() + '\n\n', encoding='utf-8')  # trailing blank lines are ignored
    assert read_jsonl(path) == rows


def test_append_jsonl(tmp_path: Path) -> None:
    path = tmp_path / 'log' / 'train.jsonl'
    append_jsonl(path, {'step': 1})
    append_jsonl(path, {'step': 2})
    assert read_jsonl(path) == [{'step': 1}, {'step': 2}]


def test_atomic_path_failure_keeps_previous_file(tmp_path: Path) -> None:
    target = write_text(tmp_path / 'a.txt', 'old')
    with pytest.raises(RuntimeError), atomic_path(target) as temporary:
        temporary.write_text('half-written')
        raise RuntimeError('crash mid-write')
    assert target.read_text() == 'old'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a.txt']  # no temp file left behind


def test_atomic_path_failure_without_previous_file(tmp_path: Path) -> None:
    target = tmp_path / 'never.txt'
    with pytest.raises(RuntimeError), atomic_path(target):
        raise RuntimeError
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ fingerprint

def _reference_fingerprint(checkpoint: Path) -> str:
    """Verbatim copy of the old ``data.checkpoint_fingerprint`` (commit 0428235)."""
    root = Path(checkpoint)
    digest = hashlib.sha256()
    files = list((root / 'adapter').glob('*'))
    files += [root / name for name in ('head.pt', 'head_config.json', 'config.json', 'calibration.json')]
    for path in sorted(p for p in files if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fake_checkpoint(root: Path) -> Path:
    (root / 'adapter').mkdir(parents=True)
    (root / 'adapter' / 'adapter_model.safetensors').write_bytes(b'\x00weights')
    (root / 'adapter' / 'adapter_config.json').write_text('{"r": 16}')
    (root / 'config.json').write_text('{"model": "x"}')
    (root / 'train.jsonl').write_text('ignored: not part of the fingerprint')
    return root


def test_fingerprint_matches_old_algorithm(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint(tmp_path / 'ckpt')
    assert checkpoint_fingerprint(checkpoint) == _reference_fingerprint(checkpoint)


def test_fingerprint_ignores_non_checkpoint_files_and_tracks_weights(tmp_path: Path) -> None:
    checkpoint = _fake_checkpoint(tmp_path / 'ckpt')
    before = checkpoint_fingerprint(checkpoint)
    (checkpoint / 'train.jsonl').write_text('changed log')
    assert checkpoint_fingerprint(checkpoint) == before
    (checkpoint / 'adapter' / 'adapter_model.safetensors').write_bytes(b'\x01weights')
    assert checkpoint_fingerprint(checkpoint) != before


def test_fingerprint_missing_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        checkpoint_fingerprint(tmp_path / 'typo')


@pytest.mark.skipif(not R4_CHECKPOINT.is_dir(), reason='old repository artifacts not available')
def test_fingerprint_of_r4_checkpoint() -> None:
    assert checkpoint_fingerprint(R4_CHECKPOINT) == R4_FINGERPRINT


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / 'blob'
    path.write_bytes(b'abc' * 1_000_000)  # spans several read chunks
    assert sha256_file(path) == hashlib.sha256(b'abc' * 1_000_000).hexdigest()


@pytest.mark.parametrize('key', ['42:form__sample-1', 'group-abc', ''])
def test_hash_bucket_matches_old_expression(key: str) -> None:
    old = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    assert hash_bucket(key) == old
    assert 0 <= hash_bucket(key, 7) < 7


# ----------------------------------------------------------------------- config

def test_parse_override_types() -> None:
    assert parse_override('steps=50') == (['steps'], 50)
    assert parse_override('format.close=im_end') == (['format', 'close'], 'im_end')
    assert parse_override('resume=true') == (['resume'], True)
    assert parse_override('forms=[a, b]') == (['forms'], ['a', 'b'])
    assert parse_override('device=cuda:0') == (['device'], 'cuda:0')
    assert parse_override('limit=') == (['limit'], None)


@pytest.mark.parametrize('bad', ['steps', '=1', 'a..b=1', 'a.=1'])
def test_parse_override_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_override(bad)


def test_apply_overrides_is_a_deep_copy() -> None:
    original = {'lr': 1e-4, 'format': {'close': 'box_end'}}
    result = apply_overrides(original, ['format.close=im_end', 'new.nested.key=1'])
    assert result == {'lr': 1e-4, 'format': {'close': 'im_end'}, 'new': {'nested': {'key': 1}}}
    assert original['format']['close'] == 'box_end'


def test_apply_overrides_refuses_path_through_scalar() -> None:
    with pytest.raises(ValueError, match='lr is not a mapping'):
        apply_overrides({'lr': 1e-4}, ['lr.x=1'])


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / 'c.yaml'
    path.write_text('model: Qwen/Qwen2-VL-2B-Instruct\nsteps: 1000\n')
    assert load_config(path, ['steps=30']) == {'model': 'Qwen/Qwen2-VL-2B-Instruct', 'steps': 30}
    (tmp_path / 'empty.yaml').write_text('')
    assert load_config(tmp_path / 'empty.yaml') == {}
    (tmp_path / 'list.yaml').write_text('- a\n')
    with pytest.raises(ValueError, match='mapping'):
        load_config(tmp_path / 'list.yaml')


# ------------------------------------------------------------------------- seed

def test_seed_everything_is_reproducible() -> None:
    import numpy as np
    import torch

    from l2r4kie.utils.seed import seed_everything

    def draw() -> tuple[float, float, list[float]]:
        return random.random(), float(np.random.rand()), torch.rand(3).tolist()

    seed_everything(7)
    first = draw()
    seed_everything(7)
    assert draw() == first


# -------------------------------------------------------------------------- cli

def test_cli_fingerprint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    checkpoint = _fake_checkpoint(tmp_path / 'ckpt')
    assert main(['fingerprint', str(checkpoint)]) == 0
    assert capsys.readouterr().out.split() == [_reference_fingerprint(checkpoint), str(checkpoint)]


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])
