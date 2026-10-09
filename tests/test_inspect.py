"""Tests for ``l2r4kie.pipelines.inspect`` (token views, no model weights)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from l2r4kie.data.selection import Selection
from l2r4kie.data.types import Document, FieldSpec
from l2r4kie.model.format import KevFormat
from l2r4kie.model.markers import special_ids
from l2r4kie.pipelines.inspect import find_document, render_ids, token_stats
from l2r4kie.utils.io import write_jsonl


@pytest.fixture
def prepared(tmp_path: Path) -> Path:
    fields = (FieldSpec('/name', 'Họ tên', 'Vũ Bảo Mai Linh', 'scalar'),
              FieldSpec('/rows', 'Các dòng', [{'a': 'x' * 2000}], 'array'),
              FieldSpec('/ok', '/ok', True, 'scalar'))
    for split in ('train', 'dev', 'calibration', 'test'):
        docs = [Document('f__1', 'f__1', 'f', (), fields, ())] if split == 'dev' else []
        write_jsonl(tmp_path / f'{split}.jsonl', (d.to_json() for d in docs))
    return tmp_path


def test_find_document(prepared: Path) -> None:
    split, document = find_document(prepared, 'f__1')
    assert split == 'dev' and document.fields[0].value == 'Vũ Bảo Mai Linh'
    with pytest.raises(KeyError):
        find_document(prepared, 'missing')


def test_render_ids_collapses_repeated_specials(qwen_tokenizer: Any) -> None:
    pad = qwen_tokenizer.convert_tokens_to_ids('<|image_pad|>')
    text = qwen_tokenizer.encode('Họ tên', add_special_tokens=False)
    rendered = render_ids(qwen_tokenizer, [*text, pad, pad, pad, *text], special_ids(qwen_tokenizer, len(qwen_tokenizer)))
    assert rendered == 'Họ tên<|image_pad|>×3Họ tên'


def test_token_stats_counts_untrainable_arrays(qwen_tokenizer: Any, prepared: Path) -> None:
    stats = token_stats(KevFormat(qwen_tokenizer), Selection(prepared), 'dev', max_value_tokens=256)
    assert stats['documents'] == 1
    assert stats['target_tokens']['array']['over_max_value_tokens'] == 1
    scalar = stats['target_tokens']['scalar']
    assert (scalar['fields'], scalar['over_max_value_tokens']) == (2, 0)
