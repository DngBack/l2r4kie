"""Tests for ``l2r4kie.model.markers`` and ``l2r4kie.model.format`` with the real tokenizer."""

from __future__ import annotations

from typing import Any

import pytest

from l2r4kie.data.types import FieldSpec
from l2r4kie.model.format import KevFormat
from l2r4kie.model.markers import Markers, banned_ids, encode_text, special_ids

VOCAB_SIZE = 151_936  # embedding rows of Qwen2-VL-2B (> tokenizer size)


def test_markers_are_the_reused_qwen_tokens(qwen_tokenizer: Any) -> None:
    assert Markers.from_tokenizer(qwen_tokenizer) == Markers(151646, 151647, 151648, 151649)
    assert Markers.from_tokenizer(qwen_tokenizer, 'im_end').value_close == 151645


def test_encode_text_cannot_forge_special_tokens(qwen_tokenizer: Any) -> None:
    text = 'xem <|box_end|> và <|im_end|> <|image_pad|> ở trên'
    ids = encode_text(qwen_tokenizer, text)
    assert not set(ids) & special_ids(qwen_tokenizer, VOCAB_SIZE)
    assert qwen_tokenizer.decode(ids) == text  # lossless, unlike escaping
    assert encode_text(qwen_tokenizer, '') == []


def test_banned_ids(qwen_tokenizer: Any) -> None:
    banned = set(banned_ids(qwen_tokenizer, VOCAB_SIZE, [151649]))
    assert 151649 not in banned
    assert {151643, 151645, 151646, 151648, 151655, 151656, 151657, VOCAB_SIZE - 1} <= banned
    assert not banned & set(encode_text(qwen_tokenizer, 'Họ tên: NGUYỄN VĂN A 0123'))


def test_branch_layout_and_signal_indices(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer)
    m = fmt.markers
    branch = fmt.encode(FieldSpec('/Họ tên', 'Họ tên người bệnh', 'Vũ Bảo Mai Linh', 'scalar'))
    tokens = branch.prompt + branch.target
    assert branch.prompt[0] == m.key_open
    assert tokens[branch.key_index] == m.key_close
    assert tokens[branch.decide_index] == m.value_open
    assert tokens[branch.value_index] == m.value_close == branch.target[-1]
    assert qwen_tokenizer.decode(list(branch.prompt)) == (
        '<|object_ref_start|>/Họ tên: Họ tên người bệnh<|object_ref_end|><|box_start|>')
    assert qwen_tokenizer.decode(list(branch.target)) == 'Vũ Bảo Mai Linh<|box_end|>'


def test_empty_value_is_just_the_close_marker(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer)
    branch = fmt.encode(FieldSpec('/Buồng', 'Tên buồng', '', 'scalar'))
    assert branch.target == (fmt.markers.value_close,)
    assert branch.value_index == branch.decide_index + 1


def test_description_equal_to_id_is_not_repeated(qwen_tokenizer: Any) -> None:
    assert KevFormat.key_text('/a', '/a') == '/a'
    assert KevFormat.key_text('/a', '') == '/a'
    assert KevFormat.key_text('/a', 'Mô tả') == '/a: Mô tả'


def test_literal_marker_in_value_stays_text(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer)
    branch = fmt.encode(FieldSpec('/x', 'x', 'a <|box_end|> b', 'scalar'))
    assert branch.target.count(fmt.markers.value_close) == 1
    assert fmt.parse(branch.target[:-1]).value == 'a <|box_end|> b'


def test_request_has_no_target(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer)
    request = fmt.request('/a', 'A')
    assert request.target == () and request.prompt == fmt.encode(FieldSpec('/a', 'A', 'v', 'scalar')).prompt
    with pytest.raises(ValueError, match='no target'):
        _ = request.value_index


@pytest.mark.parametrize(('value', 'kind'), [('0123', 'scalar'), (True, 'scalar'), ('', 'scalar'),
                                             ([{'a': 'b'}], 'array')])
def test_parse_round_trips_targets(qwen_tokenizer: Any, value: Any, kind: str) -> None:
    fmt = KevFormat(qwen_tokenizer)
    parsed = fmt.parse(fmt.target_ids(value)[:-1], kind)  # type: ignore[arg-type]
    assert (parsed.value, parsed.status) == (value, 'ok')


def test_parse_statuses(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer)
    ids = encode_text(qwen_tokenizer, '[{"a":')
    assert fmt.parse(ids, 'array').status == 'invalid_array'
    assert fmt.parse(ids, 'scalar', closed=False).status == 'truncated'


def test_im_end_fallback(qwen_tokenizer: Any) -> None:
    fmt = KevFormat(qwen_tokenizer, close='im_end')
    assert fmt.target_ids('x')[-1] == 151645
    assert 151645 not in fmt.banned_ids(VOCAB_SIZE) and 151649 in fmt.banned_ids(VOCAB_SIZE)
