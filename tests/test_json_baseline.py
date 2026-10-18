"""Tests for the JSON baseline format (``l2r4kie.baselines.json_format``)."""

from __future__ import annotations

import json

import pytest

from l2r4kie.baselines import json_format as jf
from l2r4kie.data.types import FieldSpec
from l2r4kie.model.format import KevFormat

FIELDS = [FieldSpec('/a/Tên', 'Họ tên', 'Lưu Tuấn Mai', 'scalar'), FieldSpec('/b', '/b', True, 'scalar'),
          FieldSpec('/rows', 'Bảng', [{'x': '1', 'y': ''}], 'array'), FieldSpec('/n', '', '20.650', 'scalar')]


def test_schema_and_target_text() -> None:
    assert jf.schema_text(FIELDS) == 'JSON:\n/a/Tên: Họ tên\n/b\n/rows: Bảng\n/n'
    target = jf.target_text(FIELDS)
    assert json.loads(target) == {f.id: f.value for f in FIELDS}
    assert ' ' not in target.replace('Lưu Tuấn Mai', '')  # compact


def test_encode_is_one_branch_ending_on_the_close_marker(qwen_tokenizer) -> None:  # noqa: ANN001
    fmt = KevFormat(qwen_tokenizer)
    branch = jf.encode(fmt, FIELDS)
    assert list(branch.prompt) == fmt.prompt_ids(jf.schema_text(FIELDS), '')
    assert branch.target[-1] == fmt.markers.value_close
    assert qwen_tokenizer.decode(branch.target[:-1]) == jf.target_text(FIELDS)


def test_encode_within_drops_the_longest_values(qwen_tokenizer) -> None:  # noqa: ANN001
    fmt = KevFormat(qwen_tokenizer)
    long = FieldSpec('/long', '', 'x ' * 400, 'scalar')
    full = len(jf.encode(fmt, FIELDS).target)
    assert jf.encode_within(fmt, [*FIELDS, long], full + 5) == jf.encode(fmt, FIELDS)
    with pytest.raises(ValueError, match='No field'):
        jf.encode_within(fmt, [long], 10)


def test_groups() -> None:
    assert [len(g) for g in jf.groups(list(range(25)), 12)] == [12, 12, 1]
    with pytest.raises(ValueError, match='positive'):
        jf.groups([1], 0)


def test_parse_members_keeps_what_parsed_before_a_break() -> None:
    whole = jf.target_text(FIELDS)
    assert {k: v for k, (v, _) in jf.parse_members(whole).items()} == {f.id: f.value for f in FIELDS}
    cut = whole[:whole.index('"/rows"') + 12]  # inside the array
    assert list(jf.parse_members(cut)) == ['/a/Tên', '/b']
    assert jf.parse_members(' { "a" : 1.50 , "a": 2, "b": [1,]}') == {'a': (1.5, '1.50')}
    assert jf.parse_members('[1]') == {} == jf.parse_members('') == jf.parse_members('{')
    assert jf.parse_members('{}') == {}


def test_prediction_rows_statuses() -> None:
    text = '{"/a/Tên":"Lưu Tuấn Mai","/b":true,"/rows":"x","/n":20.650}'
    rows = jf.prediction_rows('d', 'f', FIELDS, text, 'ok')
    assert [(r['status'], r['correct']) for r in rows] == [('ok', True), ('ok', True), ('invalid_array', False),
                                                           ('ok', True)]
    assert rows[3]['text'] == '20.650'
    rows = jf.prediction_rows('d', 'f', FIELDS, '{"/a/Tên":["x"],"/rows":[{"x":"1","y":""}]', 'truncated')
    assert [r['status'] for r in rows] == ['wrong_type', 'missing', 'ok', 'missing']
    assert rows[2]['correct'] and rows[2]['request_status'] == 'truncated'
