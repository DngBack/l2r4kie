"""Tests for ``l2r4kie.eval.comparator`` and ``l2r4kie.eval.errors``."""

from __future__ import annotations

from typing import Any

import pytest

from l2r4kie.eval.comparator import as_text, is_correct, json_correct, text_correct
from l2r4kie.eval.errors import error_kind, generated_text, is_coordinates


def row(target: Any, prediction: Any, status: str = 'ok', **extra: Any) -> dict[str, Any]:
    return {'document_id': 'f__1', 'field_id': '/a', 'target': target, 'prediction': prediction,
            'status': status, **extra}


@pytest.mark.parametrize(('prediction', 'target', 'text', 'json'), [
    ('abc', 'abc', True, True),
    (' abc ', 'abc', True, True),                       # surrounding whitespace
    ('Cù', 'Cù', True, True),                # NFD vs NFC
    ('ABC', 'abc', False, False),                       # case matters
    ('012', '12', False, False),                        # leading zeros matter
    ('123', 123, True, False),                          # number written as text
    ('true', True, True, False),
    ('', None, True, False),                            # blank field
    ('1.5', 1.5, True, False),
    (['a', 'b'], ['a', 'b'], True, True),
    (['b', 'a'], ['a', 'b'], False, False),             # order matters
    ('a', ['a'], False, False),                         # arrays must be arrays
    ([{'x': '1', 'y': 'b'}], [{'y': 'b', 'x': 1}], True, False),
])
def test_text_and_json_comparators(prediction: Any, target: Any, text: bool, json: bool) -> None:
    assert text_correct(prediction, target) is text
    assert json_correct(prediction, target) is json


def test_as_text_writes_scalars_like_kevformat() -> None:
    assert [as_text(v) for v in ('x', 1, 1.5, True, False, None)] == ['x', '1', '1.5', 'true', 'false', '']
    assert as_text('Nguyễn') == 'Nguyễn'


def test_a_value_that_did_not_finish_is_wrong_under_both() -> None:
    for comparator in ('text', 'json'):
        assert is_correct(row('x', 'x'), comparator)
        assert not is_correct(row('x', 'x', 'truncated'), comparator)


@pytest.mark.parametrize(('text', 'expected'), [
    ('(295,434),(526,492)', True),
    (' ( 1 , 2 ) , ( 3 , 4 ) ', True),
    ('(1,2)', False),
    ('(1,2),(3,4) x', False),
    ('295', False),
])
def test_is_coordinates(text: str, expected: bool) -> None:
    assert is_coordinates(text) is expected


def test_generated_text_reads_new_and_old_rows() -> None:
    assert generated_text({'text': 'abc'}) == 'abc'
    assert generated_text({'raw': '"(1,2),(3,4)"'}) == '(1,2),(3,4)'
    assert generated_text({'raw': None}) == ''


@pytest.mark.parametrize(('target', 'prediction', 'status', 'text', 'kind'), [
    ('Hà Nội', None, 'ok', '(1,2),(3,4)', 'coordinates'),       # checked before the status
    ('x', None, 'truncated', 'x x x', 'truncated'),
    ([{'a': '1'}], 'x', 'ok', 'x', 'wrong_json_type'),
    (['a'], ['b'], 'ok', '', 'array_content'),
    ('', 'x', 'ok', 'x', 'spurious_nonempty'),
    ('x', '', 'ok', '', 'missing_value'),
    ('Nữ', 'NỮ', 'ok', 'NỮ', 'case_only'),
    ('Hà  Nội', 'Hà Nội', 'ok', 'Hà Nội', 'internal_whitespace'),
    ('Hà Nội', 'Ha Noi', 'ok', 'Ha Noi', 'diacritics_or_case'),
    ('12/03/2020', '13/03/2020', 'ok', '13/03/2020', 'digit_substitution'),
    ('Nữ', 'N', 'ok', 'N', 'other_content'),
])
def test_error_kind(target: Any, prediction: Any, status: str, text: str, kind: str) -> None:
    assert error_kind(row(target, prediction, status, text=text)) == kind
