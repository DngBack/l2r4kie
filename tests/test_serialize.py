"""Tests for ``l2r4kie.data.serialize``: value <-> plain text."""

from __future__ import annotations

from typing import Any

import pytest

from l2r4kie.data.serialize import from_text, to_text
from l2r4kie.data.types import FieldKind


@pytest.mark.parametrize(('value', 'text', 'kind'), [
    ('NGUYỄN VĂN A', 'NGUYỄN VĂN A', 'scalar'),
    ('0123', '0123', 'scalar'),                       # leading zeros kept
    ('dòng 1\ndòng 2', 'dòng 1\ndòng 2', 'scalar'),   # newlines kept
    (' a ', ' a ', 'scalar'),                         # no stripping
    ('', '', 'scalar'),
    (True, 'true', 'scalar'),
    (False, 'false', 'scalar'),
    ([{'Ngày/tháng': '18.12'}, {'Ngày/tháng': ''}], '[{"Ngày/tháng":"18.12"},{"Ngày/tháng":""}]', 'array'),
    (['a', 'b'], '["a","b"]', 'array'),
    ([], '[]', 'array'),
])
def test_round_trip(value: Any, text: str, kind: FieldKind) -> None:
    assert to_text(value) == text
    assert from_text(text, kind) == value


@pytest.mark.parametrize('value', [None, 3, 1.5, {'a': 1}])
def test_unsupported_values_are_rejected(value: Any) -> None:
    with pytest.raises(TypeError):
        to_text(value)


@pytest.mark.parametrize('text', ['[{"a":1}', '{"a":1}', 'plain'])
def test_invalid_array_text(text: str) -> None:
    with pytest.raises(ValueError):
        from_text(text, 'array')


def test_scalar_text_that_looks_like_json_stays_a_string() -> None:
    assert from_text('[1, 2]', 'scalar') == '[1, 2]'
    assert from_text('True', 'scalar') == 'True'
