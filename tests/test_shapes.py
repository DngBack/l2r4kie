"""Tests for value-shape rules (``l2r4kie.confidence.shapes``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from l2r4kie.confidence.shapes import ShapeRules, shape


def document(**values: object) -> SimpleNamespace:
    return SimpleNamespace(fields=[SimpleNamespace(id=k, kind='array' if isinstance(v, list) else 'scalar', value=v)
                                   for k, v in values.items()])


def test_shape_maps_digits_and_letters() -> None:
    assert shape('12/03/2024') == '99/99/9999'
    assert shape('Số 5, Đường B') == 'aa 9, aaaaa a'
    assert shape('3_x') == '9_a'


def test_fit_keeps_stable_fields_only() -> None:
    documents = [document(**{'/date': f'{i % 28 + 1:02d}/01/2024', '/name': f'Lô {i}' * (1 + i % 7),
                             '/note': 'no digits', '/list': [str(i)]}) for i in range(40)]
    rules = ShapeRules.fit(documents)
    assert rules.allowed == {'/date': frozenset({'99/99/9999'})}  # /name has too many shapes
    assert ShapeRules.fit(documents[:19]).allowed == {}  # too few examples
    assert rules.violates('/date', 'scalar', '1/01/2024')
    assert not rules.violates('/date', 'scalar', '07/01/2024')
    assert not rules.violates('/date', 'scalar', 'không rõ')  # no digit: not judged
    assert not rules.violates('/date', 'scalar', '1/01/2024', 'truncated')
    assert not rules.violates('/date', 'array', ['1'])
    assert not rules.violates('/other', 'scalar', '1')


def test_rare_shapes_are_dropped_but_must_leave_coverage() -> None:
    common = [document(**{'/a': '12-34'}) for _ in range(97)]
    rare = [document(**{'/a': '1234'}) for _ in range(1)] + [document(**{'/a': '12 34'}) for _ in range(2)]
    rules = ShapeRules.fit(common + rare)
    assert rules.allowed['/a'] == frozenset({'99-99', '99 99'})  # 2% kept, 1% dropped
    assert ShapeRules.fit(common + rare, coverage=.999).allowed == {}


def test_rules_round_trip_and_flag_rows() -> None:
    rules = ShapeRules({'/a': frozenset({'99'})}, source='test')
    assert ShapeRules.from_dict(rules.to_dict()) == rules
    rows = [{'field_id': '/a', 'kind': 'scalar', 'prediction': '123', 'status': 'ok'},
            {'field_id': '/a', 'kind': 'scalar', 'prediction': '12', 'status': 'ok'}]
    assert [r['shape_violation'] for r in rules.flag_rows(rows)] == [True, False]
    assert 'shape_violation' not in rows[0]
    with pytest.raises(ValueError):
        ShapeRules.fit([], min_share=0)
