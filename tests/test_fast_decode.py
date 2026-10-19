"""Tests for ``l2r4kie.model.fast_decode`` on the tiny Qwen2-VL (CPU, FP32: the static step runs eagerly).

In FP32 the fast path must give the eager decode's tokens, statuses, stops,
signals and traces: the static-cache tail (masks, positions, cache copies,
bucket growth) and the teacher-forced trace pass are exact re-implementations.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from test_decode import constant_head, make_extractor, overfit, requests, text_prefix
from tiny import tiny_model

from l2r4kie.data.types import FieldRequest
from l2r4kie.model import fast_decode
from l2r4kie.model.decode import DecodeResult, decode_prefix


@pytest.fixture(scope='module')
def processor(qwen_processor):  # noqa: ANN001, ANN201
    return qwen_processor


def fast(extractor):  # noqa: ANN001, ANN201
    clone = make_extractor(extractor.processor, extractor.model)
    clone.config = dataclasses.replace(extractor.config, fast_decode=True)
    return clone


def assert_same(eager: list[DecodeResult], quick: list[DecodeResult], trace: bool = True) -> None:
    assert [(r.text, r.status, r.stop) for r in quick] == [(r.text, r.status, r.stop) for r in eager]
    for a, b in zip(eager, quick, strict=True):
        for name in ('key', 'decide', 'value', 'layers'):
            x, y = getattr(a.signals, name), getattr(b.signals, name)
            assert (x is None) == (y is None), name
            if x is not None:
                torch.testing.assert_close(y, x, atol=1e-5, rtol=0, equal_nan=True)
        if trace:
            assert b.trace.tokens == a.trace.tokens
            torch.testing.assert_close(b.trace.hidden, a.trace.hidden, atol=1e-5, rtol=0)
            torch.testing.assert_close(b.trace.stats, a.trace.stats, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(('tail_rows', 'min_cache'), [(4, 4096), (1, 16), (0, 16)])
def test_fast_decode_equals_eager_decode(processor, monkeypatch, tail_rows: int, min_cache: int) -> None:  # noqa: ANN001
    """All tail, eager then a growing tail (several cache buckets), and eager only."""
    monkeypatch.setattr(fast_decode, 'TAIL_ROWS', tail_rows)
    monkeypatch.setattr(fast_decode, 'MIN_CACHE', min_cache)
    monkeypatch.setattr(fast_decode, 'RESERVE', 2 if min_cache == 16 else fast_decode.RESERVE)
    extractor = make_extractor(processor, max_branches=2)
    prefix = text_prefix(extractor)
    values = {'/ho_ten': 'A', '/ngay_sinh': '01/02/1990', '/dia_chi': 'Số 12 đường Láng, Đống Đa, Hà Nội'}
    overfit(extractor, prefix, values, steps=200)
    eager = decode_prefix(extractor, prefix, requests(), max_value_tokens=48, trace=True, layers=(1, 2))
    assert {r.field_id: r.value for r in eager} == values
    quick = fast(extractor)
    assert_same(eager, decode_prefix(quick, prefix, requests(), max_value_tokens=48, trace=True, layers=(1, 2)))
    shapes = set(getattr(quick, '_static_graphs', {}))
    if tail_rows == 0:
        assert not shapes
    else:  # with a 2-slot reserve, the long value outgrows its first bucket
        assert len({length for _, length in shapes}) >= (2 if min_cache == 16 else 1)


def test_warm_up_builds_the_buckets_decoding_uses(processor, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(fast_decode, 'MIN_CACHE', 16)
    monkeypatch.setattr(fast_decode, 'RESERVE', 2)
    extractor = make_extractor(processor, max_branches=2)
    prefix = text_prefix(extractor)
    eager = decode_prefix(extractor, prefix, requests(), max_value_tokens=24, trace=True)
    quick = fast(extractor)
    longest = prefix['input_ids'].shape[1] + 256  # + branch prompt + budget
    shapes = fast_decode.warm_up(quick, longest)
    assert {rows for rows, _ in shapes} == {1, 2, 4}
    assert min(length for _, length in shapes) == 16 and max(length for _, length in shapes) >= longest
    assert_same(eager, decode_prefix(quick, prefix, requests(), max_value_tokens=24, trace=True))
    assert set(quick._static_graphs) == set(shapes)  # no graph built while decoding


def test_fast_decode_keeps_budget_and_early_stop(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor)
    x = extractor.format.tokenizer('x', add_special_tokens=False)['input_ids'][0]
    constant_head(extractor, x)
    fields = [FieldRequest('/table', 'Bảng', 'array'), FieldRequest('/name', 'Tên')]
    prefix = text_prefix(extractor)
    for early_stop in (True, False):
        eager = decode_prefix(extractor, prefix, fields, max_value_tokens=150, trace=True, early_stop=early_stop)
        assert_same(eager, decode_prefix(fast(extractor), prefix, fields, max_value_tokens=150, trace=True,
                                         early_stop=early_stop))


def test_fast_decode_without_trace_on_a_random_model(processor) -> None:  # noqa: ANN001
    extractor = make_extractor(processor, tiny_model(seed=3))
    prefix = text_prefix(extractor)
    eager = decode_prefix(extractor, prefix, requests(), max_value_tokens=24)
    quick = decode_prefix(fast(extractor), prefix, requests(), max_value_tokens=24)
    assert_same(eager, quick, trace=False)
    assert all(r.trace is None for r in quick)
