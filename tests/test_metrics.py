"""Tests for ``l2r4kie.eval.extraction``, ``eval.compare`` and ``eval.metrics``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from l2r4kie.eval.compare import paired_delta
from l2r4kie.eval.extraction import array_rows, score
from l2r4kie.eval.metrics import confidence_metrics


def row(document: str, field: str, target: Any, prediction: Any, status: str = 'ok',
        text: str = '') -> dict[str, Any]:
    return {'document_id': document, 'field_id': field, 'target': target, 'prediction': prediction,
            'status': status, 'text': text}


ROWS = [
    row('a__1', '/x', 'x', 'x'),
    row('a__1', '/y', '1', 1),                                  # right as text, wrong as JSON
    row('a__2', '/x', 'x', None, 'truncated', 'x x x'),
    row('a__2', '/y', 'y', 'y'),
    row('b__1', '/x', 'x', 'q', text='(1,2),(3,4)'),
    row('b__1', '/t', [{'k': 'v'}], [{'k': 'v'}]),
]


def test_score_headline_numbers() -> None:
    metrics = score(ROWS)
    assert (metrics['fields'], metrics['documents'], metrics['forms']) == (6, 3, 2)
    assert metrics['exact_match'] == 4 / 6
    assert metrics['scalar_exact_match'] == 3 / 5
    assert metrics['array_exact_match'] == 1
    assert metrics['macro_form_exact_match'] == (3 / 4 + 1 / 2) / 2
    assert metrics['complete_document_rate'] == 1 / 3
    assert metrics['coordinate_rate'] == 1 / 6
    assert metrics['truncated_rate'] == 1 / 6
    assert metrics['status'] == {'ok': 5, 'truncated': 1}
    assert metrics['errors'] == {'truncated': 1, 'coordinates': 1}
    assert metrics['by_form']['a'] == {'count': 4, 'exact_match': 3 / 4}


def test_json_comparator_rescores_the_old_way() -> None:
    assert score(ROWS, 'json')['exact_match'] == 3 / 6


def test_score_rejects_empty_and_duplicated_rows() -> None:
    with pytest.raises(ValueError, match='No prediction'):
        score([])
    with pytest.raises(ValueError, match='Duplicated'):
        score([ROWS[0], ROWS[0]])


def test_array_rows_gives_partial_credit() -> None:
    target = [{'a': '1', 'b': '2'}, {'a': '3', 'b': '4'}]
    metrics = array_rows([
        row('d', '/t', target, [{'a': '3', 'b': '4'}, {'a': '1', 'b': '9'}]),   # one row right, swapped order
        row('d', '/u', ['p', 'q'], None, 'truncated'),                          # counts as empty
    ])
    assert metrics['fields'] == 2
    assert metrics['length_match'] == 1 / 2
    assert metrics['row_precision'] == 1 / 2
    assert metrics['row_recall'] == 1 / 4
    assert metrics['row_f1'] == pytest.approx(2 * 1 / (2 + 4))
    assert metrics['cell_accuracy'] == 0 / 6


def test_paired_delta_compares_the_same_fields() -> None:
    baseline = [row(f'd{d}', f'/{f}', 'x', 'x' if (d + f) % 3 else 'y') for d in range(10) for f in range(4)]
    candidate = [dict(r, prediction='x') for r in baseline]
    result = paired_delta(candidate, baseline, draws=500)
    assert result['candidate_exact_match'] == 1
    assert result['delta'] == pytest.approx(1 - result['baseline_exact_match'])
    low, high = result['document_bootstrap_95_ci']
    assert 0 < low <= result['delta'] <= high
    assert paired_delta(candidate, baseline, draws=500) == result               # seeded
    assert paired_delta(baseline, baseline, draws=100)['document_bootstrap_95_ci'] == [0, 0]
    with pytest.raises(ValueError, match='differ'):
        paired_delta(candidate[1:], baseline)


def test_paired_delta_restricted_to_scalar_fields() -> None:
    result = paired_delta(ROWS, ROWS, draws=10, kinds=['scalar'])
    assert result['fields'] == 5 and result['kinds'] == ['scalar']


def reference_metrics(logits: list[float], labels: list[int]) -> dict[str, Any]:
    """The old torch ``application.metrics``, copied verbatim (reformatted)."""
    p = torch.sigmoid(torch.tensor(logits, dtype=torch.float64))
    y = torch.tensor(labels, dtype=torch.float64)
    ece = 0.
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
        if mask.any():
            ece += float(mask.double().mean() * abs(p[mask].mean() - y[mask].mean()))
    positive, negative = p[y == 1], p[y == 0]
    if len(positive) and len(negative):
        ordered = negative.sort().values
        below = torch.searchsorted(ordered, positive, right=False)
        below_or_equal = torch.searchsorted(ordered, positive, right=True)
        auroc = float(((below + below_or_equal).double() / (2 * len(negative))).mean())
    else:
        auroc = None
    acceptance = {}
    for threshold in (.5, .8, .9, .95):
        accepted = p >= threshold
        acceptance[str(threshold)] = {'coverage': float(accepted.double().mean()),
                                      'precision': float(y[accepted].mean()) if accepted.any() else None}
    nll = float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits, dtype=torch.float64), y))
    return {'count': len(labels), 'accuracy': float(y.mean()), 'brier': float(((p - y) ** 2).mean()),
            'ece': ece, 'auroc': auroc, 'acceptance': acceptance, 'nll': nll}


def test_confidence_metrics_equal_the_old_torch_ones() -> None:
    rng = np.random.default_rng(0)
    logits = (rng.normal(size=500) * 4).round(1).tolist()                      # rounded: ties
    labels = (rng.random(500) < 0.7).astype(int).tolist()
    new, old = confidence_metrics(logits, labels), reference_metrics(logits, labels)
    assert new.keys() == old.keys()
    for key in new:
        if isinstance(new[key], float):
            assert new[key] == pytest.approx(old[key], rel=1e-12, abs=1e-15), key
        else:
            assert new[key] == old[key], key
    assert confidence_metrics([50., -50.], [1, 0])['nll'] == pytest.approx(reference_metrics([50., -50.], [1, 0])['nll'])


def test_confidence_metrics_edge_cases() -> None:
    assert confidence_metrics([], []) == {'count': 0}
    assert confidence_metrics([0, 0], [0, 1])['auroc'] == .5
    assert confidence_metrics([-1, 1], [0, 1])['auroc'] == 1.
    assert confidence_metrics([1, -1], [0, 1])['auroc'] == 0.
    assert confidence_metrics([1, 1], [1, 1])['auroc'] is None
    assert confidence_metrics([-5.], [0])['acceptance']['0.5'] == {'coverage': 0., 'precision': None}
