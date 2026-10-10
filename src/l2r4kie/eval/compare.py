"""Paired comparison of two prediction sets on the same fields.

The bootstrap resamples *documents*, not fields: fields of one document share
the page images and are correlated, so a field-level bootstrap would give
intervals that are too narrow.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .comparator import Comparator, is_correct


def paired_delta(candidate: Sequence[dict[str, Any]], baseline: Sequence[dict[str, Any]],
                 comparator: Comparator = 'text', draws: int = 10_000, seed: int = 42,
                 kinds: Sequence[str] | None = None) -> dict[str, Any]:
    """EM of ``candidate`` minus EM of ``baseline``, with a document bootstrap 95% CI.

    Args:
        candidate: Prediction rows of the new system.
        baseline: Prediction rows of the reference system.
        comparator: How rows are scored (rows are re-scored, any stored
            ``correct`` is ignored).
        draws: Bootstrap resamples.
        seed: Bootstrap seed.
        kinds: Restrict to these field kinds (e.g. ``['scalar']`` for gate G1).

    Raises:
        ValueError: If the two sets do not cover exactly the same unique
            (document, field) pairs.
    """
    from .extraction import row_kind

    def flags(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str], int]:
        chosen = [r for r in rows if kinds is None or row_kind(r) in kinds]
        result = {(r['document_id'], r['field_id']): int(is_correct(r, comparator)) for r in chosen}
        if len(result) != len(chosen):
            raise ValueError('Duplicated (document_id, field_id) rows')
        return result

    a, b = flags(candidate), flags(baseline)
    if a.keys() != b.keys():
        only_a, only_b = len(a.keys() - b.keys()), len(b.keys() - a.keys())
        raise ValueError(f'Prediction sets differ: {only_a} pairs only in candidate, {only_b} only in baseline')
    if not a:
        raise ValueError('No rows to compare')
    differences: dict[str, list[int]] = defaultdict(list)
    for key in sorted(a):
        differences[key[0]].append(a[key] - b[key])
    sums = np.array([sum(v) for v in differences.values()])
    counts = np.array([len(v) for v in differences.values()])
    indices = np.random.default_rng(seed).integers(0, len(differences), (draws, len(differences)))
    samples = sums[indices].sum(1) / counts[indices].sum(1)
    return {'comparator': comparator, 'kinds': list(kinds) if kinds else 'all', 'fields': len(a),
            'documents': len(differences), 'candidate_exact_match': sum(a.values()) / len(a),
            'baseline_exact_match': sum(b.values()) / len(b), 'delta': (sum(a.values()) - sum(b.values())) / len(a),
            'document_bootstrap_95_ci': np.quantile(samples, [0.025, 0.975]).tolist(), 'draws': draws, 'seed': seed}
