"""Confidence metrics of binary correctness predictions (old ``application.metrics``).

NumPy only: the old version needed torch (and through its module,
transformers) just to compute these numbers. Values are identical (float64).
Used from step 6 on, to score confidence heads and calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

#: Acceptance thresholds reported by :func:`confidence_metrics`.
THRESHOLDS: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-width bins; the last bin includes 1.0."""
    ece = 0.0
    for i in range(bins):
        upper = p <= 1 if i == bins - 1 else p < (i + 1) / bins
        mask = (p >= i / bins) & upper
        if mask.any():
            ece += float(mask.mean() * abs(p[mask].mean() - y[mask].mean()))
    return ece


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Area under the ROC curve (ties count half); ``None`` without both classes."""
    positive, negative = p[y == 1], np.sort(p[y == 0])
    if not len(positive) or not len(negative):
        return None
    below = np.searchsorted(negative, positive, side='left')
    below_or_equal = np.searchsorted(negative, positive, side='right')
    return float(((below + below_or_equal) / (2 * len(negative))).mean())


def confidence_metrics(logits: Sequence[float], labels: Sequence[float]) -> dict[str, Any]:
    """Score confidence logits against 0/1 correctness labels.

    Returns:
        ``count``, ``accuracy`` (mean label), ``brier``, ``ece``, ``auroc``,
        ``nll`` (binary cross-entropy of the logits) and, per threshold in
        :data:`THRESHOLDS`, the ``coverage`` and ``precision`` of accepting
        fields with ``sigmoid(logit) >= threshold``.
    """
    if not len(logits):
        return {'count': 0}
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    p = sigmoid(x)
    acceptance = {}
    for threshold in THRESHOLDS:
        accepted = p >= threshold
        acceptance[str(threshold)] = {'coverage': float(accepted.mean()),
                                      'precision': float(y[accepted].mean()) if accepted.any() else None}
    # BCE with logits, stable form: max(x, 0) - x * y + log(1 + exp(-|x|)).
    nll = float((np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))).mean())
    return {'count': len(y), 'accuracy': float(y.mean()), 'brier': float(((p - y) ** 2).mean()),
            'ece': expected_calibration_error(p, y), 'auroc': auroc(p, y), 'acceptance': acceptance, 'nll': nll}
