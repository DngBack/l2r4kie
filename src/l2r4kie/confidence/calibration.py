"""Calibration of head logits into probabilities (old ``review_optimization``).

The algorithm is unchanged: three families (``identity``, ``temperature``,
``affine``), the family chosen by 4-fold cross-validated NLL with folds
grouped by document, then refit on the whole calibration cohort.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

METHODS: tuple[str, ...] = ('identity', 'temperature', 'affine')
IDENTITY: dict[str, Any] = {'temperature': 1., 'bias': 0., 'method': 'identity'}


def fit_calibration(logits: torch.Tensor, labels: Sequence[float], method: str) -> dict[str, Any]:
    """Fit ``sigmoid(logit / temperature + bias)`` by L-BFGS on binary cross-entropy.

    Args:
        logits: Head logits.
        labels: 1.0 for correct values, 0.0 for wrong ones.
        method: ``identity`` (nothing fitted), ``temperature`` (scale only)
            or ``affine`` (scale and bias).

    Returns:
        ``{'temperature', 'bias', 'method'}``; ``identity_single_class`` when
        the labels hold one class only.
    """
    x = logits.double().detach()
    y = torch.tensor(labels, dtype=torch.float64)
    if len(set(labels)) < 2:
        return {'temperature': 1., 'bias': 0., 'method': 'identity_single_class'}
    if method == 'identity':
        return {'temperature': 1., 'bias': 0., 'method': method}
    if method not in METHODS:
        raise ValueError(f'Unknown calibration method {method!r}')
    log_scale = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=method == 'affine')
    parameters = [log_scale, bias] if method == 'affine' else [log_scale]
    optimizer = torch.optim.LBFGS(parameters, lr=.2, max_iter=100, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = x * log_scale.exp().clamp(.05, 20) + bias.clamp(-10, 10)
        # Mild regularisation stabilises small calibration cohorts.
        loss = F.binary_cross_entropy_with_logits(scaled, y) + .001 * (log_scale.square() + bias.square())
        loss.backward()
        return loss

    optimizer.step(closure)
    return {'temperature': float(1 / log_scale.detach().exp().clamp(.05, 20)),
            'bias': float(bias.detach().clamp(-10, 10)), 'method': method}


def select_calibration(logits: torch.Tensor, labels: Sequence[float], documents: Sequence[str],
                       folds: int = 4, seed: int = 42) -> dict[str, Any]:
    """Choose the calibration family by document-grouped cross-validation, then refit on all.

    Args:
        logits: Head logits of the calibration cohort's valid fields.
        labels: Their correctness (0/1).
        documents: Their document ids (folds never split a document).
        folds: Number of folds.
        seed: Seed of the document shuffle that assigns folds.

    Returns:
        The refit calibration plus ``cv_nll`` per family, ``fields`` and
        ``documents`` counts.
    """
    if not (len(logits) == len(labels) == len(documents)):
        raise ValueError('logits, labels and documents must align')
    ordered = sorted(set(documents))
    random.Random(seed).shuffle(ordered)
    fold_of = {document: i % folds for i, document in enumerate(ordered)}
    results = {}
    for method in METHODS:
        losses: list[float] = []
        for fold in range(folds):
            train = [i for i, d in enumerate(documents) if fold_of[d] != fold]
            valid = [i for i, d in enumerate(documents) if fold_of[d] == fold]
            if not train or not valid:
                continue
            fitted = fit_calibration(logits[train], [labels[i] for i in train], method)
            scaled = logits[valid].double() / fitted['temperature'] + fitted['bias']
            losses.extend(F.binary_cross_entropy_with_logits(
                scaled, torch.tensor([labels[i] for i in valid], dtype=torch.float64), reduction='none').tolist())
        results[method] = sum(losses) / len(losses) if losses else float('inf')
    method = min(results, key=results.get)
    return {**fit_calibration(logits, labels, method), 'cv_nll': results, 'split': 'calibration',
            'fields': len(labels), 'documents': len(ordered),
            'selection': f'{folds}-fold grouped by document; refit on all calibration documents'}


def calibrate(logits: torch.Tensor, calibration: Mapping[str, Any] | None = None) -> torch.Tensor:
    """Calibrated logits (float64): ``logit / temperature + bias``."""
    calibration = calibration or IDENTITY
    return logits.double() / calibration['temperature'] + calibration.get('bias', 0.)


def probabilities(logits: torch.Tensor, calibration: Mapping[str, Any] | None = None) -> list[float]:
    """Calibrated probabilities of being correct."""
    return torch.sigmoid(calibrate(logits, calibration)).tolist()
