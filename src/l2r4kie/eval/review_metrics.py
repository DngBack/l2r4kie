"""Workload and recall of review policies (old ``review`` + ``review_optimization``).

A *review row* is one requested field with ``document_id``, ``field_id``,
``status`` and ``correct``; *scores* are calibrated confidences aligned with
the rows (``None`` where there is none, e.g. a truncated value).

New over the old reports:

* ``head_error_recall`` / ``head_review_rate``: the same numbers restricted to
  fields the head actually decides on (not mandatory reviews). The old error
  recall mixed in truncated values, which are caught whatever the head does.
* :func:`oracle_review_rate`: the least review any ranking could achieve at
  the target recall, the limit set by the extractor's own error rate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..confidence.policy import ReviewPolicy, fit_policy
from .metrics import auroc, confidence_metrics

Row = Mapping[str, Any]


def wilson(successes: int, total: int, z: float = 1.96) -> list[float | None]:
    """Wilson score interval of a proportion (``[None, None]`` when ``total`` is 0)."""
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., center - width), min(1., center + width)]


def evaluate_policy(rows: Sequence[Row], scores: Sequence[float | None], policy: ReviewPolicy) -> dict[str, Any]:
    """Review workload and error recall of ``policy`` on ``rows``."""
    if len(rows) != len(scores):
        raise ValueError('One score is required per field')
    n = len(rows)
    errors = sum(not r['correct'] for r in rows)
    decisions = [policy.needs_review(s, r['status'], r.get('field_id')) for r, s in zip(rows, scores, strict=True)]
    mandatory = [policy.is_mandatory(s, r['status'], r.get('field_id')) for r, s in zip(rows, scores, strict=True)]
    reviewed = sum(decisions)
    caught = sum(flag and not r['correct'] for r, flag in zip(rows, decisions, strict=True))
    accepted = n - reviewed
    accepted_correct = sum(not flag and r['correct'] for r, flag in zip(rows, decisions, strict=True))
    head = [(r, flag) for r, flag, m in zip(rows, decisions, mandatory, strict=True) if not m]
    head_errors = sum(not r['correct'] for r, _ in head)
    head_caught = sum(flag and not r['correct'] for r, flag in head)
    documents: dict[str, list[bool]] = defaultdict(list)
    for row, flag in zip(rows, decisions, strict=True):
        documents[row['document_id']].append(flag)
    flagged = sum(any(flags) for flags in documents.values())
    return {'fields': n, 'documents': len(documents), 'errors': errors, 'reviewed_fields': reviewed,
            'review_rate': reviewed / n if n else 0., 'accepted_fields': accepted,
            'accepted_precision': accepted_correct / accepted if accepted else None,
            'accepted_precision_wilson_95': wilson(accepted_correct, accepted),
            'caught_errors': caught, 'missed_errors': errors - caught,
            'error_recall': caught / errors if errors else None,
            'error_recall_wilson_95': wilson(caught, errors),
            'review_precision': caught / reviewed if reviewed else None,
            'reviewed_documents': flagged, 'document_review_rate': flagged / len(documents) if documents else 0.,
            'mandatory_review': sum(mandatory),
            'head_fields': len(head), 'head_errors': head_errors,
            'head_review_rate': sum(flag for _, flag in head) / len(head) if head else None,
            'head_error_recall': head_caught / head_errors if head_errors else None}


def oracle_review_rate(rows: Sequence[Row], scores: Sequence[float | None], target: float = .95,
                       policy: ReviewPolicy | None = None) -> float:
    """Least review rate at which ``target`` of the errors can be caught.

    Mandatory reviews are counted; on top, a perfect ranking reviews only the
    errors still needed.
    """
    policy = policy or ReviewPolicy(0.)
    mandatory = [policy.is_mandatory(s, r['status'], r.get('field_id')) for r, s in zip(rows, scores, strict=True)]
    errors = sum(not r['correct'] for r in rows)
    forced_errors = sum(m and not r['correct'] for r, m in zip(rows, mandatory, strict=True))
    extra = max(0, math.ceil(target * errors) - forced_errors)
    return (sum(mandatory) + extra) / len(rows) if rows else 0.


def review_curve(rows: Sequence[Row], scores: Sequence[float | None]) -> dict[str, Any]:
    """Risk-coverage curve over realisable thresholds, and its area (AURC).

    Fields without a usable score are ranked last for the AURC only; no
    policy may accept them.
    """
    groups: dict[float, list[Row]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        if row['status'] == 'ok' and score is not None and math.isfinite(score):
            groups[float(score)].append(row)
    n = len(rows)
    accepted = errors = 0
    aurc, points = 0., []
    for score, group in sorted(groups.items(), reverse=True):
        accepted += len(group)
        errors += sum(not r['correct'] for r in group)
        risk = errors / accepted
        aurc += risk * len(group) / n
        points.append({'threshold': score, 'accepted': accepted, 'coverage': accepted / n, 'risk': risk})
    invalid = n - accepted
    if invalid:
        aurc += (errors + invalid) / n * invalid / n
    return {'aurc': aurc, 'points': points, 'maximum_valid_coverage': accepted / n if n else 0.}


def budgets(rows: Sequence[Row], scores: Sequence[float | None],
            fractions: Sequence[float] = (.1, .2, .3, .5, .7, .9)) -> dict[str, Any]:
    """Errors caught when reviewing a fixed share of fields (mandatory ones first).

    A tie group that would cross the budget is left out entirely; the
    achieved review rate is reported instead of breaking the tie.
    """
    mandatory = {i for i, (r, s) in enumerate(zip(rows, scores, strict=True))
                 if r['status'] != 'ok' or s is None or not math.isfinite(s)}
    groups: dict[float, list[int]] = defaultdict(list)
    for i, score in enumerate(scores):
        if i not in mandatory:
            groups[float(score)].append(i)  # type: ignore[arg-type]
    total = sum(not r['correct'] for r in rows)
    result = {}
    for fraction in fractions:
        cap = max(len(mandatory), math.floor(len(rows) * fraction))
        chosen = sorted(mandatory)
        for _, indices in sorted(groups.items()):
            if len(chosen) + len(indices) > cap:
                break
            chosen.extend(indices)
        caught = sum(not rows[i]['correct'] for i in chosen)
        result[str(fraction)] = {'review_rate': len(chosen) / len(rows), 'caught_errors': caught,
                                 'error_recall': caught / total if total else None}
    return result


def within_field_auroc(rows: Sequence[Row], logits: Sequence[float]) -> dict[str, Any]:
    """Mean AUROC within each (form, field) group that has both outcomes.

    Shows whether a head ranks values of the *same* field, not only easy
    fields above hard ones.

    Args:
        rows: Rows with a score (valid fields only), aligned with ``logits``.
        logits: Head logits.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row.get('form', ''), row['field_id'])].append(i)
    values = []
    for indices in groups.values():
        labels = np.array([float(rows[i]['correct']) for i in indices])
        if 0 < labels.sum() < len(labels):
            values.append(auroc(np.array([logits[i] for i in indices], dtype=np.float64), labels))
    return {'macro_auroc': sum(values) / len(values) if values else None, 'groups_with_both_labels': len(values),
            'note': 'Grouped by (form, field); small groups are noisy.'}


def describe(rows: Sequence[Row], scores: Sequence[float | None], calibrated_logits: Sequence[float],
             policy: ReviewPolicy | None = None, target: float = .95) -> dict[str, Any]:
    """Every review metric of one head on one cohort.

    Args:
        rows: All rows of the cohort.
        scores: Calibrated confidences aligned with ``rows`` (``None`` for invalid).
        calibrated_logits: Calibrated logits of the rows that have a score, in order.
        policy: Frozen policy; ``None`` fits one on these rows (dev only).
        target: Target error recall when fitting.
    """
    if policy is None:
        policy = fit_policy(rows, scores, target)
    labels = [float(r['correct']) for r, s in zip(rows, scores, strict=True) if s is not None]
    return {'policy': policy.to_dict(), 'review': evaluate_policy(rows, scores, policy),
            'oracle_review_rate': oracle_review_rate(rows, scores, policy.target_error_recall, policy),
            'confidence': confidence_metrics(list(calibrated_logits), labels),
            'curve': review_curve(rows, scores), 'budgets': budgets(rows, scores)}


def candidate_key(report: Mapping[str, Any]) -> tuple[float, float, float]:
    """Sort key of a dev report: review rate at the target, then AURC, then -AUROC (lower is better)."""
    return (report['review']['review_rate'], report['curve']['aurc'], -(report['confidence'].get('auroc') or 0.))


def bootstrap_comparison(rows: Sequence[Row], base_scores: Sequence[float | None],
                         new_scores: Sequence[float | None], base_policy: ReviewPolicy, new_policy: ReviewPolicy,
                         replicates: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Paired document bootstrap of review rate and error recall of two policies on the same rows."""
    documents: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        documents.setdefault(row['document_id'], []).append(i)
    aggregate = []
    for indices in documents.values():
        values = [len(indices), sum(not rows[i]['correct'] for i in indices)]
        for scores, policy in ((base_scores, base_policy), (new_scores, new_policy)):
            flags = [policy.needs_review(scores[i], rows[i]['status'], rows[i].get('field_id')) for i in indices]
            values.extend([sum(flags), sum(flag and not rows[i]['correct'] for i, flag in zip(indices, flags))])
        aggregate.append(values)
    data = np.array(aggregate, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(data), size=(replicates, len(data)))
    sums = data[draws].sum(axis=1)
    errors = np.where(sums[:, 1] > 0, sums[:, 1], np.nan)
    result = {}
    for key, values in (('review_rate_change', sums[:, 4] / sums[:, 0] - sums[:, 2] / sums[:, 0]),
                        ('baseline_error_recall', sums[:, 3] / errors), ('candidate_error_recall', sums[:, 5] / errors)):
        finite = values[np.isfinite(values)]
        result[key] = np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]
    return {'method': 'paired document bootstrap', 'replicates': replicates, 'intervals_95': result}
