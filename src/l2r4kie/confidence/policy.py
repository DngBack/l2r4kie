"""Review policies: which extracted fields a person must check (old ``review``).

A field is *reviewed* when its value did not finish (status not ``ok``), it
has no confidence, its field id was never validated, or its calibrated
confidence is below the threshold. Everything else is accepted automatically.
The goal is to catch at least ``target_error_recall`` of the wrong values
while reviewing as few fields as possible.

Algorithms are unchanged from the old repository: ties in confidence are
never split using labels, and a policy that cannot reach its target fails
closed (reviews everything).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

#: Threshold above every probability: review everything.
REVIEW_ALL = 1.0000000001


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    """A frozen review rule.

    Attributes:
        threshold: Fields with calibrated confidence below it are reviewed.
        target_error_recall: Share of wrong values the policy was fitted to catch.
        fitted_split: Cohort the threshold was chosen on.
        mode: ``'global'``: one threshold for all fields.
        supported_fields: Field ids seen when fitting; any other field is
            always reviewed. Empty: no restriction.
    """

    threshold: float
    target_error_recall: float = .95
    fitted_split: str = 'dev'
    mode: str = 'global'
    supported_fields: tuple[str, ...] = ()

    def is_mandatory(self, confidence: float | None, status: str = 'ok', field_id: str | None = None) -> bool:
        """Whether the field is reviewed whatever its confidence."""
        unknown = bool(self.supported_fields) and field_id is not None and field_id not in self.supported_fields
        return unknown or status != 'ok' or confidence is None or not math.isfinite(confidence)

    def needs_review(self, confidence: float | None, status: str = 'ok', field_id: str | None = None) -> bool:
        """Whether a person must check this field."""
        return self.is_mandatory(confidence, status, field_id) or confidence < self.threshold  # type: ignore[operator]

    def to_dict(self) -> dict[str, Any]:
        """JSON form."""
        return {**asdict(self), 'supported_fields': list(self.supported_fields)}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ReviewPolicy:
        """Read :meth:`to_dict` output, or an old ``review_policy.json`` (extra keys ignored)."""
        return cls(threshold=record['threshold'], target_error_recall=record.get('target_error_recall', .95),
                   fitted_split=record.get('fitted_split', 'dev'), mode=record.get('mode', 'global'),
                   supported_fields=tuple(record.get('supported_fields', ())))


def _usable(row: Mapping[str, Any], score: float | None) -> bool:
    return row['status'] == 'ok' and score is not None and math.isfinite(score)


def fit_policy(rows: Sequence[Mapping[str, Any]], scores: Sequence[float | None], target_error_recall: float = .95,
               split: str = 'dev') -> ReviewPolicy:
    """Lowest threshold that catches ``target_error_recall`` of the errors on ``rows``.

    Used to compare heads on dev at a common operating point. Tied scores
    are swept as one group; the threshold is placed halfway between the last
    reviewed score and the next one.

    Raises:
        ValueError: Outside dev, for a target outside (0, 1], or misaligned input.
    """
    from ..eval.review_metrics import evaluate_policy

    if split != 'dev':
        raise ValueError('fit_policy selects heads on dev only; use fit_conservative_policy on risk_validation')
    if not 0 < target_error_recall <= 1:
        raise ValueError(f'Invalid target_error_recall {target_error_recall}')
    if len(rows) != len(scores) or not rows:
        raise ValueError('Missing or unaligned dev data')
    groups: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    caught = 0
    errors = sum(not r['correct'] for r in rows)
    for row, score in zip(rows, scores, strict=True):
        if _usable(row, score):
            groups[float(score)].append(row)  # type: ignore[arg-type]
        else:
            caught += not row['correct']
    if not errors:
        return ReviewPolicy(REVIEW_ALL, target_error_recall)  # no errors observed: no evidence, fail closed
    required = math.ceil(target_error_recall * errors)
    threshold = 0.
    for score, group in sorted(groups.items()):
        if caught >= required:
            break
        caught += sum(not r['correct'] for r in group)
        threshold = math.nextafter(score, math.inf)
    if threshold:
        higher = [score for score in groups if score >= threshold]
        reviewed = math.nextafter(threshold, -math.inf)
        threshold = (reviewed + min(higher)) / 2 if higher else max(REVIEW_ALL, threshold)
    supported = tuple(sorted({r['field_id'] for r in rows if 'field_id' in r}))
    policy = ReviewPolicy(threshold, target_error_recall, supported_fields=supported)
    assert evaluate_policy(rows, scores, policy)['caught_errors'] >= required
    return policy


def fit_conservative_policy(rows: Sequence[Mapping[str, Any]], scores: Sequence[float | None],
                            target_error_recall: float = .95, split: str = 'risk_validation',
                            replicates: int = 2000, seed: int = 42) -> tuple[ReviewPolicy, dict[str, Any]]:
    """Threshold on the risk-validation cohort with two observational lower bounds.

    The threshold rises until both the 95% Wilson lower bound and the
    document-bootstrap 2.5% quantile of error recall reach the target. If
    they never do, every field is reviewed. These are empirical
    diagnostics, not a population guarantee under distribution shift.

    Returns:
        The policy and the bounds reached.

    Raises:
        ValueError: Outside risk_validation, or on empty/misaligned input.
    """
    from ..eval.review_metrics import wilson

    if split != 'risk_validation':
        raise ValueError('Use the separate risk_validation cohort, never audit or test')
    if not rows or len(rows) != len(scores):
        raise ValueError('Missing risk-validation fields')
    documents = sorted({r['document_id'] for r in rows})
    index = {d: i for i, d in enumerate(documents)}
    errors_by_document = np.zeros(len(documents))
    caught = np.zeros(len(documents))
    groups: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        document = index[row['document_id']]
        errors_by_document[document] += not row['correct']
        if _usable(row, score):
            groups[float(score)].append(row)  # type: ignore[arg-type]
        else:
            caught[document] += not row['correct']
    errors = int(errors_by_document.sum())
    draws = np.random.default_rng(seed).integers(0, len(documents), size=(replicates, len(documents)))
    denominators = errors_by_document[draws].sum(1)
    keys = sorted(groups)
    supported = tuple(sorted({r['field_id'] for r in rows}))

    def bounds() -> tuple[float, float]:
        ratio = np.divide(caught[draws].sum(1), denominators, out=np.full(replicates, np.nan),
                          where=denominators > 0)
        finite = ratio[np.isfinite(ratio)]
        cluster = float(np.quantile(finite, .025)) if len(finite) else 0.
        return wilson(int(caught.sum()), errors)[0] or 0., cluster

    threshold, met, field_bound, cluster_bound = 0., False, 0., 0.
    if errors:
        for i, score in enumerate(keys):
            field_bound, cluster_bound = bounds()
            if min(field_bound, cluster_bound) >= target_error_recall:
                met = True
                break
            for row in groups[score]:
                caught[index[row['document_id']]] += not row['correct']
            threshold = (score + keys[i + 1]) / 2 if i + 1 < len(keys) else REVIEW_ALL
        field_bound, cluster_bound = bounds()
        met = min(field_bound, cluster_bound) >= target_error_recall
    if not met:
        threshold = REVIEW_ALL
    policy = ReviewPolicy(threshold, target_error_recall, fitted_split=split, supported_fields=supported)
    return policy, {'wilson_recall_lower_95': field_bound, 'document_bootstrap_recall_lower_95': cluster_bound,
                    'bound_target_met': met, 'bootstrap_replicates': replicates,
                    'criteria': 'Both observational 95% lower bounds >= target', 'population_guarantee': False,
                    'fallback': None if met else 'review_all'}


def review_reason(policy: ReviewPolicy, confidence: float | None, status: str, field_id: str) -> str | None:
    """Why a field is reviewed, or ``None`` if it is accepted."""
    if not policy.needs_review(confidence, status, field_id):
        return None
    if policy.supported_fields and field_id not in policy.supported_fields:
        return 'field_outside_evaluated_schema'
    if policy.is_mandatory(confidence, status, field_id):
        return 'incomplete_or_invalid'
    return 'low_correctness_confidence'


def queue_fields(results: Sequence[Mapping[str, Any]], policy: ReviewPolicy) -> list[dict[str, Any]]:
    """Fields to review, most urgent first (mandatory reviews, then lowest confidence).

    Args:
        results: ``{'id', 'confidence', 'status'}`` per field.
        policy: Frozen policy.
    """
    queue = []
    for row in results:
        confidence, status = row.get('confidence'), row.get('status', 'ok')
        reason = review_reason(policy, confidence, status, row['id'])
        if reason is not None:
            queue.append({'id': row['id'], 'confidence': confidence, 'status': status, 'reason': reason,
                          'priority': 1. if reason != 'low_correctness_confidence' else 1. - confidence})
    return sorted(queue, key=lambda item: -item['priority'])
