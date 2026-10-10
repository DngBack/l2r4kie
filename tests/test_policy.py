"""Tests for review policies (``l2r4kie.confidence.policy``) and their metrics (``eval.review_metrics``)."""

from __future__ import annotations

import math
import random

import pytest

from l2r4kie.confidence.policy import (REVIEW_ALL, ReviewPolicy, fit_conservative_policy, fit_policy, queue_fields,
                                       review_reason)
from l2r4kie.eval.review_metrics import (bootstrap_comparison, budgets, candidate_key, describe, evaluate_policy,
                                         oracle_review_rate, review_curve, wilson, within_field_auroc)


def row(correct: bool, status: str = 'ok', document: str = 'd0', field: str = '/a', form: str = 'f') -> dict:
    return {'document_id': document, 'form': form, 'field_id': field, 'status': status, 'correct': correct}


# ------------------------------------------------------------------ policy

def test_policy_reviews_unfinished_unknown_and_low_confidence_fields() -> None:
    policy = ReviewPolicy(.5, supported_fields=('/a',))
    assert not policy.needs_review(.9, 'ok', '/a')
    assert policy.needs_review(.4, 'ok', '/a')
    assert policy.needs_review(.9, 'truncated', '/a') and policy.is_mandatory(.9, 'truncated', '/a')
    assert policy.needs_review(None, 'ok', '/a') and policy.needs_review(math.nan, 'ok', '/a')
    assert policy.needs_review(.99, 'ok', '/new')
    assert review_reason(policy, .99, 'ok', '/new') == 'field_outside_evaluated_schema'
    assert review_reason(policy, .9, 'invalid_array', '/a') == 'incomplete_or_invalid'
    assert review_reason(policy, .4, 'ok', '/a') == 'low_correctness_confidence'
    assert review_reason(policy, .9, 'ok', '/a') is None
    assert ReviewPolicy.from_dict({**policy.to_dict(), 'validation_bounds': {}}) == policy


def test_queue_orders_mandatory_first_then_lowest_confidence() -> None:
    queue = queue_fields([{'id': '/a', 'confidence': .9, 'status': 'ok'},
                          {'id': '/b', 'confidence': .2, 'status': 'ok'},
                          {'id': '/c', 'confidence': None, 'status': 'truncated'},
                          {'id': '/d', 'confidence': .4, 'status': 'ok'}], ReviewPolicy(.5))
    assert [q['id'] for q in queue] == ['/c', '/b', '/d']
    assert queue[1]['priority'] == pytest.approx(.8)


def test_fit_policy_never_splits_ties_with_labels() -> None:
    rows = [row(False), row(True), row(True), row(False), row(True)]
    scores = [.3, .3, .6, .6, .9]
    policy = fit_policy(rows, scores, 1.)
    # Both errors need reviewing: the whole .6 tie group goes, the threshold sits between .6 and .9.
    assert policy.threshold == pytest.approx(.75)
    assert evaluate_policy(rows, scores, policy)['reviewed_fields'] == 4
    with pytest.raises(ValueError, match='dev'):
        fit_policy(rows, scores, split='risk_validation')


def test_fit_policy_counts_mandatory_reviews_and_fails_closed_without_errors() -> None:
    rows = [row(False, 'truncated'), row(True), row(True)]
    assert fit_policy(rows, [None, .4, .8], .95).threshold == 0
    assert fit_policy([row(True), row(True)], [.1, .9]).threshold == REVIEW_ALL


def test_conservative_policy_meets_both_bounds_or_reviews_all() -> None:
    rng = random.Random(0)
    rows, scores = [], []
    for d in range(120):
        for f in range(10):
            wrong = rng.random() < .15
            rows.append(row(not wrong, document=f'd{d}', field=f'/{f}'))
            scores.append(rng.uniform(0, .5) if wrong else rng.uniform(.3, 1))
    policy, bounds = fit_conservative_policy(rows, scores, .9, replicates=500)
    assert bounds['bound_target_met']
    assert min(bounds['wilson_recall_lower_95'], bounds['document_bootstrap_recall_lower_95']) >= .9
    assert evaluate_policy(rows, scores, policy)['error_recall'] >= .9
    assert policy.fitted_split == 'risk_validation'
    # A target no ranking can certify on this sample: fail closed.
    policy, bounds = fit_conservative_policy(rows[:20], scores[:20], 1., replicates=200)
    assert policy.threshold == REVIEW_ALL and bounds['fallback'] == 'review_all'
    with pytest.raises(ValueError, match='risk_validation'):
        fit_conservative_policy(rows, scores, split='audit')


# ------------------------------------------------------------------ metrics

def test_evaluate_policy_separates_head_decisions_from_mandatory_reviews() -> None:
    rows = [row(False, 'truncated'), row(False), row(False), row(True), row(True, document='d1')]
    scores = [None, .2, .8, .9, .3]
    report = evaluate_policy(rows, scores, ReviewPolicy(.5))
    assert report['reviewed_fields'] == 3 and report['caught_errors'] == 2 and report['errors'] == 3
    assert report['error_recall'] == pytest.approx(2 / 3)
    assert report['mandatory_review'] == 1
    assert report['head_fields'] == 4 and report['head_errors'] == 2
    assert report['head_error_recall'] == pytest.approx(.5)
    assert report['head_review_rate'] == pytest.approx(.5)
    assert report['documents'] == 2 and report['reviewed_documents'] == 2


def test_oracle_review_rate_is_the_extractor_limit() -> None:
    rows = [row(False, 'truncated')] + [row(False)] * 3 + [row(True)] * 6
    scores = [None] + [.5] * 9
    # 1 forced review (an error) + ceil(.95 * 4) - 1 = 3 more errors.
    assert oracle_review_rate(rows, scores) == pytest.approx(.4)


def test_review_curve_and_budgets_keep_tie_groups_whole() -> None:
    rows = [row(False), row(True), row(True), row(False, 'truncated')]
    scores = [.1, .5, .5, None]
    curve = review_curve(rows, scores)
    assert [p['accepted'] for p in curve['points']] == [2, 3]
    assert curve['maximum_valid_coverage'] == pytest.approx(.75)
    result = budgets(rows, scores, (.5, .7))
    assert result['0.5'] == {'review_rate': .5, 'caught_errors': 2, 'error_recall': 1.}
    assert result['0.7']['review_rate'] == .5  # the .5 tie group (2 fields) would cross 70%


def test_wilson_interval() -> None:
    assert wilson(0, 0) == [None, None]
    low, high = wilson(95, 100)
    assert low == pytest.approx(.8882, abs=1e-3) and high == pytest.approx(.9785, abs=1e-3)


def test_within_field_auroc_ignores_single_outcome_groups() -> None:
    rows = [row(True, field='/a'), row(False, field='/a'), row(True, field='/b'), row(True, field='/b')]
    result = within_field_auroc(rows, [1., 0., 0., 1.])
    assert result['macro_auroc'] == 1. and result['groups_with_both_labels'] == 1


def test_describe_and_candidate_key_prefer_less_review() -> None:
    rows = [row(i % 3 != 0, document=f'd{i // 4}') for i in range(24)]
    good = [.1 if not r['correct'] else .9 for r in rows]
    bad = [.5] * 24
    logit = lambda p: math.log(p / (1 - p))  # noqa: E731
    reports = [describe(rows, scores, [logit(s) for s in scores]) for scores in (good, bad)]
    assert reports[0]['review']['review_rate'] == pytest.approx(1 / 3)
    assert candidate_key(reports[0]) < candidate_key(reports[1])
    assert reports[0]['oracle_review_rate'] == pytest.approx(1 / 3)


def test_bootstrap_comparison_is_paired() -> None:
    rows = [row(i % 4 != 0, document=f'd{i // 4}') for i in range(80)]
    scores = [.1 if not r['correct'] else .9 for r in rows]
    same = bootstrap_comparison(rows, scores, scores, ReviewPolicy(.5), ReviewPolicy(.5), replicates=200)
    assert same['intervals_95']['review_rate_change'] == [0., 0.]
    more = bootstrap_comparison(rows, scores, scores, ReviewPolicy(.5), ReviewPolicy(REVIEW_ALL), replicates=200)
    assert more['intervals_95']['review_rate_change'][0] > 0
