"""One final measurement of the frozen heads on the audit cohort (``l2r4kie audit``).

Rule kept from the old ``audit_experiment``: the audit is opened only after
heads, calibration and thresholds are frozen (``frozen.json``), and nothing is
tuned on it. New:

* it runs once per selection: an existing ``audit_report.json`` is an error
  (a second look invites tuning);
* bundle fingerprints are checked, so a head or policy edited after freezing
  is refused;
* intervals: document bootstrap of review rate and error recall;
* subsets of the audit named in the cohort plan (e.g. r4's ``fresh_audit`` /
  ``seen_audit``) are reported separately;
* ``head_error_recall`` and the oracle review rate (the extractor's limit);
* optional paired comparison with another system's audit predictions on the
  same fields (e.g. r4's ``audit_predictions.jsonl``) for gate G4.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..confidence.cache import load_cache
from ..confidence.calibration import calibrate
from ..confidence.heads import load_head
from ..confidence.policy import ReviewPolicy
from ..data.cohorts import CohortPlan
from ..eval.review_metrics import (bootstrap_comparison, describe, evaluate_policy, within_field_auroc)
from ..utils.io import PathLike, dumps_jsonl, read_json, read_jsonl, write_json
from .finalize import bundle_fingerprint, logits_of
from .head_selection import compact


def bootstrap_review(rows: Sequence[Mapping[str, Any]], flags: Sequence[bool], replicates: int = 2000,
                     seed: int = 42) -> dict[str, list[float]]:
    """Document-bootstrap 95% intervals of review rate and error recall."""
    documents: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for row, flag in zip(rows, flags, strict=True):
        totals = documents[row['document_id']]
        totals[0] += 1
        totals[1] += flag
        totals[2] += not row['correct']
        totals[3] += flag and not row['correct']
    data = np.array(list(documents.values()), dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(data), size=(replicates, len(data)))
    sums = data[draws].sum(1)
    recall = np.divide(sums[:, 3], sums[:, 2], out=np.full(replicates, np.nan), where=sums[:, 2] > 0)
    finite = recall[np.isfinite(recall)]
    return {'review_rate': np.quantile(sums[:, 1] / sums[:, 0], [.025, .975]).tolist(),
            'error_recall': np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]}


def paired_systems(rows_a: Sequence[Mapping[str, Any]], flags_a: Sequence[bool],
                   rows_b: Sequence[Mapping[str, Any]], flags_b: Sequence[bool], replicates: int = 2000,
                   seed: int = 42) -> dict[str, Any]:
    """Review rate and recall of two systems (own values, own errors) on their shared fields.

    Fields are matched by (document, field id); the bootstrap resamples documents.
    """
    b = {(r['document_id'], r['field_id']): (r, f) for r, f in zip(rows_b, flags_b, strict=True)}
    documents: dict[str, list[int]] = defaultdict(lambda: [0] * 7)
    for row, flag in zip(rows_a, flags_a, strict=True):
        key = (row['document_id'], row['field_id'])
        if key not in b:
            continue
        other, other_flag = b[key]
        t = documents[row['document_id']]
        t[0] += 1
        t[1] += flag
        t[2] += not row['correct']
        t[3] += flag and not row['correct']
        t[4] += other_flag
        t[5] += not other['correct']
        t[6] += other_flag and not other['correct']
    if not documents:
        raise ValueError('The two systems share no (document, field) pair')
    data = np.array(list(documents.values()), dtype=float)
    total = data.sum(0)
    draws = np.random.default_rng(seed).integers(0, len(data), size=(replicates, len(data)))
    sums = data[draws].sum(1)
    change = sums[:, 1] / sums[:, 0] - sums[:, 4] / sums[:, 0]
    return {'fields': int(total[0]), 'documents': len(data),
            'review_rate': total[1] / total[0], 'baseline_review_rate': total[4] / total[0],
            'errors': int(total[2]), 'baseline_errors': int(total[5]),
            'error_recall': total[3] / total[2] if total[2] else None,
            'baseline_error_recall': total[6] / total[5] if total[5] else None,
            'review_rate_change': total[1] / total[0] - total[4] / total[0],
            'review_rate_change_95': np.quantile(change, [.025, .975]).tolist(), 'replicates': replicates}


def audit(output: PathLike, cache: PathLike | None = None, cohort_plan: PathLike | None = None,
          baseline_predictions: PathLike | None = None, device: str = 'cpu') -> dict[str, Any]:
    """Score every frozen candidate on the audit cohort and write ``audit_report.json``.

    Args:
        output: Frozen selection directory.
        cache: Trace cache directory (default: the frozen one).
        cohort_plan: Plan whose extra keys may name audit subsets (lists of
            document ids under keys ending in ``_audit``).
        baseline_predictions: Another system's audit rows (``document_id``,
            ``field_id``, ``correct``, ``needs_review``) for a paired comparison.
        device: Where heads run.

    Raises:
        ValueError: If not frozen, already audited, a bundle changed, the
            audit cache comes from another extractor, or overlaps a development cohort.
    """
    out = Path(output)
    frozen = read_json(out / 'frozen.json')
    if (out / 'audit_report.json').exists():
        raise ValueError(f'{out} was already audited; the audit is measured once')
    cache = Path(cache or frozen['cache'])
    data = load_cache(cache, 'audit')
    if data.metadata.get('source_fingerprint') != frozen['source_fingerprint']:
        raise ValueError('Audit traces come from another extractor than the frozen heads')
    for split in ('train', 'dev', 'calibration', 'risk_validation'):
        if data.documents & load_cache(cache, split).documents:
            raise ValueError(f'Audit overlaps the {split} cohort')
    subsets: dict[str, set[str]] = {}
    if cohort_plan is not None:
        extra = CohortPlan.load(cohort_plan).extra
        subsets = {k: set(v) for k, v in extra.items() if k.endswith('_audit') and isinstance(v, list)}
    baseline = read_jsonl(baseline_predictions) if baseline_predictions else None
    report: dict[str, Any] = {'primary': frozen['primary'], 'metadata': {
        k: v for k, v in data.metadata.items() if k not in ('timings_seconds', 'document_ids')},
        'frozen_before_audit': True, 'candidates': {}}
    flags_by: dict[str, list[bool]] = {}
    scores_by: dict[str, list[float | None]] = {}
    policies: dict[str, ReviewPolicy] = {}
    for family, record in frozen['candidates'].items():
        folder = Path(record['folder'])
        if bundle_fingerprint(folder) != record['bundle_fingerprint']:
            raise ValueError(f'{folder} changed after freezing')
        head = load_head(folder, device)
        logits = logits_of(head, data, device)
        policy = ReviewPolicy.from_dict(record['policy'])
        scores = data.scores(logits, record['calibration'])
        flags = [policy.needs_review(s, r['status'], r['field_id']) for r, s in zip(data.rows, scores, strict=True)]
        entry = compact(describe(data.rows, scores, calibrate(logits, record['calibration']).tolist(), policy))
        entry['intervals_95'] = bootstrap_review(data.rows, flags)
        entry['within_field'] = within_field_auroc(data.valid_rows, logits.tolist())
        entry['subsets'] = {}
        for name, ids in subsets.items():
            chosen = [i for i, r in enumerate(data.rows) if r['document_id'] in ids]
            if chosen:
                rows = [data.rows[i] for i in chosen]
                entry['subsets'][name] = {**evaluate_policy(rows, [scores[i] for i in chosen], policy),
                                          'intervals_95': bootstrap_review(rows, [flags[i] for i in chosen])}
        if baseline is not None:
            entry['versus_baseline'] = paired_systems(data.rows, flags, baseline,
                                                      [bool(r['needs_review']) for r in baseline])
            for name, ids in subsets.items():
                chosen = [i for i, r in enumerate(data.rows) if r['document_id'] in ids]
                if chosen:
                    entry['versus_baseline_' + name] = paired_systems(
                        [data.rows[i] for i in chosen], [flags[i] for i in chosen], baseline,
                        [bool(r['needs_review']) for r in baseline])
        report['candidates'][family] = entry
        flags_by[family], scores_by[family], policies[family] = flags, scores, policy
    primary = frozen['primary']
    report['paired_versus_primary'] = {
        family: bootstrap_comparison(data.rows, scores_by[primary], scores_by[family], policies[primary],
                                     policies[family])
        for family in frozen['candidates'] if family != primary}
    write_json(out / 'audit_report.json', report)
    for family in frozen['candidates']:
        lines = [dumps_jsonl({**row, 'confidence': score, 'needs_review': flag})
                 for row, score, flag in zip(data.rows, scores_by[family], flags_by[family], strict=True)]
        (out / 'heads' / family / 'audit_predictions.jsonl').write_text(''.join(lines), encoding='utf-8')
    print(json.dumps({family: {'review_rate': e['review']['review_rate'], 'error_recall': e['review']['error_recall'],
                               'intervals_95': e['intervals_95']} for family, e in report['candidates'].items()},
                     indent=2), flush=True)
    return report
