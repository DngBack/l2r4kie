"""Calibrate the selected heads and fix their review thresholds (``l2r4kie finalize``).

Second half of the old ``train_experiment``. For every family kept by
:mod:`.head_selection`:

1. calibration chosen and fitted on the ``calibration`` cohort
   (:func:`~l2r4kie.confidence.calibration.select_calibration`);
2. review threshold fitted on ``risk_validation`` so that both observational
   95% lower bounds of error recall reach the target
   (:func:`~l2r4kie.confidence.policy.fit_conservative_policy`).

Then everything is frozen (``frozen.json`` with file fingerprints) before the
audit cohort is decoded or read. ``--dry-run`` computes and prints without
writing. :func:`finalize_legacy` replays the same steps on an old selection
and cache, to check that calibration and policy were ported unchanged.

Each family folder becomes a *confidence bundle* usable by ``infer
--confidence``: ``head.pt``, ``head_config.json``, ``calibration.json``,
``review_policy.json`` and ``bundle.json`` (extractor fingerprint, cached
layers, trace settings).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..confidence.cache import TraceCache, check_disjoint, check_source, load_cache
from ..confidence.calibration import calibrate, select_calibration
from ..confidence.heads import Head, load_head, load_legacy_head, score_head
from ..confidence.policy import fit_conservative_policy
from ..eval.review_metrics import describe
from ..utils.io import PathLike, read_json, write_json
from .head_selection import compact

#: Files of a confidence bundle covered by its fingerprint.
BUNDLE_FILES: tuple[str, ...] = ('head.pt', 'head_config.json', 'calibration.json', 'review_policy.json',
                                 'bundle.json')


def bundle_fingerprint(folder: PathLike) -> str:
    """SHA-256 over the bundle files that exist (name, then bytes)."""
    digest = hashlib.sha256()
    for name in BUNDLE_FILES:
        path = Path(folder) / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def logits_of(head: Head, cache: TraceCache, device: str | torch.device = 'cpu') -> torch.Tensor:
    """Head logits of a cache's valid rows."""
    signals = head.config.required_signals() if head.config.signals else ('value',)
    return score_head(head, cache.store(signals, device), [r['field_id'] for r in cache.valid_rows])


def calibrate_and_fit(head: Head, caches: Mapping[str, TraceCache], target: float, replicates: int,
                      device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Calibration, risk policy and dev/risk reports of one head."""
    logits = {split: logits_of(head, caches[split], device) for split in ('calibration', 'risk_validation', 'dev')}
    cal = caches['calibration']
    calibration = select_calibration(logits['calibration'], cal.labels, [r['document_id'] for r in cal.valid_rows])
    risk = caches['risk_validation']
    policy, bounds = fit_conservative_policy(risk.rows, risk.scores(logits['risk_validation'], calibration), target,
                                             replicates=replicates)
    reports = {split: describe(caches[split].rows, caches[split].scores(logits[split], calibration),
                               calibrate(logits[split], calibration).tolist(), policy)
               for split in ('dev', 'risk_validation')}
    return {'calibration': calibration, 'policy': policy, 'bounds': bounds, 'reports': reports}


def _summary(results: Mapping[str, Mapping[str, Any]], primary: str) -> dict[str, Any]:
    return {'primary': primary, 'candidates': {
        family: {'threshold': r['policy'].threshold, 'calibration': r['calibration']['method'],
                 'temperature': r['calibration']['temperature'], 'bias': r['calibration']['bias'],
                 'bound_target_met': r['bounds']['bound_target_met'],
                 'risk_review_rate': r['reports']['risk_validation']['review']['review_rate'],
                 'risk_error_recall': r['reports']['risk_validation']['review']['error_recall'],
                 'dev_review_rate': r['reports']['dev']['review']['review_rate']}
        for family, r in results.items()}}


def finalize(output: PathLike, cache: PathLike | None = None, target: float | None = None,
             replicates: int = 2000, dry_run: bool = False, layers: tuple[int, ...] | None = None,
             max_trace_tokens: int | None = None, device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Calibrate and threshold every family of ``<output>/selection.json``.

    Args:
        output: Directory written by :func:`~l2r4kie.pipelines.head_selection.select_heads`.
        cache: Trace cache directory (default: the one recorded in the selection).
        target: Target error recall (default: the selection's).
        replicates: Bootstrap draws of the risk lower bound.
        dry_run: Compute and print, write nothing.
        layers: Cached intermediate layers, recorded in each bundle (default:
            from the cache metadata).
        max_trace_tokens: Token states per field, recorded in each bundle.
        device: Where heads run.

    Returns:
        Short summary per family (thresholds, risk review rate and recall).

    Raises:
        ValueError: If already frozen (and not a dry run), or caches overlap
            or come from another extractor.
    """
    out = Path(output)
    selection = read_json(out / 'selection.json')
    if (out / 'frozen.json').exists() and not dry_run:
        raise ValueError(f'{out} is already frozen; use a new output directory')
    cache = Path(cache or selection['cache'])
    target = selection['target_error_recall'] if target is None else target
    caches = {s: load_cache(cache, s) for s in ('train', 'dev', 'calibration', 'risk_validation')}
    check_disjoint(caches)
    source = check_source(caches, selection['source_fingerprint'])
    meta = caches['risk_validation'].metadata
    layers = tuple(meta.get('layers') or ()) if layers is None else layers
    max_trace_tokens = meta.get('max_trace_tokens', 512) if max_trace_tokens is None else max_trace_tokens
    results = {}
    for family in selection['families']:
        head = load_head(out / 'heads' / family, device)
        results[family] = calibrate_and_fit(head, caches, target, replicates, device)
    summary = _summary(results, selection['primary'])
    print(json.dumps(summary, indent=2), flush=True)
    if dry_run:
        return summary
    frozen: dict[str, Any] = {'primary': selection['primary'], 'source_fingerprint': source, 'cache': str(cache),
                              'target_error_recall': target, 'audit_seen': False, 'candidates': {},
                              'cohorts': {s: c.metadata.get('documents') for s, c in caches.items()}}
    for family, result in results.items():
        folder = out / 'heads' / family
        write_json(folder / 'calibration.json', result['calibration'])
        write_json(folder / 'review_policy.json', {**result['policy'].to_dict(), 'validation_bounds': result['bounds'],
                                                   'population_guarantee': False})
        write_json(folder / 'bundle.json', {'family': family, 'source_fingerprint': source, 'layers': list(layers),
                                            'max_trace_tokens': max_trace_tokens, 'target_error_recall': target})
        frozen['candidates'][family] = {
            'folder': str(folder), 'calibration': result['calibration'], 'policy': result['policy'].to_dict(),
            'bounds': result['bounds'], 'dev': compact(result['reports']['dev']),
            'risk_validation': compact(result['reports']['risk_validation']),
            'bundle_fingerprint': bundle_fingerprint(folder)}
    write_json(out / 'frozen.json', frozen)
    return summary


def finalize_legacy(selected: PathLike, cache: PathLike, replicates: int = 2000,
                    device: str | torch.device = 'cpu') -> dict[str, Any]:
    """Replay calibration and risk thresholds on an old selection and cache (writes nothing).

    Every old trained head (``frozen_selection.json`` candidates except the
    removed ``baseline``) is loaded with
    :func:`~l2r4kie.confidence.heads.load_legacy_head` and finalised on the
    old ``calibration`` and ``risk_validation`` caches. The thresholds must
    equal the old ``review_policy.json`` ones (r4 attention: 0.9597).

    Returns:
        The summary, with the old threshold of each family next to the new one.
    """
    selected, cache = Path(selected), Path(cache)
    old = read_json(selected / 'frozen_selection.json')
    caches = {s: load_cache(cache, s) for s in ('dev', 'calibration', 'risk_validation')}
    check_disjoint(caches)
    results = {}
    for mode, record in old['candidates'].items():
        if mode == 'baseline':
            continue  # the untrained checkpoint head: removed
        head = load_legacy_head(selected / mode, device).head
        results[mode] = calibrate_and_fit(head, caches, old['target_error_recall'], replicates, device)
    summary = _summary(results, old['primary_selected_by_dev'])
    for mode, entry in summary['candidates'].items():
        entry['old_threshold'] = old['candidates'][mode]['policy']['threshold']
        entry['old_temperature'] = old['candidates'][mode]['calibration']['temperature']
        entry['threshold_delta'] = entry['threshold'] - entry['old_threshold']
    print(json.dumps(summary, indent=2), flush=True)
    return summary
