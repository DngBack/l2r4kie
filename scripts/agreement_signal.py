"""Experiment: does agreement between sampled candidates add to the frozen confidence head?

For every field of a cached cohort, the extractor samples ``copies`` more
values over the same prefix (``l2r4kie.model.decode.sample_prefix``). The
agreement of a field is the share of candidates equal to its greedy value
under the text comparator. The analysis compares, on the same fields:

* ``head``: the frozen primary head of ``frozen.json``, calibrated;
* ``head_cv``: a logistic model on the head logit alone (fold-wise recalibration);
* ``agreement``: the agreement share alone (ties kept, never split);
* ``stack``: a logistic model on the head logit and the agreement features.

The logistic models are fitted with document-grouped cross-validation, so
every field is scored by a model that never saw its document. Review at the
target recall uses :func:`~l2r4kie.confidence.policy.fit_policy` on the
pooled scores, as in head selection. Protocol and decision rule:
``docs/results/agreement_signal.md``.

Usage::

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 uv run python scripts/agreement_signal.py sample \\
        --config configs/confidence/kev.yaml --split dev --output artifacts/kev-r4/agreement
    uv run python scripts/agreement_signal.py analyze \\
        --config configs/confidence/kev.yaml --split dev --samples artifacts/kev-r4/agreement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

FEATURES: tuple[str, ...] = ('logit', 'agreement', 'distinct')


def document_seed(seed: int, document_id: str) -> int:
    """Sampling seed of one document, independent of the order documents are decoded in."""
    return int(hashlib.sha256(f'{seed}:{document_id}'.encode()).hexdigest()[:8], 16)


def sample(args: argparse.Namespace) -> int:
    import dataclasses

    from l2r4kie.cli import _confidence_config
    from l2r4kie.confidence.cache import load_cache
    from l2r4kie.data.selection import Selection
    from l2r4kie.data.types import FieldRequest
    from l2r4kie.model.decode import sample_prefix
    from l2r4kie.model.extractor import Extractor
    from l2r4kie.pipelines.trace_cache import resolve_cohorts
    from l2r4kie.train.config import TrainConfig
    from l2r4kie.utils.config import load_config

    config = _confidence_config(args)
    extractor_config = TrainConfig.from_dict(load_config(config.extractor))
    s = extractor_config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    documents = resolve_cohorts(config.cohort_plan, selection, [args.split],
                                extractor_config.train_documents)[args.split]
    cache = load_cache(config.cache, args.split)
    fields_of: dict[str, list[str]] = defaultdict(list)
    for row in cache.rows:
        fields_of[row['document_id']].append(row['field_id'])
    documents = [d for d in documents if d.id in fields_of][:args.limit]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.split}.jsonl'
    settings = {'split': args.split, 'copies': args.copies, 'temperature': args.temperature, 'seed': args.seed,
                'max_value_tokens': args.max_value_tokens, 'max_branches': args.max_branches,
                'adapter': config.adapter, 'cache': config.cache}
    settings_path = out / f'{args.split}.settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError(f'{path} was sampled with other settings; use a new --output')
    settings_path.write_text(json.dumps(settings, indent=2))
    done = {json.loads(line)['document_id'] for line in path.open()} if path.exists() else set()

    extractor = Extractor.load(dataclasses.replace(extractor_config.extractor(), device=config.device,
                                                   max_branches=args.max_branches), config.adapter)
    started = time.monotonic()
    for n, document in enumerate(documents, 1):
        if document.id in done:
            continue
        by_id = {f.id: f for f in document.fields}
        requests = [FieldRequest.from_field(by_id[i]) for i in fields_of[document.id]]
        begin = time.monotonic()
        prefix = extractor.encode_prefix(document.pages)
        candidates = sample_prefix(extractor, prefix, requests, args.copies, temperature=args.temperature,
                                   seed=document_seed(args.seed, document.id),
                                   max_value_tokens=args.max_value_tokens)
        record = {'document_id': document.id, 'seconds': time.monotonic() - begin,
                  'fields': [{'field_id': r.id, 'candidates': [
                      {'text': c.text, 'value': c.value, 'status': c.status, 'stop': c.stop} for c in field]}
                      for r, field in zip(requests, candidates, strict=True)]}
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps({'document': n, 'of': len(documents), 'id': document.id,
                          'seconds': round(record['seconds'], 1),
                          'elapsed': round(time.monotonic() - started)}), flush=True)
    return 0


def agreement_features(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Agreement share with the greedy value, and the number of distinct candidate values (share of copies)."""
    from l2r4kie.eval.comparator import text_canonical, text_correct

    agree = sum(c['status'] == 'ok' and text_correct(c['value'], row['prediction']) for c in candidates)
    keys = {json.dumps(text_canonical(c['value']), ensure_ascii=False, sort_keys=True) if c['status'] == 'ok'
            else '<' + c['status'] + '>' for c in candidates}
    return {'agreement': agree / len(candidates), 'distinct': len(keys) / len(candidates)}


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor,
                                                                               torch.Tensor, torch.Tensor]:
    """Standardised L2 logistic regression by L-BFGS; returns (weights, bias, mean, std)."""
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = (x - mean) / std
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], lr=.5, max_iter=200, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z @ w + b, y) + l2 * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return w.detach(), b.detach(), mean, std


def cross_validated(x: torch.Tensor, y: torch.Tensor, documents: Sequence[str], folds: int,
                    seed: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Out-of-fold logits of a logistic model; folds never split a document."""
    ordered = sorted(set(documents))
    random.Random(seed).shuffle(ordered)
    fold_of = {d: i % folds for i, d in enumerate(ordered)}
    out = torch.empty(len(y), dtype=torch.float64)
    models = []
    for fold in range(folds):
        train = torch.tensor([fold_of[d] != fold for d in documents])
        w, b, mean, std = fit_logistic(x[train], y[train])
        out[~train] = ((x[~train] - mean) / std) @ w + b
        models.append({'fold': fold, 'weights_standardised': w.tolist(), 'bias': float(b)})
    return out, models


def analyze(args: argparse.Namespace) -> int:
    from l2r4kie.cli import _confidence_config
    from l2r4kie.confidence.cache import load_cache
    from l2r4kie.confidence.calibration import calibrate
    from l2r4kie.confidence.heads import load_head
    from l2r4kie.confidence.policy import fit_policy
    from l2r4kie.eval.review_metrics import (bootstrap_comparison, describe, oracle_review_rate,
                                             within_field_auroc)
    from l2r4kie.pipelines.finalize import logits_of
    from l2r4kie.pipelines.head_selection import compact
    from l2r4kie.utils.io import read_json

    config = _confidence_config(args)
    frozen = read_json(Path(config.output) / 'frozen.json')
    record = frozen['candidates'][frozen['primary']]
    cache = load_cache(config.cache, args.split)
    if cache.metadata.get('source_fingerprint') != frozen['source_fingerprint']:
        raise ValueError('The cache comes from another extractor than the frozen head')
    head = load_head(record['folder'])
    logits = calibrate(logits_of(head, cache), record['calibration'])  # calibrated logits of valid rows

    sampled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seconds = []
    for line in (Path(args.samples) / f'{args.split}.jsonl').open():
        doc = json.loads(line)
        seconds.append(doc['seconds'])
        for field in doc['fields']:
            sampled[(doc['document_id'], field['field_id'])] = field['candidates']
    settings = json.loads((Path(args.samples) / f'{args.split}.settings.json').read_text())

    # Fields of the documents that were sampled; every row of such a document must have candidates.
    documents = {d for d, _ in sampled}
    rows = [r for r in cache.rows if r['document_id'] in documents]
    missing = [r for r in rows if (r['document_id'], r['field_id']) not in sampled]
    if missing:
        raise ValueError(f'{len(missing)} fields without candidates, e.g. {missing[0]["field_id"]}')
    valid = [r for r in rows if r['trace_index'] >= 0]
    features = []
    for r in valid:
        f = agreement_features(r, sampled[(r['document_id'], r['field_id'])])
        features.append([float(logits[r['trace_index']]), f['agreement'], f['distinct']])
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor([float(r['correct']) for r in valid], dtype=torch.float64)
    docs = [r['document_id'] for r in valid]

    head_cv, head_models = cross_validated(x[:, :1], y, docs, args.folds, args.seed)
    stack, stack_models = cross_validated(x, y, docs, args.folds, args.seed)
    valid_logits = {
        'head': x[:, 0],
        'head_cv': head_cv,
        # A probability-like score in (0, 1) for policies; ties stay ties.
        'agreement': torch.logit(x[:, 1].clamp(1e-4, 1 - 1e-4)),
        'stack': stack,
    }
    position = {id(r): i for i, r in enumerate(valid)}
    reports, scores_by, policies = {}, {}, {}
    for name, values in valid_logits.items():
        probs = torch.sigmoid(values).tolist()
        scores = [probs[position[id(r)]] if id(r) in position else None for r in rows]
        policy = fit_policy(rows, scores, config.target_error_recall)
        report = compact(describe(rows, scores, values.tolist(), policy))
        report['within_field'] = within_field_auroc(valid, values.tolist())
        reports[name], scores_by[name], policies[name] = report, scores, policy
    comparisons = {f'{new}_minus_{base}': bootstrap_comparison(rows, scores_by[base], scores_by[new], policies[base],
                                                               policies[new], replicates=args.replicates,
                                                               seed=args.seed)
                   for base, new in (('head_cv', 'stack'), ('head', 'stack'), ('head', 'agreement'))}
    agreement = x[:, 1]
    summary = {
        'split': args.split, 'sampling': settings, 'documents': len(documents), 'fields': len(rows),
        'valid_fields': len(valid), 'errors': int(sum(not r['correct'] for r in rows)),
        'oracle_review_rate': oracle_review_rate(rows, scores_by['head'], config.target_error_recall),
        'sampling_seconds_per_document': {'mean': sum(seconds) / len(seconds),
                                          'median': sorted(seconds)[len(seconds) // 2], 'max': max(seconds)},
        'agreement_distribution': {
            'all_agree_share': float((agreement == 1).double().mean()),
            'correct_given_all_agree': float(y[agreement == 1].mean()) if (agreement == 1).any() else None,
            'correct_given_not_all_agree': float(y[agreement < 1].mean()) if (agreement < 1).any() else None},
        'systems': {name: {'review_rate': r['review']['review_rate'], 'error_recall': r['review']['error_recall'],
                           'auroc': r['confidence'].get('auroc'), 'aurc': r['curve']['aurc'],
                           'within_field_auroc': r['within_field']['macro_auroc']} for name, r in reports.items()},
        'comparisons': comparisons,
        'models': {'head_cv': head_models, 'stack': stack_models, 'features': list(FEATURES)},
        'reports': reports,
    }
    out = Path(args.samples) / f'{args.split}.analysis.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: summary[k] for k in ('documents', 'fields', 'errors', 'oracle_review_rate',
                                              'sampling_seconds_per_document', 'agreement_distribution',
                                              'systems')}, indent=2, ensure_ascii=False))
    print(json.dumps({k: v['intervals_95'] for k, v in comparisons.items()}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('sample', 'analyze'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, required=True)
        p.add_argument('--set', action='append', default=[])
        p.add_argument('--split', default='dev')
        p.add_argument('--seed', type=int, default=0)
    s = sub.choices['sample']
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--copies', type=int, default=8)
    s.add_argument('--temperature', type=float, default=1.)
    s.add_argument('--max-value-tokens', type=int, default=512)
    s.add_argument('--max-branches', type=int, default=96)
    s.add_argument('--limit', type=int)
    a = sub.choices['analyze']
    a.add_argument('--samples', type=Path, required=True)
    a.add_argument('--folds', type=int, default=5)
    a.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    return sample(args) if args.command == 'sample' else analyze(args)


if __name__ == '__main__':
    raise SystemExit(main())
