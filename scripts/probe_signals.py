"""Which decode signals tell a right value from a wrong one? (exploratory)

Decodes a small form-balanced cohort with an adapter (``cache-traces``
machinery, resumable), then fits one L2 logistic probe per signal set and
reports, with 5-fold cross-validation grouped by document:

* AUROC of the out-of-fold scores;
* review rate needed to catch 95% of the wrong values (``fit_policy`` on the
  out-of-fold scores; mandatory reviews of unfinished values included).

Signal sets: each marker state (``key``, ``decide``, ``value``) at each cached
layer, the value summary (token statistics), the close decision, and a few
combinations. The penalty is picked per set from a small grid by the same CV
(optimistic for every set alike); use it to rank signals, not as a result.

Usage::

    HF_HUB_OFFLINE=1 uv run python scripts/probe_signals.py --config configs/extractor/kev_smoke.yaml \\
        --adapter artifacts/kev-smoke --output artifacts/kev-smoke/probe --layers 7 14 21 --set device=cuda:0
    uv run python scripts/probe_signals.py --output artifacts/kev-smoke/probe --analyse-only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from l2r4kie.confidence.cache import load_cache
from l2r4kie.confidence.features import TraceStore
from l2r4kie.confidence.policy import fit_policy
from l2r4kie.eval.metrics import auroc
from l2r4kie.eval.review_metrics import oracle_review_rate


def decode(args: argparse.Namespace) -> None:
    """Decode the probe cohort into ``<output>/probe.pt``."""
    from l2r4kie.data.selection import Selection
    from l2r4kie.model.extractor import Extractor
    from l2r4kie.pipelines.predict import adapter_fingerprint
    from l2r4kie.pipelines.trace_cache import cache_traces
    from l2r4kie.train.config import TrainConfig
    from l2r4kie.utils.config import load_config

    config = TrainConfig.from_dict(load_config(args.config, args.set))
    s = config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    documents, seen = [], defaultdict(int)
    # Never documents of extractor gradients: the train reserve, then dev; per_form of each form.
    for split in ('train_reserve', 'dev'):
        for document in selection.documents(split):
            if seen[(split, document.form)] < args.per_form:
                seen[(split, document.form)] += 1
                documents.append(document)
    extractor = Extractor.load(config.extractor(), args.adapter)
    cache_traces(extractor, documents, args.output, 'probe', args.fields, args.max_value_tokens, args.layers,
                 provenance={'adapter': args.adapter, 'source_fingerprint': adapter_fingerprint(args.adapter)})


def fit_logistic(x: torch.Tensor, y: torch.Tensor, penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """L2 logistic regression (bias unpenalised) by L-BFGS."""
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=200, line_search_fn='strong_wolfe')

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weight + bias, y) + penalty * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach(), bias.detach()


def out_of_fold(x: torch.Tensor, y: torch.Tensor, folds: list[int], penalty: float) -> torch.Tensor:
    """Out-of-fold logits; features standardised on each training fold."""
    scores = torch.zeros(len(y), dtype=torch.float64)
    fold = torch.tensor(folds)
    for k in sorted(set(folds)):
        train, test = fold != k, fold == k
        mean, std = x[train].mean(0), x[train].std(0).clamp_min(1e-3)
        weight, bias = fit_logistic((x[train] - mean) / std, y[train], penalty)
        scores[test] = ((x[test] - mean) / std) @ weight + bias
    return scores


def analyse(args: argparse.Namespace) -> dict:
    """Probe every signal set on the cached cohort; print and write ``probe_report.json``."""
    cache = load_cache(args.output, 'probe')
    valid = cache.valid_rows
    layers = cache.layers
    bases = ('key', 'decide', 'value')
    signals = [*bases, *(f'{b}@{layer}' for b in bases for layer in layers)]
    store = TraceStore(cache.records, signals, layers)
    y = torch.tensor(cache.labels, dtype=torch.float64)
    documents = sorted({r['document_id'] for r in valid})
    random.Random(0).shuffle(documents)
    fold_of = {d: i % 5 for i, d in enumerate(documents)}
    folds = [fold_of[r['document_id']] for r in valid]
    summary, close = store.summary.double(), store.close.double()
    sets: dict[str, torch.Tensor] = {'summary (token statistics)': summary, 'close decision': close}
    for name in signals:
        sets[name] = store.vectors[name].double()
    sets['value + summary'] = torch.cat((store.vectors['value'].double(), summary), 1)
    sets['key + decide + value + summary'] = torch.cat(
        [store.vectors[b].double() for b in bases] + [summary], 1)
    for layer in layers:
        sets[f'value@{layer} + summary'] = torch.cat((store.vectors[f'value@{layer}'].double(), summary), 1)
    report = {'fields': len(cache.rows), 'valid': len(valid), 'documents': len(cache.documents),
              'wrong_valid': int((y == 0).sum()), 'mandatory': len(cache.rows) - len(valid),
              'oracle_review_rate': oracle_review_rate(cache.rows, [0. if r['trace_index'] >= 0 else None
                                                                   for r in cache.rows]),
              'layers': list(layers), 'probes': {}}
    print(json.dumps({k: v for k, v in report.items() if k != 'probes'}), flush=True)
    for name, x in sets.items():
        best = None
        for penalty in args.penalties:
            logits = out_of_fold(x, y, folds, penalty)
            area = auroc(logits.numpy(), y.numpy())
            if best is None or area > best['auroc']:
                best = {'auroc': area, 'penalty': penalty, 'logits': logits}
        probabilities = torch.sigmoid(best['logits'])
        scores = cache.scores(best['logits'])
        policy = fit_policy(cache.rows, scores)
        reviewed = sum(policy.needs_review(s, r['status'], r['field_id']) for r, s in zip(cache.rows, scores))
        report['probes'][name] = {'dims': x.shape[1], 'auroc': best['auroc'], 'penalty': best['penalty'],
                                  'review_rate_at_95': reviewed / len(cache.rows),
                                  'brier': float(((probabilities - y) ** 2).mean())}
        print(json.dumps({'signal': name, **report['probes'][name]}), flush=True)
    (Path(args.output) / 'probe_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', help='extractor run config (selection, model, format)')
    parser.add_argument('--adapter', help='trained run directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--layers', type=int, nargs='*', default=[7, 14, 21])
    parser.add_argument('--per-form', type=int, default=1, help='documents per form from each source split')
    parser.add_argument('--fields', type=int, default=24)
    parser.add_argument('--max-value-tokens', type=int, default=256)
    parser.add_argument('--penalties', type=float, nargs='*', default=[1e-3, 1e-2, 1e-1])
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--analyse-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    if not args.analyse_only:
        decode(args)
    analyse(args)
    np.seterr(all='ignore')


if __name__ == '__main__':
    main()
