"""Train confidence heads on cached traces and select them on dev (``l2r4kie select-heads``).

First half of the old ``token_review_experiment.train_experiment``; the
second half (calibration and risk thresholds) is :mod:`.finalize`. The old
function then waited up to two hours for the risk cache to appear: the two
steps are now separate commands, and nothing here reads calibration, risk
or audit traces.

Procedure (unchanged): for every grid combination and seed, a head is
trained with Adam on the train cohort (BCE, optional same-field ranking
loss, L2) and scored on dev at ``eval_steps``. A snapshot's dev score is the
review rate needed to catch ``target_error_recall`` of the errors, then AURC,
then AUROC. Each *family* (mode + signals + prior) keeps its best snapshot;
the best family is the primary head. Untrained heuristics compete too.

Output (``config.output``)::

    selection.json        primary family, each family's best snapshot and dev report
    trials.json           every snapshot scored (curve points dropped)
    heads/<family>/       head.pt, head_config.json
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..confidence.cache import TraceCache, check_disjoint, check_source, load_cache
from ..confidence.calibration import calibrate
from ..confidence.config import ConfidenceConfig
from ..confidence.features import TraceStore
from ..confidence.heads import ConfidenceHead, HeadConfig, HeuristicHead, Head, save_head, score_head
from ..eval.review_metrics import candidate_key, describe
from ..utils.io import write_json


def ranking_pairs(cache: TraceCache) -> torch.Tensor:
    """``(P, 2)`` record indices (right, wrong) of the same (form, field)."""
    groups: dict[tuple[str, str], tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for row in cache.valid_rows:
        groups[(row['form'], row['field_id'])][int(row['correct'])].append(row['trace_index'])
    pairs = [(p, n) for wrong, right in groups.values() for p in right for n in wrong]
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def dev_report(head: Head, cache: TraceCache, store: TraceStore, target: float) -> tuple[dict[str, Any], torch.Tensor]:
    """Dev report of a head (identity calibration, policy fitted on dev) and its logits."""
    logits = score_head(head, store, [r['field_id'] for r in cache.valid_rows])
    return describe(cache.rows, cache.scores(logits), calibrate(logits).tolist(), target=target), logits


def compact(report: dict[str, Any]) -> dict[str, Any]:
    """Report without the curve points and the long field list (they make trials.json huge)."""
    return {**report, 'curve': {k: v for k, v in report['curve'].items() if k != 'points'},
            'policy': {k: v for k, v in report['policy'].items() if k != 'supported_fields'}}


def train_head(config: HeadConfig, settings: Any, seed: int, l2: float, rank: float, store: TraceStore,
               labels: torch.Tensor, field_ids: Sequence[str], pairs: torch.Tensor,
               on_eval: Any) -> ConfidenceHead:
    """Train one head; call ``on_eval(step, head, loss)`` at each eval step.

    Raises:
        FloatingPointError: On a non-finite loss or gradient.
    """
    device = store.device
    torch.manual_seed(seed)
    head = ConfidenceHead(config, store.hidden_size, store.summary_size, store.stats_size).to(device)
    head.normalize(store)
    index = head.field_indices(field_ids, device) if config.field_keys else None
    lr = settings.lr_linear if config.mode in ('end', 'hybrid') else settings.lr_mlp
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)
    pairs = pairs.to(device)
    size, pair_size = settings.batch_size, settings.pair_batch
    for step in range(1, settings.steps + 1):
        head.train()
        indices = torch.randint(len(store), (size,), generator=generator, device=device)
        if rank and len(pairs):
            chosen = pairs[torch.randint(len(pairs), (pair_size,), generator=generator, device=device)]
            indices = torch.cat((indices, chosen[:, 0], chosen[:, 1]))
        batch = store.batch(indices, tokens=config.needs_tokens)
        if index is not None:
            batch['field_index'] = index[indices]
        optimizer.zero_grad(set_to_none=True)
        logits = head(batch)
        loss = F.binary_cross_entropy_with_logits(logits[:size], labels[indices[:size]])
        if rank and len(pairs):
            # Wrong values of a field must score below right values of the same field.
            loss = loss + rank * F.softplus(logits[size + pair_size:] - logits[size:size + pair_size]).mean()
        loss = loss + l2 * sum(p.square().sum() for p in head.parameters())
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite head loss at step {step}')
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(head.parameters(), settings.grad_clip)):
            raise FloatingPointError(f'Non-finite head gradient at step {step}')
        optimizer.step()
        if step in settings.eval_steps:
            on_eval(step, head, float(loss.detach()))
    return head


def select_heads(config: ConfidenceConfig, fingerprint: str | None = None) -> dict[str, Any]:
    """Train the grid, keep each family's best dev snapshot, freeze the selection.

    Args:
        config: Confidence run.
        fingerprint: Expected extractor fingerprint of the caches (``None``: not checked).

    Returns:
        The content of ``selection.json``.

    Raises:
        ValueError: If a selection already exists in ``config.output``, or
            caches overlap or come from different extractors.
    """
    out = Path(config.output)
    if (out / 'selection.json').exists():
        raise ValueError(f'{out} already holds a frozen selection; use a new output directory')
    torch.set_num_threads(4)
    caches = {s: load_cache(config.cache, s) for s in ('train', 'dev')}
    check_disjoint(caches)
    source = check_source(caches, fingerprint)
    settings, target = config.heads, config.target_error_recall
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    signals = settings.signals()
    stores = {s: c.store(signals, device) for s, c in caches.items()}
    fields = {s: [r['field_id'] for r in c.valid_rows] for s, c in caches.items()}
    labels = torch.tensor(caches['train'].labels, device=device)
    if len(set(labels.tolist())) < 2:
        raise ValueError('Head training needs both right and wrong values in the train cohort')
    counts = Counter(fields['train'])
    field_keys = tuple(sorted(k for k, n in counts.items() if n >= settings.min_field_count))
    pairs = ranking_pairs(caches['train'])
    print(json.dumps({'train_valid': len(stores['train']), 'dev_valid': len(stores['dev']),
                      'train_wrong': int((labels == 0).sum()), 'same_field_pairs': len(pairs),
                      'field_prior_keys': len(field_keys), 'signals': list(signals)}), flush=True)

    winners: dict[str, dict[str, Any]] = {}
    trials: list[dict[str, Any]] = []

    def consider(name: str, head: Head, report: dict[str, Any], extra: dict[str, Any]) -> None:
        trials.append({'name': name, 'family': head.config.family, **extra, 'dev': compact(report)})
        family = head.config.family
        if family not in winners or candidate_key(report) < candidate_key(winners[family]['dev']):
            winners[family] = {'name': name, 'dev': report, 'head': head.config, 'extra': extra,
                               'state': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                               'sizes': getattr(head, 'sizes', {})}

    for heuristic in settings.heuristics:
        head = HeuristicHead(HeadConfig(mode=heuristic, signals=()))
        consider(heuristic, head, dev_report(head, caches['dev'], stores['dev'], target)[0], {})
    started = time.monotonic()
    for seed in settings.seeds:
        for entry in settings.grid:
            for mode, signal_set, l2, rank, prior in entry.combinations():
                head_config = HeadConfig(mode=mode, signals=signal_set, width=settings.width,
                                         dropout=settings.dropout, field_keys=field_keys if prior else ())
                stem = f'{head_config.family}_l2{l2}_rank{rank}_seed{seed}'

                def on_eval(step: int, head: ConfidenceHead, loss: float, stem: str = stem, l2: float = l2,
                            rank: float = rank, seed: int = seed) -> None:
                    report = dev_report(head, caches['dev'], stores['dev'], target)[0]
                    consider(f'{stem}_step{step}', head, report,
                             {'l2': l2, 'rank': rank, 'seed': seed, 'step': step, 'loss': loss})

                train_head(head_config, settings, seed, l2, rank, stores['train'], labels, fields['train'],
                           pairs, on_eval)
                best = winners[head_config.family]
                print(json.dumps({'trained': stem, 'family_best': best['name'],
                                  'dev_review_rate': best['dev']['review']['review_rate'],
                                  'seconds': round(time.monotonic() - started, 1)}), flush=True)

    primary = min(winners, key=lambda family: candidate_key(winners[family]['dev']))
    for family, winner in winners.items():
        if winner['head'].mode in settings.heuristics:
            head: Head = HeuristicHead(winner['head'])
        else:
            head = ConfidenceHead(winner['head'], **winner['sizes'])
            head.load_state_dict(winner['state'])
        save_head(head, out / 'heads' / family)
    selection = {
        'primary': primary, 'source_fingerprint': source, 'cache': str(config.cache),
        'target_error_recall': target, 'selection_rule': 'dev review rate at the target recall, then AURC, then AUROC',
        'seeds': list(settings.seeds), 'trials': len(trials), 'seconds_training': time.monotonic() - started,
        'cohorts': {s: {k: v for k, v in c.metadata.items() if k not in ('timings_seconds', 'document_ids')}
                    for s, c in caches.items()},
        'families': {family: {'name': w['name'], 'head_config': w['head'].to_dict(), **w['extra'],
                              'dev': compact(w['dev'])} for family, w in winners.items()},
        'config': config.to_dict(), 'audit_seen': False}
    write_json(out / 'trials.json', trials)
    write_json(out / 'selection.json', selection)
    print(json.dumps({'primary': primary, 'dev_review_rate': {f: w['dev']['review']['review_rate']
                                                              for f, w in sorted(winners.items())}}, indent=2),
          flush=True)
    return selection
