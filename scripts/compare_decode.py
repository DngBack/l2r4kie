"""Compare two decode settings on the same documents: speed, values, confidences, review decisions.

Each document is decoded with both settings back to back (all fields, traces
and layers as served with a bundle), so a shared GPU slows both alike. The
first ``--warmup`` documents are decoded but not counted. With ``fast_decode``,
every tail graph is compiled first (``fast_decode.warm_up``, as at server
start); its time is reported as ``warm_up_seconds``.

    CUDA_VISIBLE_DEVICES=0 python scripts/compare_decode.py --a merge_adapter --b merge_adapter,fast_decode \\
        --bundle 'artifacts/kev-r4/confidence/selected-shapes/heads/attention[value@14+value@21]' \\
        --output artifacts/kev-r4/profile/fast_vs_merged.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import torch

from l2r4kie.confidence.bundle import ConfidenceBundle
from l2r4kie.confidence.config import ConfidenceConfig
from l2r4kie.data.selection import Selection
from l2r4kie.data.types import FieldRequest
from l2r4kie.model import fast_decode
from l2r4kie.model.decode import extract
from l2r4kie.model.extractor import Extractor
from l2r4kie.pipelines.trace_cache import resolve_cohorts
from l2r4kie.train.config import TrainConfig
from l2r4kie.utils.config import load_config


def settings(text: str) -> dict[str, bool]:
    return {name: True for name in text.split(',') if name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/confidence/kev.yaml')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--documents', type=int, default=24)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--a', default='merge_adapter', help='comma-separated settings switched on for A')
    parser.add_argument('--b', default='merge_adapter,fast_decode', help='settings for B')
    parser.add_argument('--bundle', required=True, help='confidence bundle scoring both')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = ConfidenceConfig.from_dict(load_config(args.config))
    extractor_config = TrainConfig.from_dict(load_config(config.extractor))
    s = extractor_config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    documents = sorted(resolve_cohorts(config.cohort_plan, selection, [args.split],
                                       extractor_config.train_documents)[args.split], key=lambda d: d.id)
    documents = documents[::max(1, len(documents) // args.documents)][:args.documents]
    a, b = settings(args.a), settings(args.b)
    if a.get('merge_adapter') != b.get('merge_adapter'):
        raise SystemExit('A and B must share merge_adapter (one model in memory)')
    base = dataclasses.replace(extractor_config.extractor(), device='cuda:0', merge_adapter=a.get('merge_adapter', False))
    extractor = Extractor.load(base, config.adapter)
    bundle = ConfidenceBundle.load(args.bundle, 'cuda:0')
    configs = {'a': dataclasses.replace(base, **a), 'b': dataclasses.replace(base, **b)}
    warm = 0.0
    if a.get('fast_decode') or b.get('fast_decode'):  # prefix (< 8k tokens) + branch prompt + budget
        t = time.perf_counter()
        shapes = fast_decode.warm_up(extractor, 8192 + base.max_value_tokens)
        warm = time.perf_counter() - t
        print(json.dumps({'warm_up_seconds': round(warm, 1), 'graphs': shapes}), flush=True)
    totals = {'a': [], 'b': []}
    fields = same_text = same_status = flips = 0
    diffs = []
    for k, document in enumerate([*documents[:args.warmup], *documents]):
        requests = [FieldRequest(f.id, f.description, f.kind) for f in document.fields]
        out = {}
        for name, cfg in configs.items():
            extractor.config = cfg
            torch.cuda.synchronize()
            t = time.perf_counter()
            results = extract(extractor, list(document.pages), requests, trace=True, layers=bundle.layers)
            scores = bundle.score(results, requests, extractor.format.tokenizer)
            reviewed = {q['id'] for q in bundle.review(results, scores, requests)}
            torch.cuda.synchronize()
            out[name] = (results, scores, reviewed, time.perf_counter() - t)
        if k < args.warmup:
            continue
        (ra, sa, va, ta), (rb, sb, vb, tb) = out['a'], out['b']
        totals['a'].append(ta)
        totals['b'].append(tb)
        for x, y, p, q in zip(ra, rb, sa, sb, strict=True):
            fields += 1
            same_status += x.status == y.status
            same_text += x.status == y.status and x.text == y.text
            if p is not None and q is not None:
                diffs.append(abs(p - q))
            flips += (x.field_id in va) != (y.field_id in vb)
        print(json.dumps({'document_id': document.id, 'fields': len(requests), 'a': round(ta, 2),
                          'b': round(tb, 2)}), flush=True)
    diffs.sort()
    report = {'a': a, 'b': b, 'documents': len(totals['a']), 'fields': fields, 'warm_up_seconds': warm,
              'seconds_a': sum(totals['a']), 'seconds_b': sum(totals['b']),
              'median_a': statistics.median(totals['a']), 'median_b': statistics.median(totals['b']),
              'speedup_total': sum(totals['a']) / sum(totals['b']),
              'same_status': same_status / fields, 'same_text': same_text / fields, 'review_flips': flips,
              'confidence_abs_diff': {'median': diffs[len(diffs) // 2], 'p99': diffs[int(.99 * len(diffs))],
                                      'max': diffs[-1]} if diffs else None}
    Path(args.output).write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
