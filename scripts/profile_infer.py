"""Where serving time goes: prefix encode, prefill, decode steps, host overhead (one GPU).

Runs the served decode (``extract``, all fields of each document, trace and
layers as with a confidence bundle) on a fixed sample of a cohort and times:

* ``encode``: page loading and processor (``Extractor.encode_prefix``);
* ``prefill``: the prefix forward (vision + text), timed again on its own;
* ``branch``: the branch-prompt forward of each chunk;
* ``steps``: decode-step forwards (GPU time, synchronised);
* ``host``: everything else in ``decode_prefix`` (lm_head, argmax, token
  statistics, device-to-host copies, the per-row Python loop, compaction).

Per step it also keeps the batch rows and KV length, to show the tail where
a few long values keep the batch running. Writes ``<output>.jsonl`` (one line
per document) and prints a summary.

    CUDA_VISIBLE_DEVICES=0 python scripts/profile_infer.py --split dev --documents 24 \\
        --output artifacts/kev-r4/profile/dev
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import time
from pathlib import Path

import torch

from l2r4kie.confidence.config import ConfidenceConfig
from l2r4kie.data.selection import Selection
from l2r4kie.data.types import FieldRequest
from l2r4kie.model import decode
from l2r4kie.model.extractor import Extractor
from l2r4kie.model.packing import prefix_positions
from l2r4kie.pipelines.trace_cache import resolve_cohorts
from l2r4kie.train.config import TrainConfig
from l2r4kie.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default='configs/confidence/kev.yaml')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--documents', type=int, default=24, help='every k-th document of the cohort, k = n / this')
    parser.add_argument('--no-trace', action='store_true', help='decode without traces/layers (no confidence)')
    parser.add_argument('--max-branches', type=int)
    parser.add_argument('--max-value-tokens', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config = ConfidenceConfig.from_dict(load_config(args.config))
    extractor_config = TrainConfig.from_dict(load_config(config.extractor))
    s = extractor_config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    documents = sorted(resolve_cohorts(config.cohort_plan, selection, [args.split],
                                       extractor_config.train_documents)[args.split], key=lambda d: d.id)
    documents = documents[::max(1, len(documents) // args.documents)][:args.documents]
    overrides = {'device': args.device}
    if args.max_branches:
        overrides['max_branches'] = args.max_branches
    extractor = Extractor.load(dataclasses.replace(extractor_config.extractor(), **overrides), config.adapter)
    layers = () if args.no_trace else tuple(config.layers)

    steps: list[tuple[int, int, int, float]] = []  # (rows, query length, KV length, seconds)
    original = decode._forward

    def timed(core, cache, state, **inputs):  # noqa: ANN001, ANN202
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = original(core, cache, state, **inputs)
        torch.cuda.synchronize()
        ids = inputs['input_ids']
        steps.append((ids.shape[0], ids.shape[1], cache.get_seq_length(), time.perf_counter() - start))
        return output

    decode._forward = timed
    out = Path(args.output).with_suffix('.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, document in enumerate([documents[0], *documents]):  # the first run warms up and is dropped
        requests = [FieldRequest(f.id, f.description, f.kind) for f in document.fields]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefix = extractor.encode_prefix(list(document.pages))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        steps.clear()
        results = decode.decode_prefix(extractor, prefix, requests, max_value_tokens=args.max_value_tokens,
                                       trace=not args.no_trace, layers=layers)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.inference_mode():  # the prefix forward alone (the decode above ran it once too)
            positions = prefix_positions(extractor.core, prefix)
            extractor.core(**prefix, position_ids=positions, use_cache=True)
        torch.cuda.synchronize()
        prefill = time.perf_counter() - t2
        if k == 0:
            continue
        branch = sum(t for _, q, _, t in steps if q > 1)
        decode_steps = [(r, kv, t) for r, q, kv, t in steps if q == 1]
        step_time = sum(t for *_, t in decode_steps)
        tokens = sorted(len(r.text) for r in results)
        line = {'document_id': document.id, 'form': document.form, 'pages': len(document.pages),
                'fields': len(requests), 'prefix_tokens': int(prefix['input_ids'].shape[1]),
                'seconds': t2 - t0, 'encode': t1 - t0, 'prefill': prefill, 'branch': branch, 'steps': step_time,
                'host': (t2 - t1) - prefill - branch - step_time, 'decode_steps': len(decode_steps),
                'step_ms_median': 1000 * statistics.median(t for *_, t in decode_steps) if decode_steps else None,
                'steps_with_le_2_rows': sum(r <= 2 for r, *_ in decode_steps),
                'seconds_with_le_2_rows': sum(t for r, _, t in decode_steps if r <= 2),
                'status': {s: sum(r.status == s for r in results) for s in {r.status for r in results}},
                'stop': {s: sum(r.stop == s for r in results) for s in {r.stop for r in results}},
                'longest_text_chars': tokens[-1] if tokens else 0,
                'trace_chunks': [[r, kv, round(1000 * t, 2)] for r, kv, t in decode_steps[::25]]}
        lines.append(line)
        print(json.dumps({k: line[k] for k in ('document_id', 'fields', 'prefix_tokens', 'seconds', 'encode',
                                               'prefill', 'branch', 'steps', 'host', 'decode_steps',
                                               'step_ms_median', 'seconds_with_le_2_rows', 'stop')}), flush=True)
    out.write_text(''.join(json.dumps(line) + '\n' for line in lines), encoding='utf-8')
    total = sum(line['seconds'] for line in lines)
    print(json.dumps({'documents': len(lines), 'total_seconds': total,
                      'median_seconds': statistics.median(line['seconds'] for line in lines),
                      'share': {k: sum(line[k] for line in lines) / total
                                for k in ('encode', 'prefill', 'branch', 'steps', 'host')},
                      'share_le_2_rows': sum(line['seconds_with_le_2_rows'] for line in lines) / total}, indent=2))


if __name__ == '__main__':
    main()
