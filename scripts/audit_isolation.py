"""Isolation audit on real pretrained weights and a real document.

Checks, on one prepared document:

1. **No leak**: changing branch A's value leaves every hidden state of
   branch B unchanged (teacher-forced packing).
2. **Order**: reversing the branch order does not change the signals.
3. **Cached = packed**: the signals of cached decoding equal the
   teacher-forced signals of packing on the same generated tokens.
4. **One vision forward** per document, also when fields are decoded in
   several chunks, and chunking does not change any generated value.

Usage::

    uv run python scripts/audit_isolation.py --prepared artifacts/data --doc <id> [--adapter <dir>] \\
        [--device cuda:0] [--precision float32] [--max-pixels N] [--fields 4] [--output report.json]

Exit code 1 if a check fails. Check 1 must be exact in any precision.
Checks 2-3 must agree to 1e-3 (absolute) in float32; in bfloat16 batch
layout changes kernel accumulation, so they are judged on relative L2
(≤ 5%) and chunking may change some generated values.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

import torch

from l2r4kie.data.types import FieldRequest
from l2r4kie.model.decode import decode_prefix
from l2r4kie.model.extractor import DEFAULT_MODEL, Extractor, ExtractorConfig
from l2r4kie.model.format import DEFAULT_MAX_PIXELS, EncodedBranch
from l2r4kie.model.packing import Packed, pack, prefix_positions
from l2r4kie.pipelines.inspect import find_document
from l2r4kie.utils.io import write_json


def max_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest absolute difference, in float32 on CPU (signals live on CPU, packed states on the model device)."""
    return float((a.float().cpu() - b.float().cpu()).abs().max())


def relative_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest row-wise ``||a - b|| / ||b||``, in float32 on CPU.

    The meaningful bfloat16 measure: last-layer states have entries in the
    hundreds, where one bfloat16 ulp is already 1, so absolute deltas are
    large even when the states agree to a fraction of a percent.
    """
    a, b = a.float().cpu().reshape(-1, a.shape[-1]), b.float().cpu().reshape(-1, b.shape[-1])
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--prepared', default='artifacts/data', help='prepared directory (default: artifacts/data)')
    parser.add_argument('--doc', required=True, help='document id')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--adapter', help='LoRA adapter (default: base model)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
                        help=f'page pixel budget (default: {DEFAULT_MAX_PIXELS}; lower it for a quick CPU run)')
    parser.add_argument('--fields', type=int, default=4, help='fields audited (default: 4)')
    parser.add_argument('--max-value-tokens', type=int, default=64, help='decode budget (default: 64)')
    parser.add_argument('--output', help='write the report JSON here')
    args = parser.parse_args()

    config = ExtractorConfig(model=args.model, device=args.device, precision=args.precision,
                             max_pixels=args.max_pixels)
    extractor = Extractor.load(config, args.adapter)
    fmt, core = extractor.format, extractor.core
    _, document = find_document(args.prepared, args.doc)
    fields = [f for f in document.fields if len(fmt.target_ids(f.value)) <= 256][:args.fields]
    if len(fields) < 2:
        print(f'error: {args.doc} has fewer than 2 short fields', file=sys.stderr)
        return 1
    requests = [FieldRequest.from_field(f) for f in fields]

    visual_calls: list[int] = []
    hook = core.visual.register_forward_hook(lambda *_: visual_calls.append(1))
    with torch.inference_mode():
        prefix = extractor.encode_prefix(document.pages)
        positions = prefix_positions(core, prefix)

        def hidden_of(branches: list[EncodedBranch]) -> tuple[torch.Tensor, Packed]:
            packed = pack(prefix, positions, branches, max_value_tokens=10_000, dtype=extractor.dtype)
            return core(**packed.inputs).last_hidden_state[0], packed

        # 1. leak: change every value token of the first branch (same length, so the
        # sequence layout is identical) and compare every state of the second branch
        branches = [fmt.encode(f) for f in fields]
        first = branches[0]
        changed = [dataclasses.replace(first, target=(*(t + 1 for t in first.target[:-1]), first.target[-1])),
                   *branches[1:]]
        h1, p1 = hidden_of(branches)
        h2, p2 = hidden_of(changed)
        start = int(p1.key_positions[1]) - branches[1].key_index
        leak = max_delta(h1[start:start + branches[1].length], h2[start:start + branches[1].length])
        assert not torch.equal(h1[:start], h2[:start]), 'the changed branch did not change'

        # 2. order: reversed branches give the same signals
        h3, p3 = hidden_of(branches[::-1])
        signal_names = ('key_positions', 'decide_positions', 'value_positions')
        order = max(max_delta(h1[getattr(p1, n)], h3[getattr(p3, n)].flip(0)) for n in signal_names)
        order_relative = max(relative_delta(h1[getattr(p1, n)], h3[getattr(p3, n)].flip(0)) for n in signal_names)

        # 3-4. decode (one chunk, then one field per chunk) vs packing on the generated tokens
        visual_calls.clear()
        results = decode_prefix(extractor, extractor.encode_prefix(document.pages), requests,
                                max_value_tokens=args.max_value_tokens, trace=True)
        decode_visual_calls = len(visual_calls)
        extractor.config = dataclasses.replace(config, max_branches=1)
        chunked = decode_prefix(extractor, prefix, requests, max_value_tokens=args.max_value_tokens)
        closed = [(r, req) for r, req in zip(results, requests, strict=True) if r.signals.value is not None]
        parity: list[float] = []
        parity_relative: list[float] = []
        if closed:
            generated = [EncodedBranch(req.id, fmt.request(req.id, req.description).prompt, r.trace.tokens)
                         for r, req in closed]
            h4, p4 = hidden_of(generated)
            for i, (r, _) in enumerate(closed):
                pairs = [(r.signals.key, h4[p4.key_positions[i]]), (r.signals.decide, h4[p4.decide_positions[i]]),
                         (r.signals.value, h4[p4.value_positions[i]])]
                parity.append(max(max_delta(a, b) for a, b in pairs))
                parity_relative.append(max(relative_delta(a, b) for a, b in pairs))
    hook.remove()

    # float32 must agree almost exactly; bfloat16 is judged on relative L2 (see relative_delta).
    exact = extractor.dtype == torch.float32
    within = (lambda absolute, relative: absolute <= 1e-3) if exact else (lambda absolute, relative: relative <= 0.05)
    report = {
        'document': document.id, 'dtype': str(extractor.dtype), 'fields': [r.id for r in requests],
        'other_branch_hidden_max_delta': leak,
        'reordered_signal_max_delta': order,
        'reordered_signal_relative_l2_delta': order_relative,
        'cached_vs_packed_signal_max_delta': parity,
        'cached_vs_packed_signal_relative_l2_delta': parity_relative,
        'visual_forward_calls_per_document': decode_visual_calls,
        'status': [r.status for r in results],
        'text': [r.text for r in results],
        'chunked_values_match': [r.text for r in chunked] == [r.text for r in results],
    }
    failures = [name for name, ok in (
        ('leak', leak <= 1e-6),
        ('order', within(order, order_relative)),
        ('cached_vs_packed', all(within(a, r) for a, r in zip(parity, parity_relative, strict=True))),
        ('one_visual_forward', decode_visual_calls == 1),
        ('chunked_values_match', report['chunked_values_match'] or extractor.dtype != torch.float32),
    ) if not ok]
    report['failures'] = failures
    report['passed'] = not failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, report)
    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
