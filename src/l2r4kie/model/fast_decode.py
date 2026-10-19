"""Fast decoding: generate values without traces, then trace them in one teacher-forced pass.

Serving time is spent almost entirely in the tail of a document's decode,
when one or two long values (tables) are still running token by token, and
each eager step costs the same launch overhead whatever the batch
(``docs/results/infer_speed.md``). This module splits decoding in two:

1. :func:`generate` picks the tokens greedily from the same masked logits as
   :func:`~l2r4kie.model.decode.decode_prefix`, with the same early stop.
   Wide batches run eagerly; once at most :data:`TAIL_ROWS` branches are
   left, they continue on a :class:`~transformers.StaticCache` whose step is
   captured as a CUDA graph (``torch.compile(mode='reduce-overhead')``, one
   graph per (rows, cache length) bucket, kept for the process). No state
   is copied to the host but the chosen tokens.
2. :func:`trace_tokens` feeds every branch's prompt and generated tokens in
   one forward over the prefix cache (branches as left-padded rows, as the
   branch prompts are fed in decoding) and reads the marker states,
   intermediate layers, post-token states and pre-token statistics from it.
   In FP32 these equal the incremental decode's (``tests/test_fast_decode.py``).

The traces are those of the generated tokens, so a cache built with
``fast_decode`` and a server using it see the same features; values can
differ from eager decoding where bfloat16 kernels break a near tie.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..data.types import FieldRequest
from .decode import TOKEN_STATS, DecodeResult, Signals, Trace, _forward, token_stats
from .early_stop import BranchStop, token_piece
from .extractor import Extractor
from .format import ParsedValue
from .packing import prefix_positions

#: Branches still running when generation moves to the static-cache graph.
TAIL_ROWS = 4
#: Most tokens (rows x longest row) fed at once by :func:`trace_tokens`.
TRACE_TOKENS = 16_384
#: Smallest static-cache length bucket.
MIN_CACHE = 4096
#: Free slots a bucket must have when chosen (a full bucket moves to the next one).
RESERVE = 1024


@torch.inference_mode()
def fast_decode_prefix(extractor: Extractor, prefix: Mapping[str, torch.Tensor], requests: Sequence[FieldRequest],
                       limit: int, trace: bool, layers: tuple[int, ...], early_stop: bool) -> list[DecodeResult]:
    """:func:`generate`, then :func:`trace_tokens` (same results as ``decode_prefix``, see the module doc)."""
    extractor.model.eval()
    positions = prefix_positions(extractor.core, prefix)
    prefill = extractor.core(**prefix, position_ids=positions, use_cache=True)
    shared = (prefill.past_key_values, int(positions.max()) + 1, prefix['input_ids'].shape[1])
    tokens, stops = generate(extractor, shared, requests, limit, early_stop)
    return trace_tokens(extractor, shared, requests, tokens, stops, trace, layers)


def _copy(cache: Any) -> Any:
    return copy.deepcopy(cache)


def generate(extractor: Extractor, shared: tuple[Any, int, int], requests: Sequence[FieldRequest], limit: int,
             early_stop: bool) -> tuple[list[list[int]], list[str | None]]:
    """Greedy tokens of every branch, close marker included when reached, and early-stop reasons.

    Args:
        extractor: Loaded model and format.
        shared: Prefix KV cache (not modified), first branch position, prefix length.
        requests: Fields to decode.
        limit: Decode budget per field.
        early_stop: Stop branches that can no longer be ``'ok'``.
    """
    tokens: list[list[int]] = []
    stops: list[str | None] = []
    chunk = extractor.config.max_branches
    for start in range(0, len(requests), chunk):
        part_tokens, part_stops = _generate_chunk(extractor, shared, requests[start:start + chunk], limit, early_stop)
        tokens.extend(part_tokens)
        stops.extend(part_stops)
    return tokens, stops


def _branch_inputs(extractor: Extractor, rows: Sequence[Sequence[int]], base: int, shared: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-padded ids, M-RoPE positions (3, n, longest) and attention mask over prefix + rows."""
    device, pad = extractor.device, extractor.format.tokenizer.pad_token_id
    longest = max(map(len, rows))
    ids = torch.full((len(rows), longest), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, longest - len(row):] = torch.tensor(row, device=device)
        mask[i, longest - len(row):] = 1
    positions = (base + (mask.cumsum(-1) - 1).clamp_min(0))[None].expand(3, -1, -1)
    attention = torch.cat([torch.ones((len(rows), shared), dtype=torch.long, device=device), mask], -1)
    return ids, positions, attention


def _masked_argmax(extractor: Extractor, hidden: torch.Tensor, banned: torch.Tensor) -> torch.Tensor:
    logits = extractor.lm_head(hidden)
    logits[:, banned] = torch.finfo(logits.dtype).min
    return logits.argmax(-1)


def _generate_chunk(extractor: Extractor, shared: tuple[Any, int, int], requests: Sequence[FieldRequest],
                    limit: int, early_stop: bool) -> tuple[list[list[int]], list[str | None]]:
    fmt, core, device = extractor.format, extractor.core, extractor.device
    prefix_cache, base, length = shared
    n, close, pad = len(requests), fmt.markers.value_close, fmt.tokenizer.pad_token_id
    banned = torch.tensor(fmt.banned_ids(extractor.vocab_size), dtype=torch.long, device=device)
    cache = _copy(prefix_cache)
    cache.batch_repeat_interleave(n)
    prompts = [fmt.request(r.id, r.description).prompt for r in requests]
    ids, positions, attention = _branch_inputs(extractor, prompts, base, length)
    state: dict[str, bool] = {}
    hidden = _forward(core, cache, state, input_ids=ids, attention_mask=attention, position_ids=positions,
                      use_cache=True).last_hidden_state[:, -1]
    lengths = torch.tensor([len(p) for p in prompts], device=device)
    tokens: list[list[int]] = [[] for _ in range(n)]
    stops: list[str | None] = [None] * n
    done = [False] * n
    piece = token_piece(fmt.tokenizer) if early_stop else None
    detectors = [BranchStop(r.kind, piece) for r in requests] if early_stop else None

    def record(i: int, token: int) -> None:
        tokens[i].append(token)
        if token == close:
            done[i] = True
        elif detectors is not None:
            stops[i] = detectors[i].push(token)
            done[i] = stops[i] is not None

    active = list(range(n))
    step = 0
    while step < limit:
        running = [row for row, i in enumerate(active) if not done[i]]
        if not running:
            break
        if len(running) < len(active) and (len(running) <= len(active) // 2 or len(running) <= TAIL_ROWS):
            keep = torch.tensor(running, device=device)
            cache.batch_select_indices(keep)
            attention, hidden, lengths = attention[keep], hidden[keep], lengths[keep]
            active = [active[row] for row in running]
        if len(active) <= TAIL_ROWS:
            _graph_tail(extractor, cache, attention, hidden, base + lengths, step, limit, active, record, done,
                        banned)
            break
        chosen = _masked_argmax(extractor, hidden, banned)
        chosen = chosen.masked_fill(torch.tensor([done[i] for i in active], device=device), pad)
        attention = torch.cat([attention, torch.ones((len(active), 1), dtype=torch.long, device=device)], -1)
        step_positions = (base + lengths + step)[None, :, None].expand(3, -1, -1)
        for row, token in enumerate(chosen.tolist()):
            if not done[active[row]]:
                record(active[row], token)
        hidden = _forward(core, cache, state, input_ids=chosen[:, None], attention_mask=attention,
                          position_ids=step_positions, use_cache=True).last_hidden_state[:, 0]
        step += 1
    return tokens, stops


class _Graph:
    """A static KV cache and its decode step for one (rows, length) shape."""

    def __init__(self, extractor: Extractor, rows: int, length: int, banned: torch.Tensor) -> None:
        from transformers import StaticCache

        core, device = extractor.core, extractor.device
        self.rows, self.length = rows, length
        self.cache = StaticCache(config=core.config, max_batch_size=rows, max_cache_len=length, device=device,
                                 dtype=extractor.dtype)
        self.mask = torch.full((rows, 1, 1, length), torch.finfo(extractor.dtype).min, dtype=extractor.dtype,
                               device=device)
        text, head, minimum = core.language_model, extractor.lm_head, torch.finfo(extractor.dtype).min

        def step(token: torch.Tensor, positions: torch.Tensor, cache_position: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
            hidden = text(input_ids=token, position_ids=positions, attention_mask=mask, past_key_values=self.cache,
                          use_cache=True, cache_position=cache_position).last_hidden_state[:, -1]
            logits = head(hidden)
            logits[:, banned] = minimum
            return logits.argmax(-1)

        # Static layers are allocated (and marked as static addresses) on their first update: run it
        # eagerly, so the cache lives outside the CUDA-graph pool (one visible slot: no all-masked softmax).
        self.mask[:, 0, 0, 0] = 0
        pad = extractor.format.tokenizer.pad_token_id
        warmup = (torch.full((rows, 1), pad, dtype=torch.long, device=device),
                  torch.zeros((3, rows, 1), dtype=torch.long, device=device),
                  torch.zeros(1, dtype=torch.long, device=device), self.mask)
        step(*warmup)
        self.step = step
        if device.type == 'cuda':
            compiled = torch.compile(step, mode='reduce-overhead', dynamic=False)

            def graphed(*inputs: torch.Tensor) -> torch.Tensor:
                torch.compiler.cudagraph_mark_step_begin()  # outputs of earlier runs are already cloned
                return compiled(*inputs)

            self.step = graphed
            for _ in range(2):  # compile, then record the graph
                self.step(*warmup)

    def load(self, cache: Any, attention: torch.Tensor) -> None:
        """Copy a dynamic cache's rows (and their padding mask) into this graph's cache."""
        rows, filled = attention.shape[0], attention.shape[1]
        for target, source in zip(self.cache.layers, cache.layers, strict=True):
            target.keys[:rows, :, :filled].copy_(source.keys[:rows])
            target.values[:rows, :, :filled].copy_(source.values[:rows])
        self.mask.fill_(torch.finfo(self.mask.dtype).min)
        self.mask[:rows, 0, 0, :filled].masked_fill_(attention.bool(), 0)
        self.mask[rows:, 0, 0, 0] = 0  # spare rows attend to one slot: no all-masked softmax


def _graphs(extractor: Extractor) -> dict[tuple[int, int], _Graph]:
    graphs = getattr(extractor, '_static_graphs', None)
    if graphs is None:
        graphs = extractor._static_graphs = {}  # type: ignore[attr-defined]
        for name in ('cache_size_limit', 'recompile_limit'):  # one compiled entry per shape bucket
            if hasattr(torch._dynamo.config, name):
                setattr(torch._dynamo.config, name, max(64, getattr(torch._dynamo.config, name)))
    return graphs


def warm_up(extractor: Extractor, longest: int) -> list[tuple[int, int]]:
    """Build (and capture) every tail graph up to a KV length of ``longest``, as a server does at start.

    Graphs are otherwise compiled on first use (several seconds each). Returns the shapes built.
    """
    banned = torch.tensor(extractor.format.banned_ids(extractor.vocab_size), dtype=torch.long,
                          device=extractor.device)
    graphs = _graphs(extractor)
    lengths = [MIN_CACHE]
    while lengths[-1] < longest:
        lengths.append(2 * lengths[-1])
    shapes = sorted({_bucket(rows, length) for rows in range(1, TAIL_ROWS + 1) for length in lengths})
    with torch.inference_mode():
        for shape in shapes:
            if shape not in graphs:
                graphs[shape] = _Graph(extractor, *shape, banned)
    return shapes


def _bucket(rows: int, length: int) -> tuple[int, int]:
    size = 1
    while size < rows:
        size *= 2
    cache = MIN_CACHE
    while cache < length:
        cache *= 2
    return size, cache


def _graph_tail(extractor: Extractor, cache: Any, attention: torch.Tensor, hidden: torch.Tensor,
                starts: torch.Tensor, step: int, limit: int, active: list[int], record: Any, done: list[bool],
                banned: torch.Tensor) -> None:
    """Finish the last branches on a static cache (see the module doc); ``record`` takes each token."""
    device, pad = extractor.device, extractor.format.tokenizer.pad_token_id
    graphs = _graphs(extractor)
    rows = len(active)
    filled = attention.shape[1]
    graph = None
    token = _masked_argmax(extractor, hidden, banned)
    while step < limit:
        for row, value in enumerate(token.tolist()[:rows]):
            if not done[active[row]]:
                record(active[row], value)
        if all(done[i] for i in active):
            return
        if graph is None or filled >= graph.length:  # first use, or the bucket is full: move to a longer one
            shape = _bucket(rows, filled + 1 + min(RESERVE, limit - step))
            moved = graph
            if shape not in graphs:
                graphs[shape] = _Graph(extractor, *shape, banned)
            graph = graphs[shape]
            if moved is None:
                graph.load(cache, attention)
            else:
                graph.load(_StaticView(moved, filled), torch.ones((rows, filled), dtype=torch.long, device=device)
                           * (moved.mask[:rows, 0, 0, :filled] == 0))
        fed = torch.full((graph.rows, 1), pad, dtype=torch.long, device=device)
        fed[:rows, 0] = token[:rows].masked_fill(torch.tensor([done[i] for i in active], device=device), pad)
        positions = torch.zeros((3, graph.rows, 1), dtype=torch.long, device=device)
        positions[:, :rows, 0] = (starts + step)[None]
        graph.mask[:rows, 0, 0, filled] = 0
        token = graph.step(fed, positions, torch.tensor([filled], device=device), graph.mask).clone()
        filled += 1
        step += 1


class _StaticView:
    """A filled static cache seen as the dynamic cache :meth:`_Graph.load` copies from."""

    def __init__(self, graph: _Graph, filled: int) -> None:
        self.layers = [_Layer(layer.keys[:, :, :filled], layer.values[:, :, :filled]) for layer in graph.cache.layers]


class _Layer:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys, self.values = keys, values


def trace_tokens(extractor: Extractor, shared: tuple[Any, int, int], requests: Sequence[FieldRequest],
                 tokens: Sequence[Sequence[int]], stops: Sequence[str | None], trace: bool = True,
                 layers: tuple[int, ...] = ()) -> list[DecodeResult]:
    """Decode results of given tokens, with the signals (and traces) of feeding them in one pass.

    Args:
        extractor: Loaded model and format.
        shared: Prefix KV cache (not modified), first branch position, prefix length.
        requests: The fields.
        tokens: Generated tokens per field, close marker included when reached.
        stops: Early-stop reason per field (``None``: closed or budget).
        trace: Also return per-token traces.
        layers: ``hidden_states`` indices of the extra marker states.
    """
    fmt, core, device = extractor.format, extractor.core, extractor.device
    prefix_cache, base, length = shared
    close = fmt.markers.value_close
    banned = torch.tensor(fmt.banned_ids(extractor.vocab_size), dtype=torch.long, device=device)
    allowed = extractor.vocab_size - len(banned)
    prompts = [fmt.request(r.id, r.description).prompt for r in requests]
    rows = [[*p, *t] for p, t in zip(prompts, tokens, strict=True)]
    # Similar lengths together, few rows per pass when rows are long.
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    groups: list[list[int]] = []
    for i in order:
        if groups and len(groups[-1]) < extractor.config.max_branches and \
                (len(groups[-1]) + 1) * len(rows[i]) <= TRACE_TOKENS:
            groups[-1].append(i)
        else:
            groups.append([i])
    states: dict[int, tuple[Any, ...]] = {}
    deep = bool(layers)
    for group in groups:
        cache = _copy(prefix_cache)
        cache.batch_repeat_interleave(len(group))
        ids, positions, attention = _branch_inputs(extractor, [rows[i] for i in group], base, length)
        output = _forward(core, cache, {}, input_ids=ids, attention_mask=attention, position_ids=positions,
                          use_cache=True, output_hidden_states=deep)
        longest = ids.shape[1]
        for row, i in enumerate(group):
            start = longest - len(rows[i])
            key, decide = start + len(prompts[i]) - 2, start + len(prompts[i]) - 1
            count = len(tokens[i])
            hidden = output.last_hidden_state[row]
            closed = count > 0 and tokens[i][-1] == close
            value = decide + count if closed else None
            stats = None
            if trace and count:
                chosen = torch.tensor(tokens[i], device=device)
                pieces = []
                for at in range(0, count, 2048):  # logits over the vocabulary, a slice at a time
                    logits = extractor.lm_head(hidden[decide + at:decide + min(count, at + 2048)])
                    logits[:, banned] = torch.finfo(logits.dtype).min
                    pieces.append(token_stats(logits, chosen[at:at + 2048], allowed, close).cpu())
                stats = torch.cat(pieces)
            by_layer = None
            if deep:
                by_layer = torch.full((3, len(layers), hidden.shape[-1]), torch.nan)
                for j, layer in enumerate(layers):
                    states_at = output.hidden_states[layer][row]
                    by_layer[0, j], by_layer[1, j] = states_at[key].float().cpu(), states_at[decide].float().cpu()
                    if value is not None:
                        by_layer[2, j] = states_at[value].float().cpu()
            states[i] = (hidden[key].float().cpu(), hidden[decide].float().cpu(),
                         hidden[value].float().cpu() if value is not None else None, by_layer,
                         hidden[decide + 1:decide + 1 + count].float().cpu() if trace else None, stats)
        del output, cache
    results = []
    for i, request in enumerate(requests):
        h_key, h_decide, h_value, by_layer, token_hidden, stats = states[i]
        closed = h_value is not None
        parsed = fmt.parse(list(tokens[i][:-1] if closed else tokens[i]), request.kind, closed=closed)
        if stops[i] == 'invalid_json':
            parsed = ParsedValue(parsed.text, None, 'invalid_array')
        record = None
        if trace:
            size = h_key.shape[-1]
            record = Trace(tuple(tokens[i]), token_hidden if len(tokens[i]) else torch.empty((0, size)),
                           stats if stats is not None else torch.empty((0, len(TOKEN_STATS))))
        results.append(DecodeResult(request.id, parsed.value, parsed.text, parsed.status,
                                    Signals(h_key, h_decide, h_value, by_layer), record,
                                    'close' if closed else stops[i] or 'budget'))
    return results
