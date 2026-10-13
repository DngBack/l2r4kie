"""Greedy decoding of isolated field branches over one shared prefix.

The decode mirrors :func:`~l2r4kie.model.packing.pack`, so that inference sees
exactly what training saw:

1. The prefix (system prompt, all page images, instruction) is run once with
   ``use_cache=True``; the vision encoder runs only here.
2. Its KV cache is repeated once per branch (``batch_repeat_interleave``).
   Fields beyond ``max_branches`` are decoded in further chunks, each on a
   copy of the prefix cache, so the prefix is never recomputed.
3. Branch prompts are left-padded into one batch. Their M-RoPE positions
   restart at ``base = prefix_positions.max() + 1``, as in packing; padding
   is masked out and does not shift positions.
4. Tokens are chosen greedily from logits where every special token except
   the close marker is masked out, so a value is plain text ended by
   the close marker or by the token budget.

Signals are the hidden states of the last layer (``h_key`` at ``key_close``,
``h_decide`` at ``value_open``, ``h_value`` after feeding the close marker);
they equal the teacher-forced states of :func:`~l2r4kie.model.packing.pack`
on the same tokens (exactly in FP32, see ``tests/test_decode.py``). The same
three states can also be read at intermediate layers (``layers=``), which
costs one extra output per step and no extra forward.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ..data.types import FieldRequest
from .extractor import Extractor
from .packing import prefix_positions

#: Names of the per-token statistics in :attr:`Trace.stats`, in column order.
TOKEN_STATS: tuple[str, ...] = ('log_probability', 'normalized_entropy', 'probability_margin', 'logit_margin',
                                'close_log_probability')

#: Attention backends used once cuDNN attention has failed in a chunk (see :func:`_forward`).
FALLBACK_BACKENDS: tuple[SDPBackend, ...] = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH)


@dataclass(frozen=True, slots=True)
class Signals:
    """Hidden states of one branch at the marker positions (float32, CPU).

    Attributes:
        key: ``h_key`` at ``key_close``: the field has been read.
        decide: ``h_decide`` at ``value_open``: about to write the value.
        value: ``h_value`` after feeding the close marker: the complete value
            has been read. ``None`` when decoding was truncated.
        layers: ``(3, len(layers), hidden)``: the key, decide and value
            states at the intermediate layers requested from :func:`extract`
            (index into the model's ``hidden_states``: 0 is the embedding
            output, the last one equals the signals above). The value row is
            NaN when decoding was truncated. ``None`` when no layer was requested.
    """

    key: torch.Tensor
    decide: torch.Tensor
    value: torch.Tensor | None
    layers: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class Trace:
    """Per-token record of a decoded value, for token-level confidence (step 6).

    Rows are aligned with :attr:`tokens`; the close marker, when generated, is
    the last row (so ``hidden[-1]`` is ``h_value``).

    Attributes:
        tokens: Generated ids, close marker included if generated.
        hidden: ``(n, hidden)`` last-layer state after feeding each token.
        stats: ``(n, 5)`` statistics of the distribution the token was
            chosen from, computed *before* feeding it (see :data:`TOKEN_STATS`
            and :func:`token_stats`). For the close marker's row they describe
            the decision to stop.
    """

    tokens: tuple[int, ...]
    hidden: torch.Tensor
    stats: torch.Tensor


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """Decoded value of one field.

    Attributes:
        field_id: Requested field id.
        value: Parsed value (``str``, ``bool`` or ``list``); ``None`` unless
            ``status == 'ok'``.
        text: Generated text, close marker excluded.
        status: ``'ok'``, ``'truncated'`` (no close marker within
            ``max_value_tokens``) or ``'invalid_array'`` (an array field whose
            text is not a JSON array).
        signals: Marker hidden states.
        trace: Per-token record; only when requested.
    """

    field_id: str
    value: Any
    text: str
    status: str
    signals: Signals
    trace: Trace | None = None


def token_stats(logits: torch.Tensor, chosen: torch.Tensor, allowed: int, close: int) -> torch.Tensor:
    """Statistics of the decoding distribution at the moment a token is chosen.

    Computed in FP32 on the masked logits (the distribution actually decoded
    from). Banned entries must hold a large negative finite value, not
    ``-inf``, so that the entropy stays finite.

    Args:
        logits: ``(batch, vocab)`` masked logits.
        chosen: ``(batch,)`` chosen ids.
        allowed: Number of ids that are not banned (entropy normalisation).
        close: Id of the close marker.

    Returns:
        ``(batch, 5)``: log-probability of the chosen id, entropy divided by
        ``log(allowed)``, top-1 minus top-2 probability, top-1 minus top-2
        logit, and log-probability of the close marker (how much the model
        wanted to stop here; on the close row it equals the first column).
    """
    logits = logits.float()
    logp = logits.log_softmax(-1)
    prob = logp.exp()
    top_logits = logits.topk(2, dim=-1).values
    top_prob = prob.topk(2, dim=-1).values
    return torch.stack((logp.gather(-1, chosen[:, None]).squeeze(-1),
                        -(prob * logp).sum(-1) / math.log(allowed),
                        top_prob[:, 0] - top_prob[:, 1],
                        top_logits[:, 0] - top_logits[:, 1],
                        logp[:, close]), -1)


@torch.inference_mode()
def extract(extractor: Extractor, pages: Sequence[str | Path], requests: Sequence[FieldRequest], *,
            max_value_tokens: int | None = None, trace: bool = False,
            layers: Sequence[int] = ()) -> list[DecodeResult]:
    """Extract fields from the page images of one document.

    Args:
        extractor: Loaded model, processor and format.
        pages: Page image paths, in reading order.
        requests: Fields to extract; ids must be unique.
        max_value_tokens: Decode budget per field, close marker included;
            defaults to ``extractor.config.max_value_tokens``.
        trace: Also return the per-token :class:`Trace` of every field.
        layers: Also return the marker states at these ``hidden_states``
            indices (:attr:`Signals.layers`).

    Returns:
        One result per request, in request order.
    """
    if not requests:
        return []
    return decode_prefix(extractor, extractor.encode_prefix(pages), requests,
                         max_value_tokens=max_value_tokens, trace=trace, layers=layers)


@torch.inference_mode()
def decode_prefix(extractor: Extractor, prefix: Mapping[str, torch.Tensor], requests: Sequence[FieldRequest], *,
                  max_value_tokens: int | None = None, trace: bool = False,
                  layers: Sequence[int] = ()) -> list[DecodeResult]:
    """Decode fields over an already encoded prefix (see :func:`extract`).

    Args:
        extractor: Loaded model and format.
        prefix: Processor outputs on the model device (``input_ids``,
            ``attention_mask`` and, with images, ``pixel_values`` and
            ``image_grid_thw``).
        requests: Fields to extract; ids must be unique.
        max_value_tokens: Decode budget per field (see :func:`extract`).
        trace: Also return per-token traces.
        layers: Intermediate layers of the marker states (see :func:`extract`).

    Raises:
        ValueError: On duplicate field ids, a non-positive budget or a layer
            index outside ``[0, num_hidden_layers]``.
    """
    ids = [r.id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate field ids in request: {sorted({i for i in ids if ids.count(i) > 1})}')
    limit = extractor.config.max_value_tokens if max_value_tokens is None else max_value_tokens
    if limit < 1:
        raise ValueError(f'max_value_tokens must be positive, got {limit}')
    depth = extractor.core.config.text_config.num_hidden_layers
    if any(not 0 <= layer <= depth for layer in layers):
        raise ValueError(f'layers must lie in [0, {depth}], got {list(layers)}')
    if not requests:
        return []
    extractor.model.eval()
    positions = prefix_positions(extractor.core, prefix)
    base = int(positions.max()) + 1
    prefill = extractor.core(**prefix, position_ids=positions, use_cache=True)
    results: list[DecodeResult] = []
    chunk = extractor.config.max_branches
    for start in range(0, len(requests), chunk):
        # The last chunk may consume the prefix cache; earlier ones work on copies.
        last = start + chunk >= len(requests)
        cache = prefill.past_key_values if last else copy.deepcopy(prefill.past_key_values)
        results.extend(_decode_chunk(extractor, cache, base, prefix['input_ids'].shape[1],
                                     requests[start:start + chunk], limit, trace, tuple(layers)))
    return results


def _forward(core: torch.nn.Module, cache: Any, state: dict[str, bool], **inputs: Any) -> Any:
    """Run ``core`` over ``cache``; if cuDNN attention fails, redo the step without it.

    cuDNN fused attention (PyTorch's default SDPA backend on Hopper) can fail
    on long KV caches with ``mha_graph.execute(...).is_good()`` (seen on a
    shared H200 after thousands of decode steps). The failure is raised
    inside some layer, after the layers before it (and that layer itself)
    have appended this step to the cache, so the cache is cropped back to its
    length before the step and the step is redone with :data:`FALLBACK_BACKENDS`.
    The chunk then stays on them (``state['fallback']``). Steps where cuDNN
    works are untouched, so results only change where decoding would crash.

    Args:
        core: The Qwen2-VL core model.
        cache: KV cache, appended to by this step.
        state: Per-chunk flags; ``fallback`` is set after the first failure.
        **inputs: Forward arguments other than ``past_key_values``.

    Raises:
        RuntimeError: Any other error, or a failure on the fallback backends.
    """
    if state.get('fallback'):
        with sdpa_kernel(list(FALLBACK_BACKENDS)):
            return core(past_key_values=cache, **inputs)
    length = cache.get_seq_length()
    try:
        return core(past_key_values=cache, **inputs)
    except RuntimeError as error:
        if 'mha_graph' not in str(error):
            raise
        cache.crop(length)
        state['fallback'] = True
        warnings.warn(f'cuDNN attention failed at KV length {length}; continuing the chunk on '
                      f'{[b.name for b in FALLBACK_BACKENDS]}', RuntimeWarning, stacklevel=2)
        with sdpa_kernel(list(FALLBACK_BACKENDS)):
            return core(past_key_values=cache, **inputs)


def _decode_chunk(extractor: Extractor, cache: Any, base: int, shared: int, requests: Sequence[FieldRequest],
                  limit: int, trace: bool, layers: tuple[int, ...] = ()) -> list[DecodeResult]:
    """Decode up to ``max_branches`` fields in one batch over a prefix cache.

    Args:
        extractor: Loaded model and format.
        cache: Prefix KV cache for one sequence; repeated in place per branch.
        base: First branch position (prefix positions max + 1).
        shared: Prefix length in tokens.
        requests: Fields of this chunk.
        limit: Decode budget per field, close marker included.
        trace: Record per-token traces.
        layers: ``hidden_states`` indices of the extra marker states.
    """
    fmt, core, lm_head, device = extractor.format, extractor.core, extractor.lm_head, extractor.device
    n = len(requests)
    close = fmt.markers.value_close
    pad = fmt.tokenizer.pad_token_id
    banned = torch.tensor(fmt.banned_ids(extractor.vocab_size), dtype=torch.long, device=device)
    allowed = extractor.vocab_size - len(banned)
    cache.batch_repeat_interleave(n)

    # Branch prefill: left-padded prompts, positions counted over real tokens only.
    prompts = [fmt.request(r.id, r.description).prompt for r in requests]
    longest = max(map(len, prompts))
    ids = torch.full((n, longest), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, longest - len(prompt):] = torch.tensor(prompt, device=device)
        mask[i, longest - len(prompt):] = 1
    positions = base + (mask.cumsum(-1) - 1).clamp_min(0)
    attention = torch.cat([torch.ones((n, shared), dtype=torch.long, device=device), mask], -1)
    deep = bool(layers)  # ask for every layer's output only when some are kept
    state: dict[str, bool] = {}  # attention fallback flag of this chunk (see _forward)
    output = _forward(core, cache, state, input_ids=ids, attention_mask=attention,
                      position_ids=positions[None].expand(3, -1, -1), use_cache=True, output_hidden_states=deep)
    hidden = output.last_hidden_state
    # Prompts end with key_close, value_open; left padding puts them last in every row.
    h_key, h_decide = hidden[:, -2].float().cpu(), hidden[:, -1].float().cpu()
    # (n, 3, len(layers), hidden): key, decide, value (NaN until the value closes).
    by_layer = None
    if deep:
        by_layer = torch.full((n, 3, len(layers), hidden.shape[-1]), torch.nan)
        for j, layer in enumerate(layers):
            by_layer[:, 0, j] = output.hidden_states[layer][:, -2].float().cpu()
            by_layer[:, 1, j] = output.hidden_states[layer][:, -1].float().cpu()
    hidden = hidden[:, -1]

    lengths = torch.tensor([len(p) for p in prompts], device=device)
    tokens: list[list[int]] = [[] for _ in range(n)]
    h_value: list[torch.Tensor | None] = [None] * n
    done = [False] * n  # per request; host copy of ``finished``, avoids a device sync per row
    token_hidden: list[list[torch.Tensor]] = [[] for _ in range(n)]
    token_stat: list[list[torch.Tensor]] = [[] for _ in range(n)]
    # Batch row -> request. Finished rows stay in the batch, fed padding, until
    # half of the rows are finished; then the batch is compacted (one copy of
    # the KV cache). Dropping every finished row at once copies the KV at each
    # step and was measured slower on short values, but without compaction a
    # single value that runs to the budget (a looping array, up to 16k tokens)
    # drags every finished row along: each step then appends to, and attends
    # over, the KV of all n rows. Halving bounds the copies to log2(n) per chunk.
    active = list(range(n))
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    for step in range(limit):
        logits = lm_head(hidden)
        logits[:, banned] = torch.finfo(logits.dtype).min
        chosen = logits.argmax(-1).masked_fill(finished, pad)
        stats = token_stats(logits, chosen, allowed, close) if trace else None
        attention = torch.cat([attention, torch.ones((len(active), 1), dtype=torch.long, device=device)], -1)
        step_positions = (base + lengths + step)[None, :, None].expand(3, -1, -1)
        output = _forward(core, cache, state, input_ids=chosen[:, None], attention_mask=attention,
                          position_ids=step_positions, use_cache=True, output_hidden_states=deep)
        hidden = output.last_hidden_state[:, 0]
        # One device-to-host copy per step for the whole batch, not one per row:
        # each copy synchronises, and on a GPU time-sliced with other processes
        # a synchronisation waits for their slices.
        step_hidden = hidden.float().cpu() if trace else None
        step_stats = stats.cpu() if trace else None
        for row, token in enumerate(chosen.tolist()):
            i = active[row]
            if done[i]:
                continue
            tokens[i].append(token)
            if trace:
                token_hidden[i].append(step_hidden[row].clone())  # clone: a view would keep the whole batch row
                token_stat[i].append(step_stats[row].clone())
            if token == close:
                h_value[i] = step_hidden[row].clone() if trace else hidden[row].float().cpu()
                if by_layer is not None:
                    for j, layer in enumerate(layers):
                        by_layer[i, 2, j] = output.hidden_states[layer][row, 0].float().cpu()
                done[i] = True
        running = [row for row, i in enumerate(active) if not done[i]]
        if not running:
            break
        if len(running) <= len(active) // 2:
            keep = torch.tensor(running, device=device)
            cache.batch_select_indices(keep)
            attention, hidden, lengths = attention[keep], hidden[keep], lengths[keep]
            active = [active[row] for row in running]
        finished = torch.tensor([done[i] for i in active], device=device)

    results = []
    for i, request in enumerate(requests):
        closed = h_value[i] is not None
        parsed = fmt.parse(tokens[i][:-1] if closed else tokens[i], request.kind, closed=closed)
        record = None
        if trace:
            size = h_key.shape[-1]
            record = Trace(tuple(tokens[i]),
                           torch.stack(token_hidden[i]) if token_hidden[i] else torch.empty((0, size)),
                           torch.stack(token_stat[i]) if token_stat[i] else torch.empty((0, len(TOKEN_STATS))))
        results.append(DecodeResult(request.id, parsed.value, parsed.text, parsed.status,
                                    Signals(h_key[i], h_decide[i], h_value[i],
                                            None if by_layer is None else by_layer[i]), record))
    return results
