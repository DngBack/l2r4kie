"""Confidence features of one decoded field, and batches of them.

Everything comes from the decode that produced the value; there is no
second pass over the document (old ``token_confidence``). A *trace record*
holds, for one field whose status is ``ok``:

==============  ==============================================================
``value``       ``h_value``: state after feeding the close marker, the whole
                value read (the old ``end`` feature)
``key``         ``h_key``: state at ``key_close``, the field read, no value yet
``decide``      ``h_decide``: state at ``value_open``, about to write
``layers``      ``(3, L, H)``: the same three states at intermediate layers
                (optional, see :attr:`~l2r4kie.model.decode.Signals.layers`)
``tokens``      ``(n, H)`` states after each value token (close excluded),
                at most ``max_tokens`` of them (see :func:`trace_record`)
``stats``       ``(n, K)`` statistics of the distribution each token was
                chosen from (:data:`~l2r4kie.model.decode.TOKEN_STATS`)
``mask``        ``(n,)`` tokens carrying content (not whitespace or JSON syntax)
``close``       ``(K,)`` statistics of the decision to stop
``summary``     fixed-size description of the whole value (:func:`summarize`)
``length``      number of value tokens before any cut
==============  ==============================================================

Signals are named ``value``, ``key``, ``decide`` (last layer) or
``value@14`` (``hidden_states[14]``); a head concatenates the ones it is
configured with.

:class:`TraceStore` keeps the token rows of all records in one flat tensor and
pads only per batch: values are no longer capped at 256 tokens, so padding a
whole cache to its longest value would not fit in memory.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..model.decode import TOKEN_STATS, DecodeResult

#: Kinds of a generated value, one-hot in the summary.
VALUE_KINDS: tuple[str, ...] = ('empty', 'numeric', 'text', 'array')
#: Base signals; ``<base>@<layer>`` selects an intermediate layer.
BASE_SIGNALS: tuple[str, ...] = ('key', 'decide', 'value')

_NUMERIC = re.compile(r'[\d\s.,/:%+\-()]*\d[\d\s.,/:%+\-()]*')
#: Characters that are syntax, not content, in JSON array values (old ``build_trace``).
_JSON_SYNTAX = ' \t\r\n"{}[],:'


def summary_names(stats: Sequence[str] = TOKEN_STATS) -> list[str]:
    """Names of the :func:`summarize` columns for these token statistics."""
    return [*(f'mean_{s}' for s in stats), *(f'std_{s}' for s in stats),
            'min_log_probability', 'q10_log_probability', 'max_normalized_entropy', 'min_probability_margin',
            'max_close_log_probability', *(f'close_{s}' for s in stats),
            'log1p_tokens', 'content_share', 'digit_share', *(f'kind_{k}' for k in VALUE_KINDS)]


def value_kind(text: str, kind: str) -> str:
    """One of :data:`VALUE_KINDS` for a generated value."""
    if kind == 'array':
        return 'array'
    if not text.strip():
        return 'empty'
    return 'numeric' if _NUMERIC.fullmatch(text) else 'text'


def content_mask(pieces: Sequence[str], kind: str) -> list[bool]:
    """Which value tokens carry content.

    Whitespace-only tokens never do; in array values, JSON syntax
    (quotes, brackets, commas, colons) does not either. When no token does
    (e.g. an empty array ``[]``), every token is kept so pooling has inputs.
    """
    strip = _JSON_SYNTAX if kind == 'array' else ' \t\r\n'
    mask = [bool(piece.strip(strip)) for piece in pieces]
    return mask if any(mask) else [True] * len(mask)


def summarize(stats: torch.Tensor, close: torch.Tensor, mask: Sequence[bool], text: str, kind: str) -> torch.Tensor:
    """Fixed-size description of a value (see :func:`summary_names`).

    Args:
        stats: ``(n, K)`` statistics of the value tokens (close excluded),
            columns as :data:`~l2r4kie.model.decode.TOKEN_STATS`.
        close: ``(K,)`` statistics of the close decision.
        mask: Content mask of the value tokens.
        text: Generated value text.
        kind: Requested field kind.
    """
    stats, close = stats.float(), close.float()
    width = close.numel()
    if len(stats):
        logp, entropy, margin, stop = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, -1]
        parts = [stats.mean(0), stats.std(0, unbiased=False),
                 torch.stack((logp.min(), torch.quantile(logp, .1), entropy.max(), margin.min(), stop.max()))]
    else:  # empty value: the close decision is the only one made
        parts = [stats.new_zeros(width), stats.new_zeros(width), stats.new_zeros(5)]
    kinds = torch.zeros(len(VALUE_KINDS))
    kinds[VALUE_KINDS.index(value_kind(text, kind))] = 1
    shape = torch.tensor([math.log1p(len(stats)), sum(mask) / len(mask) if len(mask) else 0.,
                          sum(c.isdigit() for c in text) / max(1, len(text))])
    return torch.cat((*parts, close, shape, kinds))


def trace_record(result: DecodeResult, pieces: Sequence[str], kind: str, max_tokens: int = 512,
                 token_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """Trace record of a field decoded with ``trace=True`` and status ``ok``.

    Args:
        result: Decode result with a trace, closed by the close marker.
        pieces: Decoded text of each value token (close excluded).
        kind: Requested field kind.
        max_tokens: Keep at most this many token rows: the content tokens with
            the lowest log-probability, in their original order. Pooling is
            order-free and errors sit in uncertain tokens; the summary is
            computed on all tokens before the cut.
        token_dtype: Storage dtype of the token states (bf16: what the model
            computed in, half the size of FP32).

    Raises:
        ValueError: If the field did not close or has no trace.
    """
    if result.trace is None or result.signals.value is None:
        raise ValueError(f'{result.field_id}: trace records need a closed value decoded with trace=True')
    trace = result.trace
    hidden, stats = trace.hidden[:-1], trace.stats[:-1]
    if len(pieces) != len(hidden):
        raise ValueError(f'{result.field_id}: {len(pieces)} token pieces for {len(hidden)} value tokens')
    mask = content_mask(pieces, kind)
    record: dict[str, Any] = {
        'key': result.signals.key.float(), 'decide': result.signals.decide.float(),
        'value': result.signals.value.float(), 'close': trace.stats[-1].float(),
        'summary': summarize(stats, trace.stats[-1], mask, result.text, kind), 'length': len(hidden)}
    if result.signals.layers is not None:
        record['layers'] = result.signals.layers.float()
    keep = torch.arange(len(hidden))
    if len(hidden) > max_tokens:
        # Lowest log-probability content tokens first; non-content tokens only if needed.
        order = sorted(range(len(hidden)), key=lambda i: (not mask[i], float(stats[i, 0])))
        keep = torch.tensor(sorted(order[:max_tokens]))
    record['tokens'] = hidden[keep].to(token_dtype)
    record['stats'] = stats[keep].float()
    record['mask'] = torch.tensor(mask, dtype=torch.bool)[keep]
    return record


def parse_signal(name: str) -> tuple[str, int | None]:
    """``'value'`` → ``('value', None)``; ``'value@14'`` → ``('value', 14)``.

    Raises:
        ValueError: On an unknown base signal or a malformed layer.
    """
    base, _, layer = name.partition('@')
    if base not in BASE_SIGNALS or (layer and not layer.isdigit()):
        raise ValueError(f'Unknown signal {name!r}: use key, decide or value, optionally @<layer>')
    return base, int(layer) if layer else None


def signal_vector(record: Mapping[str, Any], name: str, layers: Sequence[int]) -> torch.Tensor:
    """One signal of a trace record.

    Args:
        record: Trace record.
        name: Signal name (see :func:`parse_signal`).
        layers: The ``hidden_states`` indices stored in ``record['layers']``.

    Raises:
        KeyError: If the layer was not cached.
    """
    base, layer = parse_signal(name)
    if layer is None:
        return record[base]
    if 'layers' not in record or layer not in layers:
        raise KeyError(f'Signal {name!r} needs layer {layer} in the trace cache (cached: {list(layers)})')
    return record['layers'][BASE_SIGNALS.index(base), list(layers).index(layer)]


class TraceStore:
    """Trace records of one cohort, ready for batching on a device.

    Per-record tensors are stacked (``vectors[name]``: ``(N, H)``,
    ``summary``, ``close``); token rows of all records sit in one flat tensor
    and are gathered into a padded batch on demand. A record without value
    tokens (empty value) gets one pseudo-token, its ``value`` state with zero
    statistics, so pooling always has an input (as the old ``collate_traces``).

    Args:
        records: Trace records (see :func:`trace_record`).
        signals: Signal names to stack (all heads trained on this store).
        layers: ``hidden_states`` indices stored in the records' ``layers``.
        device: Where batches are built.
    """

    def __init__(self, records: Sequence[Mapping[str, Any]], signals: Sequence[str] = ('value',),
                 layers: Sequence[int] = (), device: str | torch.device = 'cpu') -> None:
        if not records:
            raise ValueError('Cannot build a trace store without records')
        self.device = torch.device(device)
        self.signals = tuple(dict.fromkeys(signals))
        self.vectors = {name: torch.stack([signal_vector(r, name, layers) for r in records]).float().to(device)
                        for name in self.signals}
        self.summary = torch.stack([r['summary'] for r in records]).float().to(device)
        stats_size = records[0]['stats'].shape[-1]
        self.close = torch.stack([r['close'] if 'close' in r else torch.zeros(stats_size)
                                  for r in records]).float().to(device)
        dtype = records[0]['tokens'].dtype
        tokens, stats, mask, lengths = [], [], [], []
        for record in records:
            if len(record['tokens']):
                tokens.append(record['tokens'].to(dtype))
                stats.append(record['stats'].float())
                mask.append(record['mask'])
            else:
                tokens.append(record['value'].to(dtype)[None])
                stats.append(torch.zeros(1, stats_size))
                mask.append(torch.ones(1, dtype=torch.bool))
            lengths.append(len(tokens[-1]))
        # Row 0 is padding: gathered wherever a batch row is shorter than the longest.
        size = tokens[0].shape[-1]
        self.tokens = torch.cat([torch.zeros(1, size, dtype=dtype), *tokens]).to(device)
        self.stats = torch.cat([torch.zeros(1, stats_size), *stats]).to(device)
        self.mask = torch.cat([torch.zeros(1, dtype=torch.bool), *mask]).to(device)
        self.lengths = torch.tensor(lengths, device=device)
        self.offsets = torch.cumsum(self.lengths, 0) - self.lengths + 1

    def __len__(self) -> int:
        return len(self.lengths)

    @property
    def hidden_size(self) -> int:
        """Width of the hidden states."""
        return self.tokens.shape[-1]

    @property
    def summary_size(self) -> int:
        """Width of the summary."""
        return self.summary.shape[-1]

    @property
    def stats_size(self) -> int:
        """Number of per-token statistics."""
        return self.stats.shape[-1]

    def batch(self, indices: torch.Tensor | Sequence[int], tokens: bool = True) -> dict[str, Any]:
        """Features of the given records.

        Returns:
            ``vectors`` (``{signal: (B, H)}``), ``summary``, ``close``,
            ``lengths`` and, with ``tokens``, ``tokens`` ``(B, T, H)``,
            ``stats`` ``(B, T, K)``, ``mask`` and ``valid`` ``(B, T)``
            padded to the longest record of the batch.
        """
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        batch: dict[str, Any] = {'vectors': {k: v[indices] for k, v in self.vectors.items()},
                                 'summary': self.summary[indices], 'close': self.close[indices],
                                 'lengths': self.lengths[indices]}
        if tokens:
            lengths = batch['lengths']
            steps = torch.arange(int(lengths.max()), device=self.device)
            valid = steps[None] < lengths[:, None]
            rows = torch.where(valid, self.offsets[indices][:, None] + steps[None], 0)
            batch.update(tokens=self.tokens[rows], stats=self.stats[rows], mask=self.mask[rows], valid=valid)
        return batch

    def normalizers(self, signals: Sequence[str]) -> dict[str, torch.Tensor]:
        """Means and standard deviations for a head using ``signals`` (old ``normalize``).

        Floors: 0.1 for states and the summary, 0.05 for token statistics.
        Token states and statistics use content tokens only, as before.
        """
        content = self.mask[1:]  # skip the padding row
        tokens, stats = self.tokens[1:][content].float(), self.stats[1:][content]
        vectors = torch.stack([self.vectors[k] for k in signals])
        return {'vector_mean': vectors.mean(1), 'vector_std': vectors.std(1).clamp_min(.1),
                'summary_mean': self.summary.mean(0), 'summary_std': self.summary.std(0).clamp_min(.1),
                'token_mean': tokens.mean(0), 'token_std': tokens.std(0).clamp_min(.1),
                'stats_mean': stats.mean(0), 'stats_std': stats.std(0).clamp_min(.05)}
