"""Correctness heads on the trace of a decoded value (backbone frozen).

A head maps the features of one field (:mod:`.features`) to a logit of
"the extracted value is correct". Modes, from the old ``TokenConfidenceHead``:

==============  ==============================================================
``end``         linear on the signal states (old: ``h_end`` only)
``hybrid``      linear on the signal states + the value summary
``hybrid_mlp``  MLP on the same inputs
``mean``        MLP on states + summary + mean-pooled token states/statistics
``attention``   as ``mean`` with learned attention pooling (r4's winner)
``query``       new: attention pooling whose query is a field state
                (``h_key`` by default), i.e. "which value tokens matter for
                *this* field", the query-key reading of KevFormat
==============  ==============================================================

``signals`` picks the states fed to the head (old heads: ``('value',)``,
the state after the end of the value). Two training-free *heuristics*
(``min_log_probability``, ``mean_log_probability``) are kept as baselines.

Fixes over the old code: the feature-only ``FeatureHead`` and the
"head already in the checkpoint" candidate are gone (the latter was an
untrained head on the new format); old ``token`` heads still load
(:func:`load_legacy_head`), so old selections can be re-checked.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..utils.io import PathLike, read_json, write_json
from .features import TOKEN_STATS, TraceStore, parse_signal, summary_names

MODES: tuple[str, ...] = ('end', 'hybrid', 'hybrid_mlp', 'mean', 'attention', 'query')
HEURISTICS: tuple[str, ...] = ('min_log_probability', 'mean_log_probability')
POOLING: tuple[str, ...] = ('mean', 'attention', 'query')


@dataclass(frozen=True, slots=True)
class HeadConfig:
    """Architecture of a head.

    Attributes:
        mode: One of :data:`MODES` or :data:`HEURISTICS`.
        signals: States concatenated into the head input (see
            :func:`~l2r4kie.confidence.features.parse_signal`).
        query: State that queries the token pool in ``query`` mode.
        width: Hidden width of the MLP and of the token projection.
        dropout: Dropout of the MLP.
        field_keys: Field ids with a learned bias (field prior); others and
            unseen fields use none.
    """

    mode: str = 'attention'
    signals: tuple[str, ...] = ('value',)
    query: str = 'key'
    width: int = 32
    dropout: float = .1
    field_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in MODES + HEURISTICS:
            raise ValueError(f'Unknown head mode {self.mode!r}; use one of {MODES + HEURISTICS}')
        if self.mode not in HEURISTICS and not self.signals:
            raise ValueError('A trained head needs at least one signal')
        for name in (*self.signals, self.query):
            parse_signal(name)

    @property
    def family(self) -> str:
        """Name shared by every configuration of one architecture, e.g. ``attention[value+key]``."""
        if self.mode in HEURISTICS:
            return self.mode
        return f"{self.mode}[{'+'.join(self.signals)}]" + ('+prior' if self.field_keys else '')

    @property
    def needs_tokens(self) -> bool:
        """Whether the head reads token rows (pooling modes)."""
        return self.mode in POOLING

    def required_signals(self) -> tuple[str, ...]:
        """Every state the head reads."""
        return (*self.signals, self.query) if self.mode == 'query' else self.signals

    def to_dict(self) -> dict[str, Any]:
        """JSON form (tuples as lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> HeadConfig:
        """Read :meth:`to_dict` output (sizes and unknown keys are ignored)."""
        known = {k: tuple(v) if isinstance(v, list) else v for k, v in record.items() if k in cls.__slots__}
        return cls(**known)


class ConfidenceHead(nn.Module):
    """A trained head (any mode of :data:`MODES`).

    Args:
        config: Architecture.
        hidden_size: Width of the model states.
        summary_size: Width of the value summary.
        stats_size: Number of per-token statistics.
    """

    def __init__(self, config: HeadConfig, hidden_size: int, summary_size: int, stats_size: int) -> None:
        super().__init__()
        if config.mode not in MODES:
            raise ValueError(f'{config.mode!r} is not a trained head mode')
        self.config = config
        self.sizes = {'hidden_size': hidden_size, 'summary_size': summary_size, 'stats_size': stats_size}
        signals, width = len(config.signals), config.width
        self.register_buffer('vector_mean', torch.zeros(signals, hidden_size))
        self.register_buffer('vector_std', torch.ones(signals, hidden_size))
        self.register_buffer('token_mean', torch.zeros(hidden_size))
        self.register_buffer('token_std', torch.ones(hidden_size))
        self.register_buffer('summary_mean', torch.zeros(summary_size))
        self.register_buffer('summary_std', torch.ones(summary_size))
        self.register_buffer('stats_mean', torch.zeros(stats_size))
        self.register_buffer('stats_std', torch.ones(stats_size))
        inputs = signals * hidden_size + (0 if config.mode == 'end' else summary_size)
        if config.needs_tokens:
            self.project = nn.Linear(hidden_size + stats_size, width)
            inputs += width
        if config.mode == 'attention':
            self.attention = nn.Linear(width, 1, bias=False)
        if config.mode == 'query':
            self.register_buffer('query_mean', torch.zeros(hidden_size))
            self.register_buffer('query_std', torch.ones(hidden_size))
            self.query = nn.Linear(hidden_size, width)
        if config.mode in ('end', 'hybrid'):
            self.network: nn.Module = nn.Linear(inputs, 1)
        else:
            self.network = nn.Sequential(nn.Linear(inputs, width), nn.GELU(), nn.Dropout(config.dropout),
                                         nn.Linear(width, 1))
        if config.field_keys:
            # Index 0 is the fallback for unknown fields: no learned offset.
            self.field_bias = nn.Embedding(len(config.field_keys) + 1, 1, padding_idx=0)
            nn.init.zeros_(self.field_bias.weight)

    def normalize(self, store: TraceStore) -> None:
        """Set the input normalisation from a training store (old ``normalize``)."""
        values = store.normalizers(self.config.signals)
        for name, value in values.items():
            getattr(self, name).copy_(value)
        if self.config.mode == 'query':
            query = store.vectors[self.config.query]
            self.query_mean.copy_(query.mean(0))
            self.query_std.copy_(query.std(0).clamp_min(.1))

    def field_indices(self, field_ids: Sequence[str], device: str | torch.device = 'cpu') -> torch.Tensor:
        """Field-prior index of each field id (0 for fields without a prior)."""
        lookup = {key: i + 1 for i, key in enumerate(self.config.field_keys)}
        return torch.tensor([lookup.get(key, 0) for key in field_ids], dtype=torch.long, device=device)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Logits ``(B,)`` of a :meth:`TraceStore.batch` (plus ``field_index`` with a field prior)."""
        config = self.config
        parts = [(batch['vectors'][name] - self.vector_mean[i]) / self.vector_std[i]
                 for i, name in enumerate(config.signals)]
        if config.mode != 'end':
            parts.append((batch['summary'] - self.summary_mean) / self.summary_std)
        if config.needs_tokens:
            tokens = (batch['tokens'].float() - self.token_mean) / self.token_std
            stats = (batch['stats'] - self.stats_mean) / self.stats_std
            values = torch.tanh(self.project(torch.cat((tokens, stats), -1)))
            mask = batch['mask'] & batch['valid']
            if config.mode == 'attention':
                weight = self.attention(values).squeeze(-1).masked_fill(~mask, -torch.inf).softmax(-1)
            elif config.mode == 'query':
                query = self.query((batch['vectors'][config.query] - self.query_mean) / self.query_std)
                scores = (values * query[:, None]).sum(-1) / math.sqrt(config.width)
                weight = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            else:
                weight = mask.float() / mask.sum(-1, keepdim=True).clamp_min(1)
            parts.append((values * weight[..., None]).sum(1))
        output = self.network(torch.cat(parts, -1)).squeeze(-1)
        if config.field_keys:
            index = batch.get('field_index')
            if index is None:
                index = torch.zeros(len(output), dtype=torch.long, device=output.device)
            output = output + self.field_bias(index).squeeze(-1)
        return output


class HeuristicHead(nn.Module):
    """Training-free baselines from the token statistics of the decode.

    ``min_log_probability``: the least likely decision (any value token or
    the close marker). ``mean_log_probability``: their average.
    """

    def __init__(self, config: HeadConfig, stats: Sequence[str] = TOKEN_STATS) -> None:
        super().__init__()
        if config.mode not in HEURISTICS:
            raise ValueError(f'{config.mode!r} is not a heuristic')
        self.config = config
        names = summary_names(stats)
        self._mean, self._min = names.index(f'mean_{stats[0]}'), names.index('min_log_probability')
        self._length = names.index('log1p_tokens')

    def normalize(self, store: TraceStore) -> None:  # noqa: ARG002
        """Nothing to fit."""

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Score ``(B,)``: higher means more likely correct."""
        summary, close = batch['summary'], batch['close'][:, 0]
        count = torch.expm1(summary[:, self._length]).round()
        if self.config.mode == 'mean_log_probability':
            return (summary[:, self._mean] * count + close) / (count + 1)
        return torch.where(count > 0, torch.minimum(summary[:, self._min], close), close)


Head = ConfidenceHead | HeuristicHead


def make_head(config: HeadConfig, hidden_size: int, summary_size: int, stats_size: int) -> Head:
    """Build a head of any mode."""
    if config.mode in HEURISTICS:
        return HeuristicHead(config)
    return ConfidenceHead(config, hidden_size, summary_size, stats_size)


@torch.inference_mode()
def score_head(head: Head, store: TraceStore, field_ids: Sequence[str] | None = None,
               batch_size: int = 128) -> torch.Tensor:
    """Logits of every record of ``store`` (CPU, float32), in eval mode.

    Args:
        head: Head to apply.
        store: Records to score.
        field_ids: Field id of each record (needed by heads with a field prior).
        batch_size: Records per forward.
    """
    head.eval()
    index = None
    if head.config.field_keys:
        if field_ids is None:
            raise ValueError('A head with a field prior needs the field ids')
        index = head.field_indices(field_ids, store.device)
    outputs = []
    for start in range(0, len(store), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(store)), device=store.device)
        batch = store.batch(indices, tokens=head.config.needs_tokens)
        if index is not None:
            batch['field_index'] = index[indices]
        outputs.append(head(batch).float().cpu())
    return torch.cat(outputs)


def save_head(head: Head, folder: PathLike) -> Path:
    """Write ``head.pt`` and ``head_config.json`` (with the input sizes) to ``folder``."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    sizes = getattr(head, 'sizes', {})
    write_json(folder / 'head_config.json', {**head.config.to_dict(), **sizes})
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, folder / 'head.pt')
    return folder


def load_head(folder: PathLike, device: str | torch.device = 'cpu') -> Head:
    """Read a head written by :func:`save_head`."""
    folder = Path(folder)
    record = read_json(folder / 'head_config.json')
    config = HeadConfig.from_dict(record)
    if config.mode in HEURISTICS:
        return HeuristicHead(config).to(device)
    head = ConfidenceHead(config, record['hidden_size'], record['summary_size'], record['stats_size'])
    head.load_state_dict(torch.load(folder / 'head.pt', map_location='cpu', weights_only=True))
    return head.to(device).eval()


@dataclass(frozen=True, slots=True)
class LegacyHead:
    """An old ``kind: token`` head converted to :class:`ConfidenceHead`."""

    head: ConfidenceHead
    source: str
    notes: list[str] = field(default_factory=list)


def load_legacy_head(folder: PathLike, device: str | torch.device = 'cpu') -> LegacyHead:
    """Load an old ``head.pt`` + ``head_config.json`` (``kind: token``) as a new head.

    The old ``end`` feature becomes the ``value`` signal; buffers ``mean``/``std``
    become ``vector_mean``/``vector_std``; every other weight keeps its name.

    Raises:
        ValueError: For the removed ``linear``/``feature`` head kinds.
    """
    folder = Path(folder)
    old = read_json(folder / 'head_config.json')
    if old.get('kind') != 'token':
        raise ValueError(f'{folder}: only old "token" heads are supported, got kind={old.get("kind")!r}')
    state = torch.load(folder / 'head.pt', map_location='cpu', weights_only=True)
    state['vector_mean'], state['vector_std'] = state.pop('mean')[None], state.pop('std')[None]
    config = HeadConfig(mode=old['mode'], signals=('value',), width=old.get('width', 32),
                        dropout=old.get('dropout', .1), field_keys=tuple(old.get('field_keys', ())))
    head = ConfidenceHead(config, state['vector_mean'].shape[-1], state['summary_mean'].numel(),
                          state['stats_mean'].numel())
    head.load_state_dict(state)
    return LegacyHead(head.to(device).eval(), str(folder))
