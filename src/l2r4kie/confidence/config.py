"""Typed, strictly validated configuration of a confidence run (``configs/confidence/*.yaml``).

Layout::

    extractor, adapter, cohort_plan, cache, output, device, ...   # top level
    heads:
      seeds, steps, eval_steps, batch_size, ...                   # training
      heuristics: [min_log_probability, ...]                      # untrained baselines
      grid:                                                        # architectures tried
        - {mode: [attention], signals: [[value], [value, key]], l2: [0.01], rank: [0.0, 0.3], prior: [false]}

Every combination of a grid entry is trained once per seed; dev keeps the
best snapshot of each *family* (mode + signals + prior) and picks the
primary head among families (see :mod:`l2r4kie.pipelines.head_selection`).
Unknown keys at any level are errors, as in :mod:`l2r4kie.train.config`.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .heads import HEURISTICS, MODES


@dataclass(frozen=True, slots=True)
class GridEntry:
    """One block of the head grid (every combination is tried).

    Attributes:
        mode: Head modes (:data:`~l2r4kie.confidence.heads.MODES`).
        signals: Signal sets, each a list of signal names.
        l2: L2 penalties on all head parameters.
        rank: Weights of the same-field ranking loss (0 disables it).
        prior: Whether to learn a per-field bias.
    """

    mode: tuple[str, ...]
    signals: tuple[tuple[str, ...], ...] = (('value',),)
    l2: tuple[float, ...] = (.001, .01)
    rank: tuple[float, ...] = (0., .3)
    prior: tuple[bool, ...] = (False,)

    def __post_init__(self) -> None:
        unknown = [m for m in self.mode if m not in MODES]
        if unknown:
            raise ValueError(f'Unknown head mode(s) {unknown}; use {MODES}')

    def combinations(self) -> Iterator[tuple[str, tuple[str, ...], float, float, bool]]:
        """``(mode, signals, l2, rank, prior)`` in a fixed order."""
        return itertools.product(self.mode, self.signals, self.l2, self.rank, self.prior)


@dataclass(frozen=True, slots=True)
class HeadTraining:
    """How heads are trained and compared (defaults: the r4 v2 run).

    Attributes:
        seeds: Every grid combination is trained once per seed.
        steps: Optimiser steps per head.
        eval_steps: Steps at which dev is scored (snapshots compete).
        batch_size: Records per step.
        pair_batch: Same-field (right, wrong) pairs per step for the ranking loss.
        width: Hidden width of MLPs and token projections.
        dropout: MLP dropout.
        lr_linear: Adam rate of ``end``/``hybrid`` heads.
        lr_mlp: Adam rate of the other heads.
        grad_clip: Gradient norm clip.
        min_field_count: Fields with at least this many training records get a
            prior bias (heads with ``prior``).
        heuristics: Untrained baselines scored on dev alongside the heads.
        grid: Architectures tried.
    """

    seeds: tuple[int, ...] = (107, 108, 109)
    steps: int = 300
    eval_steps: tuple[int, ...] = (75, 150, 300)
    batch_size: int = 128
    pair_batch: int = 32
    width: int = 32
    dropout: float = .1
    lr_linear: float = .005
    lr_mlp: float = .001
    grad_clip: float = 5.
    min_field_count: int = 16
    heuristics: tuple[str, ...] = HEURISTICS
    grid: tuple[GridEntry, ...] = (GridEntry(mode=('end', 'hybrid', 'hybrid_mlp', 'mean', 'attention')),)

    def __post_init__(self) -> None:
        unknown = [h for h in self.heuristics if h not in HEURISTICS]
        if unknown:
            raise ValueError(f'Unknown heuristic(s) {unknown}; use {HEURISTICS}')
        if not set(self.eval_steps) <= set(range(1, self.steps + 1)):
            raise ValueError(f'eval_steps {self.eval_steps} must lie in 1..steps ({self.steps})')

    def signals(self) -> tuple[str, ...]:
        """Every signal any grid head reads (``key`` too when a ``query`` head is tried)."""
        names: list[str] = ['value']
        for entry in self.grid:
            for mode, signals, *_ in entry.combinations():
                names.extend(signals)
                if mode == 'query':
                    names.append('key')
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """A confidence run on one frozen extractor.

    Attributes:
        extractor: Extractor run config (model, selection, format).
        adapter: Trained extractor run directory.
        cohort_plan: Frozen ``cohort_plan.json`` (``l2r4kie plan-cohorts``,
            or an old plan to reuse its documents).
        cache: Trace cache directory.
        output: Selection directory (heads, calibration, policies, audit).
        device: Device for decoding and head training.
        fields_per_document: First N fields per document (r4: 24).
        max_value_tokens: Decode budget; ``None`` takes the extractor's.
        layers: Intermediate ``hidden_states`` indices cached for the marker
            states (``value@14`` in a grid needs 14 here).
        max_trace_tokens: Token states kept per field.
        part_size: Documents per resumable cache part.
        target_error_recall: Share of wrong values the policy must catch.
        risk_replicates: Bootstrap draws of the risk-validation lower bound.
        heads: Head training and grid.
    """

    extractor: str
    adapter: str
    cohort_plan: str
    cache: str
    output: str
    device: str = 'cuda:0'
    fields_per_document: int | None = 24
    max_value_tokens: int | None = None
    layers: tuple[int, ...] = ()
    max_trace_tokens: int = 512
    part_size: int = 16
    target_error_recall: float = .95
    risk_replicates: int = 2000
    heads: HeadTraining = field(default_factory=HeadTraining)

    def __post_init__(self) -> None:
        if not 0 < self.target_error_recall <= 1:
            raise ValueError(f'target_error_recall must be in (0, 1], got {self.target_error_recall}')
        from .features import parse_signal

        for name in self.heads.signals():
            _, layer = parse_signal(name)
            if layer is not None and layer not in self.layers:
                raise ValueError(f'Grid signal {name!r} needs layer {layer} in layers {list(self.layers)}')

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> ConfidenceConfig:
        """Build from a loaded YAML mapping; unknown keys at any level are errors."""
        values = _checked(cls, config, 'config')
        heads = _checked(HeadTraining, values.pop('heads', {}) or {}, 'heads')
        if 'grid' in heads:
            heads['grid'] = tuple(GridEntry(**_grid_entry(e, i)) for i, e in enumerate(heads['grid']))
        for key in ('seeds', 'eval_steps', 'heuristics'):
            if key in heads:
                heads[key] = tuple(heads[key])
        if 'layers' in values:
            values['layers'] = tuple(values['layers'] or ())
        return cls(**values, heads=HeadTraining(**heads))

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible dict."""
        return _plain(dataclasses.asdict(self))


def _checked(cls: type, values: Mapping[str, Any], where: str) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f'{where} must be a mapping, got {type(values).__name__}')
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f'Unknown key(s) in {where}: {unknown}; valid keys: {sorted(names)}')
    return dict(values)


def _grid_entry(entry: Mapping[str, Any], index: int) -> dict[str, Any]:
    values = _checked(GridEntry, entry, f'heads.grid[{index}]')
    if 'mode' not in values:
        raise ValueError(f'heads.grid[{index}] needs a mode list')
    out: dict[str, Any] = {k: tuple(v) if isinstance(v, list) else (v,) for k, v in values.items()}
    if 'signals' in values:
        out['signals'] = tuple(tuple(s) if isinstance(s, list) else (s,) for s in values['signals'])
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
