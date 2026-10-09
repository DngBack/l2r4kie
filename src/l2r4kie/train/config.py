"""Typed, strictly validated configuration of an extractor training run.

A run config is YAML (``configs/extractor/*.yaml``) loaded with
:func:`~l2r4kie.utils.config.load_config`, so it can be tweaked from the CLI
(``--set steps=50 --set format.close=im_end``). :meth:`TrainConfig.from_dict`
rejects unknown keys at every level: a typo such as ``gradient_acumulation``
used to be silently ignored by ``config.get``.

Layout::

    model, output, device, precision, seed, steps, ...   # top level
    selection: {prepared, seed, holdout_percent, balanced_forms, forms}
    format:    {close, max_value_tokens, max_pixels}
    lora:      {r, alpha, dropout, target_modules}
    monitor:   {every, splits, documents, fields, max_value_tokens}
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..model.extractor import DEFAULT_MODEL, ExtractorConfig, Precision
from ..model.format import DEFAULT_MAX_PIXELS, DEFAULT_MAX_VALUE_TOKENS
from ..model.markers import CloseToken

#: ``'token'``: mean cross-entropy over all target tokens of a step (as r4), so a
#: long array outweighs short scalars. ``'field'``: mean over fields of each
#: field's mean token loss, so every field counts the same.
LossWeighting = Literal['token', 'field']

#: Keys that may differ between a run and its resume: they change where and
#: how often things are written or measured, not what is trained.
OPERATIONAL_KEYS: frozenset[str] = frozenset({'device', 'output', 'resume', 'save_every', 'log_every', 'monitor'})


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Training documents (see :class:`~l2r4kie.data.selection.Selection`)."""

    prepared: str
    seed: int = 42
    holdout_percent: int = 0
    balanced_forms: bool = False
    forms: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class FormatConfig:
    """Input format and image budget (see :class:`~l2r4kie.model.format.KevFormat`).

    Attributes:
        close: Value close token; ``'im_end'`` is the fallback if the model
            keeps writing box coordinates.
        max_value_tokens: Longest trained target, close included; longer
            fields are skipped. Also the decode budget of the trained model.
        max_pixels: Page pixel budget of the image processor.
    """

    close: CloseToken = 'box_end'
    max_value_tokens: int = DEFAULT_MAX_VALUE_TOKENS
    max_pixels: int = DEFAULT_MAX_PIXELS


@dataclass(frozen=True, slots=True)
class LoraSettings:
    """LoRA adapter of the language model (r4: r16, alpha 32, attention projections)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Periodic greedy decoding during training, logged to ``monitor.jsonl``.

    Attributes:
        every: Run after every ``every`` steps (and once before training);
            0 disables monitoring.
        splits: Which documents: ``'train'`` (the first training documents,
            i.e. seen ones) and/or ``'dev'``.
        documents: Documents per split.
        fields: First fields of each document.
        max_value_tokens: Decode budget; kept small so an untrained model's
            runaway outputs stay cheap (such fields count as truncated).
    """

    every: int = 0
    splits: tuple[Literal['train', 'dev'], ...] = ('dev',)
    documents: int = 4
    fields: int = 8
    max_value_tokens: int = 256


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Everything that defines an extractor training run.

    Attributes:
        output: Run directory (adapter, logs, snapshots).
        selection: Which training documents.
        model: Base model name or path.
        device: Torch device.
        precision: Base model dtype on CUDA (CPU always trains in float32).
        seed: Seed of the global RNGs (LoRA init) and of field sampling.
        train_documents: Use only the first N selected documents (smoke runs).
        format: Input format.
        lora: Adapter shape.
        trainable_markers: Also train the embedding rows of the four marker
            tokens (decision K6; off by default, an ablation).
        init_adapter: Start from this adapter instead of a fresh LoRA.
        steps: Training steps; one step is one document.
        fields_per_document: Fields sampled per step (all if fewer fit).
        gradient_accumulation: Steps per optimizer update.
        lr: Peak learning rate (AdamW).
        weight_decay: AdamW weight decay (torch default 0.01, as r4).
        warmup_updates: Linear warmup length in optimizer updates.
        cosine: Cosine decay after warmup; otherwise constant.
        max_grad_norm: Gradient clipping norm.
        loss_weighting: See :data:`LossWeighting`.
        loss_chunk: Target positions per cross-entropy chunk; bounds the
            memory of vocabulary logits.
        gradient_checkpointing: Recompute layers in backward (needed for long
            packed sequences with dense masks).
        log_every: Print every N steps (all steps go to ``train.jsonl``).
        save_every: Snapshot every N steps (0: only at the end).
        resume: Resume from the latest snapshot in ``output`` if there is one.
        monitor: Periodic decoding.
    """

    output: str
    selection: SelectionConfig
    model: str = DEFAULT_MODEL
    device: str = 'cuda:0'
    precision: Precision = 'bfloat16'
    seed: int = 42
    train_documents: int | None = None
    format: FormatConfig = field(default_factory=FormatConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    trainable_markers: bool = False
    init_adapter: str | None = None
    steps: int = 100
    fields_per_document: int = 4
    gradient_accumulation: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_updates: int = 0
    cosine: bool = False
    max_grad_norm: float = 1.0
    loss_weighting: LossWeighting = 'token'
    loss_chunk: int = 1024
    gradient_checkpointing: bool = False
    log_every: int = 5
    save_every: int = 0
    resume: bool = True
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self) -> None:
        positive = ('steps', 'fields_per_document', 'gradient_accumulation', 'log_every', 'loss_chunk')
        for name in positive:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer, got {getattr(self, name)!r}')
        for name in ('save_every', 'warmup_updates'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a non-negative integer, got {getattr(self, name)!r}')
        if self.warmup_updates > self.total_updates:
            raise ValueError(f'warmup_updates {self.warmup_updates} exceeds the {self.total_updates} updates')
        if not 0 <= self.selection.holdout_percent < 100:
            raise ValueError('selection.holdout_percent must be in [0, 100)')
        if self.precision not in ('bfloat16', 'float32'):
            raise ValueError(f'precision must be bfloat16 or float32, got {self.precision!r}')
        if self.format.close not in ('box_end', 'im_end'):
            raise ValueError(f'format.close must be box_end or im_end, got {self.format.close!r}')
        if self.loss_weighting not in ('token', 'field'):
            raise ValueError(f'loss_weighting must be token or field, got {self.loss_weighting!r}')
        if not set(self.monitor.splits) <= {'train', 'dev'}:
            raise ValueError(f'monitor.splits must be train and/or dev, got {self.monitor.splits!r}')

    @property
    def total_updates(self) -> int:
        """Optimizer updates of the run (a final partial accumulation counts)."""
        return math.ceil(self.steps / self.gradient_accumulation)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> TrainConfig:
        """Build from a loaded YAML mapping.

        Raises:
            ValueError: On an unknown key (at any level), a missing required
                key, or an invalid value.
        """
        return _build(cls, config, '')

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-compatible dict (tuples become lists); inverse of :meth:`from_dict`."""
        return _plain(dataclasses.asdict(self))

    def comparable(self) -> dict[str, Any]:
        """The keys that must match for a resume (all but :data:`OPERATIONAL_KEYS`)."""
        return {k: v for k, v in self.to_dict().items() if k not in OPERATIONAL_KEYS}

    def extractor(self) -> ExtractorConfig:
        """Extractor settings for loading and decoding this run's model."""
        return ExtractorConfig(model=self.model, device=self.device, precision=self.precision,
                               max_pixels=self.format.max_pixels, max_value_tokens=self.format.max_value_tokens,
                               close=self.format.close)


#: Nested sections and the dataclass each is parsed into.
_SECTIONS: dict[str, type] = {'selection': SelectionConfig, 'format': FormatConfig, 'lora': LoraSettings,
                              'monitor': MonitorConfig}
#: Fields stored as tuples (YAML gives lists).
_TUPLES = {'forms', 'target_modules', 'splits'}


def _build(cls: type, values: Mapping[str, Any], where: str) -> Any:
    """Instantiate dataclass ``cls`` from ``values``, rejecting unknown keys."""
    if not isinstance(values, Mapping):
        raise ValueError(f'{where or "config"} must be a mapping, got {type(values).__name__}')
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f'Unknown key(s) in {where or "config"}: {unknown}; valid keys: {sorted(names)}')
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key in _SECTIONS and cls is TrainConfig:
            value = _build(_SECTIONS[key], value or {}, key)
        elif key in _TUPLES and value is not None:
            value = tuple(value)
        kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as error:  # missing required key
        raise ValueError(f'{where or "config"}: {error}') from None


def _plain(value: Any) -> Any:
    """Recursively turn tuples into lists so the dict compares equal to reloaded JSON."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
