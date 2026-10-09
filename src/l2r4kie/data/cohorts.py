"""Pre-declared document cohorts for confidence-head development and audit.

The review policy is only trustworthy if the documents it is fitted and
audited on were never seen before. A *cohort plan* fixes, before any decoding,
which documents go to each stage:

========================  ===================================================
``train``                 fit confidence heads (from ``train_split``)
``dev``                   select a head
``calibration``           fit calibration
``risk_validation``       set the review threshold (from ``risk_split``)
``audit``                 one final, untouched measurement (test split)
========================  ===================================================

Documents already used by earlier runs are passed in as *exclusion files*.

Fix over the old ``scripts/cache_token_review.py::plan_cohorts``: it globbed a
hardcoded list of paths relative to the working directory and silently
skipped any that were missing, so running from another directory dropped the
leakage protection without warning. Exclusions are now an explicit list, and
a missing file is an error.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.fingerprint import checkpoint_fingerprint
from ..utils.io import PathLike, iter_jsonl, read_json, write_json
from .selection import Selection
from .types import Document, SelectableSplit, Split

COHORTS: tuple[str, ...] = ('train', 'dev', 'calibration', 'risk_validation', 'audit')


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """How many documents of each form go to each cohort.

    Attributes:
        seed: Seed of the per-form shuffles.
        train_per_type: Head-train documents per form.
        per_type: Dev and audit documents per form.
        calibration_per_type: Calibration documents per form (default: ``per_type``).
        risk_per_type: Risk-validation documents per form.
        train_split: Source of the head-train cohort: ``'train'`` or
            ``'train_reserve'`` (never in extractor gradients).
        risk_split: Source of the risk cohort: ``'train'`` (meaning
            ``train_split``) or ``'calibration'``. Risk documents are taken
            *after* that split's own cohort, so the two never overlap.
        exclude_forms: Forms left out of every cohort (e.g. forms with no
            unseen documents left).
    """

    seed: int
    train_per_type: int
    per_type: int = 2
    calibration_per_type: int | None = None
    risk_per_type: int = 4
    train_split: SelectableSplit = 'train'
    risk_split: Split = 'train'
    exclude_forms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.train_split not in ('train', 'train_reserve'):
            raise ValueError(f'train_split must be train or train_reserve, got {self.train_split!r}')
        if self.risk_split not in ('train', 'calibration'):
            raise ValueError(f'risk_split must be train or calibration, got {self.risk_split!r}')

    @property
    def calibration_count(self) -> int:
        """Calibration documents per form, resolving the default."""
        return self.calibration_per_type or self.per_type


@dataclass(frozen=True, slots=True)
class Exclusions:
    """Documents already used elsewhere, with the files they came from."""

    ids: frozenset[str] = frozenset()
    sources: tuple[str, ...] = ()


def _ids_in_file(path: Path) -> set[str]:
    """Document ids referenced by one exclusion file (see :func:`load_exclusions`)."""
    if path.suffix == '.pt':
        import torch  # only trace caches need torch

        cache = torch.load(path, map_location='cpu', weights_only=True)
        return {row['document_id'] for row in cache['rows']}
    if path.suffix == '.json':
        plan = read_json(path)
        return set(plan.get('excluded_documents', ())) | {i for ids in plan['cohorts'].values() for i in ids}
    ids = {row.get('document_id', row.get('id')) for row in iter_jsonl(path)}
    ids.discard(None)  # e.g. training logs: rows without a document reference
    return ids


def load_exclusions(paths: Iterable[PathLike]) -> Exclusions:
    """Collect every document id referenced by the given files.

    Supported files:

    * ``*.jsonl``: predictions, evaluation sets or training logs; each row's
      ``document_id`` (or ``id``).
    * ``*.pt``: trace caches; ``cache['rows'][*]['document_id']``.
    * ``*.json``: an earlier cohort plan; all its cohorts plus its own
      ``excluded_documents``, so a new plan avoids everything the old one touched.

    Raises:
        FileNotFoundError: If any path does not exist. Never skipped silently.
    """
    ids: set[str] = set()
    sources: list[str] = []
    for path in map(Path, paths):
        if not path.is_file():
            raise FileNotFoundError(f'Exclusion file not found: {path}')
        ids |= _ids_in_file(path)
        sources.append(str(path))
    return Exclusions(frozenset(ids), tuple(sources))


def plan_cohorts(selection: Selection, spec: CohortSpec, exclusions: Exclusions = Exclusions(),
                 checkpoint: PathLike | None = None) -> CohortPlan:
    """Draw disjoint, form-balanced cohorts of unseen documents.

    For each source split, remaining documents (not excluded, not in
    ``spec.exclude_forms``) are grouped by form and each group is shuffled
    with one RNG seeded by ``spec.seed``. Each cohort then takes a fixed
    number of documents per form, forms in sorted order. The draw is
    identical to the old implementation, so old plans can be reproduced.

    Args:
        selection: Where and how to read prepared documents.
        spec: Cohort sizes and sources.
        exclusions: Documents that must not appear in any cohort.
        checkpoint: Extractor the cohorts are planned for; its fingerprint
            is stored so caches built later can be checked against it.

    Returns:
        The frozen plan; use :meth:`CohortPlan.resolve` to get documents.

    Raises:
        ValueError: If two cohorts share a document or a cohort contains an
            excluded document (both guarded by construction; checked anyway).
    """
    rng = random.Random(spec.seed)
    pools: dict[Split, dict[str, list[Document]]] = {}
    for split in ('train', 'dev', 'calibration', 'test'):
        by_form: dict[str, list[Document]] = defaultdict(list)
        for document in selection.documents(spec.train_split if split == 'train' else split):
            if document.id not in exclusions.ids and document.form not in spec.exclude_forms:
                by_form[document.form].append(document)
        # One RNG for all groups, in insertion order: the order matters for parity.
        for group in by_form.values():
            rng.shuffle(group)
        pools[split] = by_form
    forms = sorted(pools['train'])

    def take(split: Split, count: int, offset: int = 0) -> list[str]:
        return [d.id for form in forms for d in pools[split].get(form, [])[offset:offset + count]]

    risk_offset = spec.train_per_type if spec.risk_split == 'train' else spec.calibration_count
    cohorts = {
        'train': take('train', spec.train_per_type),
        'dev': take('dev', spec.per_type),
        'calibration': take('calibration', spec.calibration_count),
        'risk_validation': take(spec.risk_split, spec.risk_per_type, risk_offset),
        'audit': take('test', spec.per_type),
    }
    check_disjoint(cohorts, exclusions.ids)
    fingerprint = None
    if checkpoint is not None:
        fingerprint = checkpoint_fingerprint(checkpoint)
    risk_source = spec.train_split if spec.risk_split == 'train' else spec.risk_split
    return CohortPlan(
        cohorts=cohorts, seed=spec.seed, forms=forms, train_split=spec.train_split,
        excluded_documents=sorted(exclusions.ids), exclusion_sources=list(exclusions.sources),
        source_checkpoint=None if checkpoint is None else str(checkpoint), source_fingerprint=fingerprint,
        extra={'risk_origin': f'{risk_source} split, never in extractor or head gradients',
               'exclude_forms': list(spec.exclude_forms)},
    )


def check_disjoint(cohorts: Mapping[str, Sequence[str]], excluded: Iterable[str] = ()) -> None:
    """Raise ``ValueError`` if cohorts overlap or contain an excluded id."""
    sets = {name: set(ids) for name, ids in cohorts.items()}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f'Leaking cohorts: {a}/{b} share {len(sets[a] & sets[b])} documents')
    excluded = set(excluded)
    for name, ids in sets.items():
        if ids & excluded:
            raise ValueError(f'Cohort {name} contains {len(ids & excluded)} previously used documents')


@dataclass(frozen=True, slots=True)
class CohortPlan:
    """A frozen cohort plan as stored in ``cohort_plan.json``.

    Attributes:
        cohorts: ``{cohort: [document ids]}``, cohorts as in :data:`COHORTS`.
        seed: Cohort seed.
        forms: Forms with a head-train pool, sorted; cohorts follow this order.
        train_split: Source split of the ``train`` cohort.
        excluded_documents: Every excluded id (sorted).
        exclusion_sources: Files the exclusions were read from.
        source_checkpoint: Extractor whose traces these cohorts will hold.
        source_fingerprint: Fingerprint of that checkpoint, checked before reuse.
        extra: Any further keys (e.g. ``risk_origin``, ``note``), kept verbatim.
    """

    cohorts: dict[str, list[str]]
    seed: int
    forms: list[str]
    train_split: SelectableSplit = 'train'
    excluded_documents: list[str] = field(default_factory=list)
    exclusion_sources: list[str] = field(default_factory=list)
    source_checkpoint: str | None = None
    source_fingerprint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return the ``cohort_plan.json`` representation."""
        return {**{key: getattr(self, key) for key in _PLAN_KEYS}, **self.extra}

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> CohortPlan:
        """Read a plan, including plans written by the old repository."""
        known = {key: record[key] for key in _PLAN_KEYS if key in record}
        return cls(**known, extra={k: v for k, v in record.items() if k not in _PLAN_KEYS})

    def save(self, path: PathLike) -> Path:
        """Atomically write the plan as JSON."""
        return write_json(path, self.to_json())

    @classmethod
    def load(cls, path: PathLike) -> CohortPlan:
        """Read a plan from ``path``."""
        return cls.from_json(read_json(path))

    def resolve(self, selection: Selection) -> dict[str, list[Document]]:
        """Map the stored ids back to documents, in stored order.

        Raises:
            KeyError: If an id is not in any selectable split (wrong
                ``prepared`` directory or selection settings).
        """
        splits: tuple[SelectableSplit, ...] = (self.train_split, 'dev', 'calibration', 'test')
        lookup = {d.id: d for split in splits for d in selection.documents(split)}
        return {name: [lookup[i] for i in ids] for name, ids in self.cohorts.items()}


#: Keys of ``cohort_plan.json`` mapped to :class:`CohortPlan` fields; others go to ``extra``.
_PLAN_KEYS = ('cohorts', 'seed', 'forms', 'train_split', 'excluded_documents', 'exclusion_sources',
              'source_checkpoint', 'source_fingerprint')


def plan_from_config(config: Mapping[str, Any]) -> CohortPlan:
    """Plan cohorts from a cohort config (see ``configs/cohorts/*.yaml``).

    Expected keys::

        selection:  {prepared, seed, holdout_percent, forms, balanced_forms}
        cohorts:    CohortSpec fields (seed, train_per_type, per_type, ...)
        exclusions: [files of previously used documents]
        checkpoint: extractor directory (optional)

    Raises:
        KeyError: If ``selection`` or ``cohorts`` is missing.
        TypeError: On an unknown key under ``cohorts``.
        FileNotFoundError: If an exclusion file is missing.
        ValueError: If the top level has unknown keys.
    """
    unknown = set(config) - {'selection', 'cohorts', 'exclusions', 'checkpoint'}
    if unknown:
        raise ValueError(f'Unknown cohort config keys: {sorted(unknown)}')
    options = dict(config['cohorts'])
    options['exclude_forms'] = tuple(options.get('exclude_forms') or ())
    return plan_cohorts(
        Selection.from_config(config['selection']),
        CohortSpec(**options),
        load_exclusions(config.get('exclusions') or ()),
        config.get('checkpoint'),
    )
