"""Trace caches: decoded fields of one cohort with their confidence features.

One file per cohort, ``<cache>/<split>.pt`` (``torch.save``, loadable with
``weights_only=True``)::

    {'rows':     [{document_id, form, field_id, kind, target, prediction, text,
                   status, correct, split, trace_index}, ...],   # every field
     'traces':   [trace record, ...],    # fields with status ok (see .features)
     'metadata': {split, source_fingerprint, layers, ...}}

``trace_index`` is the row's position in ``traces``, ``-1`` without one.
Caches written by the old repository (``feature_index``, an ``end`` state,
``token_stats``) are read too, so old selections can be re-checked.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..utils.io import PathLike
from .calibration import probabilities
from .features import TraceStore


@dataclass
class TraceCache:
    """One cohort's rows and trace records.

    Attributes:
        split: Cohort name (``train``, ``dev``, ``calibration``,
            ``risk_validation``, ``audit``).
        rows: Every decoded field.
        records: Trace records of the rows with ``trace_index >= 0``.
        metadata: Provenance written with the cache.
        legacy: Read from an old-repository cache.
    """

    split: str
    rows: list[dict[str, Any]]
    records: list[dict[str, Any]]
    metadata: dict[str, Any]
    legacy: bool = False

    @property
    def valid_rows(self) -> list[dict[str, Any]]:
        """Rows with a trace record, in record order."""
        return sorted((r for r in self.rows if r['trace_index'] >= 0), key=lambda r: r['trace_index'])

    @property
    def labels(self) -> list[float]:
        """Correctness of the valid rows (1.0 or 0.0)."""
        return [float(r['correct']) for r in self.valid_rows]

    @property
    def documents(self) -> set[str]:
        """Document ids of every row."""
        return {r['document_id'] for r in self.rows}

    @property
    def layers(self) -> tuple[int, ...]:
        """``hidden_states`` indices of the intermediate states in the records."""
        return tuple(self.metadata.get('layers') or ())

    def store(self, signals: Sequence[str], device: str | torch.device = 'cpu') -> TraceStore:
        """Batching view of the records with the given signals."""
        return TraceStore(self.records, signals, self.layers, device)

    def scores(self, logits: torch.Tensor, calibration: Mapping[str, Any] | None = None) -> list[float | None]:
        """Calibrated confidence of every row (``None`` without a trace record)."""
        p = probabilities(logits, calibration)
        return [p[r['trace_index']] if r['trace_index'] >= 0 else None for r in self.rows]


def _from_legacy(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows and records of an old-repository cache, renamed to the new layout."""
    rows = [{**{k: v for k, v in r.items() if k != 'feature_index'}, 'trace_index': r['feature_index']}
            for r in cache['rows']]
    records = [{'value': t['end'].float(), 'tokens': t['tokens'], 'stats': t['token_stats'].float(),
                'mask': t['mask'], 'summary': t['summary'].float(), 'length': len(t['tokens'])}
               for t in cache['traces']]
    return rows, records


def load_cache(directory: PathLike, split: str) -> TraceCache:
    """Read ``<directory>/<split>.pt`` (new or old layout).

    Raises:
        FileNotFoundError: If the cache does not exist.
        ValueError: If its rows or metadata name another split, or trace
            indices do not match the records.
    """
    path = Path(directory) / f'{split}.pt'
    if not path.is_file():
        raise FileNotFoundError(f'Trace cache not found: {path}')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    legacy = 'features' in cache  # old caches also stored the end states as one matrix
    rows, records = _from_legacy(cache) if legacy else (cache['rows'], cache['traces'])
    if cache['metadata']['split'] != split or any(r['split'] != split for r in rows):
        raise ValueError(f'{path}: cache split provenance mismatch')
    indices = sorted(r['trace_index'] for r in rows if r['trace_index'] >= 0)
    if indices != list(range(len(records))):
        raise ValueError(f'{path}: trace indices do not cover the {len(records)} records')
    return TraceCache(split, rows, records, dict(cache['metadata']), legacy)


def save_cache(path: PathLike, rows: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
               metadata: Mapping[str, Any]) -> Path:
    """Atomically write a cache (tmp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({'rows': list(rows), 'traces': list(records), 'metadata': dict(metadata)}, tmp)
    tmp.replace(path)
    return path


def check_disjoint(caches: Mapping[str, TraceCache]) -> None:
    """Raise ``ValueError`` if two cohorts share a document."""
    names = list(caches)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = caches[a].documents & caches[b].documents
            if shared:
                raise ValueError(f'Document leakage between {a} and {b}: {len(shared)} documents')


def check_source(caches: Mapping[str, TraceCache], fingerprint: str | None = None) -> str:
    """Return the extractor fingerprint shared by all caches.

    Raises:
        ValueError: If caches come from different extractors, or not from ``fingerprint``.
    """
    found = {c.metadata.get('source_fingerprint') for c in caches.values()}
    if len(found) != 1 or None in found:
        raise ValueError(f'Trace caches come from different or unknown extractors: {sorted(map(str, found))}')
    (source,) = found
    if fingerprint is not None and source != fingerprint:
        raise ValueError(f'Trace caches belong to extractor {source[:12]}..., expected {fingerprint[:12]}...')
    return source
