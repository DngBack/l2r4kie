"""Decode a cohort once and cache every field's confidence features (``l2r4kie cache-traces``).

For each document, one prefix encode (one vision forward, checked) and one
decode of all requested fields with ``trace=True``; the value, status,
correctness and, for ``ok`` fields, the trace record (see
:mod:`l2r4kie.confidence.features`) are kept. Nothing is decoded twice: heads,
calibration and policies are all fitted on these caches.

Fixes over the old ``scripts/cache_token_review.py``:

* resumable: documents are written in parts of ``part_size`` and merged at
  the end, so a crash (e.g. an out-of-memory error on a shared GPU) loses at
  most one part;
* the three marker states, intermediate layers and the close decision are
  kept, not only the end state;
* long values are kept whole for the summary, with at most
  ``max_trace_tokens`` token states stored;
* within a part, each decoded document is checkpointed
  (``<split>.partNNNN.docs/``), so a worker killed mid-part (out of memory
  on a shared GPU) loses at most the document it was decoding;
* several workers (processes, GPUs) can fill the same cache: each part is
  claimed with a lock file, and the worker that writes the last part merges.
  Decoding is bound by Python and kernel launches, not by the GPU, so a few
  workers per GPU scale almost linearly.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ..confidence.cache import save_cache
from ..confidence.features import summary_names, trace_record
from ..data.cohorts import CohortPlan
from ..data.selection import Selection
from ..data.types import Document, FieldRequest
from ..eval.comparator import is_correct
from ..model.decode import TOKEN_STATS, extract
from ..model.extractor import Extractor
from ..utils.io import PathLike, read_json, write_json


def resolve_cohorts(plan: PathLike, selection: Selection, splits: Sequence[str],
                    train_documents: int | None = None) -> dict[str, list[Document]]:
    """Documents of the requested cohorts of a frozen plan, in plan order.

    Args:
        plan: ``cohort_plan.json``.
        selection: The extractor's selection (same prepared directory and partition).
        splits: Cohorts to resolve.
        train_documents: The extractor's ``train_documents`` limit.

    Raises:
        KeyError: If a split is not in the plan.
        ValueError: If a cohort holds a document the extractor was trained on
            (its errors would be optimistic and every head fitted on them biased).
    """
    cohorts = CohortPlan.load(plan).resolve(selection)
    missing = [s for s in splits if s not in cohorts]
    if missing:
        raise KeyError(f'Cohort(s) {missing} not in {plan} (has {sorted(cohorts)})')
    trained = {d.id for d in selection.documents('train', train_documents)}
    for split in splits:
        seen = [d.id for d in cohorts[split] if d.id in trained]
        if seen:
            raise ValueError(f'{split} holds {len(seen)} extractor training documents, e.g. {seen[:3]}')
    return {split: cohorts[split] for split in splits}


def document_rows(extractor: Extractor, document: Document, split: str, fields_per_document: int | None,
                  max_value_tokens: int, layers: Sequence[int], max_trace_tokens: int
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
    """Decode one document; return its rows, trace records, seconds and vision forwards.

    ``trace_index`` of the rows counts from 0 within the document; the caller
    offsets it.

    Raises:
        AssertionError: If the vision encoder did not run exactly once.
    """
    fields = list(document.fields[:fields_per_document])
    calls: list[int] = []
    hook = extractor.core.visual.register_forward_hook(lambda *_: calls.append(1))
    started = time.monotonic()
    try:
        results = extract(extractor, document.pages, [FieldRequest.from_field(f) for f in fields],
                          max_value_tokens=max_value_tokens, trace=True, layers=layers)
    finally:
        hook.remove()
    seconds = time.monotonic() - started
    if len(calls) != 1:
        raise AssertionError(f'{document.id}: expected one vision forward, got {len(calls)}')
    tokenizer = extractor.format.tokenizer
    rows, records = [], []
    for field, result in zip(fields, results, strict=True):
        row = {'document_id': document.id, 'form': document.form, 'field_id': field.id, 'kind': field.kind,
               'target': field.value, 'prediction': result.value, 'text': result.text, 'status': result.status,
               'split': split, 'trace_index': -1}
        row['correct'] = is_correct(row, 'text')
        if result.status == 'ok':
            pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in result.trace.tokens[:-1]]
            records.append(trace_record(result, pieces, field.kind, max_trace_tokens))
            row['trace_index'] = len(records) - 1
        rows.append(row)
    return rows, records, seconds, len(calls)


def _claim(lock: Path) -> bool:
    """Create ``lock`` for this process; take over a lock left by a dead process on this host.

    Returns:
        Whether this process now holds the lock.
    """
    owner = f'{socket.gethostname()} {os.getpid()}'
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                host, pid = lock.read_text().split()
            except (OSError, ValueError):
                return False  # being written by its owner, or vanished: let the owner have it
            if host != socket.gethostname():
                return False
            try:
                os.kill(int(pid), 0)
                return False  # owner alive
            except ProcessLookupError:
                lock.unlink(missing_ok=True)  # stale: its owner crashed
                continue
            except PermissionError:
                return False
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(owner)
        return True
    return False


def cache_traces(extractor: Extractor, documents: Sequence[Document], directory: PathLike, split: str,
                 fields_per_document: int | None = 24, max_value_tokens: int | None = None,
                 layers: Sequence[int] = (), max_trace_tokens: int = 512,
                 provenance: Mapping[str, Any] | None = None, part_size: int = 16) -> Path | None:
    """Decode ``documents`` and write ``<directory>/<split>.pt`` (resumable).

    Args:
        extractor: Loaded extractor (adapter included).
        documents: The cohort, in order.
        directory: Cache directory.
        split: Cohort name.
        fields_per_document: First N fields per document (r4: 24); ``None`` for all.
        max_value_tokens: Decode budget (default: the extractor's).
        layers: Intermediate ``hidden_states`` indices of the marker states.
        max_trace_tokens: Token states kept per field.
        provenance: Extra metadata (adapter, fingerprint, cohort plan, ...);
            must match on resume.
        part_size: Documents per resumable part.

    Returns:
        Path of the cache, or ``None`` if parts are still being decoded by
        other workers (the last of them merges).

    Raises:
        ValueError: If the directory holds a cache or parts of a different run.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    budget = extractor.config.max_value_tokens if max_value_tokens is None else max_value_tokens
    record = {**(provenance or {}), 'split': split, 'documents': [d.id for d in documents],
              'fields_per_document': fields_per_document, 'max_value_tokens': budget, 'layers': list(layers),
              'max_trace_tokens': max_trace_tokens, 'max_pixels': extractor.config.max_pixels,
              'close': extractor.config.close, 'dtype': str(extractor.dtype), 'part_size': part_size}
    claim = directory / f'{split}.provenance.json'
    if claim.exists():
        stored = read_json(claim)
        changed = sorted(k for k in set(stored) | set(record) if stored.get(k) != record.get(k))
        if changed:
            raise ValueError(f'{directory} holds {split} traces of a different run (differs in {changed})')
    else:
        write_json(claim, record)
    path = directory / f'{split}.pt'
    if path.exists():
        print(json.dumps({'split': split, 'reuse': str(path)}), flush=True)
        return path
    parts = [documents[i:i + part_size] for i in range(0, len(documents), part_size)]
    started = time.monotonic()
    for number, part in enumerate(parts):
        part_path = directory / f'{split}.part{number:04d}.pt'
        lock = part_path.with_suffix('.lock')
        if part_path.exists() or not _claim(lock):
            continue
        try:
            if part_path.exists():  # finished by another worker between the check and the claim
                continue
            rows, records, timings, vision = [], [], [], []
            checkpoints = part_path.with_suffix('.docs')
            for index, document in enumerate(part):
                document_rows_, document_records, seconds, calls = _document_checkpoint(
                    checkpoints / f'{index:03d}.pt', document,
                    lambda d=document: document_rows(extractor, d, split, fields_per_document, budget, layers,
                                                     max_trace_tokens))
                for row in document_rows_:
                    if row['trace_index'] >= 0:
                        row['trace_index'] += len(records)
                rows.extend(document_rows_)
                records.extend(document_records)
                timings.append(seconds)
                vision.append(calls)
            save_cache(part_path, rows, records, {'split': split, 'part': number, 'timings_seconds': timings,
                                                  'vision_calls_per_document': vision})
            shutil.rmtree(checkpoints, ignore_errors=True)
        finally:
            lock.unlink(missing_ok=True)
        done = min(len(documents), (number + 1) * part_size)
        print(json.dumps({'split': split, 'part': number, 'documents': done, 'of': len(documents),
                          'fields': len(rows), 'valid': len(records),
                          'seconds': round(time.monotonic() - started, 1)}), flush=True)
    missing = [n for n in range(len(parts)) if not (directory / f'{split}.part{n:04d}.pt').exists()]
    merge_lock = directory / f'{split}.merge.lock'
    if missing or path.exists() or not _claim(merge_lock):
        print(json.dumps({'split': split, 'parts_left_to_other_workers': missing}), flush=True)
        return path if path.exists() else None
    try:
        return _merge(directory, split, len(parts), record)
    finally:
        merge_lock.unlink(missing_ok=True)


def _document_checkpoint(path: Path, document: Document,
                         decode: Callable[[], tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]]
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, int]:
    """Return a document's rows, records, seconds and vision forwards, from ``path`` or by decoding.

    A part holds 16 documents, and documents with long arrays take many
    minutes each at the serving budget; on a shared GPU a worker may die
    (out of memory) in the middle of a part. Each decoded document is saved
    to ``path`` (atomically), so a restarted worker resumes at the next
    document instead of decoding the whole part again. Row ``trace_index``
    values are stored as returned by :func:`document_rows` (from 0 within
    the document), before the caller offsets them.

    Raises:
        ValueError: If ``path`` holds another document (the cohort changed).
    """
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['document_id'] != document.id:
            raise ValueError(f'{path} holds {saved["document_id"]}, expected {document.id}')
        return saved['rows'], saved['traces'], saved['seconds'], saved['vision_calls']
    rows, records, seconds, calls = decode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({'document_id': document.id, 'rows': [dict(r) for r in rows], 'traces': records,
                'seconds': seconds, 'vision_calls': calls}, tmp)
    tmp.replace(path)
    return rows, records, seconds, calls


def _merge(directory: Path, split: str, parts: int, record: Mapping[str, Any]) -> Path:
    """Concatenate the parts into ``<split>.pt``, write a summary, delete the parts."""
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    timings: list[float] = []
    vision: list[int] = []
    for number in range(parts):
        part = torch.load(directory / f'{split}.part{number:04d}.pt', map_location='cpu', weights_only=True)
        for row in part['rows']:
            rows.append({**row, 'trace_index': row['trace_index'] + len(records) if row['trace_index'] >= 0 else -1})
        records.extend(part['traces'])
        timings.extend(part['metadata']['timings_seconds'])
        vision.extend(part['metadata']['vision_calls_per_document'])
    metadata = {**record, 'documents': len(record['documents']), 'document_ids': record['documents'],
                'forms': len({r['form'] for r in rows}), 'timings_seconds': timings,
                'vision_calls_per_document': vision, 'token_stats': list(TOKEN_STATS),
                'summary_names': summary_names(),
                'feature_origin': 'Real decode; marker states, post-token states and pre-token distributions; '
                                  'no extra VLM pass'}
    path = save_cache(directory / f'{split}.pt', rows, records, metadata)
    write_json(directory / f'{split}.summary.json',
               {**{k: v for k, v in metadata.items() if k not in ('timings_seconds', 'document_ids')},
                'fields': len(rows), 'valid': len(records), 'correct': sum(r['correct'] for r in rows),
                'seconds': sum(timings)})
    for number in range(parts):
        (directory / f'{split}.part{number:04d}.pt').unlink()
        # Left by a worker that died before another one redid the part from scratch.
        shutil.rmtree(directory / f'{split}.part{number:04d}.docs', ignore_errors=True)
    return path
