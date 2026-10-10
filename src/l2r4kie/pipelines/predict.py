"""Decode labelled documents and write scored prediction rows (``l2r4kie evaluate``).

Output directory::

    provenance.json     what produced the rows (model, adapter fingerprint, documents, limits)
    predictions.jsonl   one row per evaluated field, written document by document
    metrics.json        :func:`~l2r4kie.eval.extraction.score` of the rows

Evaluation can be interrupted and rerun: documents already in
``predictions.jsonl`` are skipped, provided ``provenance.json`` matches.
Ground truth is never used to choose which fields are decoded or how long
they may be: every requested field is decoded with the same budget.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data.selection import Selection
from ..data.types import Document, FieldRequest, SelectableSplit
from ..eval.comparator import Comparator, is_correct
from ..eval.extraction import score
from ..model.decode import DecodeResult, extract
from ..model.extractor import Extractor, ExtractorConfig
from ..utils.fingerprint import checkpoint_fingerprint, sha256_file
from ..utils.io import PathLike, dumps_jsonl, read_json, read_jsonl, write_json, write_jsonl


def evaluation_documents(selection: Selection, split: SelectableSplit, documents_file: PathLike | None = None,
                         limit: int | None = None) -> list[Document]:
    """Documents to evaluate, in order.

    Args:
        selection: How documents are read and ordered.
        split: Split to read.
        documents_file: JSON with ``{"documents": [ids]}`` (e.g. the old
            ``cohort_plan.json`` of the r4 dev set) or a plain id list; the
            documents are taken in that order. ``None`` takes the split in
            selection order.
        limit: Keep the first N.

    Raises:
        KeyError: If the file names a document that is not in ``split``.
    """
    if documents_file is None:
        return selection.documents(split, limit)
    plan = read_json(documents_file)
    ids = plan['documents'] if isinstance(plan, dict) else plan
    lookup = {d.id: d for d in selection.documents(split)}
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise KeyError(f'{len(missing)} documents of {documents_file} are not in split {split}: {missing[:3]}...')
    chosen = [lookup[i] for i in ids]
    return chosen if limit is None else chosen[:limit]


def adapter_fingerprint(adapter: PathLike | None) -> str | None:
    """Fingerprint of an adapter: a run directory (containing ``adapter/``) or the adapter directory itself."""
    if adapter is None:
        return None
    path = Path(adapter)
    if (path / 'adapter').is_dir():
        return checkpoint_fingerprint(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.glob('*') if p.is_file()):
        digest.update(file.name.encode())
        digest.update(sha256_file(file).encode())
    return digest.hexdigest()


def prediction_rows(document: Document, fields: Sequence[Any], results: Sequence[DecodeResult],
                    comparator: Comparator = 'text') -> list[dict[str, Any]]:
    """Rows of one document: target, prediction, generated text, status and correctness."""
    rows = []
    for field, result in zip(fields, results, strict=True):
        row = {'document_id': document.id, 'form': document.form, 'field_id': field.id, 'kind': field.kind,
               'target': field.value, 'prediction': result.value, 'text': result.text, 'status': result.status}
        row['correct'] = is_correct(row, comparator)
        rows.append(row)
    return rows


def run_record(config: ExtractorConfig, documents: Sequence[Document], fields_per_document: int | None = None,
               max_value_tokens: int | None = None, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything that determines the rows of an evaluation, as stored in ``provenance.json``.

    Built from the config alone, so a mismatch can be reported before the
    model is loaded.

    Args:
        config: Extractor settings (``max_value_tokens`` is the default budget).
        documents: Documents to evaluate, in order.
        fields_per_document: First N fields per document; ``None`` for all.
        max_value_tokens: Decode budget; ``None`` takes ``config.max_value_tokens``.
        provenance: Extra entries (model, adapter, ...).
    """
    budget = config.max_value_tokens if max_value_tokens is None else max_value_tokens
    return {**(provenance or {}), 'documents': [d.id for d in documents],
            'fields_per_document': fields_per_document, 'max_value_tokens': budget,
            'max_pixels': config.max_pixels, 'close': config.close, 'dtype': str(config.dtype)}


def claim_output(output: PathLike, record: dict[str, Any]) -> None:
    """Make ``output`` the directory of the run described by ``record``.

    A new directory gets ``provenance.json``; an existing one must hold the
    same record (then the run resumes).

    Raises:
        ValueError: If ``output`` holds predictions of a different run, or
            predictions without ``provenance.json``.
    """
    output = Path(output)
    stored_path, predictions = output / 'provenance.json', output / 'predictions.jsonl'
    if stored_path.exists():
        stored = read_json(stored_path)
        changed = sorted(k for k in set(stored) | set(record) if stored.get(k) != record.get(k))
        if changed:
            raise ValueError(f'{output} holds predictions of a different run (differs in {changed}); '
                             'use a new output directory')
        return
    if predictions.exists():
        raise ValueError(f'{predictions} exists without provenance.json; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(stored_path, record)


def predict(extractor: Extractor, documents: Sequence[Document], output: PathLike,
            fields_per_document: int | None = None, max_value_tokens: int | None = None,
            provenance: dict[str, Any] | None = None, comparator: Comparator = 'text') -> dict[str, Any]:
    """Decode the documents, append their rows to ``predictions.jsonl`` and score them.

    Args:
        extractor: Loaded model.
        documents: Documents to evaluate, in order.
        output: Output directory.
        fields_per_document: Evaluate the first N fields of each document
            (r4 used 24); ``None`` evaluates all.
        max_value_tokens: Decode budget; defaults to the extractor's.
        provenance: Extra provenance (model, adapter, ...), checked on resume.
        comparator: Comparator of the stored ``correct`` flag and the metrics.

    Returns:
        The metrics, also written to ``metrics.json``.

    Raises:
        ValueError: If ``output`` holds predictions with a different provenance.
    """
    output = Path(output)
    record = run_record(extractor.config, documents, fields_per_document, max_value_tokens, provenance)
    claim_output(output, record)
    budget = record['max_value_tokens']
    predictions = output / 'predictions.jsonl'
    done = _keep_complete(predictions, {d.id: len(d.fields[:fields_per_document]) for d in documents})
    started = time.monotonic()
    for n, document in enumerate(documents, 1):
        if document.id in done:
            continue
        fields = list(document.fields[:fields_per_document])
        results = extract(extractor, document.pages, [FieldRequest.from_field(f) for f in fields],
                          max_value_tokens=budget)
        rows = prediction_rows(document, fields, results, comparator)
        # One write per document: a crash loses at most the document in progress.
        with predictions.open('a', encoding='utf-8') as handle:
            handle.write(''.join(dumps_jsonl(row) for row in rows))
        print(json.dumps({'document': n, 'of': len(documents), 'id': document.id,
                          'correct': sum(r['correct'] for r in rows), 'fields': len(rows),
                          'seconds': round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
    rows = read_jsonl(predictions)
    expected = {d.id for d in documents}
    if {r['document_id'] for r in rows} != expected:
        raise ValueError(f'{predictions} does not cover exactly the requested documents')
    metrics = score(rows, comparator)
    write_json(output / 'metrics.json', metrics)
    return metrics


def _keep_complete(predictions: Path, expected: dict[str, int]) -> set[str]:
    """Drop rows of incompletely written documents (and a torn last line); return the complete ids.

    Args:
        predictions: ``predictions.jsonl`` (may not exist yet).
        expected: Number of rows each document must have.
    """
    if not predictions.exists():
        return set()
    rows = []
    for line in predictions.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break
    counts: dict[str, int] = {}
    for row in rows:
        counts[row['document_id']] = counts.get(row['document_id'], 0) + 1
    complete = {i for i, n in counts.items() if expected.get(i) == n}
    kept = [r for r in rows if r['document_id'] in complete]
    if len(kept) != len(rows) or len(rows) != len(predictions.read_text(encoding='utf-8').splitlines()):
        write_jsonl(predictions, kept)
    return complete
