"""Extraction metrics over prediction rows.

A *prediction row* is one requested field of one document::

    {"document_id", "field_id", "target", "prediction", "status", ...}

Rows written by :mod:`l2r4kie.pipelines.predict` also carry ``form``,
``kind`` and ``text``; for old rows (``raw`` instead of ``text``) the form is
the document id's prefix and the kind follows from the target, so the old
r4 prediction files can be re-scored unchanged.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from .comparator import Comparator, is_correct, text_canonical
from .errors import error_kind, generated_text, is_coordinates


def row_form(row: dict[str, Any]) -> str:
    """Form of a row: ``form`` if present, else the document id before ``'__'``."""
    return row.get('form') or row['document_id'].split('__')[0]


def row_kind(row: dict[str, Any]) -> str:
    """``'array'`` or ``'scalar'``: ``kind`` if present, else from the target type."""
    return row.get('kind') or ('array' if isinstance(row['target'], list) else 'scalar')


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def array_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Row-level metrics of array fields, partial credit that exact match hides.

    A prediction that is not ``ok`` counts as an empty array.

    Returns:
        ``fields``; ``length_match`` (share with as many rows as the target);
        micro ``row_precision``/``row_recall``/``row_f1`` (rows matched as a
        multiset after text canonicalisation, order ignored); ``cell_accuracy``
        (target cells equal to the prediction's cell at the same row index and
        key; a scalar row is one cell).
    """
    fields = matched = predicted = expected = 0
    length_match = cells = cells_right = 0
    for row in rows:
        target = row['target']
        prediction = row['prediction'] if row['status'] == 'ok' and isinstance(row['prediction'], list) else []
        fields += 1
        length_match += len(prediction) == len(target)
        keys_t = Counter(json.dumps(text_canonical(r), ensure_ascii=False, sort_keys=True) for r in target)
        keys_p = Counter(json.dumps(text_canonical(r), ensure_ascii=False, sort_keys=True) for r in prediction)
        matched += sum((keys_t & keys_p).values())
        predicted += len(prediction)
        expected += len(target)
        for i, target_row in enumerate(target):
            predicted_row = prediction[i] if i < len(prediction) else None
            if isinstance(target_row, dict):
                for key, value in target_row.items():
                    cells += 1
                    other = predicted_row.get(key) if isinstance(predicted_row, dict) else None
                    cells_right += other is not None and text_canonical(other) == text_canonical(value)
            else:
                cells += 1
                cells_right += predicted_row is not None and text_canonical(predicted_row) == text_canonical(target_row)
    precision = matched / predicted if predicted else None
    recall = matched / expected if expected else None
    f1 = 2 * matched / (predicted + expected) if predicted + expected else None
    return {'fields': fields, 'length_match': length_match / fields if fields else None,
            'row_precision': precision, 'row_recall': recall, 'row_f1': f1,
            'cell_accuracy': cells_right / cells if cells else None}


def score(rows: Sequence[dict[str, Any]], comparator: Comparator = 'text') -> dict[str, Any]:
    """Extraction metrics of prediction rows under ``comparator``.

    Returns:
        ``exact_match`` (micro over fields; non-``ok`` counts as wrong),
        ``macro_form_exact_match``, ``complete_document_rate`` (every evaluated
        field right), ``scalar_exact_match``/``array_exact_match`` (gate G1
        uses the scalar one), ``coordinate_rate`` (G2), ``truncated_rate``
        (G3), ``by_kind``, ``by_form``, ``status`` and ``errors`` counts, and
        ``array_rows`` (see :func:`array_rows`).

    Raises:
        ValueError: On an empty row list or a duplicated (document, field) pair.
    """
    if not rows:
        raise ValueError('No prediction rows to score')
    keys = [(r['document_id'], r['field_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicated (document_id, field_id) rows')
    forms: dict[str, list[int]] = defaultdict(list)
    kinds: dict[str, list[int]] = defaultdict(list)
    documents: dict[str, list[int]] = defaultdict(list)
    errors: Counter[str] = Counter()
    for row in rows:
        right = int(is_correct(row, comparator))
        forms[row_form(row)].append(right)
        kinds[row_kind(row)].append(right)
        documents[row['document_id']].append(right)
        if not right:
            errors[error_kind(row)] += 1
    flags = [v for values in documents.values() for v in values]
    status = Counter(r['status'] for r in rows)
    return {
        'comparator': comparator, 'fields': len(rows), 'documents': len(documents), 'forms': len(forms),
        'exact_match': _mean(flags),
        'macro_form_exact_match': _mean([_mean(v) for v in forms.values()]),
        'complete_document_rate': _mean([float(all(v)) for v in documents.values()]),
        'scalar_exact_match': _mean(kinds.get('scalar', [])),
        'array_exact_match': _mean(kinds.get('array', [])),
        'coordinate_rate': sum(is_coordinates(generated_text(r)) for r in rows) / len(rows),
        'truncated_rate': status.get('truncated', 0) / len(rows),
        'status': dict(sorted(status.items())),
        'errors': dict(errors.most_common()),
        'by_kind': {k: {'count': len(v), 'exact_match': _mean(v)} for k, v in sorted(kinds.items())},
        'by_form': {k: {'count': len(v), 'exact_match': _mean(v)} for k, v in sorted(forms.items())},
        'array_rows': array_rows(r for r in rows if row_kind(r) == 'array'),
    }
