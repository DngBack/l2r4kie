"""Request → response inference for one document (``l2r4kie infer``).

Request (JSON)::

    {"pages": ["page1.png", "page2.png"],
     "fields": [{"id": "/ho_ten", "description": "Họ và tên"},
                {"id": "/thanh_vien", "description": "...", "kind": "array"}]}

``description`` defaults to the id and ``kind`` to ``"scalar"``. Relative page
paths are resolved against the request file's directory.

Response (JSON), the old repository's schema::

    {"result":     {"/ho_ten": "NGUYỄN VĂN A", ...},   # null unless status is ok
     "confidence": {"/ho_ten": 0.97, ...},             # null without a bundle or unless ok
     "status":     {"/ho_ten": "ok", ...},             # ok | truncated | invalid_array
     "calibrated": true}

With a confidence bundle (``--confidence``, see
:mod:`l2r4kie.confidence.bundle`) the response also holds the review queue
of the frozen policy, most urgent first, and whether the document needs a
human at all::

     "review": [{"id": "/so_tien", "confidence": 0.41, "status": "ok",
                 "reason": "low_correctness_confidence", "priority": 0.59}, ...],
     "document_needs_review": true
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..confidence.bundle import ConfidenceBundle
from ..data.types import FieldRequest
from ..model.decode import DecodeResult, extract
from ..model.extractor import Extractor
from ..utils.io import PathLike, read_json, write_json


def parse_request(record: Mapping[str, Any], root: PathLike | None = None) -> tuple[list[Path], list[FieldRequest]]:
    """Validate a request and return its page paths and field requests.

    Args:
        record: Parsed request JSON.
        root: Directory that relative page paths are resolved against.

    Raises:
        ValueError: If pages or fields are missing or malformed, a field id
            repeats, or a kind is unknown.
        FileNotFoundError: If a page image does not exist.
    """
    pages, fields = record.get('pages'), record.get('fields')
    if not isinstance(pages, list) or not pages or not all(isinstance(p, str) for p in pages):
        raise ValueError('request.pages must be a non-empty list of image paths')
    if not isinstance(fields, list) or not fields:
        raise ValueError('request.fields must be a non-empty list of {"id", "description"?, "kind"?}')
    paths = [Path(root, p) if root is not None else Path(p) for p in pages]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Page images not found: {missing}')
    requests = []
    for field in fields:
        if not isinstance(field, Mapping) or not isinstance(field.get('id'), str) or not field['id']:
            raise ValueError(f'Each field needs a non-empty string "id", got {field!r}')
        kind = field.get('kind', 'scalar')
        if kind not in ('scalar', 'array'):
            raise ValueError(f'Field {field["id"]!r}: kind must be "scalar" or "array", got {kind!r}')
        requests.append(FieldRequest(field['id'], str(field.get('description') or field['id']), kind))
    ids = [r.id for r in requests]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate field ids: {sorted({i for i in ids if ids.count(i) > 1})}')
    return paths, requests


def build_response(results: Sequence[DecodeResult], scores: Sequence[float | None] | None = None,
                   review: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Response JSON of decoded fields.

    Args:
        results: Decode results, in request order.
        scores: Calibrated confidence per field (``None``: no bundle, all ``null``).
        review: Review queue of the bundle's policy (added with ``document_needs_review``).
    """
    response = {'result': {r.field_id: r.value for r in results},
                'confidence': {r.field_id: (scores[i] if scores is not None else None)
                               for i, r in enumerate(results)},
                'status': {r.field_id: r.status for r in results},
                'calibrated': scores is not None}
    if review is not None:
        response['review'] = review
        response['document_needs_review'] = bool(review)
    return response


def infer(extractor: Extractor, request: Mapping[str, Any], root: PathLike | None = None,
          bundle: ConfidenceBundle | None = None) -> dict[str, Any]:
    """Extract the requested fields of one document and build the response.

    Args:
        extractor: Loaded extractor.
        request: Parsed request JSON.
        root: Directory that relative page paths are resolved against.
        bundle: Frozen confidence bundle; its extractor fingerprint must have
            been checked by the caller (:meth:`ConfidenceBundle.check_source`).
    """
    pages, requests = parse_request(request, root)
    if bundle is None:
        return build_response(extract(extractor, pages, requests))
    results = extract(extractor, pages, requests, trace=True, layers=bundle.layers)
    scores = bundle.score(results, requests, extractor.format.tokenizer)
    return build_response(results, scores, bundle.review(results, scores))


def infer_file(extractor: Extractor, request: PathLike, output: PathLike,
               bundle: ConfidenceBundle | None = None) -> dict[str, Any]:
    """Read a request file, run :func:`infer`, write the response atomically and return it."""
    response = infer(extractor, read_json(request), Path(request).parent, bundle)
    write_json(output, response)
    return response
