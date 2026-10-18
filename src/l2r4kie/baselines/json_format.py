"""JSON baseline (ablation of KevFormat): all requested fields in one sequence, as one JSON object.

KevFormat decodes every field in its own isolated branch. The usual way to
fine-tune a VLM for KIE instead asks for all fields at once and trains the
model to write ``{"field": value, ...}``. To change nothing else, the baseline
is written as a *single* KevFormat branch over the same prefix (system
prompt, pages, instruction): its key text lists the requested fields, its
value is the compact JSON object of their values, and it ends on the same
close marker. Training (same LoRA, loss, documents and sampled fields) and
greedy decoding then reuse :mod:`~l2r4kie.train` and :mod:`~l2r4kie.model.decode`
unchanged; only the branch layout differs.

Parsing is lenient, to the baseline's advantage: when the object is broken
(cut off by the token budget or malformed), every member that parsed before
the break is kept.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from ..data.serialize import to_text
from ..data.types import FieldSpec
from ..eval.comparator import Comparator, is_correct
from ..model.format import EncodedBranch, KevFormat
from ..model.markers import encode_text

#: First line of the key text; the lines after it are the requested fields.
HEADER = 'JSON:'
_WHITESPACE = ' \t\n\r'


def schema_text(fields: Sequence[FieldSpec]) -> str:
    """Key text of a JSON request: the header, then one ``id: description`` line per field."""
    return '\n'.join([HEADER, *(KevFormat.key_text(f.id, f.description) for f in fields)])


def target_text(fields: Sequence[FieldSpec]) -> str:
    """The object the model is trained to write: ``{"id": value, ...}`` in field order, compact."""
    return json.dumps({f.id: f.value for f in fields}, ensure_ascii=False, separators=(',', ':'))


def encode(fmt: KevFormat, fields: Sequence[FieldSpec]) -> EncodedBranch:
    """The training branch of ``fields`` (prompt + JSON object + close marker)."""
    return EncodedBranch(HEADER, tuple(fmt.prompt_ids(schema_text(fields), '')),
                         (*encode_text(fmt.tokenizer, target_text(fields)), fmt.markers.value_close))


def groups(fields: Sequence[FieldSpec], size: int) -> list[list[FieldSpec]]:
    """Consecutive groups of at most ``size`` fields (one request each)."""
    if size < 1:
        raise ValueError(f'size must be positive, got {size}')
    return [list(fields[i:i + size]) for i in range(0, len(fields), size)]


def parse_members(text: str) -> dict[str, tuple[Any, str]]:
    """Members of a JSON object, up to the first break.

    Returns:
        ``key -> (value, source text of the value)``; the first occurrence of
        a key wins. Empty when ``text`` does not start with an object.
    """
    decoder = json.JSONDecoder()
    members: dict[str, tuple[Any, str]] = {}

    def skip(i: int) -> int:
        while i < len(text) and text[i] in _WHITESPACE:
            i += 1
        return i

    i = skip(0)
    if not text.startswith('{', i):
        return members
    i = skip(i + 1)
    try:
        while i < len(text) and text[i] != '}':
            key, i = decoder.raw_decode(text, i)
            i = skip(i)
            if not isinstance(key, str) or not text.startswith(':', i):
                break
            start = skip(i + 1)
            value, i = decoder.raw_decode(text, start)
            members.setdefault(key, (value, text[start:i]))
            i = skip(i)
            if not text.startswith(',', i):
                break
            i = skip(i + 1)
    except json.JSONDecodeError:
        pass
    return members


def field_value(field: FieldSpec, members: dict[str, tuple[Any, str]]) -> tuple[Any, str, str]:
    """``(prediction, text, status)`` of one field from the parsed members.

    Statuses: ``'ok'``; ``'missing'`` (no member for the field);
    ``'invalid_array'`` (an array field whose member is not a list);
    ``'wrong_type'`` (a scalar field given a list, object or null). Numbers
    are read as their source text, as the text comparator would see them.
    """
    if field.id not in members:
        return None, '', 'missing'
    value, source = members[field.id]
    if field.kind == 'array':
        return (value, to_text(value), 'ok') if isinstance(value, list) else (None, source, 'invalid_array')
    if isinstance(value, str | bool):
        return value, to_text(value), 'ok'
    if isinstance(value, int | float):
        return source, source, 'ok'
    return None, source, 'wrong_type'


def prediction_rows(document_id: str, form: str, fields: Sequence[FieldSpec], text: str, decode_status: str,
                    comparator: Comparator = 'text') -> list[dict[str, Any]]:
    """Rows of one JSON request, shaped like :func:`~l2r4kie.pipelines.predict.prediction_rows`.

    A member that parsed is scored even when the object as a whole was cut
    off (``decode_status`` is kept as ``request_status``).
    """
    members = parse_members(text)
    rows = []
    for field in fields:
        prediction, value_text, status = field_value(field, members)
        row = {'document_id': document_id, 'form': form, 'field_id': field.id, 'kind': field.kind,
               'target': field.value, 'prediction': prediction, 'text': value_text, 'status': status,
               'request_status': decode_status}
        row['correct'] = is_correct(row, comparator)
        rows.append(row)
    return rows


def encode_within(fmt: KevFormat, fields: Sequence[FieldSpec], limit: int) -> EncodedBranch:
    """:func:`encode` of ``fields``, dropping the longest values until the target fits ``limit`` tokens.

    Raises:
        ValueError: If not even one field fits.
    """
    kept = list(fields)
    while kept:
        branch = encode(fmt, kept)
        if len(branch.target) <= limit:
            return branch
        kept.remove(max(kept, key=lambda f: len(to_text(f.value))))
    raise ValueError(f'No field fits a JSON target of {limit} tokens')
