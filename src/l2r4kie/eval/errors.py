"""Recognisable failure modes of generated values.

:func:`error_kind` is the old report's taxonomy (``scripts/report_training_optimization.py``)
with one addition, ``coordinates``: Qwen2-VL's grounding output written
instead of the value, the specific risk of reusing ``<|box_start|>``.
It is for analysis after evaluation only, never for loosening the comparator.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

#: Qwen2-VL's grounding output inside ``<|box_start|>...<|box_end|>``, e.g.
#: ``(295,434),(526,492)``. The base model writes this for every field until
#: fine-tuning replaces it with text (see ``docs/notes/marker_selection.md``).
COORDINATES = re.compile(r'\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*')


def is_coordinates(text: str) -> bool:
    """Whether ``text`` is a box in Qwen2-VL's grounding format and nothing else."""
    return COORDINATES.fullmatch(text) is not None


def generated_text(row: dict[str, Any]) -> str:
    """Generated text of a prediction row (``text`` in new rows, ``raw`` in old ones)."""
    text = row.get('text', row.get('raw')) or ''
    # Old rows hold JSON; a coordinate string would be quoted there.
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] == '"' else text


def _strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))


def error_kind(row: dict[str, Any]) -> str:
    """Classify a wrong prediction row.

    Kinds, first match wins: ``coordinates``; the status when not ``ok``
    (``truncated``, ``invalid_array``, old ``invalid_json``); ``wrong_json_type``;
    ``array_content``; for strings ``spurious_nonempty`` (target blank),
    ``missing_value`` (prediction blank), ``case_only``,
    ``internal_whitespace``, ``diacritics_or_case``, ``digit_substitution``;
    otherwise ``other_content``.
    """
    if is_coordinates(generated_text(row)):
        return 'coordinates'
    if row['status'] != 'ok':
        return row['status']
    target, prediction = row['target'], row['prediction']
    if type(target) is not type(prediction):
        return 'wrong_json_type'
    if isinstance(target, list):
        return 'array_content'
    if isinstance(target, str):
        a, b = (unicodedata.normalize('NFC', s).strip() for s in (target, prediction))
        if not a:
            return 'spurious_nonempty'
        if not b:
            return 'missing_value'
        if a.casefold() == b.casefold():
            return 'case_only'
        if re.sub(r'\s+', ' ', a) == re.sub(r'\s+', ' ', b):
            return 'internal_whitespace'
        if _strip_accents(a).casefold() == _strip_accents(b).casefold():
            return 'diacritics_or_case'
        if any(c.isdigit() for c in a) and re.sub(r'\d', '#', a) == re.sub(r'\d', '#', b):
            return 'digit_substitution'
    return 'other_content'
