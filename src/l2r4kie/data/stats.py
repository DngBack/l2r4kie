"""Summary statistics of prepared splits.

Besides sizes, this counts the value shapes and string features that the
plain-text value format (KevFormat, step 2) has to handle: empty strings,
booleans, arrays, multi-line values, leading zeros and text that looks like a
special token.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from ..utils.io import PathLike, iter_jsonl
from .types import SPLITS, Document

#: Text shaped like a chat/special token, e.g. ``<|box_end|>``.
SPECIAL_TOKEN_PATTERN = re.compile(r'<\|[^|<>]*\|>')


def value_type(value: Any) -> str:
    """Classify a ground-truth value.

    Returns one of ``'string'``, ``'empty_string'``, ``'bool'``, ``'null'``,
    ``'number'``, ``'array_of_objects'``, ``'array_of_scalars'``,
    ``'empty_array'`` or ``'object'``.
    """
    match value:
        case str():
            return 'string' if value else 'empty_string'
        case bool():  # before int: bool is a subclass of int
            return 'bool'
        case None:
            return 'null'
        case int() | float():
            return 'number'
        case []:
            return 'empty_array'
        case list():
            return 'array_of_objects' if any(isinstance(v, dict) for v in value) else 'array_of_scalars'
        case _:
            return 'object'


def string_features(value: str) -> list[str]:
    """Return the format-relevant features present in a string value."""
    features = []
    if value.isdigit() and len(value) > 1 and value.startswith('0'):
        features.append('leading_zero_digits')
    if '\n' in value:
        features.append('multiline')
    if value != value.strip():
        features.append('surrounding_whitespace')
    if '<' in value or '>' in value:
        features.append('angle_brackets')
    if SPECIAL_TOKEN_PATTERN.search(value):
        features.append('special_token_like')
    return features


def document_stats(documents: Iterable[Document]) -> dict[str, Any]:
    """Count documents, forms, fields, value types and string features."""
    counts: Counter[str] = Counter()
    forms: Counter[str] = Counter()
    types: Counter[str] = Counter()
    features: Counter[str] = Counter()
    for document in documents:
        counts['documents'] += 1
        counts['pages'] += len(document.pages)
        forms[document.form] += 1
        for field in document.fields:
            counts['fields'] += 1
            types[value_type(field.value)] += 1
            if isinstance(field.value, str):
                features.update(string_features(field.value))
    return {**counts, 'forms': len(forms), 'value_types': dict(types.most_common()),
            'string_features': dict(features.most_common())}


def prepared_stats(prepared: PathLike) -> dict[str, Any]:
    """Statistics of every split of a prepared directory, plus their total.

    Returns:
        ``{'splits': {split: stats}, 'total': stats}`` where each ``stats``
        is the output of :func:`document_stats`.
    """
    root = Path(prepared)

    def load(split: str) -> Iterator[Document]:
        return (Document.from_json(r) for r in iter_jsonl(root / f'{split}.jsonl'))

    splits = {split: document_stats(load(split)) for split in SPLITS}
    total = document_stats(d for split in SPLITS for d in load(split))
    return {'splits': splits, 'total': total}
