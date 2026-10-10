"""Exact-match comparison of extracted values.

Two comparators, both conservative (case, punctuation and leading zeros
matter; only Unicode normalisation (NFC) and surrounding whitespace of strings
are ignored):

* ``json`` (:func:`json_correct`, the old ``domain.correct``): JSON types must
  match too. Kept to re-score old predictions exactly as reported.
* ``text`` (:func:`text_correct`, the default): scalars are compared by their
  text, so ``"123"`` equals ``123`` and ``"true"`` equals ``true``. KevFormat
  writes values as plain text, so a JSON type is not something it can get
  wrong; comparing r4 (JSON values) with Kev (text) fairly needs this one.
  Arrays must still be arrays and match element by element.

A prediction whose status is not ``'ok'`` is wrong under both.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from typing import Any, Literal

Comparator = Literal['text', 'json']


def canonical(value: Any) -> Any:
    """NFC-normalise and strip strings, recursively; other values unchanged."""
    if isinstance(value, str):
        return unicodedata.normalize('NFC', value).strip()
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def _key(value: Any) -> str:
    """Order-independent serialisation of a canonical value (dict keys sorted)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def correct(prediction: Any, target: Any) -> bool:
    """Whether ``prediction`` equals ``target`` after :func:`canonical`; JSON types must match."""
    return _key(canonical(prediction)) == _key(canonical(target))


def as_text(value: Any) -> str:
    """Text of a scalar as KevFormat writes it: strings verbatim, ``true``/``false``,
    numbers as JSON, ``None`` as the empty string (a blank field)."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def text_canonical(value: Any) -> Any:
    """Canonical form for the text comparator: scalars as text, arrays/objects element-wise."""
    if isinstance(value, list):
        return [text_canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: text_canonical(v) for k, v in value.items()}
    return canonical(as_text(value))


def text_correct(prediction: Any, target: Any) -> bool:
    """Whether the values are equal as text (see the module docstring)."""
    if isinstance(target, list) != isinstance(prediction, list):
        return False
    return _key(text_canonical(prediction)) == _key(text_canonical(target))


def json_correct(prediction: Any, target: Any) -> bool:
    """Alias of :func:`correct`, the comparator of the old reports."""
    return correct(prediction, target)


COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {'text': text_correct, 'json': json_correct}


def is_correct(row: dict[str, Any], comparator: Comparator = 'text') -> bool:
    """Score one prediction row (``status``, ``prediction``, ``target``)."""
    return row['status'] == 'ok' and COMPARATORS[comparator](row['prediction'], row['target'])
