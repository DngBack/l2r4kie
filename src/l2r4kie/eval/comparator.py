"""Exact-match comparison of extracted values (old ``domain.canonical``/``correct``).

Conservative on purpose: case, punctuation, leading zeros and JSON types are
kept; only Unicode normalisation (NFC) and surrounding whitespace of strings
are ignored. Step 5 adds the evaluation variants on top of this.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any


def canonical(value: Any) -> Any:
    """NFC-normalise and strip strings, recursively; other values unchanged."""
    if isinstance(value, str):
        return unicodedata.normalize('NFC', value).strip()
    if isinstance(value, list):
        return [canonical(v) for v in value]
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    return value


def correct(prediction: Any, target: Any) -> bool:
    """Whether ``prediction`` equals ``target`` after :func:`canonical` (types must match)."""
    return json.dumps(canonical(prediction), ensure_ascii=False, sort_keys=True) == json.dumps(
        canonical(target), ensure_ascii=False, sort_keys=True)
