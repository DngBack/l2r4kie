"""Plain-text form of field values (the KevFormat output).

The extractor generates a value as plain text followed by a close marker,
never as JSON. This module fixes the mapping between ground-truth JSON values
and that text:

==========================  ======================  ==========================
value                       text                    note
==========================  ======================  ==========================
``"NGUYỄN VĂN A"``          ``NGUYỄN VĂN A``        kept verbatim (newlines,
                                                    leading zeros, spaces)
``""``                      (empty)                 field blank on the page
``True`` / ``False``        ``true`` / ``false``    checkboxes
``[{"a": "1"}, ...]``       ``[{"a":"1"},...]``     arrays: compact JSON
==========================  ======================  ==========================

The dataset contains no ``null`` or numbers; :func:`to_text` rejects them
rather than guessing a spelling.
"""

from __future__ import annotations

import json
from typing import Any

from .types import FieldKind

TRUE, FALSE = 'true', 'false'


def to_text(value: Any) -> str:
    """Render a ground-truth value as the text the model should generate.

    Raises:
        TypeError: For ``None``, numbers or objects (absent from the data;
            supporting them needs an explicit decision on their spelling).
    """
    match value:
        case str():
            return value
        case bool():
            return TRUE if value else FALSE
        case list():
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        case _:
            raise TypeError(f'No text form for {type(value).__name__} values: {value!r}')


def from_text(text: str, kind: FieldKind) -> Any:
    """Turn generated text back into a JSON value.

    Args:
        text: Generated text, without the close marker.
        kind: ``'array'`` parses ``text`` as JSON; ``'scalar'`` returns the
            string, except that exactly ``true`` / ``false`` become booleans
            (no string value in the data is spelled that way).

    Raises:
        ValueError: If an array's text is not a JSON array.
    """
    if kind == 'array':
        value = json.loads(text)  # json.JSONDecodeError is a ValueError
        if not isinstance(value, list):
            raise ValueError(f'Expected a JSON array, got {type(value).__name__}')
        return value
    if text == TRUE:
        return True
    if text == FALSE:
        return False
    return text
