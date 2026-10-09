"""Flatten a nested label into extraction branches.

Each leaf of a label object becomes one :class:`~l2r4kie.data.types.FieldSpec`
(one isolated decode branch). Arrays are *not* expanded into per-row fields:
the number of rows is unknown at inference time, so deriving prompts from the
ground-truth row count would leak the answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .types import FieldSpec


def escape_pointer(key: str) -> str:
    """Escape one object key as a JSON-Pointer segment (RFC 6901).

    ``~`` becomes ``~0`` and ``/`` becomes ``~1``; the order matters so that a
    literal ``~1`` in a key round-trips.
    """
    return key.replace('~', '~0').replace('/', '~1')


def array_description(description: Any) -> str | None:
    """Return the description of an array field from its schema entry, if any.

    Schemas describe an array by a one-element list holding either the item
    description (``["..."]``, arrays of scalars) or an item object whose
    ``__desc`` describes the array (``[{"__desc": "...", ...}]``).
    """
    if isinstance(description, list) and len(description) == 1:
        item = description[0]
        if isinstance(item, str):
            return item
        if isinstance(item, dict) and isinstance(item.get('__desc'), str):
            return item['__desc']
    return None


def iter_branches(value: Any, description: Any = None, path: str = '',
                  describe_arrays: bool = True) -> Iterator[FieldSpec]:
    """Yield one branch per leaf of ``value``, depth-first in key order.

    Args:
        value: Label value; objects are recursed into, anything else is a leaf.
        description: Description tree parallel to ``value``. A string at a
            leaf is used as that field's description; for an array, see
            :func:`array_description`. Anything else falls back to the path.
        path: Pointer prefix of ``value`` (``''`` at the root).
        describe_arrays: Read array descriptions from the schema. ``False``
            reproduces the old repository, where every array field (all
            15,562 of them) silently got its path as description.

    Yields:
        Fields with ids such as ``'/a~1b/x~0y'``.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            child = description.get(key) if isinstance(description, dict) else None
            yield from iter_branches(item, child, f'{path}/{escape_pointer(key)}', describe_arrays)
        return
    text = description if isinstance(description, str) else None
    if text is None and describe_arrays and isinstance(value, list):
        text = array_description(description)
    yield FieldSpec(id=path, description=path if text is None else text, value=value,
                    kind='array' if isinstance(value, list) else 'scalar')
