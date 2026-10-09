"""YAML config loading with command-line overrides.

Configs stay plain nested dicts at this layer. Typed validation (unknown keys,
required fields) belongs to the consumer, e.g. ``train.config.TrainConfig``,
because only it knows which keys are valid.

Overrides use dotted keys and YAML-typed values, so the CLI can tweak a run
without copying the file::

    l2r4kie train --config kev_smoke.yaml --set steps=50 --set format.close=im_end
"""

from __future__ import annotations

import copy
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

PathLike = str | os.PathLike[str]
Config = dict[str, Any]


def load_config(path: PathLike, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML mapping from ``path`` and apply ``key=value`` overrides.

    Args:
        path: YAML file whose top level must be a mapping.
        overrides: Strings in the form accepted by :func:`parse_override`.

    Returns:
        The resulting config as a new dict.

    Raises:
        ValueError: If the file's top level is not a mapping, or an override
            is malformed.
    """
    loaded = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Config {path} must be a YAML mapping, got {type(loaded).__name__}')
    return apply_overrides(loaded, overrides)


def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``'a.b.c=value'`` into ``(['a', 'b', 'c'], value)``.

    The value is parsed as YAML, so ``steps=50`` gives an ``int``,
    ``forms=[a,b]`` a list, ``resume=true`` a bool and ``device=cuda:0`` a
    string. An empty value (``key=``) gives ``None``.

    Raises:
        ValueError: If there is no ``=`` or the key has an empty segment.
    """
    key, separator, raw = text.partition('=')
    parts = key.strip().split('.')
    if not separator or not all(parts):
        raise ValueError(f'Override must look like key.sub=value, got {text!r}')
    return parts, yaml.safe_load(raw)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> Config:
    """Return a deep copy of ``config`` with each override applied in order.

    Intermediate mappings are created when missing. Overriding *through* a
    non-mapping value (e.g. ``lr.x=1`` when ``lr`` is a float) is an error
    rather than a silent replacement.

    Raises:
        ValueError: On a malformed override or a path through a non-mapping.
    """
    result: Config = copy.deepcopy(dict(config))
    for text in overrides:
        parts, value = parse_override(text)
        node: dict[str, Any] = result
        for depth, part in enumerate(parts[:-1]):
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                prefix = '.'.join(parts[:depth + 1])
                raise ValueError(f'Cannot override {text!r}: {prefix} is not a mapping')
            node = child
        node[parts[-1]] = value
    return result
