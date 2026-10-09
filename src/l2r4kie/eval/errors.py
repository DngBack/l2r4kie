"""Recognisable failure modes of generated values."""

from __future__ import annotations

import re

#: Qwen2-VL's grounding output inside ``<|box_start|>...<|box_end|>``, e.g.
#: ``(295,434),(526,492)``. The base model writes this for every field until
#: fine-tuning replaces it with text (see ``docs/notes/marker_selection.md``).
COORDINATES = re.compile(r'\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*')


def is_coordinates(text: str) -> bool:
    """Whether ``text`` is a box in Qwen2-VL's grounding format and nothing else."""
    return COORDINATES.fullmatch(text) is not None
