"""Read chat-style KIE labels (one user prompt, one JSON answer) as documents.

Layout of one dataset directory (e.g. ``IDP-Training-Pipeline/Data/kie/<name>``)::

    image_label_details.json   [{"label_path", "image_paths" | "image_path", "split"?, ...}]
    labels/<doc>__v1.json      {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}

The answer is a nested JSON object; it becomes branches exactly as a training
label does (:func:`~l2r4kie.data.schema.iter_branches`): one field per leaf,
arrays atomic. Descriptions come from the prompt when it lists the fields::

    - "Hành chính" (object)
      - "Họ và tên": Họ tên bệnh nhân viết in hoa.
    - "Xét nghiệm" (array of objects): Gồm cả dòng tiêu đề nhóm
      - "Kết quả": Có thể kèm ký hiệu H/L

A prompt in prose (CORD, SROIE, FUNSD) gives no per-field description, so the
field id is its own description. ``null`` answers (an absent field) become
``""``, the blank value of the training labels, and numbers become text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..utils.io import PathLike, read_json
from .schema import iter_branches
from .types import Document, FieldSpec

_FIELD = re.compile(r'^(?P<indent>\s*)- "(?P<key>.+?)"(?: \((?P<kind>[^)]*)\))?(?::\s*(?P<desc>.*))?$')


def field_descriptions(prompt: str) -> dict[str, Any] | None:
    """Description tree of a field-list prompt, or ``None`` for a prose prompt.

    Objects map to dicts of their children; an array of objects to
    ``[{"__desc": ..., <children>}]`` and an array of scalars to ``[desc]``
    (the schema convention of :func:`~l2r4kie.data.schema.array_description`),
    in English (``array of objects``) or Vietnamese (``mảng object``) prompts;
    a ``(true/false)`` field to its description prefixed with that type; any
    other scalar to its description, or ``None`` when it has none.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    found = False
    for line in prompt.splitlines():
        match = _FIELD.match(line)
        if match is None:
            continue
        found = True
        depth = len(match['indent'])
        kind, desc = (match['kind'] or '').lower(), (match['desc'] or '').strip() or None
        while stack[-1][0] >= depth:
            stack.pop()
        parent = stack[-1][1]
        container = parent[0] if isinstance(parent, list) else parent
        array = kind.startswith(('array', 'mảng'))
        if kind == 'object':
            node: Any = {}
        elif array and 'object' in kind:
            node = [{'__desc': desc} if desc else {}]
        elif array:
            node = [desc] if desc else None
        elif kind == 'true/false':
            # The prompt's type annotation is the only hint that the answer is a boolean.
            node = f'(true/false) {desc}' if desc else '(true/false)'
        else:
            node = desc
        if isinstance(container, dict):
            container[match['key']] = node
        if isinstance(node, (dict, list)) and not (isinstance(node, list) and not isinstance(node[0], dict)):
            stack.append((depth, node))
    return root if found else None


def _plain(value: Any) -> Any:
    """Label value with ``null`` as ``""`` and numbers as text (arrays/objects recursively)."""
    if value is None:
        return ''
    if isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    raise TypeError(f'Unsupported label value {value!r}')


def conversation_fields(label: Mapping[str, Any]) -> list[FieldSpec]:
    """Branches of one chat label (prompt descriptions, answer values)."""
    turns = label['conversations']
    prompt = next(t['content'] for t in turns if t['role'] == 'user')
    answer = next(t['content'] for t in reversed(turns) if t['role'] == 'assistant')
    value = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(value, dict):
        raise TypeError('The answer is not a JSON object')
    descriptions = field_descriptions(prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False))
    return list(iter_branches(_plain(value), descriptions))


def iter_dataset(directory: PathLike, variant: str | None = 'v1', split: str | None = None) -> Iterator[Document]:
    """Documents of one dataset directory, in ``image_label_details.json`` order.

    Args:
        directory: Dataset directory.
        variant: Keep labels whose stem ends with ``__<variant>`` (datasets
            with several prompt variants per document); labels without a
            variant suffix are always kept. ``None`` keeps all.
        split: Keep entries of this ``split`` only (when they have one).
    """
    directory = Path(directory)
    for entry in read_json(directory / 'image_label_details.json'):
        if split is not None and entry.get('split', split) != split:
            continue
        label_path = Path(entry['label_path'])
        if not label_path.is_file():
            label_path = directory / 'labels' / label_path.name
        stem = label_path.stem
        if variant is not None and re.search(r'__v\d+$', stem) and not stem.endswith(f'__{variant}'):
            continue
        pages = entry.get('image_paths') or [entry['image_path']]
        pages = [p if Path(p).is_file() else str(directory / 'images' / Path(p).name) for p in pages]
        fields = conversation_fields(read_json(label_path))
        if fields:
            name = f'{directory.name}__{stem}'
            yield Document(name, name, directory.name, tuple(pages), tuple(fields), ())
