"""Human-readable views of what the model is fed (``show-input``, ``token-stats``).

These need the tokenizer/processor only, never model weights, so they run in
seconds on CPU and are the quickest way to check the input format by eye.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.selection import Selection
from ..data.types import SPLITS, Document
from ..model.format import DEFAULT_MAX_VALUE_TOKENS, EncodedBranch, KevFormat
from ..model.markers import special_ids
from ..utils.io import PathLike, iter_jsonl


def find_document(prepared: PathLike, document_id: str) -> tuple[str, Document]:
    """Return ``(split, document)`` for an id, searching every prepared split.

    Raises:
        KeyError: If no split contains the id.
    """
    for split in SPLITS:
        for record in iter_jsonl(Path(prepared) / f'{split}.jsonl'):
            if record['id'] == document_id:
                return split, Document.from_json(record)
    raise KeyError(f'Document {document_id!r} not found in {prepared}')


def render_ids(tokenizer: Any, ids: Sequence[int], special: set[int]) -> str:
    """Decode ids, collapsing runs of one repeated special token to ``token×n``.

    Ordinary text is decoded in whole segments (decoding token by token would
    break multi-byte characters such as Vietnamese diacritics).
    """
    parts: list[str] = []
    segment: list[int] = []
    i = 0
    while i < len(ids):
        j = i
        while j < len(ids) and ids[j] == ids[i]:
            j += 1
        if ids[i] in special and j - i > 1:
            parts.append(tokenizer.decode(segment, skip_special_tokens=False))
            parts.append(f'{tokenizer.decode([ids[i]], skip_special_tokens=False)}×{j - i}')
            segment = []
        else:
            segment.extend(ids[i:j])
        i = j
    parts.append(tokenizer.decode(segment, skip_special_tokens=False))
    return ''.join(parts)


def show_input(processor: Any, fmt: KevFormat, prepared: PathLike, document_id: str,
               field_ids: Sequence[str] = (), fields: int = 3, max_value_tokens: int = DEFAULT_MAX_VALUE_TOKENS) -> str:
    """Render the prefix and some branches of one document as the model sees them.

    Args:
        processor: Qwen2-VL processor (see :func:`~l2r4kie.model.format.load_processor`).
        fmt: The input format.
        prepared: Prepared directory.
        document_id: Document to show.
        field_ids: Fields to show; empty shows the first ``fields`` fields.
        fields: Number of fields shown when ``field_ids`` is empty.
        max_value_tokens: Training limit; longer targets are flagged.

    Returns:
        Multi-line text. Positions are indices in the packed training sequence.
    """
    split, document = find_document(prepared, document_id)
    tokenizer = processor.tokenizer
    special = special_ids(tokenizer, len(tokenizer))
    prefix_ids = fmt.encode_prefix(processor, document.pages)['input_ids'][0].tolist()
    chosen = ([f for f in document.fields if f.id in set(field_ids)] if field_ids else list(document.fields[:fields]))
    lines = [f'Document {document.id}  (split {split}, form {document.form}, {len(document.pages)} page(s), '
             f'{len(document.fields)} fields)',
             '', f'PREFIX  {len(prefix_ids)} tokens, positions 0-{len(prefix_ids) - 1}',
             render_ids(tokenizer, prefix_ids, special), '']
    start = len(prefix_ids)
    for n, field in enumerate(chosen, 1):
        branch: EncodedBranch = fmt.encode(field)
        packed = len(branch.target) <= max_value_tokens
        lines += [
            f'BRANCH {n}  {field.id}  ({field.kind})',
            f'  prompt {len(branch.prompt):>4} tokens  {tokenizer.decode(list(branch.prompt), skip_special_tokens=False)}',
            f'  target {len(branch.target):>4} tokens  {tokenizer.decode(list(branch.target), skip_special_tokens=False)}',
        ]
        if packed:
            lines.append(f'  h_key @ {start + branch.key_index}   h_decide @ {start + branch.decide_index}   '
                         f'h_value @ {start + branch.value_index}   (branch spans {start}-{start + branch.length - 1})')
            start += branch.length
        else:
            # Training drops such fields before packing, so they take no positions.
            lines.append(f'  ⚠ target {len(branch.target)} > max_value_tokens {max_value_tokens}: '
                         'not packed, skipped in training')
        lines.append('')
    return '\n'.join(lines)


def _summary(lengths: Sequence[int], limit: int | None = None) -> dict[str, Any]:
    """Count and length percentiles of token counts; with ``limit``, also how many exceed it."""
    if not lengths:
        return {'fields': 0}
    values = np.asarray(lengths)
    summary = {'fields': len(values), 'p50': int(np.percentile(values, 50)), 'p95': int(np.percentile(values, 95)),
               'max': int(values.max())}
    if limit is not None:
        summary['over_max_value_tokens'] = int((values > limit).sum())
    return summary


def token_stats(fmt: KevFormat, selection: Selection, split: str, max_value_tokens: int = DEFAULT_MAX_VALUE_TOKENS,
                limit: int | None = None) -> dict[str, Any]:
    """Token lengths of prompts and targets of a split, per field kind.

    Reports how many targets exceed ``max_value_tokens`` (those fields cannot
    be trained on and are skipped, as in the old ``usable`` filter) and checks
    that no caller text produced a special token.

    Raises:
        AssertionError: If a key, description or value encoded to a special
            token other than the markers (would mean the escaping is broken).
    """
    tokenizer = fmt.tokenizer
    special = special_ids(tokenizer, len(tokenizer))
    m = fmt.markers
    prompts: dict[str, list[int]] = defaultdict(list)
    targets: dict[str, list[int]] = defaultdict(list)
    documents = selection.documents(split, limit)  # type: ignore[arg-type]
    for document in documents:
        for field in document.fields:
            branch = fmt.encode(field)
            inner = branch.prompt[1:-2] + branch.target[:-1]
            assert not special & set(inner), f'special token inside text of {document.id} {field.id}'
            assert branch.prompt[0] == m.key_open and branch.target[-1] == m.value_close
            prompts[field.kind].append(len(branch.prompt))
            targets[field.kind].append(len(branch.target))
    kinds = sorted(targets)
    return {
        'split': split, 'documents': len(documents), 'max_value_tokens': max_value_tokens,
        'target_tokens': {k: _summary(targets[k], max_value_tokens) for k in kinds},
        'prompt_tokens': {k: _summary(prompts[k]) for k in kinds},
    }
