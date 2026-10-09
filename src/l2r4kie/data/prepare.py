"""Turn the raw dataset into deduplicated train/dev/calibration/test splits.

Raw layout (read-only)::

    <root>/manifest.jsonl          one row per document: form, sample, pages, fields
    <root>/images/<page>           page images
    <root>/kie-labels/<fields>     {"fields": {...}, "unprinted": [...]}
    <root>/schemas/<form>.descriptions.json   optional description tree

Splitting is leak-proof against duplicated scans: pages are hashed by
content, documents sharing any page are merged into one group, and the split
is a hash of the group. The output is byte-identical to the old repository's
``data.prepare`` for the same input and seed.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..utils.fingerprint import hash_bucket, sha256_file
from ..utils.io import PathLike, read_jsonl, write_json, write_jsonl
from .schema import iter_branches
from .types import SPLITS, Document, FieldSpec, Split

#: Upper bucket bounds (exclusive) of each split, out of 100: 80/7/7/6 percent.
SPLIT_BOUNDS: tuple[tuple[Split, int], ...] = (('train', 80), ('dev', 87), ('calibration', 94), ('test', 100))

ARRAY_POLICY = 'Atomic JSON branch; no GT-derived row count or row location used as prompt.'

#: Per-document errors that reject the document instead of aborting the run.
_REJECTABLE = (ValueError, KeyError, OSError, TypeError)


def document_id(row: Mapping[str, Any]) -> str:
    """Return the ``'<form>__<sample>'`` id of a manifest row."""
    return f"{row['form']}__{row['sample']}"


def assign_split(group_key: str, seed: int) -> Split:
    """Return the split of a duplicate group, stable for a given seed."""
    bucket = hash_bucket(f'{seed}:{group_key}')
    return next(split for split, bound in SPLIT_BOUNDS if bucket < bound)


def hash_pages(images: Path, names: Iterable[str], workers: int = 8) -> dict[str, str]:
    """SHA-256 every existing page file, in parallel.

    Args:
        images: Directory holding the page images.
        names: Page file names; duplicates and missing files are allowed.
        workers: Hashing threads (``hashlib`` releases the GIL on large
            buffers, so threads scale on fast storage).

    Returns:
        ``{name: hex digest}`` for each name whose file exists. Missing pages
        are left out here and reported later as a rejected document.
    """
    unique = [name for name in dict.fromkeys(names) if (images / name).is_file()]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        digests = pool.map(lambda name: sha256_file(images / name), unique)
        return dict(zip(unique, digests, strict=True))


def group_duplicates(manifest: Sequence[Mapping[str, Any]], page_hashes: Mapping[str, str]) -> list[str]:
    """Group documents that share any page image, transitively.

    Uses union-find over manifest rows: two rows are joined when one of their
    pages has the same content hash. A multi-page document can thereby link
    several otherwise unrelated documents into one group.

    Args:
        manifest: Manifest rows, in file order.
        page_hashes: Output of :func:`hash_pages`.

    Returns:
        For each manifest row, its group key: the smallest document id in its
        group. The key does not depend on row order.
    """
    parents = list(range(len(manifest)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]  # path halving
            i = parents[i]
        return i

    first_row: dict[str, int] = {}
    for i, row in enumerate(manifest):
        for name in row['pages']:
            digest = page_hashes.get(name)
            if digest is None:
                continue
            if digest in first_row:
                a, b = find(i), find(first_row[digest])
                parents[max(a, b)] = min(a, b)
            else:
                first_row[digest] = i
    members: dict[int, list[str]] = {}
    for i, row in enumerate(manifest):
        members.setdefault(find(i), []).append(document_id(row))
    return [min(members[find(i)]) for i in range(len(manifest))]


def load_fields(root: Path, row: Mapping[str, Any]) -> list[FieldSpec]:
    """Read a document's label and flatten it into extraction branches.

    Fields listed under the label's ``unprinted`` key are dropped rather than
    kept as "confidently absent": they may be hidden or omitted on the page,
    so their ground truth is not observable from the image.

    Raises:
        ValueError: If no usable field remains.
        OSError, KeyError, TypeError: On a missing or malformed label.
    """
    label = json.loads((root / 'kie-labels' / row['fields']).read_text())
    description_file = root / 'schemas' / f"{row['form'].removeprefix('lift-')}.descriptions.json"
    descriptions = json.loads(description_file.read_text()) if description_file.exists() else {}
    fields = list(iter_branches(label['fields'], descriptions))
    unprinted = label.get('unprinted', [])
    if isinstance(unprinted, list):
        fields = [f for f in fields if f.id not in unprinted]
    if not fields:
        raise ValueError('No usable fields')
    return fields


def prepare(root: PathLike, output: PathLike, seed: int = 42, workers: int = 8) -> dict[str, Any]:
    """Build ``<output>/{train,dev,calibration,test}.jsonl`` and ``report.json``.

    Documents with a missing page, an unreadable label or no usable field are
    skipped and listed under ``report['rejected']``. Every file is written
    atomically, after all documents are processed.

    Args:
        root: Raw dataset directory (never written to).
        output: Destination directory; must lie outside ``root``.
        seed: Split seed. ``42`` reproduces the published splits.
        workers: Threads for hashing page images.

    Returns:
        The report, also written to ``<output>/report.json``.

    Raises:
        ValueError: If ``output`` is ``root`` or inside it.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside the read-only source dataset')
    images = root / 'images'
    manifest = read_jsonl(root / 'manifest.jsonl')
    page_hashes = hash_pages(images, (name for row in manifest for name in row['pages']), workers)
    groups = group_duplicates(manifest, page_hashes)

    records: dict[Split, list[Document]] = {split: [] for split in SPLITS}
    # Counters keep first-seen key order, which the report format relies on.
    split_counts: Counter[str] = Counter()
    form_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    rejected: list[dict[str, str]] = []
    for row, group_key in zip(manifest, groups, strict=True):
        doc_id = document_id(row)
        try:
            pages = [images / name for name in row['pages']]
            if not pages or not all(page.is_file() for page in pages):
                raise ValueError('Missing image')
            fields = load_fields(root, row)
            split = assign_split(group_key, seed)
            records[split].append(Document(
                id=doc_id, group_id=group_key, form=row['form'],
                pages=tuple(str(page) for page in pages), fields=tuple(fields),
                image_sha256=tuple(page_hashes[name] for name in row['pages']),
            ))
            split_counts[split] += 1
            form_counts[row['form']] += 1
            branch_counts['array_branches'] += sum(f.kind == 'array' for f in fields)
            branch_counts['scalar_branches'] += sum(f.kind == 'scalar' for f in fields)
        except _REJECTABLE as error:
            rejected.append({'id': doc_id, 'reason': str(error)})

    for split, documents in records.items():
        write_jsonl(output / f'{split}.jsonl', (d.to_json() for d in documents))
    report = {
        'source': str(root), 'seed': seed, 'documents': split_counts.total(), 'splits': dict(split_counts),
        'forms': dict(form_counts), 'branches': dict(branch_counts), 'rejected': rejected,
        # Page files whose content already appeared under another name.
        'duplicate_page_occurrences': len(page_hashes) - len(set(page_hashes.values())),
        'array_policy': ARRAY_POLICY,
    }
    write_json(output / 'report.json', report)
    return report

