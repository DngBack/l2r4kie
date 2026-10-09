"""JSON / JSONL helpers and atomic file writes.

Every artifact the pipeline produces (prepared splits, reports, trace caches,
policies) is written through :func:`atomic_path`. A crash or ``Ctrl-C`` mid-write
then leaves either the previous complete file or no file at all, never a
truncated one that a later step would silently read.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PathLike = str | os.PathLike[str]


@contextmanager
def atomic_path(path: PathLike) -> Iterator[Path]:
    """Yield a temporary path that replaces ``path`` only if the block succeeds.

    The temporary file is created in the destination directory, so the final
    :func:`os.replace` is a same-filesystem rename and therefore atomic. Parent
    directories are created as needed.

    Example::

        with atomic_path(out / 'train.pt') as tmp:
            torch.save(cache, tmp)

    Args:
        path: Final destination of the file.

    Yields:
        A path to write to. It does not exist yet when the block starts.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp reserves a unique name; close the descriptor and remove the file
    # so writers that refuse to overwrite (or that open by name) behave normally.
    fd, name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    temporary.unlink()
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        # On success the rename already consumed the file; on failure, clean up.
        temporary.unlink(missing_ok=True)


def write_text(path: PathLike, text: str) -> Path:
    """Atomically write UTF-8 ``text`` to ``path`` and return the path."""
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding='utf-8')
    return Path(path)


def read_json(path: PathLike) -> Any:
    """Read one JSON document from ``path``."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: PathLike, value: Any) -> Path:
    """Atomically write ``value`` as pretty, UTF-8 JSON (non-ASCII kept as-is).

    The format (2-space indent, trailing newline) matches the old repository's
    reports so files can be diffed against them directly.
    """
    return write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def iter_jsonl(path: PathLike) -> Iterator[Any]:
    """Lazily yield one parsed record per non-blank line of a JSONL file."""
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: PathLike) -> list[Any]:
    """Read every record of a JSONL file into a list."""
    return list(iter_jsonl(path))


def dumps_jsonl(record: Any) -> str:
    """Serialize one record as a single JSONL line (with trailing newline)."""
    return json.dumps(record, ensure_ascii=False) + '\n'


def write_jsonl(path: PathLike, records: Iterable[Any]) -> Path:
    """Atomically write ``records`` as JSONL, one compact record per line."""
    with atomic_path(path) as temporary, temporary.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(dumps_jsonl(record))
    return Path(path)


def append_jsonl(path: PathLike, record: Any) -> None:
    """Append one record to a JSONL log, flushing it to the OS immediately.

    Used for running logs (e.g. per-step training metrics) where the file grows
    over time; appending is not atomic, but a torn final line is the only
    possible damage and :func:`iter_jsonl` callers can detect it.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a', encoding='utf-8') as handle:
        handle.write(dumps_jsonl(record))
        handle.flush()
