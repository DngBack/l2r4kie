"""Content hashes: file digests, checkpoint fingerprints and hash buckets.

These hashes are part of the experiment protocol, not an implementation detail:

* :func:`checkpoint_fingerprint` ties cached traces and frozen review policies
  to the exact extractor that produced them. Its byte-for-byte algorithm is
  kept from the old repository so existing fingerprints (e.g. the r4
  checkpoint's ``596bba7c...``) still verify.
* :func:`hash_bucket` assigns documents to splits and hold-outs
  deterministically, independent of file order or Python's hash seed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

PathLike = str | os.PathLike[str]

#: Files at the checkpoint root that affect decoding or cached baseline scores.
#: Old checkpoints may carry ``head.pt`` / ``calibration.json``; new ones may
#: not. Missing files are simply skipped, matching the original behaviour.
CHECKPOINT_ROOT_FILES: tuple[str, ...] = ('head.pt', 'head_config.json', 'config.json', 'calibration.json')

_CHUNK_SIZE = 1 << 20


def sha256_file(path: PathLike) -> str:
    """Return the hex SHA-256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(checkpoint: PathLike) -> str:
    """Fingerprint an extractor checkpoint directory.

    The digest covers every file directly inside ``adapter/`` plus the
    :data:`CHECKPOINT_ROOT_FILES` that exist. Files are visited in sorted order
    of their path, and for each one the relative path and then its bytes are
    fed into a single SHA-256.

    Args:
        checkpoint: Checkpoint directory (the one containing ``adapter/``).

    Returns:
        Hex SHA-256 digest.

    Raises:
        FileNotFoundError: If ``checkpoint`` is not a directory. (The old code
            silently fingerprinted nothing, which made a typo look like a
            valid, empty checkpoint.)
    """
    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f'Checkpoint directory not found: {root}')
    candidates = [*(root / 'adapter').glob('*'), *(root / name for name in CHECKPOINT_ROOT_FILES)]
    digest = hashlib.sha256()
    # Sorting Path objects (not strings) is what the original implementation
    # did; keep it so the visiting order, and therefore the digest, is unchanged.
    for path in sorted(p for p in candidates if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def hash_bucket(key: str, modulo: int = 100) -> int:
    """Map ``key`` to a stable bucket in ``[0, modulo)``.

    Uses the first 32 bits of the key's SHA-256, as the old split and
    hold-out code did, so assignments reproduce exactly. Callers that need a
    seed include it in the key, e.g. ``hash_bucket(f'{seed}:{group_id}')``.
    """
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % modulo
