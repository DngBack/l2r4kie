"""Choose which prepared documents a run uses, and in what order.

Everything downstream (extractor training, evaluation, confidence cohorts)
reads documents through :class:`Selection`, so the train/reserve partition and
the document order are defined in exactly one place.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.fingerprint import hash_bucket
from ..utils.io import iter_jsonl
from .types import Document, SelectableSplit


@dataclass(frozen=True, slots=True)
class Selection:
    """How to read documents from a prepared directory.

    Attributes:
        prepared: Directory written by :func:`~l2r4kie.data.prepare.prepare`.
        seed: Seed of the document shuffle.
        holdout_percent: Share (0-100) of train *groups* moved to
            ``'train_reserve'``: never used for extractor gradients, kept to
            give the confidence head realistic errors. Hash-based, so the
            partition does not depend on ``seed`` or file order.
        forms: Keep only these forms; ``None`` or empty keeps all.
        balanced_forms: For ``'train'`` only, interleave forms round-robin
            (one document of each form, then the next one of each, ...) so a
            truncated run still sees every form.
    """

    prepared: Path
    seed: int = 42
    holdout_percent: int = 0
    forms: tuple[str, ...] | None = None
    balanced_forms: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Selection:
        """Read the selection keys of a run config, ignoring all other keys.

        Recognised keys: ``prepared`` (required), ``seed``, ``holdout_percent``,
        ``forms``, ``balanced_forms``.
        """
        forms = config.get('forms')
        return cls(
            prepared=Path(config['prepared']),
            seed=config.get('seed', 42),
            holdout_percent=config.get('holdout_percent', 0),
            forms=tuple(forms) if forms else None,
            balanced_forms=config.get('balanced_forms', False),
        )

    def is_reserved(self, document: Document) -> bool:
        """Whether ``document`` belongs to the train reserve (decided per group)."""
        return hash_bucket(document.group_id) < self.holdout_percent

    def documents(self, split: SelectableSplit, limit: int | None = None) -> list[Document]:
        """Return the documents of ``split`` in their run order.

        Steps, in order: read the split file (``'train_reserve'`` reads
        ``train.jsonl``), apply the train/reserve partition, filter ``forms``,
        shuffle with ``seed``, interleave forms (``'train'`` with
        ``balanced_forms`` only), then truncate to ``limit``.

        Args:
            split: Prepared split, or ``'train_reserve'``.
            limit: Keep at most this many documents; ``None`` keeps all.
        """
        reserve = split == 'train_reserve'
        documents = [Document.from_json(r) for r in iter_jsonl(self.prepared / f"{'train' if reserve else split}.jsonl")]
        if split in ('train', 'train_reserve'):
            documents = [d for d in documents if self.is_reserved(d) == reserve]
        if self.forms:
            documents = [d for d in documents if d.form in self.forms]
        random.Random(self.seed).shuffle(documents)
        if split == 'train' and self.balanced_forms:
            documents = interleave_forms(documents)
        return documents if limit is None else documents[:limit]


def interleave_forms(documents: Sequence[Document]) -> list[Document]:
    """Round-robin over forms, keeping each form's internal order.

    Forms are visited in order of first appearance, e.g.
    ``[a1, a2, b1, a3, c1] -> [a1, b1, c1, a2, a3]``.
    """
    by_form: dict[str, list[Document]] = {}
    for document in documents:
        by_form.setdefault(document.form, []).append(document)
    rounds = max(map(len, by_form.values()), default=0)
    return [group[i] for i in range(rounds) for group in by_form.values() if i < len(group)]
