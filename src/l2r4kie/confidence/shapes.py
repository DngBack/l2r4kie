"""Value-shape rules: mandatory review of a value whose shape a field never takes.

The *shape* of a text maps every digit to ``9`` and every letter to ``a``
(``"12/03/2024"`` → ``"99/99/9999"``, ``"Số 5"`` → ``"aa 9"``). For each scalar
field whose labelled values on the extractor's training documents contain a
digit, the shapes taking at least ``min_share`` of them are kept. A field
gets a rule only when it has ``min_examples`` such values and at most
``max_shapes`` kept shapes that together cover ``coverage`` of them (a
field with stable formatting: dates, codes, amounts).

A predicted scalar that contains a digit and has none of the field's shapes
is reviewed whatever its confidence. On the kev_r4 cohorts about 99% of the
flagged values are wrong, and those are often the confident errors the head
misses (a dropped or extra digit, a misread separator). Rules are keyed by
field id only: a request at serving time carries no form, and keying by
(form, field) flagged nearly the same values on dev and risk_validation.

Rules read only labels of the extractor's training documents, never a
confidence cohort.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..eval.comparator import as_text

_DIGIT = re.compile(r'\d')
_LETTER = re.compile(r'[^\W\d_]')


def shape(text: str) -> str:
    """Digits to ``9``, letters to ``a``; everything else kept."""
    return _LETTER.sub('a', _DIGIT.sub('9', text))


@dataclass(frozen=True)
class ShapeRules:
    """Allowed value shapes per field id.

    Attributes:
        allowed: Field id → shapes its values may take.
        min_examples: Labelled values with a digit a field needs for a rule.
        min_share: Share of those values a shape needs to be kept.
        coverage: Share of the values the kept shapes must cover.
        max_shapes: Most shapes a rule may keep.
        source: Where the labels came from (free text, for the record).
    """

    allowed: Mapping[str, frozenset[str]] = field(default_factory=dict)
    min_examples: int = 20
    min_share: float = .02
    coverage: float = .95
    max_shapes: int = 5
    source: str = ''

    @classmethod
    def fit(cls, documents: Iterable[Any], min_examples: int = 20, min_share: float = .02, coverage: float = .95,
            max_shapes: int = 5, source: str = '') -> ShapeRules:
        """Learn rules from labelled documents (``.fields`` with ``id``, ``kind``, ``value``).

        Raises:
            ValueError: On parameters outside their range.
        """
        if min_examples < 1 or not 0 < min_share <= 1 or not 0 < coverage <= 1 or max_shapes < 1:
            raise ValueError('Invalid shape-rule parameters')
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for document in documents:
            for f in document.fields:
                if f.kind == 'scalar':
                    text = as_text(f.value)
                    if _DIGIT.search(text):
                        counts[f.id][shape(text)] += 1
        allowed = {}
        for field_id, shapes in counts.items():
            total = sum(shapes.values())
            keep = {s for s, n in shapes.items() if n / total >= min_share}
            if total >= min_examples and len(keep) <= max_shapes and sum(shapes[s] for s in keep) / total >= coverage:
                allowed[field_id] = frozenset(keep)
        return cls(allowed, min_examples, min_share, coverage, max_shapes, source)

    def violates(self, field_id: str, kind: str, value: Any, status: str = 'ok') -> bool:
        """Whether a decoded value breaks its field's rule (never for non-``ok`` or array values)."""
        shapes = self.allowed.get(field_id)
        if shapes is None or status != 'ok' or kind != 'scalar':
            return False
        text = as_text(value)
        return bool(_DIGIT.search(text)) and shape(text) not in shapes

    def flag_rows(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Copies of review rows (``field_id``, ``kind``, ``prediction``, ``status``) with ``shape_violation``."""
        return [{**r, 'shape_violation': self.violates(r['field_id'], r['kind'], r['prediction'], r['status'])}
                for r in rows]

    def to_dict(self) -> dict[str, Any]:
        """JSON form (shapes sorted, for a stable bundle fingerprint)."""
        return {'min_examples': self.min_examples, 'min_share': self.min_share, 'coverage': self.coverage,
                'max_shapes': self.max_shapes, 'source': self.source,
                'allowed': {k: sorted(v) for k, v in sorted(self.allowed.items())}}

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> ShapeRules:
        """Read :meth:`to_dict` output."""
        return cls({k: frozenset(v) for k, v in record['allowed'].items()}, record['min_examples'],
                   record['min_share'], record['coverage'], record['max_shapes'], record.get('source', ''))
