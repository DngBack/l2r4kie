"""Typed records for prepared documents.

A *prepared* document is one line of ``<prepared>/<split>.jsonl``. Its JSON
layout is fixed by :meth:`Document.to_json` (key order included) so that new
splits stay byte-identical to the ones the old repository produced.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

#: ``'scalar'``: a single leaf value (string or bool in this dataset).
#: ``'array'``: a variable-length list, kept as one atomic branch.
FieldKind = Literal['scalar', 'array']

#: Splits written by ``prepare``.
Split = Literal['train', 'dev', 'calibration', 'test']

#: Splits accepted by document selection: ``'train'`` is the part of the train
#: split used for extractor gradients, ``'train_reserve'`` the hash-held-out
#: rest (see :class:`~l2r4kie.data.selection.Selection`).
SelectableSplit = Literal['train', 'train_reserve', 'dev', 'calibration', 'test']

SPLITS: tuple[Split, ...] = ('train', 'dev', 'calibration', 'test')


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One extraction branch of a document: a field and its ground truth.

    Attributes:
        id: JSON-pointer-like path of the field, e.g. ``'/benh_nhan/ho_ten'``
            (segments escaped by :func:`~l2r4kie.data.schema.escape_pointer`).
        description: Human description shown to the model; falls back to ``id``.
        value: Ground-truth JSON value (string, bool or list).
        kind: ``'array'`` when ``value`` is a list, otherwise ``'scalar'``.
    """

    id: str
    description: str
    value: Any
    kind: FieldKind

    def to_json(self) -> dict[str, Any]:
        """Return the prepared-file representation (key order is part of the format)."""
        return {'id': self.id, 'description': self.description, 'value': self.value, 'kind': self.kind}

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> FieldSpec:
        """Build from a prepared-file field record."""
        return cls(record['id'], record['description'], record['value'], record['kind'])


@dataclass(frozen=True, slots=True)
class Document:
    """A prepared document: its page images and the fields to extract.

    Attributes:
        id: ``'<form>__<sample>'``, unique across the dataset.
        group_id: Smallest document id among all documents sharing a page
            image with this one (transitively). Splits and hold-outs are
            assigned per group, so duplicated pages never cross a boundary.
        form: Form type, e.g. ``'lift-a-1x'``.
        pages: Absolute paths of the page images, in reading order.
        fields: Extraction branches, in label order.
        image_sha256: SHA-256 of each page file, aligned with ``pages``.
    """

    id: str
    group_id: str
    form: str
    pages: tuple[str, ...]
    fields: tuple[FieldSpec, ...]
    image_sha256: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """Return the prepared-file representation (key order is part of the format)."""
        return {'id': self.id, 'group_id': self.group_id, 'form': self.form, 'pages': list(self.pages),
                'fields': [f.to_json() for f in self.fields], 'image_sha256': list(self.image_sha256)}

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> Document:
        """Build from one line of a prepared split."""
        return cls(record['id'], record['group_id'], record['form'], tuple(record['pages']),
                   tuple(FieldSpec.from_json(f) for f in record['fields']), tuple(record['image_sha256']))


@dataclass(frozen=True, slots=True)
class FieldRequest:
    """A field to extract at inference time (no ground truth).

    Attributes:
        id: Caller's field key, returned unchanged in the response.
        description: What to extract; defaults to the id.
        kind: ``'array'`` parses the generated text as a JSON array;
            ``'scalar'`` (default) returns text, or a bool for ``true``/``false``.
    """

    id: str
    description: str = ''
    kind: FieldKind = 'scalar'

    @classmethod
    def from_field(cls, field: FieldSpec) -> FieldRequest:
        """The request a labelled field corresponds to (for evaluation)."""
        return cls(field.id, field.description, field.kind)
