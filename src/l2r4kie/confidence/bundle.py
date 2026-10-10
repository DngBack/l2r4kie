"""A frozen confidence bundle, applied at serving time (``l2r4kie infer --confidence``).

A bundle is one family folder written by :mod:`l2r4kie.pipelines.finalize`::

    head.pt, head_config.json   the head
    calibration.json            logit calibration (calibration cohort)
    review_policy.json          review threshold (risk_validation cohort)
    bundle.json                 extractor fingerprint, cached layers, max_trace_tokens

Serving decodes with ``trace=True`` and the bundle's layers, builds the same
trace records as ``cache-traces`` (:func:`~l2r4kie.confidence.features.trace_record`),
and scores them with the same head code, so a served confidence equals the
cached one for the same document (checked in ``tests/test_pipelines.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.types import FieldRequest
from ..model.decode import DecodeResult
from ..utils.io import PathLike, read_json
from .calibration import probabilities
from .features import TraceStore, trace_record
from .heads import Head, load_head, score_head
from .policy import ReviewPolicy, queue_fields


@dataclass
class ConfidenceBundle:
    """Head, calibration and review policy of one frozen family.

    Attributes:
        folder: Bundle directory.
        head: Loaded head (eval mode).
        calibration: ``calibration.json`` content.
        policy: Frozen review policy.
        source_fingerprint: Fingerprint of the extractor the head was trained on.
        layers: Intermediate ``hidden_states`` indices the head may read.
        max_trace_tokens: Token states kept per field (as in the cache).
    """

    folder: Path
    head: Head
    calibration: dict[str, Any]
    policy: ReviewPolicy
    source_fingerprint: str
    layers: tuple[int, ...]
    max_trace_tokens: int

    @classmethod
    def load(cls, folder: PathLike, device: str | torch.device = 'cpu') -> ConfidenceBundle:
        """Read a bundle folder.

        Raises:
            FileNotFoundError: If a bundle file is missing (e.g. a folder not finalised).
        """
        folder = Path(folder)
        meta = read_json(folder / 'bundle.json')
        return cls(folder, load_head(folder, device), read_json(folder / 'calibration.json'),
                   ReviewPolicy.from_dict(read_json(folder / 'review_policy.json')), meta['source_fingerprint'],
                   tuple(meta.get('layers') or ()), int(meta.get('max_trace_tokens', 512)))

    def check_source(self, fingerprint: str) -> None:
        """Refuse an extractor other than the one the head was trained on.

        Raises:
            ValueError: If the fingerprints differ.
        """
        if fingerprint != self.source_fingerprint:
            raise ValueError(f'{self.folder} was trained on extractor {self.source_fingerprint[:12]}…, '
                             f'not {fingerprint[:12]}…; confidence would be meaningless')

    @property
    def signals(self) -> tuple[str, ...]:
        """Signals the head reads."""
        return self.head.config.required_signals() if self.head.config.signals else ('value',)

    def score(self, results: Sequence[DecodeResult], requests: Sequence[FieldRequest],
              tokenizer: Any) -> list[float | None]:
        """Calibrated confidence per field (``None`` unless the status is ``ok``).

        Args:
            results: Decode results with traces (``extract(..., trace=True, layers=self.layers)``).
            requests: The field requests, in the same order.
            tokenizer: Tokenizer that decodes each value token to its text piece.
        """
        records, positions = [], []
        for i, (result, request) in enumerate(zip(results, requests, strict=True)):
            if result.status != 'ok':
                continue
            pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in result.trace.tokens[:-1]]
            records.append(trace_record(result, pieces, request.kind, self.max_trace_tokens))
            positions.append(i)
        scores: list[float | None] = [None] * len(results)
        if not records:
            return scores
        device = next(iter(self.head.buffers()), torch.empty(0)).device
        store = TraceStore(records, self.signals, self.layers, device)
        logits = score_head(self.head, store, [results[i].field_id for i in positions])
        for i, p in zip(positions, probabilities(logits, self.calibration), strict=True):
            scores[i] = p
        return scores

    def review(self, results: Sequence[DecodeResult], scores: Sequence[float | None]) -> list[dict[str, Any]]:
        """Review queue of a document, most urgent first (see :func:`~.policy.queue_fields`)."""
        return queue_fields([{'id': r.field_id, 'confidence': s, 'status': r.status}
                             for r, s in zip(results, scores, strict=True)], self.policy)
