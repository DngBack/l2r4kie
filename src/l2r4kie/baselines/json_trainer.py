"""Train and evaluate the JSON baseline (see :mod:`.json_format`) with the extractor's trainer and decoder."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ..data.types import Document, FieldRequest
from ..eval.comparator import Comparator
from ..eval.extraction import score
from ..model.decode import decode_prefix
from ..model.extractor import Extractor
from ..model.format import EncodedBranch
from ..train.trainer import Trainer
from ..utils.io import dumps_jsonl, read_jsonl, write_json
from . import json_format


class JsonTrainer(Trainer):
    """:class:`~l2r4kie.train.trainer.Trainer` whose step trains one JSON branch over the sampled fields.

    The fields are sampled exactly as KevFormat's trainer samples branches
    (same eligibility, same random stream), so both runs see the same
    documents and fields at every step; only their layout differs.
    """

    def fields_for(self, step: int) -> tuple[Document, list[EncodedBranch]]:
        document = self.documents[step % len(self.documents)]
        fmt, limit = self.extractor.format, self.config.format.max_value_tokens
        fits = [f for f in document.fields if len(fmt.encode(f).target) <= limit]
        if not fits:
            raise ValueError(f'No field of {document.id} fits max_value_tokens={limit}')
        chosen = self.field_rng.sample(fits, min(self.config.fields_per_document, len(fits)))
        return document, [json_format.encode_within(fmt, chosen, limit)]


def evaluate(extractor: Extractor, documents: Sequence[Document], output: Path | str, fields_per_document: int = 24,
             group: int = 12, max_value_tokens: int = 16384, comparator: Comparator = 'text',
             early_stop: bool = True) -> dict[str, Any]:
    """Decode the first ``fields_per_document`` fields of each document as JSON requests of ``group`` fields.

    The requests of a document share its prefix and decode in parallel, as
    KevFormat's branches do. Appends rows to ``predictions.jsonl`` and one
    timing record per document to ``documents.jsonl`` (resumable), then
    writes ``metrics.json``.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions, timings = output / 'predictions.jsonl', output / 'documents.jsonl'
    done = {r['document_id'] for r in read_jsonl(timings)} if timings.exists() else set()
    if predictions.exists():  # keep only fully written documents
        kept = [r for r in read_jsonl(predictions) if r['document_id'] in done]
        predictions.write_text(''.join(dumps_jsonl(r) for r in kept), encoding='utf-8')
    tokenizer = extractor.format.tokenizer
    cuda = extractor.device.type == 'cuda'
    for n, document in enumerate(documents, 1):
        if document.id in done:
            continue
        chunks = json_format.groups(document.fields[:fields_per_document], group)
        requests = [FieldRequest(json_format.schema_text(c), '', 'scalar') for c in chunks]
        if cuda:
            torch.cuda.synchronize(extractor.device)
        started = time.monotonic()
        results = decode_prefix(extractor, extractor.encode_prefix(document.pages), requests,
                                max_value_tokens=max_value_tokens, early_stop=early_stop)
        if cuda:
            torch.cuda.synchronize(extractor.device)
        seconds = time.monotonic() - started
        rows = [row for c, r in zip(chunks, results, strict=True)
                for row in json_format.prediction_rows(document.id, document.form, c, r.text, r.status, comparator)]
        tokens = [len(tokenizer(r.text, add_special_tokens=False)['input_ids']) + (r.stop == 'close') for r in results]
        record = {'document_id': document.id, 'seconds': round(seconds, 3), 'requests': len(requests),
                  'tokens': tokens, 'stops': [r.stop for r in results], 'correct': sum(r['correct'] for r in rows),
                  'fields': len(rows), 'texts': [r.text for r in results]}
        with predictions.open('a', encoding='utf-8') as handle:
            handle.write(''.join(dumps_jsonl(row) for row in rows))
        with timings.open('a', encoding='utf-8') as handle:
            handle.write(dumps_jsonl(record))
        shown = {k: v for k, v in record.items() if k != 'texts'}
        print(json.dumps({'document': n, 'of': len(documents), **shown}, ensure_ascii=False), flush=True)
    rows = read_jsonl(predictions)
    if {r['document_id'] for r in rows} != {d.id for d in documents}:
        raise ValueError(f'{predictions} does not cover exactly the requested documents')
    metrics = score(rows, comparator)
    write_json(output / 'metrics.json', metrics)
    return metrics
