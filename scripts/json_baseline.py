"""Ablation: does KevFormat (one isolated branch per field) cost accuracy against plain JSON output?

The baseline (``l2r4kie.baselines.json_format``) trains the same model, LoRA,
documents, sampled fields, steps and loss, but the model writes all requested
fields as one JSON object in a single sequence. Evaluation decodes the first
24 fields of each document as requests of 12 (the training group size) over
the shared prefix, and scores the rows exactly like ``l2r4kie evaluate``.

Usage::

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 uv run python scripts/json_baseline.py train \\
        --config configs/extractor/kev_r4.yaml --set output=artifacts/json-r4 --set monitor.every=0
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 uv run python scripts/json_baseline.py evaluate \\
        --config configs/extractor/kev_r4.yaml --adapter artifacts/json-r4/adapter \\
        --documents-from artifacts/kev-r4/eval-dev/provenance.json --split dev --output artifacts/json-r4/eval-dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', type=Path, required=True)
    train.add_argument('--set', action='append', default=[])
    train.add_argument('--until', type=int, help='stop after N steps (smoke test)')
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--config', type=Path, required=True)
    evaluate.add_argument('--set', action='append', default=[])
    evaluate.add_argument('--adapter', type=Path, required=True)
    evaluate.add_argument('--split', default='dev')
    evaluate.add_argument('--documents-from', type=Path,
                          help='provenance.json of an earlier evaluation (its "documents" ids, in order)')
    evaluate.add_argument('--limit', type=int)
    evaluate.add_argument('--fields', type=int, default=24)
    evaluate.add_argument('--group', type=int, default=12)
    evaluate.add_argument('--max-value-tokens', type=int, default=16384)
    evaluate.add_argument('--no-early-stop', action='store_true')
    evaluate.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    from l2r4kie.baselines.json_trainer import JsonTrainer
    from l2r4kie.baselines.json_trainer import evaluate as evaluate_json
    from l2r4kie.data.selection import Selection
    from l2r4kie.model.extractor import Extractor
    from l2r4kie.pipelines.predict import evaluation_documents
    from l2r4kie.train.config import TrainConfig
    from l2r4kie.utils.config import load_config
    from l2r4kie.utils.io import read_json

    config = TrainConfig.from_dict(load_config(args.config, args.set))
    if args.command == 'train':
        print(json.dumps(JsonTrainer(config).run(args.until), ensure_ascii=False, indent=2))
        return 0
    s = config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    if args.documents_from:
        lookup = {d.id: d for d in selection.documents(args.split)}
        documents = [lookup[i] for i in read_json(args.documents_from)['documents']][:args.limit]
    else:
        documents = evaluation_documents(selection, args.split, None, args.limit)
    extractor = Extractor.load(config.extractor(), args.adapter)
    metrics = evaluate_json(extractor, documents, args.output, args.fields, args.group, args.max_value_tokens,
                            early_stop=not args.no_early_stop)
    keys = ('documents', 'fields', 'exact_match', 'scalar_exact_match', 'array_exact_match',
            'macro_form_exact_match', 'complete_document_rate', 'status')
    print(json.dumps({k: metrics.get(k) for k in keys}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
