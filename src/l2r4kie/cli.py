"""Command-line entry point: ``l2r4kie <command> ...``.

Each refactor step adds its commands here (``prepare`` in step 1, ``infer``
in step 3, ...). Command handlers import their heavy dependencies (torch,
transformers) lazily, so ``l2r4kie --help`` and the data commands stay fast.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

Handler = Callable[[argparse.Namespace], int]

DEFAULT_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'


def _fingerprint(args: argparse.Namespace) -> int:
    """Print the fingerprint of each checkpoint directory given."""
    from .utils.fingerprint import checkpoint_fingerprint

    for checkpoint in args.checkpoints:
        print(f'{checkpoint_fingerprint(checkpoint)}  {checkpoint}')
    return 0


def _prepare(args: argparse.Namespace) -> int:
    """Build the prepared splits and print the report (without the per-form counts)."""
    import json

    from .data.prepare import prepare

    report = prepare(args.data, args.output, args.seed, args.workers, not args.legacy_array_descriptions)
    print(json.dumps({k: v for k, v in report.items() if k != 'forms'}, ensure_ascii=False, indent=2))
    return 0


def _data_stats(args: argparse.Namespace) -> int:
    """Print split sizes, value types and string features of a prepared directory."""
    import json

    from .data.stats import prepared_stats

    print(json.dumps(prepared_stats(args.prepared), ensure_ascii=False, indent=2))
    return 0


def _plan_cohorts(args: argparse.Namespace) -> int:
    """Plan cohorts from a config and write the frozen plan; never overwrite one."""
    import json
    from pathlib import Path

    from .data.cohorts import plan_from_config
    from .utils.config import load_config

    output = Path(args.output)
    if output.exists():
        # A plan is pre-declared once; replacing it would let cohorts be re-drawn after looking at results.
        print(f'error: {output} already exists; cohort plans are never overwritten', file=sys.stderr)
        return 1
    plan = plan_from_config(load_config(args.config, args.set))
    plan.save(output)
    print(json.dumps({'output': str(output), 'cohorts': {k: len(v) for k, v in plan.cohorts.items()},
                      'forms': len(plan.forms), 'excluded_documents': len(plan.excluded_documents),
                      'source_fingerprint': plan.source_fingerprint}, indent=2))
    return 0


def _show_input(args: argparse.Namespace) -> int:
    """Print the prefix and branches of one document as token text."""
    from .model.format import KevFormat, load_processor
    from .pipelines.inspect import show_input

    processor = load_processor(args.model, args.max_pixels)
    fmt = KevFormat(processor.tokenizer, args.close)
    print(show_input(processor, fmt, args.prepared, args.doc, args.field, args.fields, args.max_value_tokens))
    return 0


def _token_stats(args: argparse.Namespace) -> int:
    """Print prompt/target token lengths of a split per field kind."""
    import json

    from transformers import AutoTokenizer

    from .data.selection import Selection
    from .model.format import KevFormat
    from .pipelines.inspect import token_stats

    fmt = KevFormat(AutoTokenizer.from_pretrained(args.model), args.close)
    stats = token_stats(fmt, Selection(Path(args.prepared)), args.split, args.max_value_tokens, args.limit)
    print(json.dumps(stats, indent=2))
    return 0


def _infer(args: argparse.Namespace) -> int:
    """Run one request through the model and write the response JSON."""
    import json

    from .model.extractor import Extractor, ExtractorConfig
    from .pipelines.infer import infer_file

    bundle = None
    if args.confidence:
        from .confidence.bundle import ConfidenceBundle
        from .pipelines.predict import adapter_fingerprint

        bundle = ConfidenceBundle.load(args.confidence)
        # Checked before loading the model: a head of another extractor would give meaningless scores.
        bundle.check_source(adapter_fingerprint(args.adapter) or 'zero-shot base model')
    config = ExtractorConfig(model=args.model, device=args.device, precision=args.precision,
                             max_pixels=args.max_pixels, max_value_tokens=args.max_value_tokens,
                             max_branches=args.max_branches, close=args.close)
    response = infer_file(Extractor.load(config, args.adapter), args.request, args.output, bundle)
    counts: dict[str, int] = {}
    for status in response['status'].values():
        counts[status] = counts.get(status, 0) + 1
    summary = {'output': str(args.output), 'fields': len(response['status']), 'status': counts}
    if bundle is not None:
        summary['review'] = len(response['review'])
    print(json.dumps(summary))
    return 0


def _train(args: argparse.Namespace) -> int:
    """Train (or resume) an extractor adapter from a run config."""
    import json

    from .train.config import TrainConfig
    from .train.trainer import train
    from .utils.config import load_config

    summary = train(TrainConfig.from_dict(load_config(args.config, args.set)))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Decode labelled documents with an adapter and write predictions and metrics."""
    import json
    from pathlib import Path

    from .data.selection import Selection
    from .model.extractor import Extractor
    from .pipelines.predict import adapter_fingerprint, claim_output, evaluation_documents, predict, run_record
    from .train.config import TrainConfig
    from .utils.config import load_config

    config = TrainConfig.from_dict(load_config(args.config, args.set))
    s = config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    documents = evaluation_documents(selection, args.split, args.documents_file, args.limit)
    provenance = {'model': config.model, 'adapter': str(args.adapter) if args.adapter else None,
                  'adapter_fingerprint': adapter_fingerprint(args.adapter), 'split': args.split,
                  'documents_file': str(args.documents_file) if args.documents_file else None}
    # Refuse a mismatching output directory before spending time (and GPU memory) on the model.
    claim_output(args.output, run_record(config.extractor(), documents, args.fields, args.max_value_tokens,
                                         provenance))
    extractor = Extractor.load(config.extractor(), args.adapter)
    metrics = predict(extractor, documents, args.output, args.fields, args.max_value_tokens, provenance,
                      args.comparator)
    print(json.dumps(_headline(metrics), ensure_ascii=False, indent=2))
    return 0


def _headline(metrics: dict) -> dict:
    """The gate-relevant numbers of a metrics dict, for printing."""
    keys = ('comparator', 'documents', 'fields', 'exact_match', 'scalar_exact_match', 'array_exact_match',
            'macro_form_exact_match', 'coordinate_rate', 'truncated_rate', 'status', 'errors')
    return {k: metrics[k] for k in keys}


def _evaluate_file(args: argparse.Namespace) -> int:
    """Re-score an existing predictions file (new or old format) with a chosen comparator."""
    import json

    from .eval.extraction import score
    from .utils.io import read_jsonl, write_json

    metrics = score(read_jsonl(args.predictions), args.comparator)
    if args.output:
        write_json(args.output, metrics)
    print(json.dumps(metrics if args.full else _headline(metrics), ensure_ascii=False, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Paired document-bootstrap EM difference between two prediction files."""
    import json

    from .eval.compare import paired_delta
    from .utils.io import read_jsonl

    result = paired_delta(read_jsonl(args.b), read_jsonl(args.a), args.comparator, args.draws,
                          kinds=args.kind or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _confidence_config(args: argparse.Namespace):  # noqa: ANN202 (lazy import of the return type)
    """The :class:`~l2r4kie.confidence.config.ConfidenceConfig` of ``--config`` with ``--set`` overrides."""
    from .confidence.config import ConfidenceConfig
    from .utils.config import load_config

    return ConfidenceConfig.from_dict(load_config(args.config, args.set))


def _cache_traces(args: argparse.Namespace) -> int:
    """Decode confidence cohorts once and cache their trace features."""
    import dataclasses

    from .data.selection import Selection
    from .model.extractor import Extractor
    from .pipelines.predict import adapter_fingerprint
    from .pipelines.trace_cache import cache_traces, resolve_cohorts
    from .train.config import TrainConfig
    from .utils.config import load_config

    config = _confidence_config(args)
    extractor_config = TrainConfig.from_dict(load_config(config.extractor))
    s = extractor_config.selection
    selection = Selection(Path(s.prepared), s.seed, s.holdout_percent, s.forms, s.balanced_forms)
    # Resolved and checked for training documents before the model is loaded.
    cohorts = resolve_cohorts(config.cohort_plan, selection, args.split, extractor_config.train_documents)
    fingerprint = adapter_fingerprint(config.adapter)
    extractor = Extractor.load(dataclasses.replace(extractor_config.extractor(), device=config.device),
                               config.adapter)
    for split, documents in cohorts.items():
        cache_traces(extractor, documents, config.cache, split, config.fields_per_document,
                     config.max_value_tokens, config.layers, config.max_trace_tokens,
                     {'adapter': config.adapter, 'source_fingerprint': fingerprint,
                      'cohort_plan': config.cohort_plan, 'extractor_config': config.extractor},
                     config.part_size)
    return 0


def _select_heads(args: argparse.Namespace) -> int:
    """Train the head grid on the train cohort and select on dev."""
    from .pipelines.head_selection import select_heads
    from .pipelines.predict import adapter_fingerprint

    config = _confidence_config(args)
    select_heads(config, adapter_fingerprint(config.adapter) if Path(config.adapter).exists() else None)
    return 0


def _finalize(args: argparse.Namespace) -> int:
    """Calibrate and threshold the selected heads, then freeze them (or replay an old selection)."""
    if args.legacy_selection:
        from .pipelines.finalize import finalize_legacy

        if not args.cache:
            print('error: --legacy-selection needs --cache (the old cache directory)', file=sys.stderr)
            return 2
        finalize_legacy(args.legacy_selection, args.cache, args.replicates or 2000, args.device)
        return 0
    if not args.config:
        print('error: finalize needs --config or --legacy-selection', file=sys.stderr)
        return 2
    from .pipelines.finalize import finalize

    config = _confidence_config(args)
    finalize(config.output, args.cache or config.cache, config.target_error_recall,
             args.replicates or config.risk_replicates, args.dry_run, config.layers, config.max_trace_tokens,
             args.device)
    return 0


def _audit(args: argparse.Namespace) -> int:
    """Measure the frozen heads once on the audit cohort."""
    from .pipelines.audit import audit

    config = _confidence_config(args)
    audit(config.output, config.cache, config.cohort_plan, args.baseline_predictions, args.device)
    return 0


def _add_confidence_arguments(parser: argparse.ArgumentParser, required: bool = True) -> None:
    """``--config`` and ``--set`` of the confidence commands."""
    parser.add_argument('--config', required=required, help='confidence config, e.g. configs/confidence/kev.yaml')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a config value, e.g. device=cuda:1 (repeatable)')


def _add_format_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by commands that encode inputs with KevFormat."""
    parser.add_argument('--prepared', required=True, help='directory written by prepare')
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'model (tokenizer) name (default: {DEFAULT_MODEL})')
    parser.add_argument('--close', choices=['box_end', 'im_end'], default='box_end',
                        help='token closing a value (default: box_end)')
    from .model.format import DEFAULT_MAX_VALUE_TOKENS

    parser.add_argument('--max-value-tokens', type=int, default=DEFAULT_MAX_VALUE_TOKENS,
                        help=f'longest trainable target, close token included (default: {DEFAULT_MAX_VALUE_TOKENS})')


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-parser per command.

    Every sub-parser stores its handler in ``args.handler``.
    """
    parser = argparse.ArgumentParser(
        prog='l2r4kie',
        description='Learning to Reject for KIE: branch-isolated extraction with a learned review policy.',
    )
    commands = parser.add_subparsers(dest='command', required=True, metavar='<command>')
    from .model.format import DEFAULT_MAX_PIXELS, DEFAULT_MAX_VALUE_TOKENS

    fingerprint = commands.add_parser(
        'fingerprint', help='print the content fingerprint of extractor checkpoints',
        description='Hash adapter/ and the root config/head files, as stored in cache and policy provenance.',
    )
    fingerprint.add_argument('checkpoints', nargs='+', metavar='CHECKPOINT', help='checkpoint directory')
    fingerprint.set_defaults(handler=_fingerprint)

    prepare = commands.add_parser(
        'prepare', help='split the raw dataset into deduplicated train/dev/calibration/test',
        description='Hash pages, group duplicates, assign splits by group hash and flatten labels into fields.',
    )
    prepare.add_argument('--data', required=True, help='raw dataset directory (read only)')
    prepare.add_argument('--output', required=True, help='prepared directory (outside --data)')
    prepare.add_argument('--seed', type=int, default=42, help='split seed (default: 42)')
    prepare.add_argument('--workers', type=int, default=8, help='page hashing threads (default: 8)')
    prepare.add_argument('--legacy-array-descriptions', action='store_true',
                         help='describe array fields by their path, as the old repository did (byte-identical output)')
    prepare.set_defaults(handler=_prepare)

    stats = commands.add_parser('data-stats', help='summarise a prepared directory')
    stats.add_argument('--prepared', required=True, help='directory written by prepare')
    stats.set_defaults(handler=_data_stats)

    cohorts = commands.add_parser(
        'plan-cohorts', help='pre-declare confidence-head cohorts of unseen documents',
        description='Draw form-balanced train/dev/calibration/risk_validation/audit cohorts, '
                    'excluding documents listed in the config\'s exclusion files.',
    )
    cohorts.add_argument('--config', required=True, help='cohort config, e.g. configs/cohorts/r4_winner.yaml')
    cohorts.add_argument('--output', required=True, help='cohort_plan.json to create (must not exist)')
    cohorts.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                         help='override a config value, e.g. cohorts.seed=108 (repeatable)')
    cohorts.set_defaults(handler=_plan_cohorts)

    show = commands.add_parser(
        'show-input', help='print the tokens of one document as the model sees them',
        description='Encode the shared prefix (with images) and some field branches, and show where '
                    'the h_key / h_decide / h_value signals sit in the packed sequence.',
    )
    _add_format_arguments(show)
    show.add_argument('--doc', required=True, help='document id, e.g. lift-tsr-a04__sample_000102')
    show.add_argument('--fields', type=int, default=3, help='show the first N fields (default: 3)')
    show.add_argument('--field', action='append', default=[], metavar='FIELD_ID',
                      help='show this field instead (repeatable)')
    show.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
                      help=f'page pixel budget of the image processor (default: {DEFAULT_MAX_PIXELS}, as r4)')
    show.set_defaults(handler=_show_input)

    lengths = commands.add_parser('token-stats', help='token lengths of prompts and targets of a split')
    _add_format_arguments(lengths)
    lengths.add_argument('--split', default='dev', choices=['train', 'dev', 'calibration', 'test'])
    lengths.add_argument('--limit', type=int, help='only the first N documents')
    lengths.set_defaults(handler=_token_stats)

    run = commands.add_parser(
        'infer', help='extract the fields of one document request',
        description='Encode the pages once, decode every field in its own isolated branch, and write '
                    '{"result", "confidence", "status", "calibrated"} JSON.',
    )
    run.add_argument('--model', default=DEFAULT_MODEL, help=f'base model (default: {DEFAULT_MODEL})')
    run.add_argument('--adapter', help='LoRA adapter directory, or a checkpoint containing adapter/ '
                                       '(default: zero-shot base model)')
    run.add_argument('--request', required=True, help='request JSON {"pages": [...], "fields": [...]}')
    run.add_argument('--output', required=True, help='response JSON to write')
    run.add_argument('--device', default='cuda:0', help='torch device (default: cuda:0)')
    run.add_argument('--precision', choices=['bfloat16', 'float32'], default='bfloat16',
                     help='bfloat16 (default) or float32 (bit-stable across batch layouts, slower)')
    run.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
                     help=f'page pixel budget (default: {DEFAULT_MAX_PIXELS})')
    run.add_argument('--max-value-tokens', type=int, default=DEFAULT_MAX_VALUE_TOKENS,
                     help=f'decode budget per field, close token included (default: {DEFAULT_MAX_VALUE_TOKENS})')
    run.add_argument('--max-branches', type=int, default=64,
                     help='fields decoded together; bounds KV-cache memory (default: 64)')
    run.add_argument('--close', choices=['box_end', 'im_end'], default='box_end',
                     help='token closing a value (default: box_end)')
    run.add_argument('--confidence', metavar='BUNDLE',
                     help='frozen confidence bundle (a heads/<family>/ folder after finalize): fills '
                          'confidence and adds the review queue; must belong to --adapter')
    run.set_defaults(handler=_infer)

    fit = commands.add_parser(
        'train', help='train a LoRA extractor adapter (resumes from the latest snapshot)',
        description='Branch-packed teacher forcing with cross-entropy on value tokens and the close marker. '
                    'Writes adapter/, train.jsonl, monitor.jsonl and snapshots/ to the config\'s output.',
    )
    fit.add_argument('--config', required=True, help='run config, e.g. configs/extractor/kev_smoke.yaml')
    fit.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                     help='override a config value, e.g. steps=50 or format.close=im_end (repeatable)')
    fit.set_defaults(handler=_train)

    evaluate = commands.add_parser(
        'evaluate', help='decode labelled documents and score them',
        description='Writes provenance.json, predictions.jsonl and metrics.json to --output; '
                    'an interrupted run resumes when rerun with the same arguments.',
    )
    evaluate.add_argument('--config', required=True, help='run config (selection, model, format), e.g. kev_r4.yaml')
    evaluate.add_argument('--adapter', help='trained run or adapter directory (default: zero-shot base model)')
    evaluate.add_argument('--split', default='dev', choices=['train_reserve', 'dev', 'calibration', 'test'])
    evaluate.add_argument('--documents-file', help='JSON {"documents": [ids]} fixing the documents and their '
                                                   'order, e.g. the r4 dev cohort_plan.json')
    evaluate.add_argument('--limit', type=int, help='first N documents (r4 test: 116)')
    evaluate.add_argument('--fields', type=int, help='first N fields per document (r4: 24; default: all)')
    evaluate.add_argument('--max-value-tokens', type=int, help='decode budget (default: format.max_value_tokens)')
    evaluate.add_argument('--comparator', choices=['text', 'json'], default='text')
    evaluate.add_argument('--output', required=True, help='output directory')
    evaluate.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                          help='override a config value, e.g. device=cuda:1 (repeatable)')
    evaluate.set_defaults(handler=_evaluate)

    rescore = commands.add_parser('evaluate-file', help='re-score a predictions.jsonl (also old r4 files)')
    rescore.add_argument('--predictions', required=True)
    rescore.add_argument('--comparator', choices=['text', 'json'], default='text')
    rescore.add_argument('--output', help='write the full metrics JSON here')
    rescore.add_argument('--full', action='store_true', help='print all metrics, not only the headline')
    rescore.set_defaults(handler=_evaluate_file)

    compare = commands.add_parser('compare', help='paired document-bootstrap EM difference B - A')
    compare.add_argument('--a', required=True, help='baseline predictions.jsonl')
    compare.add_argument('--b', required=True, help='candidate predictions.jsonl')
    compare.add_argument('--comparator', choices=['text', 'json'], default='text')
    compare.add_argument('--kind', action='append', choices=['scalar', 'array'],
                         help='only fields of this kind (repeatable; gate G1 uses scalar)')
    compare.add_argument('--draws', type=int, default=10_000)
    compare.set_defaults(handler=_compare)

    traces = commands.add_parser(
        'cache-traces', help='decode confidence cohorts once and cache their features',
        description='Resumable; writes <cache>/<split>.pt with every field\'s value, correctness, marker states '
                    '(key/decide/value, optional intermediate layers), token states and statistics.',
    )
    _add_confidence_arguments(traces)
    traces.add_argument('--split', nargs='+', required=True,
                        choices=['train', 'dev', 'calibration', 'risk_validation', 'audit'],
                        help='cohorts to decode (decode audit only after finalize, to keep it unseen)')
    traces.set_defaults(handler=_cache_traces)

    heads = commands.add_parser(
        'select-heads', help='train the confidence-head grid and select on dev',
        description='Reads only the train and dev caches. Writes selection.json, trials.json and '
                    'heads/<family>/ to the config\'s output; refuses an existing selection.',
    )
    _add_confidence_arguments(heads)
    heads.set_defaults(handler=_select_heads)

    final = commands.add_parser(
        'finalize', help='calibrate and threshold the selected heads, then freeze them',
        description='Calibration on the calibration cohort, review threshold on risk_validation; writes each '
                    'family\'s confidence bundle and frozen.json. --legacy-selection replays an old r4 '
                    'selection on its cache and writes nothing (port check: attention threshold 0.9597).',
    )
    _add_confidence_arguments(final, required=False)
    final.add_argument('--dry-run', action='store_true', help='compute and print, write nothing')
    final.add_argument('--legacy-selection', help='old selected/ directory (frozen_selection.json)')
    final.add_argument('--cache', help='trace cache directory (default: the config\'s)')
    final.add_argument('--replicates', type=int, help='bootstrap draws of the risk lower bound (default: 2000)')
    final.add_argument('--device', default='cpu', help='where heads run (default: cpu)')
    final.set_defaults(handler=_finalize)

    check = commands.add_parser(
        'audit', help='measure the frozen heads once on the audit cohort',
        description='Requires frozen.json; refuses a second audit. Writes audit_report.json (intervals, '
                    'plan subsets such as fresh_audit/seen_audit, paired comparisons) and '
                    'heads/<family>/audit_predictions.jsonl.',
    )
    _add_confidence_arguments(check)
    check.add_argument('--baseline-predictions',
                       help='another system\'s audit_predictions.jsonl (e.g. r4 attention) for a paired comparison')
    check.add_argument('--device', default='cpu', help='where heads run (default: cpu)')
    check.set_defaults(handler=_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, return its exit code."""
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == '__main__':
    sys.exit(main())
