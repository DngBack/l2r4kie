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
    from .model.format import DEFAULT_MAX_PIXELS

    show.add_argument('--max-pixels', type=int, default=DEFAULT_MAX_PIXELS,
                      help=f'page pixel budget of the image processor (default: {DEFAULT_MAX_PIXELS}, as r4)')
    show.set_defaults(handler=_show_input)

    lengths = commands.add_parser('token-stats', help='token lengths of prompts and targets of a split')
    _add_format_arguments(lengths)
    lengths.add_argument('--split', default='dev', choices=['train', 'dev', 'calibration', 'test'])
    lengths.add_argument('--limit', type=int, help='only the first N documents')
    lengths.set_defaults(handler=_token_stats)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, return its exit code."""
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == '__main__':
    sys.exit(main())
