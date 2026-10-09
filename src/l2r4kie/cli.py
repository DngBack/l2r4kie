"""Command-line entry point: ``l2r4kie <command> ...``.

Each refactor step adds its commands here (``prepare`` in step 1, ``infer``
in step 3, ...). Command handlers import their heavy dependencies (torch,
transformers) lazily, so ``l2r4kie --help`` and the data commands stay fast.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

Handler = Callable[[argparse.Namespace], int]


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

    report = prepare(args.data, args.output, args.seed, args.workers)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, return its exit code."""
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == '__main__':
    sys.exit(main())
