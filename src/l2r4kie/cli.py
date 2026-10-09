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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, return its exit code."""
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == '__main__':
    sys.exit(main())
