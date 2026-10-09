"""Shared pytest configuration.

Tests marked ``@pytest.mark.gpu`` are skipped when CUDA is unavailable, so the
default ``pytest`` run works on any machine.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``gpu``-marked tests when no CUDA device is visible."""
    gpu_items = [item for item in items if 'gpu' in item.keywords]
    if not gpu_items:
        return  # avoid importing torch for CPU-only selections
    import torch

    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason='CUDA is not available')
    for item in gpu_items:
        item.add_marker(skip)
