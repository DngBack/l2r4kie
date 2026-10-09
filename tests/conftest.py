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


@pytest.fixture(scope='session')
def qwen_tokenizer():  # noqa: ANN201 - transformers type
    """The real Qwen2-VL tokenizer from the local HF cache; skips if it is not cached."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', local_files_only=True)
    except OSError:
        pytest.skip('Qwen2-VL tokenizer not in the local Hugging Face cache')


@pytest.fixture(scope='session')
def qwen_processor():  # noqa: ANN201 - transformers type
    """The real Qwen2-VL processor with a small page budget; skips if it is not cached."""
    from l2r4kie.model.format import load_processor

    try:
        return load_processor('Qwen/Qwen2-VL-2B-Instruct', max_pixels=56 * 56 * 4, local_files_only=True)
    except OSError:
        pytest.skip('Qwen2-VL processor not in the local Hugging Face cache')
