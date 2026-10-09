"""Global random seeding.

Data order and split assignment do not depend on these generators: they use
explicit ``random.Random(seed)`` instances or :func:`~l2r4kie.utils.fingerprint.hash_bucket`.
Seeding the globals covers what remains: LoRA/head initialisation, dropout and
any library code that draws from the default generators.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python's, NumPy's and PyTorch's (CPU and all CUDA) global RNGs.

    Args:
        seed: Seed applied to every generator.
        deterministic: Also ask PyTorch for deterministic kernels. Off by
            default: it is slower and some ops (used by SDPA backends) raise
            when no deterministic implementation exists. Use it to debug
            reproducibility, not for normal runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also seeds every CUDA device's default generator.
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
