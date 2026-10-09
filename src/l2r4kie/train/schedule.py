"""Learning-rate schedule (unchanged from the old ``training.lr_multiplier``)."""

from __future__ import annotations

import math


def lr_multiplier(update: int, total: int, warmup: int = 0, cosine: bool = False) -> float:
    """Factor applied to the peak learning rate before optimizer update ``update``.

    Args:
        update: Index of the update about to be applied (0-based).
        total: Number of updates of the run.
        warmup: Linear warmup length: update ``u < warmup`` gets ``(u + 1) / warmup``.
        cosine: After warmup, decay with a half cosine to 0 at ``total``;
            otherwise stay at 1.
    """
    if update < warmup:
        return (update + 1) / warmup
    if not cosine:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (update - warmup) / max(1, total - warmup))))
