"""Training loss of the extractor: cross-entropy on value tokens and the close marker.

The old joint loss (BCE confidence head, pairwise ranking, synthetic
negatives) is gone (decision D1): confidence is learned afterwards from
decode traces (step 6), so the extractor is trained on generation only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from ..model.packing import Packed
from .config import LossWeighting


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Loss of one packed document and what it was computed on.

    Attributes:
        loss: Scalar to backpropagate.
        target_tokens: Number of trained positions (value tokens + closes).
        token_accuracy: Share of positions whose argmax is the target.
        close_accuracy: Share of branches whose close marker is predicted
            at the right place (the model knows where the value ends).
    """

    loss: torch.Tensor
    target_tokens: int
    token_accuracy: float
    close_accuracy: float


def branch_sizes(packed: Packed) -> torch.Tensor:
    """Number of target tokens of each branch (``value_close`` included).

    A branch predicts from ``value_open`` up to the token before
    ``value_close``, so its count is ``value_position - decide_position``.
    """
    return packed.value_positions - packed.decide_positions


def position_weights(packed: Packed, weighting: LossWeighting) -> torch.Tensor:
    """Weight of every trained position; the weights sum to 1."""
    sizes = branch_sizes(packed)
    if weighting == 'token':
        return torch.full((int(sizes.sum()),), 1 / float(sizes.sum()), device=sizes.device)
    # Positions are laid out branch by branch, in packing order.
    return torch.repeat_interleave(1 / (sizes.float() * len(sizes)), sizes)


def _chunk_loss(lm_head: torch.nn.Module, states: torch.Tensor, targets: torch.Tensor,
                weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted CE sum and correct-argmax mask of one chunk of positions (FP32 logits).

    ``lm_head`` is called as a module, not through its ``weight``: with
    trainable marker rows PEFT wraps the (tied) head to add the row deltas.
    """
    logits = lm_head(states).float()
    losses = F.cross_entropy(logits, targets, reduction='none')
    return (losses * weights).sum(), logits.argmax(-1) == targets


def value_loss(core: torch.nn.Module, lm_head: torch.nn.Module, packed: Packed,
               weighting: LossWeighting = 'token', chunk: int = 1024) -> LossOutput:
    """Teacher-forced cross-entropy of one packed document.

    The vocabulary projection runs in chunks of ``chunk`` positions under
    activation checkpointing: with long array targets a step can train ~10k
    positions, whose full FP32 logits (151,936 wide) would take ~6 GB.

    Args:
        core: Inner ``Qwen2VLModel`` (possibly with LoRA layers).
        lm_head: Output projection (bias-free, as in Qwen2-VL).
        packed: Output of :func:`~l2r4kie.model.packing.pack`.
        weighting: See :data:`~l2r4kie.train.config.LossWeighting`.
        chunk: Positions per chunk.
    """
    # Dense block masks can make the cuDNN/flash SDPA backward non-finite on this
    # torch/CUDA stack; the math backend accumulates in FP32 and is required here.
    with sdpa_kernel(SDPBackend.MATH):
        hidden = core(**packed.inputs).last_hidden_state[0]
    states = hidden[packed.predict_positions]
    weights = position_weights(packed, weighting)
    total = states.new_zeros((), dtype=torch.float32)
    hits = []
    for start in range(0, len(states), chunk):
        part = slice(start, start + chunk)
        loss, hit = checkpoint(_chunk_loss, lm_head, states[part], packed.targets[part], weights[part],
                               use_reentrant=False)
        total = total + loss
        hits.append(hit)
    correct = torch.cat(hits)
    # The close marker is the last target of each branch.
    closes = torch.cumsum(branch_sizes(packed), 0) - 1
    return LossOutput(total, len(states), float(correct.float().mean()), float(correct[closes].float().mean()))
