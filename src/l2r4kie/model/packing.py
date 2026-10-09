"""Pack a shared prefix and isolated field branches into one training sequence.

Layout of a packed sequence (one document)::

    [ prefix ........ | branch 1 (prompt + target) | branch 2 | ... ]

Isolation has two parts, both needed for training to match decoding, where
every branch is decoded on its own copy of the prefix KV cache:

* **Block-causal attention** (:func:`block_mask`): a branch token attends to
  the prefix and to earlier tokens of its own branch, never to other branches.
* **Branch-local positions**: every branch's M-RoPE positions restart at the
  first position after the prefix, so a branch's representation does not
  depend on how many branches precede it or in which order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .format import EncodedBranch


class RopeModel(Protocol):
    """Anything with Qwen2-VL's ``get_rope_index`` (the inner ``Qwen2VLModel``)."""

    def get_rope_index(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor | None = ...,
                       video_grid_thw: torch.Tensor | None = ...,
                       attention_mask: torch.Tensor | None = ...) -> tuple[torch.Tensor, torch.Tensor]: ...


def prefix_positions(model: RopeModel, prefix: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """M-RoPE positions ``(3, 1, prefix_len)`` of the prefix, image grid included."""
    positions, _ = model.get_rope_index(prefix['input_ids'], prefix.get('image_grid_thw'),
                                        attention_mask=prefix['attention_mask'])
    return positions


def block_mask(shared: int, lengths: Sequence[int], dtype: torch.dtype = torch.float32,
               device: torch.device | str = 'cpu') -> torch.Tensor:
    """Additive attention mask ``(1, 1, total, total)`` for a packed sequence.

    Args:
        shared: Prefix length; the prefix is causal over itself.
        lengths: Branch lengths, in packing order.
        dtype: Mask dtype; blocked entries hold ``finfo(dtype).min``.
        device: Mask device.

    Returns:
        ``0`` where attention is allowed: within the prefix (causal), from a
        branch to the whole prefix, and within a branch (causal).
    """
    total = shared + sum(lengths)
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    allowed[:shared, :shared] = torch.ones((shared, shared), dtype=torch.bool, device=device).tril()
    start = shared
    for length in lengths:
        allowed[start:start + length, :shared] = True
        allowed[start:start + length, start:start + length] = torch.ones(
            (length, length), dtype=torch.bool, device=device).tril()
        start += length
    mask = torch.zeros((total, total), dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


@dataclass(frozen=True, slots=True)
class Packed:
    """Model inputs and per-branch indices of one packed document.

    All index tensors are positions in the packed sequence.

    Attributes:
        inputs: Keyword arguments for the inner ``Qwen2VLModel`` forward:
            ``input_ids``, 4-D ``attention_mask``, ``position_ids`` and the
            image tensors.
        predict_positions: Positions whose next-token logits are trained.
        targets: Target id for each of ``predict_positions``.
        key_positions: ``h_key`` position of each branch.
        decide_positions: ``h_decide`` position of each branch.
        value_positions: ``h_value`` position of each branch.
        field_ids: Field of each branch, aligned with the position tensors.
    """

    inputs: dict[str, Any]
    predict_positions: torch.Tensor
    targets: torch.Tensor
    key_positions: torch.Tensor
    decide_positions: torch.Tensor
    value_positions: torch.Tensor
    field_ids: list[str]


def pack(prefix: Mapping[str, torch.Tensor], positions: torch.Tensor, branches: Sequence[EncodedBranch],
         max_value_tokens: int, dtype: torch.dtype = torch.float32) -> Packed:
    """Concatenate the prefix and labelled branches into one training sequence.

    Args:
        prefix: Output of :meth:`~l2r4kie.model.format.KevFormat.encode_prefix`,
            on the target device.
        positions: Prefix M-RoPE positions from :func:`prefix_positions`.
        branches: Labelled branches (with targets).
        max_value_tokens: Longest allowed target, close marker included.
        dtype: Dtype of the attention mask (the model's compute dtype).

    Raises:
        ValueError: If a branch has no target or a target is too long. Callers
            filter fields first (``target`` length is known before packing).
    """
    device = prefix['input_ids'].device
    ids = prefix['input_ids'][0].tolist()
    shared = len(ids)
    base = int(positions.max()) + 1
    all_positions = [positions]
    lengths: list[int] = []
    predict: list[int] = []
    targets: list[int] = []
    keys: list[int] = []
    decides: list[int] = []
    values: list[int] = []
    for branch in branches:
        if not branch.target:
            raise ValueError(f'Branch {branch.field_id!r} has no target to train on')
        if len(branch.target) > max_value_tokens:
            raise ValueError(f'Target of {branch.field_id!r} has {len(branch.target)} > {max_value_tokens} tokens')
        start = len(ids)
        ids.extend(branch.prompt)
        ids.extend(branch.target)
        lengths.append(branch.length)
        all_positions.append(torch.arange(base, base + branch.length, device=device)[None, None].expand(3, 1, -1))
        # Teacher forcing: the token at position p predicts the token at p + 1,
        # from the last prompt token (value_open) up to the token before value_close.
        predict.extend(range(start + len(branch.prompt) - 1, start + branch.length - 1))
        targets.extend(branch.target)
        keys.append(start + branch.key_index)
        decides.append(start + branch.decide_index)
        values.append(start + branch.value_index)
    inputs = {k: v for k, v in prefix.items() if k not in ('input_ids', 'attention_mask')}
    inputs.update(input_ids=torch.tensor([ids], device=device),
                  attention_mask=block_mask(shared, lengths, dtype, device),
                  position_ids=torch.cat(all_positions, dim=-1), use_cache=False)

    def as_long(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    return Packed(inputs, as_long(predict), as_long(targets), as_long(keys), as_long(decides), as_long(values),
                  [b.field_id for b in branches])
