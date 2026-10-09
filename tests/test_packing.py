"""Tests for ``l2r4kie.model.packing``: block mask, layout, and isolation on a tiny Qwen2-VL."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from l2r4kie.model.format import EncodedBranch
from l2r4kie.model.packing import Packed, block_mask, pack

# Fake marker ids inside the tiny vocabulary: key_open, key_close, value_open, value_close.
KO, KC, VO, VC = 90, 91, 92, 93


def tiny_model() -> torch.nn.Module:
    """Randomly initialised 2-layer Qwen2-VL text model (vocab 100) with M-RoPE."""
    config = Qwen2VLConfig(
        text_config={'vocab_size': 100, 'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 2,
                     'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'rope_scaling': {'type': 'mrope', 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 1, 'embed_dim': 32, 'hidden_size': 32, 'num_heads': 4, 'patch_size': 14,
                       'spatial_merge_size': 2, 'in_channels': 3})
    config._attn_implementation = 'sdpa'
    return Qwen2VLForConditionalGeneration(config).eval().model


def branch(field_id: str, key: list[int], value: list[int]) -> EncodedBranch:
    return EncodedBranch(field_id, (KO, *key, KC, VO), (*value, VC))


def prefix(length: int = 3) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ids = torch.arange(1, length + 1)[None]
    return ({'input_ids': ids, 'attention_mask': torch.ones_like(ids)},
            torch.arange(length)[None, None].expand(3, 1, -1))


def run(model: torch.nn.Module, packed: Packed) -> torch.Tensor:
    with torch.no_grad():
        return model(**packed.inputs).last_hidden_state[0]


# -------------------------------------------------------------------- block mask

def test_no_cross_branch_or_future_attention() -> None:
    mask = block_mask(3, [4, 5])[0, 0]
    assert (mask[3:7, 7:] < 0).all()   # branch 1 cannot see branch 2
    assert (mask[7:, 3:7] < 0).all()   # branch 2 cannot see branch 1
    assert (mask[:3, 3:] < 0).all()    # prefix cannot see branches
    assert (mask[7:, :3] == 0).all()   # branches see the whole prefix
    assert mask[7, 8] < 0              # causal inside a branch


# ------------------------------------------------------------------------ layout

def test_pack_layout_targets_and_signals() -> None:
    a, b = branch('/a', [10, 11], [20, 21]), branch('/b', [12], [])
    packed = pack(*prefix(), [a, b], max_value_tokens=8)
    ids = packed.inputs['input_ids'][0].tolist()
    assert ids == [1, 2, 3, KO, 10, 11, KC, VO, 20, 21, VC, KO, 12, KC, VO, VC]
    assert [ids[p] for p in packed.key_positions] == [KC, KC]
    assert [ids[p] for p in packed.decide_positions] == [VO, VO]
    assert [ids[p] for p in packed.value_positions] == [VC, VC]
    # Each predicted position is the token before its target (teacher forcing).
    assert packed.targets.tolist() == [20, 21, VC, VC]
    assert [ids[p + 1] for p in packed.predict_positions] == packed.targets.tolist()
    assert packed.predict_positions.tolist()[0] == packed.decide_positions.tolist()[0]
    assert packed.field_ids == ['/a', '/b']


def test_positions_restart_after_the_prefix_for_every_branch() -> None:
    a, b = branch('/a', [10, 11], [20]), branch('/b', [12], [21, 22])
    positions = pack(*prefix(), [a, b], max_value_tokens=8).inputs['position_ids']
    assert positions.shape == (3, 1, 3 + a.length + b.length)
    assert positions[0, 0].tolist() == [0, 1, 2, *range(3, 3 + a.length), *range(3, 3 + b.length)]
    assert (positions[0] == positions[1]).all() and (positions[0] == positions[2]).all()


def test_pack_rejects_untrainable_branches() -> None:
    with pytest.raises(ValueError, match='no target'):
        pack(*prefix(), [EncodedBranch('/a', (KO, KC, VO))], max_value_tokens=8)
    with pytest.raises(ValueError, match='> 2 tokens'):
        pack(*prefix(), [branch('/a', [10], [20, 21])], max_value_tokens=2)


# ------------------------------------------------- isolation on a real transformer

def test_changing_one_branch_does_not_leak_into_another() -> None:
    torch.manual_seed(2)
    model = tiny_model()
    b = branch('/b', [12, 13], [30, 31])
    first = pack(*prefix(), [branch('/a', [10], [20, 21]), b], max_value_tokens=8)
    second = pack(*prefix(), [branch('/a', [10], [40, 41]), b], max_value_tokens=8)
    h1, h2 = run(model, first), run(model, second)
    start = int(first.key_positions[1]) - 3  # first token of branch b
    torch.testing.assert_close(h1[start:], h2[start:], atol=0, rtol=0)
    assert not torch.equal(h1[:start], h2[:start])  # branch a itself did change


def test_branch_order_does_not_change_signals() -> None:
    torch.manual_seed(3)
    model = tiny_model()
    a, b = branch('/a', [10, 11], [20, 21, 22]), branch('/b', [12], [30])
    ab = pack(*prefix(), [a, b], max_value_tokens=8)
    ba = pack(*prefix(), [b, a], max_value_tokens=8)
    h_ab, h_ba = run(model, ab), run(model, ba)
    for name in ('key_positions', 'decide_positions', 'value_positions'):
        torch.testing.assert_close(h_ab[getattr(ab, name)], h_ba[getattr(ba, name)].flip(0), atol=1e-6, rtol=1e-5)
