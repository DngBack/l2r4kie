"""Tiny random Qwen2-VL shared by the model tests (CPU, FP32, seconds to run).

It keeps Qwen2-VL's full vocabulary (151,936 rows), so the real tokenizer,
processor and :class:`~l2r4kie.model.format.KevFormat` work unchanged.
"""

from __future__ import annotations

import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

VOCAB = 151_936
HIDDEN = 32


def tiny_model(seed: int = 0) -> Qwen2VLForConditionalGeneration:
    """Random 2-layer Qwen2-VL with a 1-block vision tower, in eval mode."""
    torch.manual_seed(seed)
    config = Qwen2VLConfig(
        text_config={'vocab_size': VOCAB, 'hidden_size': HIDDEN, 'intermediate_size': 64, 'num_hidden_layers': 2,
                     'num_attention_heads': 4, 'num_key_value_heads': 2,
                     'rope_scaling': {'type': 'mrope', 'mrope_section': [1, 1, 2]}},
        vision_config={'depth': 1, 'embed_dim': 32, 'hidden_size': HIDDEN, 'num_heads': 4, 'patch_size': 14,
                       'spatial_merge_size': 2, 'in_channels': 3})
    config._attn_implementation = 'sdpa'
    return Qwen2VLForConditionalGeneration(config).eval()
