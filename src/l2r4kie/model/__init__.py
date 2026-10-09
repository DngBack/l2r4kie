"""Qwen2-VL input format, packing and (from step 3) decoding.

* :mod:`.markers`: reused special tokens delimiting keys and values.
* :mod:`.format`: KevFormat: prefix, branch prompts, targets, signal positions.
* :mod:`.packing`: block-causal packing of isolated branches for training.
"""
