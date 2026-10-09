"""Qwen2-VL input format, packing, loading and decoding.

* :mod:`.markers`: reused special tokens delimiting keys and values.
* :mod:`.format`: KevFormat: prefix, branch prompts, targets, signal positions.
* :mod:`.packing`: block-causal packing of isolated branches for training.
* :mod:`.extractor`: load the model (optionally with a LoRA adapter) and processor.
* :mod:`.decode`: greedy KV-cache decoding of isolated branches, with signals.
"""
