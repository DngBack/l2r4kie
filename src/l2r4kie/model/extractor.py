"""Load Qwen2-VL (optionally with a LoRA adapter) and encode documents.

The :class:`Extractor` owns the model, processor and input format, and
nothing else: no confidence head or calibration (those live in
:mod:`l2r4kie.confidence` from step 6), so a checkpoint's decoding behaviour
depends on its adapter alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal

import torch

from .format import DEFAULT_MAX_PIXELS, DEFAULT_MAX_VALUE_TOKENS, KevFormat, load_processor
from .markers import CloseToken

DEFAULT_MODEL = 'Qwen/Qwen2-VL-2B-Instruct'
Precision = Literal['bfloat16', 'float32']


@dataclass(frozen=True, slots=True)
class ExtractorConfig:
    """How to load and run the extractor.

    Attributes:
        model: Hugging Face model name or local path.
        device: Torch device, e.g. ``'cuda:0'`` (no longer hardcoded to ``cuda:1``).
        precision: ``'bfloat16'`` (CUDA only; CPU always runs float32) or
            ``'float32'`` (bit-stable values across batch layouts, slower).
        max_pixels: Page pixel budget of the image processor.
        max_value_tokens: Decode budget per field, close marker included.
        max_branches: Fields decoded together; each holds a copy of the prefix
            KV cache, so this bounds memory. More fields run in further chunks
            that reuse the single prefix encode.
        close: Value close token (``'box_end'`` or the ``'im_end'`` fallback).
    """

    model: str = DEFAULT_MODEL
    device: str = 'cuda:0'
    precision: Precision = 'bfloat16'
    max_pixels: int = DEFAULT_MAX_PIXELS
    max_value_tokens: int = DEFAULT_MAX_VALUE_TOKENS
    max_branches: int = 64
    close: CloseToken = 'box_end'

    def __post_init__(self) -> None:
        if self.precision not in ('bfloat16', 'float32'):
            raise ValueError(f'precision must be bfloat16 or float32, got {self.precision!r}')
        if self.max_branches < 1 or self.max_value_tokens < 1:
            raise ValueError('max_branches and max_value_tokens must be positive')

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ExtractorConfig:
        """Read the extractor keys of a run config, ignoring all other keys."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in names})

    @property
    def dtype(self) -> torch.dtype:
        """Compute dtype: bfloat16 only when requested and on CUDA."""
        return torch.bfloat16 if self.precision == 'bfloat16' and self.device.startswith('cuda') else torch.float32


class Extractor:
    """A Qwen2-VL model with its processor and input format.

    Args:
        model: ``Qwen2VLForConditionalGeneration``, or a ``PeftModel`` wrapping one.
        processor: Its processor; may be ``None`` if :meth:`encode_prefix`
            is never called (e.g. tests that build prefixes by hand).
        fmt: Input format bound to the model's tokenizer.
        config: Decoding limits (``max_value_tokens``, ``max_branches``).
    """

    def __init__(self, model: torch.nn.Module, processor: Any, fmt: KevFormat, config: ExtractorConfig) -> None:
        self.model = model
        self.processor = processor
        self.format = fmt
        self.config = config

    @classmethod
    def load(cls, config: ExtractorConfig, adapter: str | Path | None = None) -> Extractor:
        """Load the base model, optionally with a trained LoRA adapter, for inference.

        Args:
            config: Model name, device, precision and limits.
            adapter: Adapter directory, or a checkpoint directory containing
                ``adapter/`` (the old repository's layout). ``None`` runs the
                base model zero-shot.
        """
        from transformers import Qwen2VLForConditionalGeneration

        processor = load_processor(config.model, config.max_pixels)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model, dtype=config.dtype, attn_implementation='sdpa')
        if adapter is not None:
            from peft import PeftModel

            path = Path(adapter)
            model = PeftModel.from_pretrained(model, str(path / 'adapter' if (path / 'adapter').is_dir() else path))
        model.to(config.device).eval()
        return cls(model, processor, KevFormat(processor.tokenizer, config.close), config)

    @property
    def base(self) -> torch.nn.Module:
        """The ``Qwen2VLForConditionalGeneration`` under any PEFT wrapper."""
        get_base = getattr(self.model, 'get_base_model', None)
        return get_base() if get_base is not None else self.model

    @property
    def core(self) -> torch.nn.Module:
        """Inner ``Qwen2VLModel`` (vision tower + language model, no LM head)."""
        return self.base.model

    @property
    def lm_head(self) -> torch.nn.Module:
        """Output projection to vocabulary logits."""
        return self.base.lm_head

    @property
    def vocab_size(self) -> int:
        """Number of logits (embedding rows), larger than the tokenizer."""
        return int(self.lm_head.out_features)

    @property
    def device(self) -> torch.device:
        """Device of the model parameters."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model parameters."""
        return next(self.model.parameters()).dtype

    def encode_prefix(self, pages: Sequence[str | Path]) -> dict[str, torch.Tensor]:
        """Encode the shared prefix of a document and move it to the model device."""
        prefix = self.format.encode_prefix(self.processor, pages)
        return {k: v.to(self.device) for k, v in prefix.items()}
