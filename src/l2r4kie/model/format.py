"""KevFormat: how a document and its fields become model tokens.

One document is one shared *prefix* plus one isolated *branch* per field::

    prefix  <|im_start|>system ... <|im_end|>
            <|im_start|>user <|vision_start|><|image_pad|>...<|vision_end|> ... instruction<|im_end|>
            <|im_start|>assistant\\n
    branch  <|object_ref_start|>field_id: description<|object_ref_end|><|box_start|>
    target  value as plain text<|box_end|>

The prefix (images included) is encoded once and shared; every branch sees
the prefix and itself only (see :mod:`.packing`). Three hidden states per
branch are kept as confidence signals:

========== ========================= ==========================================
signal     token                     meaning
========== ========================= ==========================================
``h_key``  ``<|object_ref_end|>``    the model has read which field is asked
``h_decide`` ``<|box_start|>``       about to write the value (predicts token 1)
``h_value`` ``<|box_end|>``          has read its own complete value
========== ========================= ==========================================

Compared with the old format (JSON value inside a chat turn, closed by
``<|im_end|>``), values are plain text, so there are no JSON syntax errors,
and the markers give fixed, format-independent signal positions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizerBase

from ..data.serialize import from_text, to_text
from ..data.types import FieldKind, FieldSpec
from .markers import CloseToken, Markers, banned_ids, encode_text

SYSTEM_PROMPT = 'You extract document fields accurately. Do not invent absent values.'
DOCUMENT_PROMPT = ('These are all pages of one document. For each requested field, write its value '
                   'exactly as printed, or nothing if the field is blank.')

#: Longest trainable target (value tokens + close marker). The longest target in
#: the data is 6,245 tokens (an array), so nothing is dropped. The old default
#: of 256 silently dropped 48% of array fields. Packed training sequences of 12
#: fields stay under ~9.4k tokens (p95 5.5k) including a 2 MP prefix.
DEFAULT_MAX_VALUE_TOKENS = 16_384

#: Pixel bounds of one page for the image processor. The maximum is the r4
#: winner's setting (2 MP); the old code's 262,144 default was always overridden.
MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 2_097_152


@dataclass(frozen=True, slots=True)
class EncodedBranch:
    """Token ids of one field's branch.

    Attributes:
        field_id: Field the branch extracts.
        prompt: ``key_open, key text..., key_close, value_open``.
        target: ``value text..., value_close``; empty for an inference request.
    """

    field_id: str
    prompt: tuple[int, ...]
    target: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        """Number of tokens of prompt plus target."""
        return len(self.prompt) + len(self.target)

    @property
    def key_index(self) -> int:
        """Index of ``key_close`` (``h_key``) within the branch."""
        return len(self.prompt) - 2

    @property
    def decide_index(self) -> int:
        """Index of ``value_open`` (``h_decide``) within the branch."""
        return len(self.prompt) - 1

    @property
    def value_index(self) -> int:
        """Index of ``value_close`` (``h_value``) within the branch; needs a target."""
        if not self.target:
            raise ValueError(f'Branch {self.field_id!r} has no target, so no value_close position')
        return self.length - 1


@dataclass(frozen=True, slots=True)
class ParsedValue:
    """Generated value text and its JSON value.

    ``value`` is ``None`` when ``status`` is not ``'ok'``.
    """

    text: str
    value: Any
    status: str


class KevFormat:
    """Encodes fields as marker-delimited branches and parses generated values.

    Args:
        tokenizer: The model's tokenizer.
        close: Token closing a value: ``'box_end'`` (default) or the
            ``'im_end'`` fallback.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, close: CloseToken = 'box_end') -> None:
        self.tokenizer = tokenizer
        self.close: CloseToken = close
        self.markers = Markers.from_tokenizer(tokenizer, close)

    # ------------------------------------------------------------------ prefix

    @staticmethod
    def prefix_messages(images: Sequence[Any]) -> list[dict[str, Any]]:
        """Chat messages of the shared prefix: system prompt, then all pages and the instruction."""
        return [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [{'type': 'image', 'image': image} for image in images]
                                        + [{'type': 'text', 'text': DOCUMENT_PROMPT}]},
        ]

    def encode_prefix(self, processor: Any, pages: Sequence[str | Path]) -> dict[str, Any]:
        """Load the page images and encode the shared prefix on CPU.

        The prefix ends with the assistant generation prompt, so the first
        branch marker starts the assistant's answer (as in the marker probe).

        Returns:
            Processor outputs: ``input_ids``, ``attention_mask``,
            ``pixel_values`` and ``image_grid_thw``.
        """
        from PIL import Image

        images = []
        for path in pages:
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        text = processor.apply_chat_template(self.prefix_messages(images), tokenize=False, add_generation_prompt=True)
        return dict(processor(text=[text], images=images, return_tensors='pt'))

    # ---------------------------------------------------------------- branches

    @staticmethod
    def key_text(field_id: str, description: str) -> str:
        """``'field_id: description'``, or just the id when the description is the id."""
        return field_id if description in ('', field_id) else f'{field_id}: {description}'

    def prompt_ids(self, field_id: str, description: str) -> list[int]:
        """``key_open key-text key_close value_open`` (the branch prompt)."""
        m = self.markers
        return [m.key_open, *encode_text(self.tokenizer, self.key_text(field_id, description)), m.key_close, m.value_open]

    def target_ids(self, value: Any) -> list[int]:
        """``value-text value_close`` (the training target)."""
        return [*encode_text(self.tokenizer, to_text(value)), self.markers.value_close]

    def encode(self, field: FieldSpec) -> EncodedBranch:
        """Prompt and target of a labelled field (training, evaluation)."""
        return EncodedBranch(field.id, tuple(self.prompt_ids(field.id, field.description)),
                             tuple(self.target_ids(field.value)))

    def request(self, field_id: str, description: str) -> EncodedBranch:
        """Prompt-only branch of a field to extract (inference)."""
        return EncodedBranch(field_id, tuple(self.prompt_ids(field_id, description)))

    # ---------------------------------------------------------------- decoding

    def banned_ids(self, vocab_size: int) -> list[int]:
        """Ids masked out while decoding: every special token except the close marker."""
        return banned_ids(self.tokenizer, vocab_size, [self.markers.value_close])

    def parse(self, generated: Sequence[int], kind: FieldKind = 'scalar', closed: bool = True) -> ParsedValue:
        """Turn generated ids (close marker excluded) into a value.

        Args:
            generated: Value tokens, without the close marker.
            kind: Expected field kind (see :func:`~l2r4kie.data.serialize.from_text`).
            closed: Whether decoding ended on the close marker.

        Returns:
            Status ``'ok'``, ``'truncated'`` (no close marker within the token
            budget) or ``'invalid_array'`` (array text that is not a JSON array).
        """
        text = self.tokenizer.decode(list(generated), skip_special_tokens=False)
        if not closed:
            return ParsedValue(text, None, 'truncated')
        try:
            return ParsedValue(text, from_text(text, kind), 'ok')
        except ValueError:
            return ParsedValue(text, None, 'invalid_array')


def load_processor(model_name: str, max_pixels: int = DEFAULT_MAX_PIXELS, **kwargs: Any) -> Any:
    """Load the Qwen2-VL processor with the page pixel bounds used in training.

    Extra keyword arguments (e.g. ``local_files_only``) go to ``from_pretrained``.
    """
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_name, min_pixels=MIN_PIXELS, max_pixels=max_pixels, use_fast=False,
                                         **kwargs)
