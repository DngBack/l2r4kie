"""Reused special tokens that delimit keys and values (no new tokens added).

Following Kev (``jaredpalmer/kev``), the format reuses rarely used special
tokens that already exist in Qwen2-VL's vocabulary instead of adding new ones.
Evidence for the choice (embedding norms, zero-shot probe) is in
``docs/notes/marker_selection.md``:

* ``<|object_ref_start|>`` / ``<|object_ref_end|>`` wrap the key;
* ``<|box_start|>`` opens the value, ``<|box_end|>`` closes it.

The base model already closes ``box_start ... box_end`` reliably (24/24 in the
probe), at the cost of a grounding prior: untrained, it writes coordinates
inside the box. Fine-tuning replaces the coordinates with text.

Caller text (keys, descriptions, values) is tokenized with
``split_special_tokens=True``, so a literal ``<|box_end|>`` inside a value is
plain text and cannot forge a marker. Unlike Kev's ``<|x|>`` -> ``<¦x¦>``
escaping this is lossless: decoding gives back the original string.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from transformers import PreTrainedTokenizerBase

#: Tokens that ``get_rope_index`` and the processor use to locate images and
#: videos. Reusing them as markers would corrupt vision positions.
VISION_TOKENS: tuple[str, ...] = (
    '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>')

#: Token that closes a value. ``'im_end'`` is the fallback if the model keeps
#: writing coordinates inside ``box_start ... box_end`` after fine-tuning.
CloseToken = Literal['box_end', 'im_end']


@dataclass(frozen=True, slots=True)
class Markers:
    """Token ids of the four markers and of the value close token.

    Attributes:
        key_open: ``<|object_ref_start|>``.
        key_close: ``<|object_ref_end|>``; its hidden state is ``h_key``.
        value_open: ``<|box_start|>``; its hidden state is ``h_decide``
            (it predicts the first value token).
        value_close: Token generated after the value; its hidden state is
            ``h_value``. ``<|box_end|>`` unless the ``im_end`` fallback is used.
    """

    key_open: int
    key_close: int
    value_open: int
    value_close: int

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase, close: CloseToken = 'box_end') -> Markers:
        """Look the markers up in ``tokenizer`` and check they are usable.

        Raises:
            ValueError: If a marker is not a single known token, or collides
                with a vision token.
        """
        names = ('<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', f'<|{close}|>')
        ids = [single_token_id(tokenizer, name) for name in names]
        vision = {single_token_id(tokenizer, name) for name in VISION_TOKENS}
        if len(set(ids)) != len(ids) or vision & set(ids):
            raise ValueError(f'Markers must be distinct non-vision tokens, got {dict(zip(names, ids))}')
        return cls(*ids)


def single_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    """Return the id of ``token``, which must encode to exactly one known id."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1 or ids[0] == tokenizer.unk_token_id:
        raise ValueError(f'{token!r} is not a single token of this tokenizer: {ids}')
    return ids[0]


def encode_text(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Tokenize caller text so that it can never produce a special token.

    This is the only way keys, descriptions and values enter the model; it
    also fixes the old code's inconsistent ``add_special_tokens`` use.
    """
    return tokenizer(text, add_special_tokens=False, split_special_tokens=True)['input_ids']


def special_ids(tokenizer: PreTrainedTokenizerBase, vocab_size: int) -> set[int]:
    """Every id that is not ordinary text.

    That is the tokenizer's added/special tokens plus the embedding rows beyond
    the tokenizer (Qwen2-VL has 151,936 rows but 151,657 tokens; the extra rows
    were never trained).
    """
    return set(tokenizer.added_tokens_decoder) | set(range(len(tokenizer), vocab_size))


def banned_ids(tokenizer: PreTrainedTokenizerBase, vocab_size: int, allowed: Iterable[int]) -> list[int]:
    """Ids to mask out while decoding a value: all specials except ``allowed``.

    Decoding with these banned guarantees a value is plain text ended by the
    close marker; no stray ``<|im_end|>``, vision token or unused row.
    """
    return sorted(special_ids(tokenizer, vocab_size) - set(allowed))
