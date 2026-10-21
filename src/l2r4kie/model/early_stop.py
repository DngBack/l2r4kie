"""Stop decoding a field once its outcome can no longer be ``'ok'``.

Values that never close run to ``max_value_tokens`` (16,384) and, in the kev-r4
caches, the documents holding them took 69-80% of all decode time while
their fields were sent to review anyway. Two detectors end such branches
early:

* :class:`JsonArrayPrefix` (array fields): an incremental validator that
  mirrors :func:`json.loads`. Once the generated text is no prefix of any
  JSON array, the field can only end ``'invalid_array'`` (if it closes) or
  ``'truncated'`` (if it does not), so stopping changes neither the value's
  correctness nor its review. The branch is reported ``'invalid_array'``.
* :class:`Repetition` (all fields): the value ends in an exact repeat of a
  short token period (a loop). Its thresholds were set on every field of the
  kev-r4 trace caches (35,174 values): it never fires on an ``'ok'`` value
  and stops 70% of the truncated ones, at token 255 (median) instead of
  16,384. The branch is reported ``'truncated'``.

Both work on what has been generated so far and never look ahead, so a
branch they leave alone decodes exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

try:
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
except ImportError:  # transformers >= 5 (Qwen3.5)
    from transformers.convert_slow_tokenizer import bytes_to_unicode

_WHITESPACE = frozenset(b' \t\n\r')
_DIGITS = frozenset(b'0123456789')
_HEX = frozenset(b'0123456789abcdefABCDEF')
_ESCAPES = frozenset(b'"\\/bfnrt')
#: First byte of each literal :func:`json.loads` accepts, and the bytes that must follow it.
_LITERALS = {ord('t'): b'rue', ord('f'): b'alse', ord('n'): b'ull', ord('N'): b'aN', ord('I'): b'nfinity'}
#: Number states in which a number may end (``0``, ``12``, ``1.5``, ``1e3``).
_NUMBER_ENDS = frozenset({'zero', 'int', 'frac', 'exp'})


class JsonArrayPrefix:
    """Incremental check that bytes are a prefix of a text :func:`json.loads` parses to a list.

    Mirrors the standard library parser: whitespace is only `` \\t\\n\\r``;
    strings reject raw control characters (``strict=True``) and need four
    hex digits after ``\\u``; ``NaN``, ``Infinity`` and ``-Infinity`` are
    values; no trailing commas; nothing but whitespace after the array.
    Bytes from 0x80 up are only accepted inside strings, where any of them
    decodes (with replacement) to a valid character.

    Attributes:
        viable: ``False`` from the first byte that no JSON array can contain
            at that point; it never turns back.
        complete: The bytes so far form a whole JSON array (plus whitespace).
    """

    __slots__ = ('_escape', '_is_key', '_literal', '_mode', '_number', '_stack', 'viable')

    def __init__(self) -> None:
        self._stack: list[int] = []  # open containers: ord('[') or ord('{')
        self._mode = 'start'
        self._is_key = False  # the open string is an object key
        self._escape = 0  # 0: none, 1: after a backslash, 2-5: hex digits of \\uXXXX still expected + 1
        self._literal = b''  # bytes a literal still needs
        self._number = ''  # number grammar state
        self.viable = True

    @property
    def complete(self) -> bool:
        return self.viable and self._mode == 'done'

    def feed(self, data: bytes) -> bool:
        """Consume ``data``; return :attr:`viable`."""
        for byte in data:
            if not self.viable:
                break
            self.viable = self._step(byte)
        return self.viable

    def _step(self, byte: int) -> bool:  # noqa: C901, PLR0911, PLR0912 (one branch per grammar state)
        mode = self._mode
        if mode == 'string':
            return self._string(byte)
        if mode == 'literal':
            if byte != self._literal[0]:
                return False
            self._literal = self._literal[1:]
            if not self._literal:
                self._mode = 'after_value'
            return True
        if mode == 'number':
            if self._number_step(byte):
                return True
            if self._number not in _NUMBER_ENDS:
                return False
            self._mode = mode = 'after_value'  # the byte ends the number; read it as what follows a value
        if byte in _WHITESPACE:
            return True
        if mode == 'start':
            if byte != ord('['):
                return False
            self._stack.append(byte)
            self._mode = 'value_or_end'
            return True
        if mode == 'done':
            return False
        if mode == 'after_value':
            top = self._stack[-1]
            if byte == ord(','):
                self._mode = 'value' if top == ord('[') else 'key'
                return True
            if byte == (ord(']') if top == ord('[') else ord('}')):
                self._stack.pop()
                self._mode = 'after_value' if self._stack else 'done'
                return True
            return False
        if mode == 'colon':
            if byte != ord(':'):
                return False
            self._mode = 'value'
            return True
        if mode in ('key', 'key_or_end'):
            if byte == ord('}') and mode == 'key_or_end':
                self._stack.pop()
                self._mode = 'after_value' if self._stack else 'done'
                return True
            if byte != ord('"'):
                return False
            self._mode, self._is_key = 'string', True
            return True
        # 'value' or 'value_or_end'
        if byte == ord(']') and mode == 'value_or_end':
            self._stack.pop()
            self._mode = 'after_value' if self._stack else 'done'
            return True
        return self._value_start(byte)

    def _value_start(self, byte: int) -> bool:
        if byte == ord('"'):
            self._mode, self._is_key = 'string', False
        elif byte in (ord('['), ord('{')):
            self._stack.append(byte)
            self._mode = 'value_or_end' if byte == ord('[') else 'key_or_end'
        elif byte in _LITERALS:
            self._mode, self._literal = 'literal', _LITERALS[byte]
        elif byte == ord('-'):
            self._mode, self._number = 'number', 'minus'
        elif byte in _DIGITS:
            self._mode, self._number = 'number', 'zero' if byte == ord('0') else 'int'
        else:
            return False
        return True

    def _string(self, byte: int) -> bool:
        if self._escape == 1:
            if byte == ord('u'):
                self._escape = 5
                return True
            self._escape = 0
            return byte in _ESCAPES
        if self._escape:
            self._escape = self._escape - 1 if self._escape > 2 else 0
            return byte in _HEX
        if byte == ord('\\'):
            self._escape = 1
        elif byte == ord('"'):
            self._mode = 'colon' if self._is_key else 'after_value'
        elif byte < 0x20:
            return False
        return True

    def _number_step(self, byte: int) -> bool:
        """Advance the number grammar ``-?(0|[1-9]\\d*)(\\.\\d+)?([eE][-+]?\\d+)?``.

        Returns ``False`` if ``byte`` is not part of the number.
        """
        state, digit = self._number, byte in _DIGITS
        if state == 'minus':
            if byte == ord('I'):  # -Infinity
                self._mode, self._literal = 'literal', b'nfinity'
                return True
            nxt = ('zero' if byte == ord('0') else 'int') if digit else None
        elif state in ('zero', 'int'):
            nxt = ('int' if digit and state == 'int' else
                   'frac0' if byte == ord('.') else 'exp0' if byte in b'eE' else None)
        elif state in ('frac0', 'frac'):
            nxt = 'frac' if digit else ('exp0' if byte in b'eE' and state == 'frac' else None)
        elif state == 'exp0':
            nxt = 'exp' if digit else ('expsign' if byte in b'+-' else None)
        else:  # 'expsign', 'exp'
            nxt = 'exp' if digit else None
        if nxt is None:
            return False
        self._number = nxt
        return True


class Repetition:
    """Detects a value that ends in an exact repeat of a short token period.

    Fires when, for some period ``p <= max_period``, the last
    ``max(min_span, min_repeats * p)`` tokens all equal the token ``p``
    positions before them. Checked every ``every`` tokens (a loop is caught
    at most that many tokens late; checking less often can only fire less).
    """

    __slots__ = ('_tokens', 'every', 'max_period', 'min_repeats', 'min_span')

    def __init__(self, max_period: int = 256, min_span: int = 128, min_repeats: int = 4, every: int = 16) -> None:
        if min(max_period, min_span, every) < 1 or min_repeats < 2:
            raise ValueError('max_period, min_span and every must be positive and min_repeats at least 2')
        self.max_period, self.min_span, self.min_repeats, self.every = max_period, min_span, min_repeats, every
        self._tokens: list[int] = []

    def push(self, token: int) -> bool:
        """Append ``token``; return whether the value now ends in a loop."""
        self._tokens.append(token)
        n = len(self._tokens)
        return n >= self.min_span and n % self.every == 0 and looping(
            self._tokens, self.max_period, self.min_span, self.min_repeats)


def looping(tokens: Sequence[int], max_period: int = 256, min_span: int = 128, min_repeats: int = 4) -> bool:
    """Whether ``tokens`` end in a loop (see :class:`Repetition`)."""
    window = np.asarray(tokens[-max(min_span, min_repeats * max_period):])
    n = len(window)
    # A period p can only qualify if the last token repeats p back.
    for p in np.flatnonzero(window[-2::-1][:max_period] == window[-1]) + 1:
        need = max(min_span, min_repeats * int(p))
        if need > n:
            continue
        tail = window[n - need:]
        if np.array_equal(tail[p:], tail[:-p]):
            return True
    return False


def token_bytes(convert_ids_to_tokens: Callable[[list[int]], list[str]], ids: Sequence[int]) -> list[bytes]:
    """Raw bytes of byte-level BPE tokens (Qwen2), so that ``b''.join`` decodes like the tokenizer.

    Args:
        convert_ids_to_tokens: The tokenizer's method of that name.
        ids: Token ids; special tokens are not supported.
    """
    decoder = _byte_decoder()
    return [bytes(decoder[ch] for ch in piece) for piece in convert_ids_to_tokens(list(ids))]


class BranchStop:
    """Both detectors for one branch: :meth:`push` each generated token (close marker excluded).

    Args:
        kind: Field kind; the JSON check only runs on ``'array'`` fields.
        piece: Raw bytes of a token id (see :func:`token_piece`).
    """

    __slots__ = ('_json', '_piece', '_repetition')

    def __init__(self, kind: str, piece: Callable[[int], bytes]) -> None:
        self._json = JsonArrayPrefix() if kind == 'array' else None
        self._repetition = Repetition()
        self._piece = piece

    def push(self, token: int) -> str | None:
        """Return why the branch should stop (``'invalid_json'`` or ``'loop'``), or ``None``."""
        if self._json is not None and not self._json.feed(self._piece(token)):
            return 'invalid_json'
        return 'loop' if self._repetition.push(token) else None


def token_piece(tokenizer: Any) -> Callable[[int], bytes]:
    """A cached ``id -> bytes`` lookup for ``tokenizer`` (byte-level BPE, special tokens excluded)."""
    cache: dict[int, bytes] = {}

    def piece(token: int) -> bytes:
        found = cache.get(token)
        if found is None:
            found = cache[token] = token_bytes(tokenizer.convert_ids_to_tokens, [token])[0]
        return found

    return piece


_BYTE_DECODER: dict[str, int] = {}


def _byte_decoder() -> dict[str, int]:
    if not _BYTE_DECODER:
        _BYTE_DECODER.update({v: k for k, v in bytes_to_unicode().items()})
    return _BYTE_DECODER
