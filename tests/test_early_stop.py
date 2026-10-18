"""Tests for ``l2r4kie.model.early_stop`` (CPU, no model)."""

from __future__ import annotations

import json
import random

import pytest

from l2r4kie.model.early_stop import JsonArrayPrefix, Repetition, looping, token_bytes

SEEDS = [
    '[]', '[ ]', ' [1, -2.5e+3, 0, -0.0, 1E9]', '[true,false,null,NaN,Infinity,-Infinity]',
    '[{"Chỉ tiêu":"Tiền","Tổng":"20.650.978.793"},{"a":[],"b":{}}]',
    '["a\\"b\\\\c\\/\\b\\f\\n\\r\\t\\u00e9\\uD83D"]',
    '[[[["deep"]]], {"k": {"k": ["v", {"x": ""}]}}]\n', '["Lưu Tuấn Mai","Huỳnh Xuân"]',
]
ALPHABET = list('[]{}",:\\ -+.eE0123456789tfnaNIulrsy\n\t\x01') + ['ư', 'é', 'Đ']


def feed_text(text: str) -> JsonArrayPrefix:
    state = JsonArrayPrefix()
    state.feed(text.encode())
    return state


def parses_to_list(text: str) -> bool:
    try:
        return isinstance(json.loads(text), list)
    except ValueError:
        return False


def completion(state: JsonArrayPrefix) -> str:
    """Some suffix that turns a viable prefix into a whole array (from the validator's state)."""
    out, mode = '', state._mode  # noqa: SLF001 - the test derives a witness from the grammar state
    if mode == 'start':
        return '[]'
    if mode == 'string':
        out += {0: '', 1: 'n', 2: '0', 3: '00', 4: '000', 5: '0000'}[state._escape] + '"'  # noqa: SLF001
        mode = 'colon' if state._is_key else 'after_value'  # noqa: SLF001
    elif mode == 'literal':
        out += state._literal.decode()  # noqa: SLF001
        mode = 'after_value'
    elif mode == 'number':
        out += '' if state._number in ('zero', 'int', 'frac', 'exp') else '0'  # noqa: SLF001
        mode = 'after_value'
    if mode == 'key':
        out += '"k"'
        mode = 'colon'
    if mode == 'colon':
        out += ':'
        mode = 'value'
    if mode == 'value':
        out += '0'
    return out + ''.join(']' if c == ord('[') else '}' for c in reversed(state._stack))  # noqa: SLF001


def mutations(rng: random.Random, n: int) -> list[str]:
    texts = []
    for _ in range(n):
        text = list(rng.choice(SEEDS))
        for _ in range(rng.randint(1, 3)):
            i = rng.randrange(len(text) + 1)
            match rng.randrange(3):
                case 0:
                    text.insert(i, rng.choice(ALPHABET))
                case 1 if text:
                    del text[min(i, len(text) - 1)]
                case _ if text:
                    text[min(i, len(text) - 1)] = rng.choice(ALPHABET)
        texts.append(''.join(text))
    return texts


@pytest.mark.parametrize('text', SEEDS)
def test_every_prefix_of_a_valid_array_is_viable(text: str) -> None:
    data = text.encode()
    for end in range(len(data) + 1):
        state = JsonArrayPrefix()
        assert state.feed(data[:end]), data[:end]
    assert feed_text(text).complete


@pytest.mark.parametrize('text', ['{}', '"a"', '1', 'x[', '[1,]', '[,1]', '{"a":1,}', '[01]', '[1.]', '[.5]', '[1e]',
                                  '["a\nb"]', '["\\x"]', '["\\u12G4"]', '[] []', '[]x', '[tru e]', '[-]', '[-a]',
                                  '[{"a" 1}]', '[{1:2}]', '[NaNa]', '﻿[]', '[ä]'])
def test_rejects_what_json_rejects(text: str) -> None:
    state = feed_text(text)
    assert not parses_to_list(text)
    assert not state.complete


def test_agrees_with_json_loads_on_mutated_arrays() -> None:
    rng = random.Random(0)
    for text in mutations(rng, 4000):
        state = feed_text(text)
        assert state.complete == parses_to_list(text), text
        if state.viable:  # viable means some continuation parses: build one from the state
            assert parses_to_list(text + completion(state)), (text, completion(state))


def test_stays_rejected() -> None:
    state = feed_text('[1,]')
    assert not state.feed(b']') and not state.viable


def test_bytes_split_inside_a_character_are_accepted_in_strings() -> None:
    data = '["Đạo"]'.encode()
    state = JsonArrayPrefix()
    for i in range(len(data)):
        assert state.feed(data[i:i + 1])
    assert state.complete
    assert not feed_text('[Đ]').viable  # non-ASCII outside a string


def test_token_bytes_join_like_the_tokenizer(qwen_tokenizer) -> None:  # noqa: ANN001
    text = '[{"Họ tên":"Lưu Tuấn Mai","Tổng":"20.650.978.793"}]\t Huỳnh'
    ids = qwen_tokenizer(text, add_special_tokens=False)['input_ids']
    pieces = token_bytes(qwen_tokenizer.convert_ids_to_tokens, ids)
    assert len(pieces) == len(ids)
    assert b''.join(pieces).decode() == qwen_tokenizer.decode(ids) == text


# ---------------------------------------------------------------- repetition

def test_repetition_fires_on_a_loop_and_not_before_min_span() -> None:
    period = [7, 8, 9, 10, 11]
    detector = Repetition(min_span=40, min_repeats=4, every=1)
    fired = [detector.push(t) for t in ([1, 2, 3] + period * 20)]
    first = fired.index(True)
    assert first == 3 + 40 - 1  # as soon as 40 tokens (8 periods) repeat
    assert not any(fired[:first])


def test_repetition_needs_min_repeats_for_long_periods() -> None:
    rng = random.Random(1)
    block = [rng.randrange(1000) for _ in range(100)]
    assert not looping(block * 3, max_period=256, min_span=128, min_repeats=4)  # 300 tokens, period 100: 3 repeats
    assert looping(block * 4, max_period=256, min_span=128, min_repeats=4)
    assert not looping(block * 8, max_period=64, min_span=128, min_repeats=4)  # period above max_period


def test_repetition_ignores_varied_text() -> None:
    rng = random.Random(2)
    detector = Repetition(every=1)
    assert not any(detector.push(rng.randrange(50_000)) for _ in range(2000))


def test_repetition_is_checked_every_n_tokens() -> None:
    detector = Repetition(min_span=8, min_repeats=2, every=16)
    assert [detector.push(5) for _ in range(16)] == [False] * 15 + [True]


def test_repetition_validates_arguments() -> None:
    with pytest.raises(ValueError, match='min_repeats'):
        Repetition(min_repeats=1)
