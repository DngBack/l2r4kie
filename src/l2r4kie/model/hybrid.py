"""Branch-isolated forward of Qwen3.5, a hybrid of Gated DeltaNet and gated attention.

Qwen3.5 interleaves three *linear-attention* layers (Gated DeltaNet: a short
causal convolution, then a gated delta-rule recurrence) with one full
attention layer. The block-causal mask of :mod:`.packing` isolates branches
only in attention layers: a recurrence and a convolution run along the
packed sequence, so through them a branch would read the branches packed
before it. Hugging Face's cache cannot isolate them either (its padding
still decays the recurrent state, the convolution mixes neighbours, and the
state is updated in place).

:class:`HybridCore` therefore runs the language model itself, layer by
layer, so that every branch continues the prefix on its own:

* **DeltaNet layers**: each branch starts from the prefix's final recurrent
  state and convolves its first tokens with the prefix's last
  ``kernel - 1`` inputs. Branches are packed without padding and run in one
  varlen ``fla`` kernel call (``cu_seqlens``, one initial state per branch).
* **Attention layers**: a branch token attends to the whole prefix and to
  the earlier tokens of its own branch (the mask of :func:`.packing.block_mask`).

The prefix runs once. Training (:meth:`HybridCore.packed_hidden`) and the
first decode call (branch prompts) share :meth:`HybridCore._branches`, so a
decoded branch is computed exactly as it was trained; the following decode
steps advance per-branch states (:class:`HybridCache`). The prefix state is
shared by all branches and never copied per branch.

The wrapper exposes what :mod:`.decode` and :mod:`.packing` use from a
Qwen2-VL core (``get_rope_index``, ``visual``, ``config`` and a forward with
``past_key_values``), so decoding, traces and intermediate layers work
unchanged. Weights are called through their modules, so LoRA layers and
trainable marker rows apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

#: ``model_type`` of the hybrid models this module runs.
HYBRID_MODEL_TYPES: tuple[str, ...] = ('qwen3_5',)


def is_hybrid(config: Any) -> bool:
    """Whether a model config (or the model's) is a Qwen3.5 hybrid."""
    return getattr(config, 'model_type', None) in HYBRID_MODEL_TYPES


def _fla() -> tuple[Any, Any]:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@dataclass
class HybridOutput:
    """Forward output with the fields :mod:`.decode` reads from a Hugging Face output."""

    last_hidden_state: torch.Tensor
    past_key_values: HybridCache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class PrefixState:
    """Per-layer state of the shared prefix.

    Attributes:
        length: Prefix tokens.
        attention: Layer → ``(key, value)``, ``(1, kv_heads, length, head_dim)``,
            after norm and rotary embedding.
        delta: Layer → ``(tail, state)``: the last ``kernel - 1`` convolution
            inputs ``(kernel - 1, channels)`` and the final recurrent state
            ``(heads, key_dim, value_dim)`` (float32).
    """

    length: int
    attention: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    delta: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


class HybridCache:
    """Decode state of a batch of branches over one prefix.

    Holds the :class:`PrefixState` (shared) and, once the branch prompts have
    run, per-row states: convolution windows and recurrent states of the
    DeltaNet layers, and left-padded keys/values of the attention layers
    (float32). Implements the cache methods :mod:`.decode` calls.
    """

    def __init__(self, prefix: PrefixState) -> None:
        self.prefix = prefix
        self.rows = 1
        self.started = False
        self.conv: dict[int, torch.Tensor] = {}
        self.recurrent: dict[int, torch.Tensor] = {}
        self.keys: dict[int, torch.Tensor] = {}
        self.values: dict[int, torch.Tensor] = {}
        self.padding: torch.Tensor | None = None  # (rows,) left padding of the branch columns
        self.width = 0  # branch columns in use
        self._prefix32: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def batch_repeat_interleave(self, n: int) -> None:
        """Prepare ``n`` branches (before their prompts run)."""
        if self.started:
            raise ValueError('Branches already started on this cache')
        self.rows = n

    def batch_select_indices(self, keep: torch.Tensor) -> None:
        """Keep only the rows ``keep`` (compaction of finished branches)."""
        for states in (self.conv, self.recurrent, self.keys, self.values):
            for layer, value in states.items():
                states[layer] = value[keep]
        if self.padding is not None:
            self.padding = self.padding[keep]
        self.rows = len(keep)

    def get_seq_length(self, layer: int = 0) -> int:
        """Prefix plus branch columns."""
        return self.prefix.length + self.width

    def crop(self, length: int) -> None:
        raise NotImplementedError('A hybrid cache cannot be cropped')

    def __deepcopy__(self, memo: dict[int, Any]) -> HybridCache:
        # The prefix state is never modified: share it.
        other = HybridCache(self.prefix)
        other.rows, other.started, other.width = self.rows, self.started, self.width
        other._prefix32 = self._prefix32
        for name in ('conv', 'recurrent', 'keys', 'values'):
            setattr(other, name, {k: v.clone() for k, v in getattr(self, name).items()})
        other.padding = None if self.padding is None else self.padding.clone()
        return other

    def prefix32(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefix keys/values of an attention layer as ``(kv_heads, length, head_dim)`` float32."""
        if layer not in self._prefix32:
            key, value = self.prefix.attention[layer]
            self._prefix32[layer] = (key[0].float(), value[0].float())
        return self._prefix32[layer]

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write one step's ``(rows, kv_heads, head_dim)`` key/value at column ``width``."""
        keys, values = self.keys[layer], self.values[layer]
        if self.width >= keys.shape[2]:
            grow = max(keys.shape[2], 64)
            keys = F.pad(keys, (0, 0, 0, grow))
            values = F.pad(values, (0, 0, 0, grow))
            self.keys[layer], self.values[layer] = keys, values
        keys[:, :, self.width] = key.float()
        values[:, :, self.width] = value.float()


class HybridCore:
    """Branch-isolated language model of a ``Qwen3_5Model`` (see the module docstring).

    Args:
        model: The inner ``Qwen3_5Model`` (vision tower + language model).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.text = model.language_model
        self.config = model.config
        self.layer_types: list[str] = list(self.config.text_config.layer_types)

    # ------------------------------------------------------------ interface

    @property
    def visual(self) -> torch.nn.Module:
        """Vision tower."""
        return self.model.visual

    def get_rope_index(self, input_ids: torch.Tensor, image_grid_thw: torch.Tensor | None = None,
                       video_grid_thw: torch.Tensor | None = None,
                       attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Qwen2-VL's signature over Qwen3.5's (``mm_token_type_ids`` from the image token)."""
        types = (input_ids == self.config.image_token_id).int()
        return self.model.get_rope_index(input_ids, types, image_grid_thw, video_grid_thw, attention_mask)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 position_ids: torch.Tensor | None = None, past_key_values: HybridCache | None = None,
                 use_cache: bool = False, output_hidden_states: bool = False,
                 pixel_values: torch.Tensor | None = None, image_grid_thw: torch.Tensor | None = None,
                 **_: Any) -> HybridOutput:
        """Prefix forward (no cache), branch prompts (fresh cache) or one decode step.

        Args:
            input_ids: ``(1, prefix)`` for the prefix; ``(rows, length)``
                left-padded branch prompts; ``(rows, 1)`` for a step.
            attention_mask: Prefix mask (all ones), or ``(rows, prefix + columns)``
                whose branch columns mark real prompt tokens.
            position_ids: ``(3, rows, length)`` M-RoPE positions.
            past_key_values: A :class:`HybridCache` for branch calls.
            use_cache: Return a cache after the prefix.
            output_hidden_states: Also return every layer's output
                (index 0: embeddings, last: normed output).
            pixel_values: Prefix images.
            image_grid_thw: Their grids.
        """
        if past_key_values is None:
            if attention_mask is not None and not bool(attention_mask.all()):
                raise ValueError('The prefix must not be padded')
            if position_ids is None:
                position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
            x = self.embed(input_ids, pixel_values, image_grid_thw)
            hidden, state, layers = self._prefix(x, position_ids, output_hidden_states)
            return HybridOutput(hidden, HybridCache(state) if use_cache else None, layers)
        cache = past_key_values
        if not cache.started:
            return self._prompts(cache, input_ids, attention_mask, position_ids, output_hidden_states)
        if input_ids.shape[1] != 1:
            raise ValueError('After the branch prompts, decode one token per step')
        return self._step(cache, input_ids, position_ids, output_hidden_states)

    def embed(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None = None,
              image_grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        """Token embeddings with the image features scattered in (vision tower frozen)."""
        x = self.text.embed_tokens(input_ids)
        if pixel_values is not None:
            with torch.no_grad():
                features = self.model.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            features = torch.cat(features).to(x.device, x.dtype)
            mask = (input_ids == self.config.image_token_id)[..., None].expand_as(x)
            if int(mask.sum()) != features.numel():
                raise ValueError('Image features and image tokens do not match')
            x = x.masked_scatter(mask, features)
        return x

    def packed_hidden(self, inputs: dict[str, Any], segments: Sequence[int]) -> torch.Tensor:
        """Normed last hidden states ``(total, hidden)`` of a packed document (training).

        Args:
            inputs: :attr:`.packing.Packed.inputs` (ids, positions, images).
            segments: Prefix length, then branch lengths, in packing order.
        """
        ids, positions = inputs['input_ids'], inputs['position_ids']
        shared, lengths = segments[0], list(segments[1:])
        x = self.embed(ids, inputs.get('pixel_values'), inputs.get('image_grid_thw'))
        prefix, state, _ = self._prefix(x[:, :shared], positions[..., :shared], False)
        if not lengths:
            return prefix[0]
        branches, _ = self._branches(x[0, shared:], positions[..., shared:], lengths, state, False)
        return torch.cat([prefix[0], branches])

    # --------------------------------------------------------------- passes

    def _checkpointed(self) -> bool:
        return torch.is_grad_enabled() and bool(getattr(self.text.layers[0], 'gradient_checkpointing', False))

    def _prefix(self, x: torch.Tensor, positions: torch.Tensor,
                deep: bool) -> tuple[torch.Tensor, PrefixState, tuple[torch.Tensor, ...] | None]:
        """Run the prefix ``(1, length, hidden)``; return its normed states and per-layer state."""
        rotary = self.text.rotary_emb(x, positions)
        state = PrefixState(x.shape[1])
        outputs = [x] if deep else None
        for i, layer in enumerate(self.text.layers):
            run = (lambda h, layer=layer, i=i: _prefix_layer(layer, self.layer_types[i], h, rotary))
            x, *captured = checkpoint(run, x, use_reentrant=False) if self._checkpointed() else run(x)
            (state.attention if self.layer_types[i] == 'full_attention' else state.delta)[i] = tuple(captured)
            if deep:
                outputs.append(x)
        x = self.text.norm(x)
        if deep:
            outputs[-1] = x
        return x, state, tuple(outputs) if deep else None

    def _branches(self, x: torch.Tensor, positions: torch.Tensor, lengths: Sequence[int], state: PrefixState,
                  deep: bool, capture: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run packed branches ``(total, hidden)`` (no padding) after the prefix.

        Returns:
            Normed states ``(total, hidden)`` and, with ``deep`` or
            ``capture``, ``'layers'`` (every layer's output) and per-layer
            branch states (``'attention'``: keys/values ``(total, kv_heads,
            head_dim)``; ``'delta'``: convolution windows and final states).
        """
        rotary = self.text.rotary_emb(x[None], positions)
        bounds = torch.tensor([0, *lengths], device=x.device).cumsum(0)
        extra: dict[str, Any] = {'layers': [x] if deep else None, 'attention': {}, 'delta': {}}
        mask = _branch_mask(state.length, lengths, x.device)
        for i, layer in enumerate(self.text.layers):
            kind = self.layer_types[i]
            captured = state.attention[i] if kind == 'full_attention' else state.delta[i]

            def run(h: torch.Tensor, a: torch.Tensor, b: torch.Tensor, layer: Any = layer,
                    kind: str = kind) -> tuple[torch.Tensor, ...]:
                return _branch_layer(layer, kind, h, rotary, (a, b), lengths, bounds, mask)

            x, *own = checkpoint(run, x, *captured, use_reentrant=False) if self._checkpointed() else run(x, *captured)
            if capture:
                extra['attention' if kind == 'full_attention' else 'delta'][i] = own
            if deep:
                extra['layers'].append(x)
        x = self.text.norm(x)
        if deep:
            extra['layers'][-1] = x
        return x, extra

    def _prompts(self, cache: HybridCache, ids: torch.Tensor, attention_mask: torch.Tensor,
                 positions: torch.Tensor, deep: bool) -> HybridOutput:
        """First branch call: left-padded prompts, run packed; fills the per-row states."""
        rows, width = ids.shape
        if rows != cache.rows:
            raise ValueError(f'{rows} prompts for a cache prepared for {cache.rows} rows')
        real = attention_mask[:, -width:].bool()
        lengths = real.sum(1).tolist()
        x = self.text.embed_tokens(ids[real])
        hidden, extra = self._branches(x, positions[:, real][:, None], lengths, cache.prefix, deep, capture=True)

        def padded(packed: torch.Tensor) -> torch.Tensor:
            out = packed.new_zeros((rows, width, *packed.shape[1:]))
            out[real] = packed
            return out

        for layer, (key, value) in extra['attention'].items():
            cache.keys[layer] = padded(key.float()).transpose(1, 2).contiguous()
            cache.values[layer] = padded(value.float()).transpose(1, 2).contiguous()
        for layer, (conv, recurrent) in extra['delta'].items():
            cache.conv[layer], cache.recurrent[layer] = conv, recurrent
        cache.padding = (width - real.sum(1)).to(ids.device)
        cache.width, cache.started = width, True
        layers = tuple(padded(h) for h in extra['layers']) if deep else None
        return HybridOutput(padded(hidden), cache, layers)

    def _step(self, cache: HybridCache, ids: torch.Tensor, positions: torch.Tensor, deep: bool) -> HybridOutput:
        """One decode token per row."""
        x = self.text.embed_tokens(ids)  # (rows, 1, hidden)
        rotary = self.text.rotary_emb(x, positions)
        outputs = [x] if deep else None
        for i, layer in enumerate(self.text.layers):
            residual = x
            h = layer.input_layernorm(x)
            if self.layer_types[i] == 'full_attention':
                h = _attention_step(layer.self_attn, h, rotary, cache, i)
            else:
                h = _delta_step(layer.linear_attn, h[:, 0], cache, i)[:, None]
            x = residual + h
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            if deep:
                outputs.append(x)
        cache.width += 1
        x = self.text.norm(x)
        if deep:
            outputs[-1] = x
        return HybridOutput(x, cache, tuple(outputs) if deep else None)


# ---------------------------------------------------------------- layers


def _mlp(layer: Any, x: torch.Tensor) -> torch.Tensor:
    return x + layer.mlp(layer.post_attention_layernorm(x))


def _prefix_layer(layer: Any, kind: str, x: torch.Tensor,
                  rotary: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """One decoder layer over the prefix ``(1, length, hidden)``; returns output and captured state."""
    h = layer.input_layernorm(x)
    if kind == 'full_attention':
        attn = layer.self_attn
        q, k, v, gate = _attention_inputs(attn, h, rotary)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn.scaling, enable_gqa=True)
        h = _attention_output(attn, o, gate)
        captured = (k, v)
    else:
        mod = layer.linear_attn
        qkv = mod.in_proj_qkv(h[0])
        context = qkv.new_zeros((mod.conv_kernel_size - 1, qkv.shape[1]))
        out, (tail, recurrent) = _delta_mix(mod, h[0], qkv, [h.shape[1]], context, None)
        h = mod.out_proj(out)[None]
        captured = (tail[0].transpose(0, 1), recurrent[0])  # tail as (kernel - 1, channels)
    return (_mlp(layer, x + h), *captured)


def _branch_layer(layer: Any, kind: str, x: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor],
                  captured: tuple[torch.Tensor, torch.Tensor], lengths: Sequence[int], bounds: torch.Tensor,
                  mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """One decoder layer over packed branches ``(total, hidden)``; returns output and branch state."""
    h = layer.input_layernorm(x)
    if kind == 'full_attention':
        attn = layer.self_attn
        prefix_key, prefix_value = captured
        q, k, v, gate = _attention_inputs(attn, h[None], rotary)
        o = F.scaled_dot_product_attention(q, torch.cat([prefix_key, k], 2), torch.cat([prefix_value, v], 2),
                                           attn_mask=mask, scale=attn.scaling, enable_gqa=True)
        h = _attention_output(attn, o, gate)[0]
        own = (k[0].transpose(0, 1), v[0].transpose(0, 1))  # (total, kv_heads, head_dim)
    else:
        mod = layer.linear_attn
        tail, recurrent = captured
        qkv = mod.in_proj_qkv(h)
        out, own = _delta_mix(mod, h, qkv, lengths, tail, recurrent, bounds)
        h = mod.out_proj(out)
    return (_mlp(layer, x + h), *own)


def _branch_mask(shared: int, lengths: Sequence[int], device: torch.device) -> torch.Tensor:
    """Boolean ``(1, 1, total, shared + total)``: all of the prefix, then causal within the own branch."""
    total = sum(lengths)
    branch = torch.repeat_interleave(torch.arange(len(lengths), device=device),
                                     torch.tensor(lengths, device=device))
    index = torch.arange(total, device=device)
    own = (branch[:, None] == branch[None, :]) & (index[:, None] >= index[None, :])
    return torch.cat([torch.ones((total, shared), dtype=torch.bool, device=device), own], 1)[None, None]


def _attention_inputs(attn: Any, h: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor]
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Queries, keys, values ``(batch, heads, length, head_dim)`` and output gate of ``h`` ``(batch, length, hidden)``."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    batch, length = h.shape[:2]
    shape = (batch, length, -1, attn.head_dim)
    q, gate = torch.chunk(attn.q_proj(h).view(batch, length, -1, attn.head_dim * 2), 2, dim=-1)
    q = attn.q_norm(q.reshape(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(h).view(shape)).transpose(1, 2)
    v = attn.v_proj(h).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *rotary)
    return q, k, v, gate.reshape(batch, length, -1)


def _attention_output(attn: Any, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    batch, _, length, _ = o.shape
    o = o.transpose(1, 2).reshape(batch, length, -1)
    return attn.o_proj(o * torch.sigmoid(gate))


def _attention_step(attn: Any, h: torch.Tensor, rotary: tuple[torch.Tensor, torch.Tensor], cache: HybridCache,
                    layer: int) -> torch.Tensor:
    """Attention of one token per row over the shared prefix and its own branch (float32)."""
    q, k, v, gate = _attention_inputs(attn, h, rotary)
    cache.append(layer, k[:, :, 0], v[:, :, 0])
    prefix_key, prefix_value = cache.prefix32(layer)
    width = cache.width + 1
    keys, values = cache.keys[layer][:, :, :width], cache.values[layer][:, :, :width]
    rows, kv_heads = keys.shape[:2]
    q = q[:, :, 0].float().view(rows, kv_heads, -1, attn.head_dim)  # (rows, kv_heads, groups, head_dim)
    scores = torch.cat([torch.einsum('rhgd,hpd->rhgp', q, prefix_key),
                        torch.einsum('rhgd,rhwd->rhgw', q, keys)], -1) * attn.scaling
    padded = torch.arange(width, device=q.device)[None] < cache.padding[:, None]  # (rows, width)
    scores[..., prefix_key.shape[1]:].masked_fill_(padded[:, None, None], float('-inf'))
    weights = scores.softmax(-1)
    shared = prefix_key.shape[1]
    o = (torch.einsum('rhgp,hpd->rhgd', weights[..., :shared], prefix_value)
         + torch.einsum('rhgw,rhwd->rhgd', weights[..., shared:], values))
    o = o.reshape(rows, -1, 1, attn.head_dim).to(h.dtype)  # (rows, heads, 1, head_dim)
    return _attention_output(attn, o, gate)


def _delta_inputs(mod: Any, h: torch.Tensor, mixed: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Queries, keys, values, decay and beta of convolved inputs ``mixed`` ``(tokens, channels)``."""
    q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    tokens = mixed.shape[0]
    q = q.reshape(1, tokens, -1, mod.head_k_dim)
    k = k.reshape(1, tokens, -1, mod.head_k_dim)
    v = v.reshape(1, tokens, -1, mod.head_v_dim)
    beta = mod.in_proj_b(h).sigmoid()[None]
    g = (-mod.A_log.float().exp() * F.softplus(mod.in_proj_a(h).float() + mod.dt_bias))[None]
    repeat = mod.num_v_heads // mod.num_k_heads
    if repeat > 1:
        q, k = q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)
    return q, k, v, g, beta


def _delta_output(mod: Any, h: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """Gated norm of the recurrence output ``(1, tokens, heads, value_dim)`` → ``(tokens, value_dim_total)``."""
    z = mod.in_proj_z(h).reshape(-1, mod.head_v_dim)
    return mod.norm(o.reshape(-1, mod.head_v_dim), z).reshape(h.shape[0], -1)


def _delta_mix(mod: Any, h: torch.Tensor, qkv: torch.Tensor, lengths: Sequence[int], context: torch.Tensor,
               initial: torch.Tensor | None,
               bounds: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Gated DeltaNet token mixer over packed segments that all continue one context.

    Every segment is convolved as if ``context`` (the previous
    ``kernel - 1`` inputs) preceded it, and its recurrence starts from
    ``initial`` (``None``: zeros).

    Args:
        mod: The ``Qwen3_5GatedDeltaNet`` module.
        h: Normed inputs ``(tokens, hidden)``.
        qkv: ``mod.in_proj_qkv(h)`` ``(tokens, channels)``, before the convolution.
        lengths: Segment lengths (sum ``tokens``).
        context: ``(kernel - 1, channels)`` inputs before every segment.
        initial: Recurrent state ``(heads, key_dim, value_dim)`` every segment starts from.
        bounds: Cumulative ``[0, *lengths]``, if already computed.

    Returns:
        Mixer output before ``out_proj`` ``(tokens, value_dim_total)``, and
        per segment the convolution window to continue from
        ``(segments, channels, kernel - 1)`` and the final recurrent state.
    """
    chunk_rule, _ = _fla()
    device, n = qkv.device, len(lengths)
    pad = context.shape[0]
    if bounds is None:
        bounds = torch.tensor([0, *lengths], device=device).cumsum(0)
    # Source rows: the context, then the tokens; each segment reads [context, own tokens].
    source = torch.cat([context.to(qkv.dtype), qkv])
    sizes = torch.tensor(lengths, device=device)
    segment = torch.repeat_interleave(torch.arange(n, device=device), sizes + pad)
    offset = torch.arange(len(segment), device=device) - torch.repeat_interleave(bounds[:-1] + pad * torch.arange(
        n, device=device), sizes + pad)
    token = bounds[:-1][segment] + offset - pad
    index = torch.where(offset < pad, offset, token + pad)
    extended = source[index]  # (tokens + segments * pad, channels)
    weight = mod.conv1d.weight
    conv = F.conv1d(extended.T[None].to(weight.dtype), weight, groups=extended.shape[1])[0].T  # valid
    # Output row r is the window ending at extended row r + pad.
    real = torch.nonzero(offset >= pad)[:, 0]
    mixed = F.silu(conv[real - pad]).to(qkv.dtype)
    ends = torch.cumsum(sizes + pad, 0)
    tails = torch.stack([extended[e - pad:e] for e in ends.tolist()]).transpose(1, 2)  # (n, channels, pad)
    q, k, v, g, beta = _delta_inputs(mod, h, mixed)
    start = None if initial is None else initial[None].expand(n, *initial.shape).contiguous()
    o, final = chunk_rule(q, k, v, g=g, beta=beta, initial_state=start, output_final_state=True,
                          use_qk_l2norm_in_kernel=True, cu_seqlens=bounds)
    return _delta_output(mod, h, o), (tails, final)


def _delta_step(mod: Any, h: torch.Tensor, cache: HybridCache, layer: int) -> torch.Tensor:
    """Gated DeltaNet of one token per row ``(rows, hidden)``; advances the row states."""
    _, recurrent_rule = _fla()
    qkv = mod.in_proj_qkv(h)
    window = torch.cat([cache.conv[layer], qkv[:, :, None]], -1)  # (rows, channels, kernel)
    weight = mod.conv1d.weight
    mixed = F.silu(F.conv1d(window.to(weight.dtype), weight, groups=window.shape[1])[:, :, 0]).to(qkv.dtype)
    cache.conv[layer] = window[:, :, 1:]
    rows = h.shape[0]
    q, k, v = torch.split(mixed, [mod.key_dim, mod.key_dim, mod.value_dim], dim=-1)
    q = q.reshape(rows, 1, -1, mod.head_k_dim)
    k = k.reshape(rows, 1, -1, mod.head_k_dim)
    v = v.reshape(rows, 1, -1, mod.head_v_dim)
    beta = mod.in_proj_b(h).sigmoid()[:, None]
    g = (-mod.A_log.float().exp() * F.softplus(mod.in_proj_a(h).float() + mod.dt_bias))[:, None]
    repeat = mod.num_v_heads // mod.num_k_heads
    if repeat > 1:
        q, k = q.repeat_interleave(repeat, dim=2), k.repeat_interleave(repeat, dim=2)
    o, cache.recurrent[layer] = recurrent_rule(q, k, v, g=g, beta=beta, initial_state=cache.recurrent[layer],
                                               output_final_state=True, use_qk_l2norm_in_kernel=True)
    z = mod.in_proj_z(h).reshape(-1, mod.head_v_dim)
    out = mod.norm(o.reshape(-1, mod.head_v_dim), z).reshape(rows, -1)
    return mod.out_proj(out)
