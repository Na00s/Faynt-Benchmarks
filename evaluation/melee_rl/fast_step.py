"""Vectorised per-frame cached policy step (P7-PLAN.md 3): Ali's math without the per-row loops.

``MeleePolicy``'s cached inference walks the batch in Python twice per layer per frame -- the ring
write (``model.py:1138-1147``) and the chronological gather (``model.py:343-352``) -- costing
``2 * n_layers * B`` iterations, ``12 * B / n_layers``-ish host syncs and a fresh capacity-sized
allocation per layer per frame.  This module reimplements only that orchestration, vectorised, and
reuses every parameterised piece of Ali's modules unchanged (weights stay live through in-place
optimizer updates):

* the write becomes one advanced-index assignment per layer (``keys[rows, write_position] = k``);
* the chronological rearrangement is dropped entirely: attention is a set operation over (K, V)
  pairs and RoPE phases are baked into K at write time (``_project``, ``model.py:1044``), so the
  ring storage is attended *in ring order* through a derived ``[B, capacity]`` validity mask
  (slot ``j`` is valid iff ``(j - start) mod capacity < length`` with
  ``start = (write - length) mod capacity`` -- exactly the index set of ``model.py:347-348``);
* a row reset clears only the metadata (``masked_fill_``); stale K/V at invalid slots are
  unobservable behind the mask (and Ali's own reader gathers valid slots only), which is proven by
  ``tests/rl/test_fast_step.py::test_fast_step_ignores_stale_invalid_slots``;
* the metadata commit copies ``model.py:1148-1156`` / ``:1539-1552`` verbatim with the step path's
  all-ones row-valid mask (``model.py:1650``);
* the head's sampling loop mirrors ``_predict``'s sample branch (``model.py:1767-1789``) and skips
  the ``codec.decode`` that ``StepOutput`` discards (``model.py:1832``, ``adapter.py:189``).

Numeric contract (pinned by ``tests/rl/test_fast_step.py``): logits match the slow path within
rtol 1e-4 / atol 1e-5, cache K/V at valid slots within rtol 1e-5 / atol 1e-6, metadata exactly;
sampling consumes the caller's ``torch.Generator`` exactly like the slow path.  The fast core issues
no host syncs and no data-dependent branches (the left-padded ``prime`` path stays on Ali's loop --
it runs at rollout boundaries only, and never in ring mode).

``attention = "lean"`` (5 Sep 2026, ``/rl-speedup`` lever 1): Ali's ``_sdpa`` with ``enable_gqa=True`` on fp32
inputs takes SDPA's math path, which expands the keys and values to every query head
(``repeat_interleave``), fills the fresh copies (deterministic mode), scales the expanded keys and runs
the ``M = 1`` products as batched GEMVs -- 44 of the 52 ms of GPU time per actor frame on the bench of
5 Sep 2026.  The lean path computes ``softmax(q k^T / sqrt(d) + mask) v`` per kv head with batched
matmuls straight over the ring (no expansion, no copies): the same formula and mask, different kernels,
so it is held to the fast step's own parity contract (``tests/rl/test_fast_attention.py``).
"""

from __future__ import annotations

import math
from typing import Final, cast

import torch

from controller_codec import ControllerLabels
from model import (
    AutoregressiveControllerHead,
    CausalTransformer,
    GroupedQueryCausalAttention,
    KVCache,
    TransformerBlock,
    _AutoregressiveComponent,
)

FAST_ATTENTIONS: Final[tuple[str, ...]] = ("sdpa", "lean")


class FastStepper:
    """One (backbone, cache) binding: vectorised ``backbone.step`` semantics for the actor.

    ``attention`` selects the cached-step attention: ``"sdpa"`` (Ali's ``_sdpa``) or ``"lean"`` (the
    copy-free per-kv-head matmuls, see the module docstring).
    """

    def __init__(self, backbone: CausalTransformer, cache: KVCache, *, attention: str = "sdpa") -> None:
        if attention not in FAST_ATTENTIONS:
            raise ValueError(f"attention must be one of {FAST_ATTENTIONS}, got {attention!r}")
        config = backbone.config
        expected = (
            config.n_layers,
            None,  # batch: taken from the cache
            config.context_length,
            config.n_kv_heads,
            config.head_dim,
        )
        shape = tuple(cache.keys.shape)
        if tuple(cache.values.shape) != shape or len(shape) != 5:
            raise ValueError("cache K/V shapes must match and have 5 dimensions")
        for axis, want in enumerate(expected):
            if want is not None and shape[axis] != want:
                raise ValueError(
                    f"cache shape {shape} does not fit the model "
                    f"(n_layers {config.n_layers}, context_length {config.context_length}, "
                    f"n_kv_heads {config.n_kv_heads}, head_dim {config.head_dim})"
                )
        if not cache.keys.is_floating_point() or not cache.values.is_floating_point():
            raise TypeError("cache keys and values must have floating dtype")
        device = cache.keys.device
        for name in ("valid_length", "write_position", "next_position"):
            tensor = cast(torch.Tensor, getattr(cache, name))
            if tensor.shape != (cache.batch_size,) or tensor.device != device:
                raise ValueError(f"cache {name} must have shape [B] on the cache device")
            if tensor.dtype != torch.long:
                raise TypeError(f"cache {name} must have torch.long dtype")
        self.backbone = backbone
        self.cache = cache
        self.attention = attention
        self.batch = cache.batch_size
        self.capacity = cache.capacity
        self._layers = [cast(TransformerBlock, layer) for layer in backbone.layers]
        self._rows = torch.arange(self.batch, device=device)
        self._slots = torch.arange(self.capacity, device=device)
        self._full = torch.full((self.batch,), self.capacity, dtype=torch.long, device=device)

    def step(self, encoded: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        """One frame for every row: hidden ``[B, d_model]``; the cache is advanced in place.

        The order of operations mirrors ``_cached_frame`` (``model.py:1497-1553``) with the step
        path's all-ones row validity; the post-write metadata is derived first (Ali attends with the
        post-write lengths, ``model.py:1148-1156``) and committed after the layers.
        """

        backbone = self.backbone
        cache = self.cache
        if encoded.shape[0] != self.batch:
            raise ValueError(f"encoded batch {encoded.shape[0]} does not match the cache {self.batch}")
        # Row resets: metadata only (stale K/V stay masked out; Ali zeroes them, model.py:293-297).
        cache.valid_length.masked_fill_(reset, 0)
        cache.write_position.masked_fill_(reset, 0)
        cache.next_position.masked_fill_(reset, 0)
        positions = cache.next_position[:, None]
        write = cache.write_position
        length_post = torch.minimum(cache.valid_length + 1, self._full)
        write_post = torch.remainder(cache.write_position + 1, self.capacity)
        start = torch.remainder(write_post - length_post, self.capacity)
        slot_valid = (
            torch.remainder(self._slots[None, :] - start[:, None], self.capacity) < length_post[:, None]
        )
        mask = slot_valid[:, None, None, :]
        bias: torch.Tensor | None = None
        if self.attention == "lean":
            # The additive mask SDPA's math path builds per layer: 0 at valid slots, -inf elsewhere.
            bias = torch.zeros((self.batch, 1, self.capacity), dtype=encoded.dtype, device=encoded.device)
            bias = bias.masked_fill_(~slot_valid[:, None, :], float("-inf"))
        projected = backbone.input_projection(encoded).unsqueeze(1)
        if backbone.config.residual_mode == "standard":
            hidden = projected
            for index, layer in enumerate(self._layers):
                attention_output = self._attend(
                    index, layer, layer.attention_norm(hidden), positions, write, mask, bias
                )
                hidden = hidden + layer.attention_dropout(attention_output)
                hidden = hidden + CausalTransformer._ffn_transform(layer, hidden)
        else:
            values: list[torch.Tensor] = [projected]
            for index, layer in enumerate(self._layers):
                if layer.attention_residual is None or layer.ffn_residual is None:
                    raise RuntimeError("full_attnres layer is missing its depth reads")
                attention_input = layer.attention_residual(values)
                attention_output = self._attend(
                    index, layer, layer.attention_norm(attention_input), positions, write, mask, bias
                )
                values.append(layer.attention_dropout(attention_output))
                ffn_input = layer.ffn_residual(values)
                values.append(CausalTransformer._ffn_transform(layer, ffn_input))
            if backbone.output_residual is None:
                raise RuntimeError("full_attnres backbone is missing its final depth read")
            hidden = backbone.output_residual(values)
        hidden = backbone.final_norm(hidden).squeeze(1)
        cache.valid_length.copy_(length_post)
        cache.write_position.copy_(write_post)
        cache.next_position.add_(1)
        return hidden

    def _attend(
        self,
        layer_index: int,
        layer: TransformerBlock,
        normalized: torch.Tensor,
        positions: torch.Tensor,
        write: torch.Tensor,
        mask: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One layer: project, write the ring slot (vectorised), attend the ring behind the mask."""

        attention = cast(GroupedQueryCausalAttention, layer.attention)
        queries, keys, values = attention._project(normalized, positions)
        layer_keys = self.cache.keys[layer_index]
        layer_values = self.cache.values[layer_index]
        layer_keys[self._rows, write] = keys[:, :, 0].detach().to(layer_keys.dtype)
        layer_values[self._rows, write] = values[:, :, 0].detach().to(layer_values.dtype)
        if bias is not None:
            head_output = self._lean_attention(queries, layer_keys, layer_values, bias)
        else:
            ring_keys = layer_keys.permute(0, 2, 1, 3).to(queries.dtype)
            ring_values = layer_values.permute(0, 2, 1, 3).to(queries.dtype)
            head_output = attention._sdpa(
                queries, ring_keys, ring_values, attention_mask=mask, is_causal=False
            )
        return attention._apply_output_gate(head_output, normalized)

    @staticmethod
    def _lean_attention(
        queries: torch.Tensor, layer_keys: torch.Tensor, layer_values: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        """``softmax(q k^T / sqrt(d) + bias) v`` per kv head over the ring, no expansion, no copies.

        ``queries`` ``[B, H, 1, Dh]`` (query head ``h`` reads kv head ``h // groups``, the order
        ``repeat_interleave`` gives), ``layer_keys`` / ``layer_values`` the ring ``[B, C, K, Dh]``, ``bias``
        ``[B, 1, C]``; returns ``[B, H, 1, Dh]``.  The kv-head slices of the ring are BLAS-compatible views
        (row-major with leading dimension ``K * Dh``), so the batched matmuls read the ring in place.
        """

        batch, heads, _, head_dim = queries.shape
        kv_heads = layer_keys.shape[2]
        groups = heads // kv_heads
        dtype = queries.dtype
        grouped = queries[:, :, 0, :].reshape(batch, kv_heads, groups, head_dim)
        additive = bias.to(dtype)
        scale = 1.0 / math.sqrt(head_dim)
        outputs = []
        for kv in range(kv_heads):
            keys = layer_keys[:, :, kv, :].to(dtype)
            values = layer_values[:, :, kv, :].to(dtype)
            scores = torch.baddbmm(additive, grouped[:, kv], keys.transpose(1, 2), alpha=scale)
            probabilities = torch.softmax(scores, dim=-1)
            outputs.append(torch.bmm(probabilities, values))
        return torch.stack(outputs, dim=1).view(batch, heads, 1, head_dim)


def fast_sample(
    head: AutoregressiveControllerHead,
    hidden: torch.Tensor,
    previous: ControllerLabels,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[ControllerLabels, dict[str, torch.Tensor]]:
    """``_predict``'s sample branch (``model.py:1767-1789``), skipping the discarded decode.

    Returns ``(labels, float32 raw logits by component)``; the generator is consumed exactly as the
    slow path consumes it (one ``multinomial`` per component on the same shapes).
    """

    residual = head.to_residual(hidden)
    previous_values = previous.as_dict()
    logits: dict[str, torch.Tensor] = {}
    selected: dict[str, torch.Tensor] = {}
    for name in head.codec.component_order:
        component = cast(_AutoregressiveComponent, head.components[name])
        component_logits = component.logits(residual, previous_values[name])
        logits[name] = component_logits.float()
        if temperature > 0.0:
            probabilities = torch.softmax(component_logits.float() / temperature, dim=-1)
            flat = probabilities.reshape(-1, probabilities.shape[-1])
            choice = torch.multinomial(flat, 1, generator=generator).reshape(component_logits.shape[:-1])
        else:
            choice = component_logits.float().argmax(dim=-1)
        selected[name] = choice
        residual = component.update(residual, choice)
    return ControllerLabels(buttons=selected["buttons"], main_stick=selected["main_stick"]), logits


__all__ = ["FAST_ATTENTIONS", "FastStepper", "fast_sample"]
