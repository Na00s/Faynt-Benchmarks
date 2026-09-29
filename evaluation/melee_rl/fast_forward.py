"""The lean learner forward (5 Sep 2026, ``/rl-learner-speedup`` lever 1; ``learner.fast_forward``).

``MeleePolicy.forward(..., prefix=)`` (``model.py:2032-2067`` -> ``CausalTransformer.forward`` ``:1642-1750``
-> ``_parallel`` ``:1514-1582``) is what the PPO learner runs three times per chunk (two training epochs and
the post pass) plus once as the frozen teacher.  On the 75m profile of 5 Sep 2026 that pass is 92 % GPU-busy
and 57 % of its GPU time is elementwise traffic from ``FullAttentionResidual`` (``model.py:1226-1255``:
``2L + 1`` depth reads per forward, each stacking ``S = r + 1`` fp32 ``[B, T, D]`` slabs, normalising the
stack, scoring it, and mixing it through a broadcast multiply), another 5 % the ``Memcpy DtoD`` of SDPA's GQA
math path, and ≈ 23 host round trips per pass (label range checks, ``codec.decode``, the cache checks, the
dynamic prefix length ``int(valid_length.max())``).

This module re-orchestrates the same forward with Ali's modules and weights (the ``fast_encode`` /
``fast_step`` pattern: nothing in ``model.py`` changes, parameters stay live through in-place optimizer
updates) for the learner's prefix window only:

* **exact parts** (``torch.equal`` with Ali's, ``tests/rl/test_fast_forward.py``): the encoder
  (``melee_rl.fast_encode``), the RoPE positions (:func:`positions`, a segmented cumulative sum instead of
  the Python loop over ``T``, ``model.py:1431-1434``), the attention mask (:func:`prefix_mask`, Ali's
  ``_prefix_attention_mask`` at a **static prefix length** ``P = context_length``: the arranged prefix is
  zero-padded and ``prefix_valid`` masks the pad, so every pass of a run has one shape and no host read
  decides it), one RoPE table per pass shared by the layers (:func:`rope_table`; Ali rebuilds it per layer),
  the arranged prefix (:func:`arrange_prefix` = ``KVCache.chronological_all`` without the ``int(max)``), and
  the head's teacher-forced logits without the discarded ``codec.decode``;
* **re-associated parts** (the same fp32 math through different kernels; measured by the learner's
  ``loss/actor_kl_pre`` and ``pre/teacher_kl`` at unchanged weights, bar 1e-6 nats): the depth reads
  (:class:`DepthSources`) and the attention (:func:`grouped_attention`), documented there.

Padded query rows (``padding_mask`` false; absent from production prefix rollouts, the actor pads nothing
there) get a self-only mask entry instead of Ali's fully masked row, so no attention kernel ever sees an empty
row; their hidden states are finite and never read: valid rows mask them as keys, the depth reads and the
FFN are per position, and the final ``where(valid, hidden, 0)`` zeroes them as ``model.py:1582`` does.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final, cast

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from controller_codec import ControllerLabels
from melee_rl.fast_encode import fast_encode
from melee_rl.frames import tree_leaves
from melee_rl.kv_cache import cache_to
from model import (
    CausalTransformer,
    FullAttentionResidual,
    GroupedQueryCausalAttention,
    KVCache,
    MeleePolicy,
    RotaryEmbedding,
    TransformerBlock,
    _AutoregressiveComponent,
)

ATTENTION_MODES: Final[tuple[str, ...]] = ("sdpa", "bmm")
COMPILE_BACKENDS: Final[tuple[str, ...]] = ("inductor", "aot_eager")
"""``torch.compile`` backends the lean forward accepts: ``inductor`` (the GPU runs) and ``aot_eager``
(dynamo + AOTAutograd without code generation -- the CPU tests' proof that the pass traces as one graph)."""


def positions(reset: torch.Tensor, valid: torch.Tensor, start: torch.Tensor | None = None) -> torch.Tensor:
    """RoPE positions ``[B, T]``: ``CausalTransformer._positions`` (``model.py:1400-1435``) without the loop.

    0 at a reset, +1 per valid frame, counting from ``start`` (``[B]``, a prefix cache's ``next_position``)
    in the first segment.  ``before[t]`` counts the valid frames strictly before ``t``; since it never
    decreases, the running maximum of its values at the resets is its value at the most recent reset, and the
    difference restarts the count there.  Integer arithmetic: bit-identical to the loop.
    """

    reset = reset.to(dtype=torch.bool)
    valid_long = valid.to(dtype=torch.long)
    before = torch.cumsum(valid_long, dim=1) - valid_long
    latest = torch.cummax(torch.where(reset, before, 0), dim=1).values
    result = before - latest
    if start is not None:
        segment = torch.cumsum(reset.long(), dim=1)
        offset = start.to(device=result.device, dtype=torch.long)[:, None]
        result = result + torch.where(segment == 0, offset, 0)
    return result


def arrange_prefix(cache: KVCache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``KVCache.chronological_all`` (``model.py:309-329``) at the static ``P = capacity``, no host read.

    Returns detached, contiguous ``keys`` and ``values`` ``[L, B, K, P, Dh]`` (the attention layout,
    ``model.py:1733``), ``valid [B, P]`` and the RoPE ``positions [B, P]`` of every slot, oldest to newest;
    slots beyond a row's ``valid_length`` are zero and invalid.  Equal to Ali's on the valid slots.
    """

    _layers, batch, capacity, _, _ = cache.keys.shape
    device = cache.keys.device
    offsets = torch.arange(capacity, device=device)
    start = torch.remainder(cache.write_position - cache.valid_length, capacity)
    slots = torch.remainder(start[:, None] + offsets[None, :], capacity)
    rows = torch.arange(batch, device=device)[:, None]
    valid = offsets[None, :] < cache.valid_length[:, None]
    keep = valid[None, :, :, None, None]
    keys = torch.where(keep, cache.keys[:, rows, slots], 0.0).detach()
    values = torch.where(keep, cache.values[:, rows, slots], 0.0).detach()
    slot_positions = cache.next_position[:, None] - cache.valid_length[:, None] + offsets[None, :]
    return (
        keys.permute(0, 1, 3, 2, 4).contiguous(),
        values.permute(0, 1, 3, 2, 4).contiguous(),
        valid,
        slot_positions,
    )


def prefix_mask(
    query_positions: torch.Tensor,
    reset: torch.Tensor,
    valid: torch.Tensor,
    prefix_valid: torch.Tensor,
    prefix_positions: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """``CausalTransformer._prefix_attention_mask`` (``model.py:1463-1496``) plus a self-only entry for
    padded rows.

    ``[B, 1, T, P + T]`` bool: a row sees a key of its own segment (the prefix is segment 0) that is not
    later, valid on both sides and within ``window`` positions.  A padded query row, fully masked in Ali's
    version, sees exactly itself here (see the module docstring).
    """

    batch, time = reset.shape
    prefix_length = prefix_valid.shape[1]
    device = reset.device
    segment = torch.cumsum(reset.long(), dim=1)
    prefix_segment = torch.zeros((batch, prefix_length), dtype=torch.long, device=device)
    same_segment = segment[:, :, None] == torch.cat((prefix_segment, segment), dim=1)[:, None, :]
    causal = torch.cat(
        (
            torch.ones((time, prefix_length), dtype=torch.bool, device=device),
            torch.ones((time, time), dtype=torch.bool, device=device).tril(),
        ),
        dim=1,
    )
    key_valid = torch.cat((prefix_valid, valid), dim=1)
    key_positions = torch.cat((prefix_positions, query_positions), dim=1)
    in_window = (query_positions[:, :, None] - key_positions[:, None, :]) < window
    allowed = same_segment & causal.unsqueeze(0) & valid[:, :, None] & key_valid[:, None, :] & in_window
    diagonal = torch.cat(
        (
            torch.zeros((time, prefix_length), dtype=torch.bool, device=device),
            torch.eye(time, dtype=torch.bool, device=device),
        ),
        dim=1,
    )
    self_only = (~valid)[:, :, None] & diagonal.unsqueeze(0)
    return (allowed | self_only).unsqueeze(1)


def rope_table(
    rope: RotaryEmbedding, query_positions: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(cos, sin)`` ``[B, 1, T, Dh]`` tables of ``RotaryEmbedding.apply_rotary`` (``model.py:426-429``),
    computed once per pass; every layer's module carries the same ``inv_freq``."""

    frequencies = query_positions.float().unsqueeze(-1) * rope.inv_freq.float()
    embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rope(
    queries: torch.Tensor, keys: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``model.py:430-431`` with a precomputed table (bit-identical)."""

    rotate = RotaryEmbedding._rotate_half
    return queries * cosine + rotate(queries) * sine, keys * cosine + rotate(keys) * sine


def _unflatten(paths: tuple[str, ...], leaves: tuple[torch.Tensor, ...]) -> dict[str, Any]:
    """The frame tree back from ``tree_leaves``' dotted paths and leaves."""

    tree: dict[str, Any] = {}
    for path, leaf in zip(paths, leaves, strict=True):
        *parents, name = path.split(".")
        node = tree
        for parent in parents:
            node = node.setdefault(parent, {})
        node[name] = leaf
    return tree


def depth_reads(backbone: CausalTransformer) -> list[FullAttentionResidual]:
    """The ``2L + 1`` depth reads in forward order: per layer its attention and FFN read, then the output
    read."""

    reads: list[FullAttentionResidual] = []
    for layer in backbone.layers:
        block = cast(TransformerBlock, layer)
        if block.attention_residual is None or block.ffn_residual is None:
            raise RuntimeError("full_attnres layer is missing its depth reads")
        reads.extend((block.attention_residual, block.ffn_residual))
    if backbone.output_residual is None:
        raise RuntimeError("full_attnres backbone is missing its final depth read")
    reads.append(backbone.output_residual)
    return reads


class DepthSources:
    """One pass's depth sources and their ``FullAttentionResidual`` reads, traffic-optimal.

    Ali's read ``r`` over sources ``v_0 .. v_{S-1}`` (``model.py:1240-1255``) stacks them ``[S, B, T, D]``,
    RMS-normalises the stack, scores it against ``depth_query_r`` (``<q_r, w_r * v_s * rms_s>``), softmaxes
    over ``S`` and mixes the **raw** sources with the weights -- ``Θ(L²)`` slab-sized intermediates per
    forward (the stack, the normalised stack, the broadcast multiply of the mix).  Here
    ``rms_s = rsqrt(mean(v_s²) + eps)`` is computed once per source (the same ops as Ali's RMSNorm,
    ``model.py:389-390``), its routing dots against **every** read's query ``q_r * w_r`` come from one
    ``[B·T, D] x [D, R]`` GEMM when the source is produced (``<q_r, w_r * v * rms> = rms * <q_r * w_r, v>``),
    and each read is a softmax over its ``[B, T, S]`` scores plus one ``[B·T, 1, S] x [B·T, S, D]`` batched
    matmul over ``torch.stack`` of its sources.  The same fp32 math, re-associated (the dot products and the
    mix reduce in a different order): held to Ali's backbone tolerances in ``tests/rl/test_fast_forward.py``
    and measured end to end by ``loss/actor_kl_pre``.
    """

    def __init__(self, reads: Sequence[FullAttentionResidual], eps: float) -> None:
        self.eps = eps
        # ``[R, D]``: every read's pseudo-query with its routing weight folded in.
        self.queries = torch.stack(
            [read.depth_query.float() * read.routing_norm.weight.float() for read in reads]
        )
        self.values: list[torch.Tensor] = []
        self._rms: list[torch.Tensor] = []
        self._dots: list[torch.Tensor] = []

    def append(self, value: torch.Tensor) -> None:
        """Register the next source ``[B, T, D]`` (fp32 in the RL precision)."""

        source = value.float()
        self.values.append(value)
        self._rms.append(torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + self.eps))
        self._dots.append(source @ self.queries.T)

    def read(self, index: int) -> torch.Tensor:
        """Read ``index`` over every source registered so far: ``FullAttentionResidual.forward(values)``."""

        count = len(self.values)
        if count == 0:
            raise ValueError("a depth read needs at least one source")
        rms = torch.cat(self._rms, dim=-1)
        dots = torch.stack([dot[..., index] for dot in self._dots], dim=-1)
        weights = torch.softmax(rms * dots, dim=-1)
        stacked = torch.stack(self.values, dim=2).float()
        batch, time, _, width = stacked.shape
        mixed = torch.bmm(weights.view(batch * time, 1, count), stacked.view(batch * time, count, width))
        return mixed.view(batch, time, width).to(self.values[0].dtype)


def grouped_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    mode: str = "sdpa",
) -> torch.Tensor:
    """``softmax(q kᵀ / √d + mask) v`` for GQA without expanding the keys and values to every query head.

    ``queries [B, H, T, Dh]``, ``keys`` / ``values [B, K, S, Dh]`` (the prefix slots then the window),
    ``mask [B, 1, T, S]`` bool (True = attend).  Query head ``h`` reads kv head ``h // G`` (the order Ali's
    ``repeat_interleave`` gives, ``model.py:1105``), so the ``G`` query heads of one kv head are folded into
    the row axis: ``[B, K, G·T, Dh]`` against ``[B, K, S, Dh]`` is plain multi-head attention.  ``"sdpa"``
    hands that to ``F.scaled_dot_product_attention`` **without** ``enable_gqa`` (fp32 + a bool mask +
    ``enable_gqa`` is SDPA's math path, ``model.py:1090-1098``, which copies K/V ``G`` times; without it a
    fused fp32 kernel is eligible); ``"bmm"`` is the ``fast_step._lean_attention`` pattern at ``M = G·T``
    (``baddbmm`` + fp32 softmax + ``bmm`` per kv head, deterministic).  The same formula as the math path
    through different kernels; measured.
    """

    if mode not in ATTENTION_MODES:
        raise ValueError(f"attention mode must be one of {ATTENTION_MODES}, got {mode!r}")
    batch, heads, time, head_dim = queries.shape
    kv_heads, length = keys.shape[1], keys.shape[2]
    groups = heads // kv_heads
    grouped = queries.reshape(batch, kv_heads, groups * time, head_dim)
    grouped_mask = mask.expand(batch, groups, time, length).reshape(batch, 1, groups * time, length)
    if mode == "sdpa":
        output = F.scaled_dot_product_attention(grouped, keys, values, attn_mask=grouped_mask)
        return output.reshape(batch, heads, time, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    bias = torch.zeros((batch, groups * time, length), dtype=queries.dtype, device=queries.device)
    bias = bias.masked_fill_(~grouped_mask[:, 0], float("-inf"))
    outputs = []
    for kv in range(kv_heads):
        scores = torch.baddbmm(bias, grouped[:, kv], keys[:, kv].transpose(1, 2), alpha=scale)
        outputs.append(torch.bmm(torch.softmax(scores, dim=-1), values[:, kv]))
    return torch.stack(outputs, dim=1).view(batch, heads, time, head_dim)


class CacheStager:
    """Moves a host-resident ``KVCache`` slice to a CUDA device without stalling the CPU (``prefix_on_host``).

    Today's path (``cache_to`` on the strided ``[L, b, C, K, Dh]`` slice of a trajectory's prefix) makes a
    pageable contiguous copy and a synchronous H2D transfer per pass: ≈ 25 ms of CPU stall per 75m chunk
    pass, ≈ 15 s per PPO step (the profile of 5 Sep 2026).  Here the slice is copied into one of two
    **pinned** staging buffers and transferred with ``non_blocking=True``; a CUDA event recorded after each
    transfer is waited on before that buffer is written again, so the staging is safe however far the CPU
    runs ahead.  On a CPU device it is ``cache_to``.  The metadata rides in one small pinned buffer.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._buffers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = [None, None]
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._turn = 0

    def stage(self, cache: KVCache) -> KVCache:
        if self.device.type != "cuda" or cache.keys.device == self.device:
            return cache_to(cache, self.device)
        slot, self._turn = self._turn, 1 - self._turn
        event = self._events[slot]
        if event is not None:
            event.synchronize()
        buffers = self._buffers[slot]
        if buffers is None or buffers[0].shape != cache.keys.shape:
            buffers = (
                torch.empty_like(cache.keys, pin_memory=True),
                torch.empty_like(cache.values, pin_memory=True),
                torch.empty((3, cache.batch_size), dtype=torch.long, pin_memory=True),
            )
            self._buffers[slot] = buffers
        keys_buffer, values_buffer, meta_buffer = buffers
        keys_buffer.copy_(cache.keys)
        values_buffer.copy_(cache.values)
        meta_buffer[0].copy_(cache.valid_length)
        meta_buffer[1].copy_(cache.write_position)
        meta_buffer[2].copy_(cache.next_position)
        keys = keys_buffer.to(self.device, non_blocking=True)
        values = values_buffer.to(self.device, non_blocking=True)
        meta = meta_buffer.to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._events[slot] = event
        return KVCache(
            keys=keys, values=values, valid_length=meta[0], write_position=meta[1], next_position=meta[2]
        )


class FastForward:
    """The learner's ``unroll_window(..., prefix=)`` for one ``MeleePolicy``: Ali's modules, our
    orchestration.

    Bound once per model (the policy, the frozen teacher); every call is one prefix-window pass (see the
    module docstring for what is exact and what is re-associated).  Refuses the configurations it does not
    cover: dropout, a non-fp32 compute dtype, player-name conditioning and the controller RNN
    (``fast_encode``).  ``attention`` selects :func:`grouped_attention`'s mode.

    ``compile`` (lever 2, ``learner.compile``) wraps the whole pass -- encoder, backbone, head -- in
    ``torch.compile(backend, dynamic=False, fullgraph=True)`` over flat tensors (the frame tree's leaves, the
    labels, the masks, the raw ring and its metadata): one graph per grad mode at the run's fixed chunk shape
    (chunk rows x T x context_length; ``finalize_config`` guarantees uniform chunks). The inputs go in
    contiguous, so strides cannot tell the arms apart, and a recompile can only come from a new shape.  It
    needs ``gradient_checkpointing`` off (non-reentrant checkpointing under dynamo stacks a second recompute
    policy on the partitioner's; the lean pass fits the L4 without it).  Inductor's fusion changes fp32
    rounding: measured like everything else.
    """

    def __init__(self, model: MeleePolicy, *, attention: str = "sdpa", compile: str | None = None) -> None:
        if attention not in ATTENTION_MODES:
            raise ValueError(f"attention must be one of {ATTENTION_MODES}, got {attention!r}")
        if compile is not None and compile not in COMPILE_BACKENDS:
            raise ValueError(f"compile must be one of {COMPILE_BACKENDS} or None, got {compile!r}")
        config = model.config
        if compile is not None and config.gradient_checkpointing:
            raise ValueError("the compiled lean forward needs gradient_checkpointing = false")
        if config.compute_dtype != "float32":
            raise NotImplementedError("the lean forward covers the RL precision (compute_dtype float32) only")
        if config.attention_dropout != 0.0 or config.residual_dropout != 0.0:
            raise NotImplementedError("the lean forward covers dropout-free models only")
        if config.condition_on_player_name or model.encoder.controller_rnn is not None:
            raise NotImplementedError(
                "the lean forward covers neither player-name conditioning nor the controller RNN"
            )
        self.model = model
        self.backbone = model.backbone
        self.attention = attention
        self.layers = [cast(TransformerBlock, layer) for layer in self.backbone.layers]
        self.reads = depth_reads(self.backbone) if config.residual_mode == "full_attnres" else None
        first = cast(GroupedQueryCausalAttention, self.layers[0].attention)
        self.rope = first.rope
        self.compile_backend = compile
        self._paths: tuple[str, ...] | None = None
        self._compiled: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None
        if compile is not None:
            self._compiled = torch.compile(self._flat_logits, backend=compile, dynamic=False, fullgraph=True)

    def logits(
        self,
        frames: Any,
        controller_prev: ControllerLabels,
        targets: ControllerLabels,
        reset: torch.Tensor,
        valid: torch.Tensor,
        cache: KVCache,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced fp32 logits ``{component: [B, T, V]}`` of the rows against the stored cache."""

        previous = ControllerLabels(controller_prev.buttons.long(), controller_prev.main_stick.long())
        taken = ControllerLabels(targets.buttons.long(), targets.main_stick.long())
        reset = reset.to(dtype=torch.bool)
        valid = valid.to(dtype=torch.bool)
        if self._compiled is not None:
            paths, leaves = zip(*tree_leaves(frames), strict=True)
            if self._paths is None:
                self._paths = tuple(paths)
            elif self._paths != tuple(paths):
                raise ValueError("the compiled lean forward saw a frame tree with a different structure")
            # Contiguous inputs (17 Sep 2026): ``dynamic=False`` guards strides, and a chunk of a prefix cache
            # (``keys[:, rows]``) carries its parent's row count in them, so arms of different sizes each
            # compiled their own graphs (the league smoke hit dynamo's recompile limit at its first gradient
            # step). ``contiguous()`` is a no-op for a tensor that already is.
            buttons, main_stick = self._compiled(
                tuple(leaf.contiguous() for leaf in leaves),
                previous.buttons.contiguous(),
                previous.main_stick.contiguous(),
                taken.buttons.contiguous(),
                taken.main_stick.contiguous(),
                reset.contiguous(),
                valid.contiguous(),
                cache.keys.contiguous(),
                cache.values.contiguous(),
                cache.valid_length.contiguous(),
                cache.write_position.contiguous(),
                cache.next_position.contiguous(),
            )
            return {"buttons": buttons, "main_stick": main_stick}
        encoded = fast_encode(self.model.encoder, frames, previous)
        hidden = self.hidden(encoded, reset, valid, cache)
        return self.head_logits(hidden, previous, taken)

    def _flat_logits(
        self,
        leaves: tuple[torch.Tensor, ...],
        prev_buttons: torch.Tensor,
        prev_stick: torch.Tensor,
        target_buttons: torch.Tensor,
        target_stick: torch.Tensor,
        reset: torch.Tensor,
        valid: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_length: torch.Tensor,
        write_position: torch.Tensor,
        next_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The pass over flat tensors -- what ``torch.compile`` traces (the frame tree is rebuilt inside)."""

        assert self._paths is not None
        frames = _unflatten(self._paths, leaves)
        previous = ControllerLabels(prev_buttons, prev_stick)
        taken = ControllerLabels(target_buttons, target_stick)
        cache = KVCache(keys, values, valid_length, write_position, next_position)
        encoded = fast_encode(self.model.encoder, frames, previous)
        hidden = self.hidden(encoded, reset, valid, cache)
        logits = self.head_logits(hidden, previous, taken)
        return logits["buttons"], logits["main_stick"]

    def hidden(
        self, encoded: torch.Tensor, reset: torch.Tensor, valid: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """``CausalTransformer.forward(encoded, reset, valid, prefix=cache)`` over the prefix window."""

        backbone = self.backbone
        config = backbone.config
        if encoded.shape[1] > config.context_length:
            raise ValueError(
                f"window length {encoded.shape[1]} exceeds context_length {config.context_length}"
            )
        if cache.keys.device != encoded.device:
            raise ValueError("the prefix cache must live on the model's device (stage it first)")
        prefix_keys, prefix_values, prefix_valid, prefix_positions = arrange_prefix(cache)
        query_positions = positions(reset, valid, cache.next_position)
        mask = prefix_mask(
            query_positions, reset, valid, prefix_valid, prefix_positions, config.context_length
        )
        cosine, sine = rope_table(self.rope, query_positions, encoded.dtype)
        projected = backbone.input_projection(encoded)
        use_checkpoint = config.gradient_checkpointing and backbone.training and torch.is_grad_enabled()

        def attend(layer: TransformerBlock, index: int, inputs: torch.Tensor) -> torch.Tensor:
            if use_checkpoint:
                return checkpoint(
                    lambda value, current=layer, extra=index: self._attention(
                        current, value, cosine, sine, mask, prefix_keys[extra], prefix_values[extra]
                    ),
                    inputs,
                    use_reentrant=False,
                )
            return self._attention(
                layer, inputs, cosine, sine, mask, prefix_keys[index], prefix_values[index]
            )

        def feed_forward(layer: TransformerBlock, inputs: torch.Tensor) -> torch.Tensor:
            if use_checkpoint:
                return checkpoint(
                    lambda value, current=layer: current.feed_forward(current.ffn_norm(value)),
                    inputs,
                    use_reentrant=False,
                )
            return layer.feed_forward(layer.ffn_norm(inputs))

        if self.reads is None:
            hidden = projected
            for index, layer in enumerate(self.layers):
                hidden = hidden + attend(layer, index, hidden)
                hidden = hidden + feed_forward(layer, hidden)
        else:
            sources = DepthSources(self.reads, config.norm_eps)
            sources.append(projected)
            for index, layer in enumerate(self.layers):
                sources.append(attend(layer, index, sources.read(2 * index)))
                sources.append(feed_forward(layer, sources.read(2 * index + 1)))
            hidden = sources.read(2 * len(self.layers))
        hidden = backbone.final_norm(hidden)
        return torch.where(valid.unsqueeze(-1), hidden, 0.0)

    def _attention(
        self,
        layer: TransformerBlock,
        inputs: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        mask: torch.Tensor,
        prefix_keys: torch.Tensor,
        prefix_values: torch.Tensor,
    ) -> torch.Tensor:
        """``layer.attention(layer.attention_norm(inputs), positions, mask, prefix)`` (``model.py:1129-1157``)
        with the shared RoPE table, :func:`grouped_attention` and the output gate inlined (``:1116-1127``)."""

        attention = cast(GroupedQueryCausalAttention, layer.attention)
        normalized = layer.attention_norm(inputs)
        batch, time, _ = normalized.shape
        heads, kv_heads, head_dim = attention.n_heads, attention.n_kv_heads, attention.head_dim
        queries = attention.q_proj(normalized).view(batch, time, heads, head_dim).transpose(1, 2)
        keys = attention.k_proj(normalized).view(batch, time, kv_heads, head_dim).transpose(1, 2)
        values = attention.v_proj(normalized).view(batch, time, kv_heads, head_dim).transpose(1, 2)
        queries = attention.q_norm(queries)
        keys = attention.k_norm(keys)
        queries, keys = apply_rope(queries, keys, cosine, sine)
        keys = torch.cat((prefix_keys.to(keys.dtype), keys), dim=2)
        values = torch.cat((prefix_values.to(values.dtype), values), dim=2)
        head_output = grouped_attention(queries, keys, values, mask, mode=self.attention)
        gate = torch.sigmoid(attention.attention_gate(normalized).float())
        gate = gate.view(batch, time, heads, head_dim).transpose(1, 2)
        gated = head_output.float() * gate
        flattened = gated.transpose(1, 2).reshape(batch, time, -1).to(normalized.dtype)
        return attention.output_projection(flattened)

    def head_logits(
        self, hidden: torch.Tensor, previous: ControllerLabels, targets: ControllerLabels
    ) -> dict[str, torch.Tensor]:
        """``AutoregressiveControllerHead._predict`` under teacher forcing (``model.py:1892-1913``) without
        ``codec.decode``: the taken labels condition the later components, fp32 logits per component."""

        head = self.model.controller_head
        residual = head.to_residual(hidden)
        previous_values = previous.as_dict()
        target_values = targets.as_dict()
        logits: dict[str, torch.Tensor] = {}
        for name in head.codec.component_order:
            component = cast(_AutoregressiveComponent, head.components[name])
            logits[name] = component.logits(residual, previous_values[name]).float()
            residual = component.update(residual, target_values[name])
        return logits


__all__ = [
    "ATTENTION_MODES",
    "COMPILE_BACKENDS",
    "CacheStager",
    "DepthSources",
    "FastForward",
    "apply_rope",
    "arrange_prefix",
    "depth_reads",
    "grouped_attention",
    "positions",
    "prefix_mask",
    "rope_table",
]
