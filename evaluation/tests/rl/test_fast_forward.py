"""The lean learner forward (5 Sep 2026, ``/rl-learner-speedup`` lever 1).

``melee_rl.fast_forward`` re-orchestrates ``MeleePolicy.forward(..., prefix=)`` for the learner's prefix
window with Ali's modules and weights: the chunk constants (positions, mask, RoPE table, the arranged prefix)
are computed without host reads and with a static prefix length, the ``FullAttentionResidual`` depth reads
become one GEMM of routing dots per source plus one batched matmul per read, the attention runs per kv head
without the GQA expansion.  The exact parts are ``torch.equal`` with Ali's; the reformulated parts are held to
the prefix forward's own tolerances and the learner's exactness monitor (``loss/actor_kl_pre``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from melee_rl import distributions, fast_forward
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.kv_cache import cache_to
from melee_rl.trajectory import Trajectory
from model import CausalTransformer, GroupedQueryCausalAttention, KVCache, ModelConfig
from tests.rl.conftest import RolloutFactory
from tests.rl.test_prefix_forward import _backbone, _frames, _prime

B = 3


def _masks(seed: int, batch: int = B, time: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    reset = torch.rand((batch, time), generator=generator) < 0.2
    valid = torch.rand((batch, time), generator=generator) < 0.85
    reset[0, 0] = True  # a row that resets on its first frame ignores the prefix
    valid[1, 0] = False  # a padded first frame
    return reset, valid


def _cache_with_rows(backbone: CausalTransformer, history: int) -> KVCache:
    """A primed ring whose rows have different lengths (one row reset to a short history)."""

    cache = _prime(backbone, history)
    with torch.no_grad():
        cache.reset_slots(torch.tensor([False, True, False]))
        _, cache = backbone(_frames(3, 99), cache=cache, use_cache=True)
    return cache


@pytest.mark.parametrize("with_start", [False, True])
def test_positions_equal_alis_loop(with_start: bool) -> None:
    """The vectorised positions reproduce ``CausalTransformer._positions`` exactly (integers)."""

    for seed in range(4):
        reset, valid = _masks(seed)
        start = torch.tensor([0, 17, 300]) if with_start else None
        expected = CausalTransformer._positions(B, 11, torch.device("cpu"), reset, valid, start=start)
        actual = fast_forward.positions(reset, valid, start)
        assert torch.equal(actual, expected), seed
    # No resets and no padding: the arange (+ start) branch.
    reset = torch.zeros((B, 5), dtype=torch.bool)
    valid = torch.ones((B, 5), dtype=torch.bool)
    start = torch.tensor([3, 0, 9])
    assert torch.equal(
        fast_forward.positions(reset, valid, start),
        CausalTransformer._positions(B, 5, torch.device("cpu"), None, None, start=start),
    )
    assert torch.equal(
        fast_forward.positions(reset, valid, None),
        CausalTransformer._positions(B, 5, torch.device("cpu"), None, None),
    )


@pytest.mark.parametrize("history", [0, 10, 24, 40], ids=["empty", "partial", "full", "overflowed"])
def test_arranged_prefix_equals_chronological_all_on_valid_slots(history: int) -> None:
    """``arrange_prefix`` is ``KVCache.chronological_all`` at the static length ``P = context_length`` with
    no host read: equal on the valid slots (permuted to ``[L, B, K, P, Dh]``), zero elsewhere, equal
    positions."""

    backbone = _backbone()
    cache = _cache_with_rows(backbone, history) if history else _prime(backbone, 0)
    keys, values, valid, positions = fast_forward.arrange_prefix(cache)
    capacity = cache.capacity
    config = backbone.config
    assert keys.shape == values.shape == (cache.n_layers, B, config.n_kv_heads, capacity, config.head_dim)
    assert valid.shape == positions.shape == (B, capacity) and valid.dtype == torch.bool
    assert keys.is_contiguous() and values.is_contiguous()
    ali_keys, ali_values, ali_valid, ali_positions = cache.chronological_all()
    length = ali_keys.shape[2]
    assert torch.equal(valid[:, :length], ali_valid) and not valid[:, length:].any()
    assert torch.equal(positions[:, :length], ali_positions)
    assert torch.equal(keys[:, :, :, :length].permute(0, 1, 3, 2, 4), ali_keys)
    assert torch.equal(values[:, :, :, :length].permute(0, 1, 3, 2, 4), ali_values)
    assert not keys[:, :, :, length:].any() and not values[:, :, :, length:].any()


def test_prefix_mask_equals_alis_at_its_length_and_the_segmented_mask_at_p0() -> None:
    """The fixed-``P`` mask restricted to Ali's ``P`` slots equals ``_prefix_attention_mask``; with an
    all-invalid prefix (an empty ring) the window block equals ``_segmented_attention_mask``; padded query
    rows differ only on their own diagonal (a self-only entry, so a fused kernel never sees a fully masked
    row)."""

    backbone = _backbone()
    window = backbone.config.context_length
    for history, seed in ((10, 1), (24, 2), (40, 3)):
        cache = _cache_with_rows(backbone, history)
        reset, valid = _masks(seed)
        _, _, prefix_valid, prefix_positions = fast_forward.arrange_prefix(cache)
        positions = fast_forward.positions(reset, valid, cache.next_position)
        mask = fast_forward.prefix_mask(positions, reset, valid, prefix_valid, prefix_positions, window)
        assert mask.shape == (B, 1, 11, cache.capacity + 11) and mask.dtype == torch.bool
        _, _, ali_valid, ali_positions = cache.chronological_all()
        length = ali_valid.shape[1]
        expected = CausalTransformer._prefix_attention_mask(
            positions, reset, valid, ali_valid, ali_positions, window
        )
        ours = torch.cat((mask[..., :length], mask[..., cache.capacity :]), dim=-1)
        padded_rows = ~valid
        assert torch.equal(ours[~padded_rows[:, None, :]], expected[~padded_rows[:, None, :]])
        # Padded rows: Ali's are fully masked, ours see exactly themselves.
        for row in range(B):
            for t in range(11):
                if not valid[row, t]:
                    assert not expected[row, 0, t].any()
                    assert int(mask[row, 0, t].sum()) == 1 and bool(mask[row, 0, t, cache.capacity + t])
        # Nothing attends to the padded slots beyond Ali's length, and a valid row never is fully masked.
        assert not mask[..., length : cache.capacity].any()
        assert bool(mask.any(dim=-1).all())
    empty = _prime(backbone, 0)
    reset, valid = _masks(5)
    _, _, prefix_valid, prefix_positions = fast_forward.arrange_prefix(empty)
    positions = fast_forward.positions(reset, valid, empty.next_position)
    mask = fast_forward.prefix_mask(positions, reset, valid, prefix_valid, prefix_positions, window)
    segmented = CausalTransformer._segmented_attention_mask(B, 11, torch.device("cpu"), reset, valid)
    assert segmented is not None and not mask[..., : empty.capacity].any()
    block = mask[..., empty.capacity :]
    assert torch.equal(block[valid[:, None, :]], segmented[valid[:, None, :]])


def test_rope_table_is_bit_identical_and_shared_by_the_layers(tiny_adapter: MeleePolicyAdapter) -> None:
    """One cos/sin table per pass (Ali rebuilds it per layer: same values); the rotated q/k are
    ``torch.equal`` with ``RotaryEmbedding.apply_rotary`` for every layer."""

    backbone = tiny_adapter.backbone
    config = backbone.config
    reset, valid = _masks(7, time=9)
    positions = fast_forward.positions(reset, valid, torch.tensor([0, 5, 40]))
    generator = torch.Generator().manual_seed(3)
    queries = torch.randn((B, config.n_heads, 9, config.head_dim), generator=generator)
    keys = torch.randn((B, config.n_kv_heads, 9, config.head_dim), generator=generator)
    first = cast(GroupedQueryCausalAttention, backbone.layers[0].attention).rope
    cosine, sine = fast_forward.rope_table(first, positions, queries.dtype)
    assert cosine.shape == sine.shape == (B, 1, 9, config.head_dim)
    for layer in backbone.layers:
        rope = cast(GroupedQueryCausalAttention, layer.attention).rope
        expected_q, expected_k = rope.apply_rotary(queries, keys, positions)
        actual_q, actual_k = fast_forward.apply_rope(queries, keys, cosine, sine)
        assert torch.equal(actual_q, expected_q) and torch.equal(actual_k, expected_k)


def _randomise_depth_reads(backbone: CausalTransformer, seed: int) -> None:
    """Ali zero-initialises every ``depth_query`` (a uniform mixture); give the reads real routing."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for read in fast_forward.depth_reads(backbone):
            read.depth_query.normal_(0.0, 0.5, generator=generator)
            read.routing_norm.weight.normal_(1.0, 0.1, generator=generator)


def test_attnres_read_matches_full_attention_residual(tiny_adapter: MeleePolicyAdapter) -> None:
    """The depth read as ``rms`` once per source + one GEMM of routing dots per source + a batched matmul
    per read equals ``FullAttentionResidual`` (forward within Ali's backbone tolerances, gradients of the
    sources and of both routing parameters within rtol 1e-4); ``S = 1 .. 2L + 1`` sources, random nonzero
    routing."""

    backbone = tiny_adapter.backbone
    _randomise_depth_reads(backbone, 31)
    reads = fast_forward.depth_reads(backbone)
    assert len(reads) == 2 * backbone.config.n_layers + 1
    config = backbone.config
    generator = torch.Generator().manual_seed(32)
    values = [torch.randn((B, 7, config.d_model), generator=generator).requires_grad_(True) for _ in reads]
    sources = fast_forward.DepthSources(reads, config.norm_eps)
    ours_total = torch.zeros(())
    for index, read in enumerate(reads):
        sources.append(values[index])
        ours = sources.read(index)
        expected = read(values[: index + 1])
        assert ours.shape == expected.shape == (B, 7, config.d_model) and ours.dtype == torch.float32
        torch.testing.assert_close(ours, expected, rtol=1.0e-5, atol=1.0e-6)
        ours_total = ours_total + (ours * (index + 1)).sum()
    ours_total.backward()
    ours_grads = [value.grad.clone() for value in values if value.grad is not None]
    ours_params = []
    for read in reads:
        assert read.depth_query.grad is not None and read.routing_norm.weight.grad is not None
        ours_params.append((read.depth_query.grad.clone(), read.routing_norm.weight.grad.clone()))
    for value in values:
        value.grad = None
    for read in reads:
        read.depth_query.grad = None
        read.routing_norm.weight.grad = None
    expected_total = torch.zeros(())
    for index, read in enumerate(reads):
        expected_total = expected_total + (read(values[: index + 1]) * (index + 1)).sum()
    expected_total.backward()
    assert len(ours_grads) == len(values)
    for value, ours_grad in zip(values, ours_grads, strict=True):
        assert value.grad is not None
        torch.testing.assert_close(ours_grad, value.grad, rtol=1.0e-4, atol=1.0e-5)
    for read, (query_grad, weight_grad) in zip(reads, ours_params, strict=True):
        assert read.depth_query.grad is not None and read.routing_norm.weight.grad is not None
        torch.testing.assert_close(query_grad, read.depth_query.grad, rtol=1.0e-4, atol=1.0e-5)
        torch.testing.assert_close(weight_grad, read.routing_norm.weight.grad, rtol=1.0e-4, atol=1.0e-5)


@pytest.mark.parametrize("mode", ["sdpa", "bmm"])
def test_grouped_attention_matches_sdpa_gqa(tiny_adapter: MeleePolicyAdapter, mode: str) -> None:
    """Attention with the query heads of one kv head folded into the row axis (no ``enable_gqa``, so a fused
    fp32 kernel is eligible; or explicit ``baddbmm`` + softmax + ``bmm`` per kv head) equals Ali's ``_sdpa``
    with ``enable_gqa=True`` -- forward and gradients -- on a prefix mask with resets and self-only padded
    rows."""

    attention = cast(GroupedQueryCausalAttention, tiny_adapter.backbone.layers[0].attention)
    config = tiny_adapter.backbone.config
    time, prefix_length = 9, 13
    generator = torch.Generator().manual_seed(41)

    def tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn((B, config.n_heads, time, config.head_dim), generator=generator)
        k = torch.randn((B, config.n_kv_heads, prefix_length + time, config.head_dim), generator=generator)
        v = torch.randn((B, config.n_kv_heads, prefix_length + time, config.head_dim), generator=generator)
        return q, k, v

    q, k, v = tensors()
    reset, valid = _masks(42, time=time)
    prefix_valid = torch.arange(prefix_length)[None, :] < torch.tensor([13, 0, 5])[:, None]
    prefix_positions = torch.arange(prefix_length)[None, :].expand(B, -1)
    query_positions = fast_forward.positions(reset, valid, torch.tensor([13, 0, 5]))
    mask = fast_forward.prefix_mask(
        query_positions, reset, valid, prefix_valid, prefix_positions, config.context_length
    )
    expected_inputs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    expected = attention._sdpa(
        expected_inputs[0], expected_inputs[1], expected_inputs[2], attention_mask=mask, is_causal=False
    )
    actual_inputs = [t.clone().requires_grad_(True) for t in (q, k, v)]
    aq, ak, av = actual_inputs
    actual = fast_forward.grouped_attention(aq, ak, av, mask, mode=mode)
    assert actual.shape == expected.shape == (B, config.n_heads, time, config.head_dim)
    torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-5)
    weights = torch.randn(expected.shape, generator=generator)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    for ours, theirs in zip(actual_inputs, expected_inputs, strict=True):
        assert ours.grad is not None and theirs.grad is not None
        torch.testing.assert_close(ours.grad, theirs.grad, rtol=1.0e-4, atol=1.0e-5)
    with pytest.raises(ValueError, match="mode"):
        fast_forward.grouped_attention(q, k, v, mask, mode="flash")


def _same_weights(adapter: MeleePolicyAdapter, config: ModelConfig) -> MeleePolicyAdapter:
    policy = MeleePolicyAdapter(config)
    policy.load_state_dict(adapter.state_dict(), strict=True)
    return policy


def _decision_logits(
    adapter: MeleePolicyAdapter, trajectory: Trajectory, fast: fast_forward.FastForward | None
) -> dict[str, torch.Tensor]:
    window = trajectory.prefix_window()
    assert window.targets is not None and trajectory.initial_cache is not None
    if fast is None:
        return adapter.unroll_window(
            window.frames,
            window.controller_prev,
            window.targets,
            window.reset_mask,
            window.padding_mask,
            prefix=trajectory.initial_cache,
        ).logits
    return fast.logits(
        window.frames,
        window.controller_prev,
        window.targets,
        window.reset_mask,
        window.padding_mask,
        trajectory.initial_cache,
    )


def _kl64(p: dict[str, torch.Tensor], q: dict[str, torch.Tensor]) -> float:
    """The largest per-row ``KL(p || q)`` in float64 (fp32 ``kl`` has a rounding floor of ≈ 3e-7 per row)."""

    total = torch.zeros(p["buttons"].shape[:-1], dtype=torch.float64)
    for name in p:
        log_p = torch.log_softmax(p[name].detach().double(), dim=-1)
        log_q = torch.log_softmax(q[name].detach().double(), dim=-1)
        total = total + (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return float(total.max())


@pytest.mark.parametrize("attention", ["sdpa", "bmm"])
def test_fast_forward_logits_match_unroll_window(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory, attention: str
) -> None:
    """The lean pass reproduces ``unroll_window(..., prefix=)`` over real prefix rollouts -- an empty ring
    (P = 0), a partial one, a full one -- with resets inside the window, for the whole batch and for a
    one-row microbatch; per component within the fast step's contract, KL between the two below 1e-8."""

    _randomise_depth_reads(tiny_adapter.backbone, 51)
    # 21-frame games give resets and short rings; a reset-free fourth rollout gives a full ring.
    rollouts = rollout_trajectories(tiny_adapter, 3, context_mode="prefix")
    full = rollout_trajectories(tiny_adapter, 4, context_mode="prefix", game_frames=10_000)[-1]
    fast = fast_forward.FastForward(tiny_adapter, attention=attention)
    tiny_adapter.train()
    lengths = []
    for trajectory in (*rollouts, full):
        assert trajectory.initial_cache is not None
        lengths.append(int(trajectory.initial_cache.valid_length.max()))
        for sub in (trajectory, trajectory.index_envs(slice(1, 2))):
            expected = _decision_logits(tiny_adapter, sub, None)
            actual = _decision_logits(tiny_adapter, sub, fast)
            assert actual.keys() == expected.keys()
            for name in expected:
                assert actual[name].shape == expected[name].shape and actual[name].dtype == torch.float32
                torch.testing.assert_close(actual[name], expected[name], rtol=1.0e-4, atol=1.0e-5)
            assert _kl64(expected, actual) < 1.0e-8
            monitor = distributions.kl(expected, actual).detach().mean()  # the learner's fp32 monitor
            assert abs(float(monitor)) < 1.0e-6
    assert lengths[0] == 0 and min(lengths[1:]) > 0 and lengths[-1] == tiny_adapter.context_length
    assert any(bool(trajectory.prefix_window().reset_mask.any()) for trajectory in rollouts)


def test_fast_forward_matches_a_standard_residual_backbone(
    tiny_config: ModelConfig, rollout_trajectories: RolloutFactory
) -> None:
    torch.manual_seed(77)
    adapter = MeleePolicyAdapter(replace(tiny_config, residual_mode="standard"))
    rollouts = rollout_trajectories(adapter, 2, context_mode="prefix")
    fast = fast_forward.FastForward(adapter)
    assert fast.reads is None
    for trajectory in rollouts:
        expected = _decision_logits(adapter, trajectory, None)
        actual = _decision_logits(adapter, trajectory, fast)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1.0e-4, atol=1.0e-5)


def _grads(policy: MeleePolicyAdapter) -> list[torch.Tensor | None]:
    return [None if p.grad is None else p.grad.clone() for p in policy.parameters()]


def test_fast_forward_gradients_match_with_and_without_checkpointing(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig, rollout_trajectories: RolloutFactory
) -> None:
    """Checkpointing on or off, the lean pass gives the same gradients bit for bit (the same kernels
    replayed); against Ali's forward every parameter's gradient agrees to 5e-5 of that tensor's scale (the
    depth reads and the attention re-associate fp32 sums; measured at ≈ 1e-6 of the scale with sharp random
    routing on this host, 1.4e-5 on the Modal L4's host, ≈ 1e-7 with Ali's zero-initialised routing)."""

    _randomise_depth_reads(tiny_adapter.backbone, 61)
    trajectory = rollout_trajectories(tiny_adapter, 3, context_mode="prefix")[-1]
    weights = {
        "buttons": torch.randn((B, 8, 728), generator=torch.Generator().manual_seed(62)),
        "main_stick": torch.randn((B, 8, 85), generator=torch.Generator().manual_seed(63)),
    }

    def run(policy: MeleePolicyAdapter, fast: fast_forward.FastForward | None) -> list[torch.Tensor | None]:
        policy.train()
        policy.zero_grad(set_to_none=True)
        logits = _decision_logits(policy, trajectory, fast)
        torch.stack([(logits[name] * weights[name]).sum() for name in logits]).sum().backward()
        return _grads(policy)

    plain = _same_weights(tiny_adapter, tiny_config)
    checkpointed = _same_weights(tiny_adapter, replace(tiny_config, gradient_checkpointing=True))
    assert not plain.config.gradient_checkpointing and checkpointed.config.gradient_checkpointing
    lean_plain = run(plain, fast_forward.FastForward(plain))
    lean_checkpointed = run(checkpointed, fast_forward.FastForward(checkpointed))
    ali = run(_same_weights(tiny_adapter, tiny_config), None)
    compared = 0
    for a, b, c in zip(lean_plain, lean_checkpointed, ali, strict=True):
        if c is None:
            assert a is None and b is None
            continue
        assert a is not None and b is not None
        assert torch.equal(a, b)
        scale = float(c.abs().max())
        torch.testing.assert_close(a, c, rtol=1.0e-4, atol=5.0e-5 * max(scale, 1.0e-3))
        assert float((a - c).norm()) <= 1.0e-4 * max(float(c.norm()), 1.0e-6)
        compared += 1
    assert compared > 20


class _Census(TorchDispatchMode):
    def __init__(self) -> None:
        self.syncs = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        if func._overloadpacket is torch.ops.aten._local_scalar_dense:
            self.syncs += 1
        return func(*args, **(kwargs or {}))


def test_fast_forward_never_reads_the_host(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory
) -> None:
    """On the CPU the only scalar reads left are ``F.one_hot``'s own input checks (the CUDA kernel makes
    none): the encoder's 9 one-hots and the head's 4 = 26; Ali's forward adds its label / cache / decode /
    causality checks on top."""

    trajectory = rollout_trajectories(tiny_adapter, 2, context_mode="prefix")[-1]
    fast = fast_forward.FastForward(tiny_adapter)
    tiny_adapter.train()
    with torch.no_grad():
        lean = _Census()
        with lean:
            _decision_logits(tiny_adapter, trajectory, fast)
        ali = _Census()
        with ali:
            _decision_logits(tiny_adapter, trajectory, None)
    assert lean.syncs == 26 and ali.syncs >= lean.syncs + 20, (lean.syncs, ali.syncs)


def test_the_compiled_pass_shares_one_graph_across_arms_of_different_sizes(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory
) -> None:
    """17 Sep 2026: the league smoke died at its first gradient step (step 7) on dynamo's recompile limit
    of 8. Its arms hold 60 / 24 / 12 / 6 rows. A 6-row chunk of an arm's prefix cache is ``keys[:, rows]``,
    whose dim-0 stride carries the parent's row count, and ``dynamic=False`` guards strides. So every arm
    layout compiled its own graph per model and grad mode: 4 teacher + 4 policy graphs in step 1, and the
    ninth hit the limit. The compiled call takes contiguous inputs, so equal chunks from any parent share one
    graph, and the logits are unchanged."""

    import torch._dynamo
    from torch._dynamo.utils import counters

    wide = rollout_trajectories(tiny_adapter, 1, num_envs=4, context_mode="prefix")[0]
    narrow = rollout_trajectories(tiny_adapter, 1, num_envs=2, context_mode="prefix", env_seed=9)[0]
    chunks = [wide.index_envs(slice(0, 2)), narrow, wide.index_envs(slice(2, 4))]
    assert chunks[0].initial_cache is not None and narrow.initial_cache is not None
    assert chunks[0].initial_cache.keys.stride() != narrow.initial_cache.keys.stride()  # the league's case
    eager = fast_forward.FastForward(tiny_adapter)
    torch._dynamo.reset()
    counters.clear()
    compiled = fast_forward.FastForward(tiny_adapter, compile="aot_eager")
    tiny_adapter.train()
    with torch.no_grad():
        for chunk in chunks:
            expected = _decision_logits(tiny_adapter, chunk, eager)
            actual = _decision_logits(tiny_adapter, chunk, compiled)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=1.0e-4, atol=1.0e-5)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
    torch._dynamo.reset()


def test_fast_forward_refuses_what_it_does_not_cover(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig
) -> None:
    with pytest.raises(ValueError, match="attention"):
        fast_forward.FastForward(tiny_adapter, attention="flash")
    with pytest.raises(NotImplementedError, match="dropout"):
        fast_forward.FastForward(MeleePolicyAdapter(replace(tiny_config, residual_dropout=0.1)))
    with pytest.raises(NotImplementedError, match="float32"):
        fast_forward.FastForward(MeleePolicyAdapter(replace(tiny_config, compute_dtype="bfloat16")))


def test_cache_stager_moves_the_slice_without_a_device_copy_on_the_cpu(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory
) -> None:
    trajectory = rollout_trajectories(tiny_adapter, 2, context_mode="prefix")[-1]
    cache = trajectory.index_envs(slice(1, 3)).initial_cache
    assert cache is not None
    staged = fast_forward.CacheStager(torch.device("cpu")).stage(cache)
    for name in ("keys", "values", "valid_length", "write_position", "next_position"):
        assert torch.equal(getattr(staged, name), getattr(cache, name))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cache_stager_matches_cache_to_on_cuda(
    tiny_adapter: MeleePolicyAdapter, rollout_trajectories: RolloutFactory
) -> None:
    rollouts = rollout_trajectories(tiny_adapter, 3, context_mode="prefix")
    stager = fast_forward.CacheStager(torch.device("cuda"))
    for trajectory in rollouts:
        for chunk in (slice(0, 2), slice(2, 3)):
            cache = trajectory.index_envs(chunk).initial_cache
            assert cache is not None
            staged = stager.stage(cache)
            expected = cache_to(cache, "cuda")
            torch.cuda.synchronize()
            for name in ("keys", "values", "valid_length", "write_position", "next_position"):
                assert getattr(staged, name).device.type == "cuda"
                assert torch.equal(getattr(staged, name), getattr(expected, name))
