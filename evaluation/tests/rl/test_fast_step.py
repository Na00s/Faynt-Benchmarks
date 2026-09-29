"""The vectorised fast policy step (P7-PLAN.md 3): parity, resets, ring wrap, syncs, wiring.

The fast path (``melee_rl/fast_step.py``) replaces the per-frame cached-step orchestration of
``model.py`` (the per-row cache-write and chronological-gather loops) with vectorised ring
attention, without editing Ali's files: it calls his layer modules (``_project`` / ``_sdpa`` /
``_apply_output_gate`` / SwiGLU / residual reads) and reimplements only the cache bookkeeping and
the head's sampling loop.  Numeric contract: fast-vs-slow logits within rtol 1e-4 / atol 1e-5 (the
actor-learner parity precedent), cache K/V at valid slots within rtol 1e-5 / atol 1e-6, cache
metadata exactly equal, same-path determinism; cross-path *sampled labels* are not promised
identical (fp reduction order differs at ~1e-7).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from controller_codec import ControllerLabels
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter, rl_model_config
from melee_rl.checkpoint import save_bc_checkpoint
from melee_rl.config import PolicyConfig, RLConfig, config_from_mapping, config_to_mapping
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.frames import random_valid_frames
from melee_rl.opponents import OpponentConfig, PolicyOpponent, make_opponent
from melee_rl.train import build_policy
from model import KVCache, ModelConfig

LOGITS_RTOL, LOGITS_ATOL = 1.0e-4, 1.0e-5
CACHE_RTOL, CACHE_ATOL = 1.0e-5, 1.0e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _labels(batch: int, seed: int) -> ControllerLabels:
    generator = torch.Generator().manual_seed(seed)
    return ControllerLabels(
        buttons=torch.randint(0, 728, (batch,), generator=generator),
        main_stick=torch.randint(0, 85, (batch,), generator=generator),
    )


def _drive(
    adapter: MeleePolicyAdapter,
    fast: bool,
    cache: KVCache,
    steps: int,
    *,
    batch: int,
    resets: dict[int, list[bool]] | None = None,
    temperature: float = 1.0,
    seed: int = 7,
) -> list[Any]:
    """``steps`` calls of ``step_sample`` with shared, input-aligned frames / prev labels."""

    adapter.fast_step = fast
    generator = torch.Generator().manual_seed(seed)
    outputs = []
    for step in range(steps):
        frames = random_valid_frames((batch,), torch.Generator().manual_seed(1000 + step))
        prev = _labels(batch, 2000 + step)
        row = (resets or {}).get(step, [False] * batch)
        reset = torch.tensor(row, dtype=torch.bool)
        outputs.append(
            adapter.step_sample(frames, prev, reset, cache, temperature=temperature, generator=generator)
        )
    return outputs


def _assert_close_logits(a: Any, b: Any) -> None:
    for name in ("buttons", "main_stick"):
        torch.testing.assert_close(a.logits[name], b.logits[name], rtol=LOGITS_RTOL, atol=LOGITS_ATOL)


def _assert_metadata_equal(a: KVCache, b: KVCache) -> None:
    assert torch.equal(a.valid_length, b.valid_length)
    assert torch.equal(a.write_position, b.write_position)
    assert torch.equal(a.next_position, b.next_position)


def _assert_valid_kv_close(a: KVCache, b: KVCache) -> None:
    _assert_metadata_equal(a, b)
    for layer in range(a.n_layers):
        keys_a, values_a, valid_a = a.chronological(layer)
        keys_b, values_b, valid_b = b.chronological(layer)
        assert torch.equal(valid_a, valid_b)
        torch.testing.assert_close(keys_a[valid_a], keys_b[valid_b], rtol=CACHE_RTOL, atol=CACHE_ATOL)
        torch.testing.assert_close(values_a[valid_a], values_b[valid_b], rtol=CACHE_RTOL, atol=CACHE_ATOL)


def _pair(
    adapter: MeleePolicyAdapter,
    steps: int,
    *,
    batch: int = 3,
    resets: dict[int, list[bool]] | None = None,
    temperature: float = 1.0,
) -> tuple[list[Any], list[Any], KVCache, KVCache]:
    slow_cache = adapter.init_cache(batch)
    fast_cache = adapter.init_cache(batch)
    slow = _drive(adapter, False, slow_cache, steps, batch=batch, resets=resets, temperature=temperature)
    fast = _drive(adapter, True, fast_cache, steps, batch=batch, resets=resets, temperature=temperature)
    return slow, fast, slow_cache, fast_cache


# ---------------------------------------------------------------------------
# 1-6: parity and cache semantics
# ---------------------------------------------------------------------------


def test_fast_step_matches_slow_from_fresh_cache(tiny_adapter: MeleePolicyAdapter) -> None:
    slow, fast, slow_cache, fast_cache = _pair(tiny_adapter, steps=8)
    for a, b in zip(slow, fast, strict=True):
        _assert_close_logits(a, b)
        assert b.labels.buttons.dtype == torch.long and b.labels.buttons.shape == (3,)
        assert bool((b.labels.buttons >= 0).all()) and bool((b.labels.buttons < 728).all())
        assert bool((b.labels.main_stick >= 0).all()) and bool((b.labels.main_stick < 85).all())
        assert b.logits["buttons"].dtype == torch.float32
    _assert_valid_kv_close(slow_cache, fast_cache)


def test_fast_step_matches_slow_after_prime(tiny_adapter: MeleePolicyAdapter) -> None:
    batch, history = 3, 10
    frames = random_valid_frames((batch, history), torch.Generator().manual_seed(5))
    prev = ControllerLabels(
        buttons=torch.randint(0, 728, (batch, history), generator=torch.Generator().manual_seed(6)),
        main_stick=torch.randint(0, 85, (batch, history), generator=torch.Generator().manual_seed(7)),
    )
    padding = torch.ones(batch, history, dtype=torch.bool)
    padding[0, :4] = False  # a left-padded row
    reset = torch.zeros(batch, history, dtype=torch.bool)
    reset[0, 4] = True  # its first valid row starts a segment
    reset[2, 6] = True  # an in-window reset
    caches = []
    for _ in range(2):
        cache = tiny_adapter.init_cache(batch)
        tiny_adapter.prime(frames, prev, reset, padding, cache)
        caches.append(cache)
    slow_cache, fast_cache = caches
    slow = _drive(tiny_adapter, False, slow_cache, 4, batch=batch)
    fast = _drive(tiny_adapter, True, fast_cache, 4, batch=batch)
    for a, b in zip(slow, fast, strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(slow_cache, fast_cache)


def test_fast_step_matches_slow_across_ring_overflow(tiny_adapter: MeleePolicyAdapter) -> None:
    capacity = tiny_adapter.context_length  # 24 in the tiny config
    steps = capacity * 2 + 5
    slow, fast, slow_cache, fast_cache = _pair(tiny_adapter, steps=steps)
    assert int(slow_cache.valid_length.max()) == capacity  # the ring really overflowed
    for a, b in zip(slow[-5:], fast[-5:], strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(slow_cache, fast_cache)


def test_fast_step_matches_slow_with_midstream_resets(tiny_adapter: MeleePolicyAdapter) -> None:
    resets = {3: [True, False, False], 9: [False, True, True], 17: [True, True, True]}
    slow, fast, slow_cache, fast_cache = _pair(tiny_adapter, steps=20, resets=resets)
    for a, b in zip(slow, fast, strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(slow_cache, fast_cache)
    assert slow_cache.valid_length.tolist() == [3, 3, 3]  # every env last reset at step 17 of 20


def test_fast_step_ignores_stale_invalid_slots(tiny_adapter: MeleePolicyAdapter) -> None:
    batch = 3
    resets = {5: [True, False, True]}
    cache = tiny_adapter.init_cache(batch)
    _drive(tiny_adapter, True, cache, 9, batch=batch, resets=resets)
    poisoned = KVCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        valid_length=cache.valid_length.clone(),
        write_position=cache.write_position.clone(),
        next_position=cache.next_position.clone(),
    )
    # Poison every slot outside the valid window (slot-space mask derived per row).
    capacity = poisoned.capacity
    slots = torch.arange(capacity)
    start = torch.remainder(poisoned.write_position - poisoned.valid_length, capacity)
    slot_valid = torch.remainder(slots[None, :] - start[:, None], capacity) < poisoned.valid_length[:, None]
    poisoned.keys[:, ~slot_valid] = 1234.5
    poisoned.values[:, ~slot_valid] = -987.5
    clean_out = _drive(tiny_adapter, True, cache, 1, batch=batch, seed=42)[0]
    poisoned_out = _drive(tiny_adapter, True, poisoned, 1, batch=batch, seed=42)[0]
    for name in ("buttons", "main_stick"):
        assert torch.equal(clean_out.logits[name], poisoned_out.logits[name])
        assert torch.equal(clean_out.labels.as_dict()[name], poisoned_out.labels.as_dict()[name])


def test_fast_reset_metadata_matches_ali_reset_slots(tiny_adapter: MeleePolicyAdapter) -> None:
    batch = 4
    slow_cache = tiny_adapter.init_cache(batch)
    fast_cache = tiny_adapter.init_cache(batch)
    _drive(tiny_adapter, False, slow_cache, 6, batch=batch)
    _drive(tiny_adapter, True, fast_cache, 6, batch=batch)
    mask = torch.tensor([True, False, True, False])
    resets = {0: mask.tolist()}
    slow = _drive(tiny_adapter, False, slow_cache, 1, batch=batch, resets=resets, seed=11)[0]
    fast = _drive(tiny_adapter, True, fast_cache, 1, batch=batch, resets=resets, seed=11)[0]
    _assert_close_logits(slow, fast)
    _assert_metadata_equal(slow_cache, fast_cache)
    assert fast_cache.valid_length.tolist() == [1, 7, 1, 7]


# ---------------------------------------------------------------------------
# 7-13: sampling, determinism, syncs, modes, validation
# ---------------------------------------------------------------------------


def test_fast_step_temperature_zero_is_argmax(tiny_adapter: MeleePolicyAdapter) -> None:
    cache = tiny_adapter.init_cache(2)
    out = _drive(tiny_adapter, True, cache, 3, batch=2, temperature=0.0)[-1]
    assert torch.equal(out.labels.buttons, out.logits["buttons"].argmax(dim=-1))
    # main_stick is conditioned on the taken buttons; its greedy label matches its own logits.
    assert torch.equal(out.labels.main_stick, out.logits["main_stick"].argmax(dim=-1))


def test_fast_step_deterministic_with_seeded_generator(tiny_adapter: MeleePolicyAdapter) -> None:
    runs = []
    for _ in range(2):
        cache = tiny_adapter.init_cache(3)
        runs.append(_drive(tiny_adapter, True, cache, 6, batch=3, seed=123))
    for a, b in zip(runs[0], runs[1], strict=True):
        for name in ("buttons", "main_stick"):
            assert torch.equal(a.labels.as_dict()[name], b.labels.as_dict()[name])
            assert torch.equal(a.logits[name], b.logits[name])


def test_fast_step_advances_generator_like_slow(tiny_adapter: MeleePolicyAdapter) -> None:
    frames = random_valid_frames((3,), torch.Generator().manual_seed(1))
    prev = _labels(3, 2)
    reset = torch.zeros(3, dtype=torch.bool)
    states = []
    for fast in (False, True):
        tiny_adapter.fast_step = fast
        cache = tiny_adapter.init_cache(3)
        generator = torch.Generator().manual_seed(77)
        tiny_adapter.step_sample(frames, prev, reset, cache, temperature=1.0, generator=generator)
        states.append(generator.get_state())
    assert torch.equal(states[0], states[1])


class _DispatchCounter(TorchDispatchMode):
    def __init__(self) -> None:
        self.syncs = 0
        self.calls = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None
    ) -> Any:
        self.calls += 1
        if func._overloadpacket is torch.ops.aten._local_scalar_dense:
            self.syncs += 1
        return func(*args, **(kwargs or {}))


def _count(adapter: MeleePolicyAdapter, fast: bool, batch: int) -> tuple[int, int]:
    adapter.fast_step = fast
    cache = adapter.init_cache(batch)
    frames = random_valid_frames((batch,), torch.Generator().manual_seed(3))
    prev = _labels(batch, 4)
    reset = torch.zeros(batch, dtype=torch.bool)
    generator = torch.Generator().manual_seed(5)
    adapter.step_sample(frames, prev, reset, cache, generator=generator)  # bind / warm the path
    counter = _DispatchCounter()
    with counter:
        adapter.step_sample(frames, prev, reset, cache, generator=generator)
    return counter.syncs, counter.calls


def test_fast_step_sync_count_is_small_and_batch_independent(
    tiny_adapter: MeleePolicyAdapter,
) -> None:
    fast_2 = _count(tiny_adapter, True, 2)
    fast_5 = _count(tiny_adapter, True, 5)
    slow_2 = _count(tiny_adapter, False, 2)
    slow_5 = _count(tiny_adapter, False, 5)
    assert fast_2[0] == fast_5[0], "fast-path host syncs must not scale with the batch"
    assert fast_5[0] <= 70  # the encoder's checks (CPU one_hot .item()s included); the core adds none
    assert slow_5[0] > slow_2[0], "the slow path's per-row loops sync per batch row"
    assert fast_5[1] < slow_5[1], "the fast path dispatches fewer ops than the loops"


def test_fast_step_restores_training_mode(tiny_adapter: MeleePolicyAdapter) -> None:
    tiny_adapter.fast_step = True
    tiny_adapter.train()
    cache = tiny_adapter.init_cache(2)
    frames = random_valid_frames((2,), torch.Generator().manual_seed(8))
    tiny_adapter.step_sample(frames, _labels(2, 9), torch.zeros(2, dtype=torch.bool), cache)
    assert tiny_adapter.training
    tiny_adapter.eval()
    tiny_adapter.step_sample(frames, _labels(2, 9), torch.zeros(2, dtype=torch.bool), cache)
    assert not tiny_adapter.training


def test_fast_step_standard_residual_mode(tiny_config: ModelConfig) -> None:
    config = rl_model_config(tiny_config, residual_mode="standard")
    torch.manual_seed(21)
    adapter = MeleePolicyAdapter(config)
    slow, fast, slow_cache, fast_cache = _pair(adapter, steps=6, batch=2)
    for a, b in zip(slow, fast, strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(slow_cache, fast_cache)


def test_fast_step_rejects_wrong_cache(tiny_adapter: MeleePolicyAdapter) -> None:
    tiny_adapter.fast_step = True
    frames = random_valid_frames((2,), torch.Generator().manual_seed(10))
    wrong_batch = tiny_adapter.init_cache(3)
    with pytest.raises(ValueError, match="cache"):
        tiny_adapter.step_sample(frames, _labels(2, 11), torch.zeros(2, dtype=torch.bool), wrong_batch)
    small = rl_model_config(tiny_adapter.config, context_length=8)
    torch.manual_seed(22)
    other = MeleePolicyAdapter(small)
    foreign = other.init_cache(2)  # wrong capacity for tiny_adapter's config
    with pytest.raises(ValueError, match="cache"):
        tiny_adapter.step_sample(frames, _labels(2, 12), torch.zeros(2, dtype=torch.bool), foreign)


# ---------------------------------------------------------------------------
# 14-16: wiring
# ---------------------------------------------------------------------------


def test_fast_flag_propagates_to_clones_and_opponents(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig, tmp_path: Any
) -> None:
    assert tiny_adapter.fast_step is False  # the default
    tiny_adapter.fast_step = True
    assert MeleePolicyAdapter.from_policy(tiny_adapter).fast_step is True
    assert tiny_adapter.clone_frozen().fast_step is True
    self_opponent = make_opponent(OpponentConfig(type="self"), tiny_adapter, batch_size=2, delay_frames=0)
    assert isinstance(self_opponent, PolicyOpponent)
    assert getattr(self_opponent.policy, "fast_step", False) is True
    path = tmp_path / "bc.pt"
    save_bc_checkpoint(path, tiny_adapter.get_state(), tiny_config)
    other = make_opponent(
        OpponentConfig(type="other", checkpoint=str(path)),
        tiny_adapter,
        batch_size=2,
        delay_frames=0,
    )
    assert isinstance(other, PolicyOpponent)
    assert getattr(other.policy, "fast_step", False) is True
    tiny_adapter.fast_step = False
    assert MeleePolicyAdapter.from_policy(tiny_adapter).fast_step is False


def test_policy_config_fast_step_default_off_and_round_trip(tiny_config: ModelConfig) -> None:
    assert PolicyConfig().fast_step is False
    config = RLConfig(policy=PolicyConfig(fast_step=True))
    mapping = config_to_mapping(config)
    assert mapping["policy"]["fast_step"] is True
    assert config_from_mapping(mapping).policy.fast_step is True
    fast = build_policy(config, tiny_config, torch.device("cpu"))
    assert fast.fast_step is True
    slow = build_policy(RLConfig(), tiny_config, torch.device("cpu"))
    assert slow.fast_step is False


def test_rollout_worker_with_fast_step_two_ports(tiny_adapter: MeleePolicyAdapter) -> None:
    tiny_adapter.fast_step = True

    def run() -> Any:
        env = DummyMeleeEnv(DummyEnvConfig(num_envs=3, players=("policy", "policy"), game_frames=21))
        worker = RolloutWorker(
            tiny_adapter,
            env,
            None,
            ActorConfig(rollout_length=8, context_frames=12, context_mode="ring", seed=5),
            ports=(0, 1),
        )
        trajectories = [worker.rollout() for _ in range(2)]
        env.close()
        return trajectories

    first = run()
    second = run()
    for trajectory in first:
        trajectory.validate()
        assert trajectory.sampled_labels.buttons.shape[0] == 6  # 3 envs x 2 ports
        for name in ("buttons", "main_stick"):
            assert torch.isfinite(trajectory.sampled_logits[name]).all()
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.sampled_labels.as_tensor(), b.sampled_labels.as_tensor())
        assert torch.equal(a.rewards, b.rewards)
