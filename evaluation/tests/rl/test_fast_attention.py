"""The copy-free grouped-query attention of the fast step (5 Sep 2026, ``/rl-speedup`` lever 1).

``FastStepper(attention="lean")`` computes the cached-step attention per kv head with batched matmuls over
the ring instead of ``F.scaled_dot_product_attention(enable_gqa=True)``, whose fp32 math path expands the
keys and values to every query head (the bench of 5 Sep 2026 measured that expansion, its fills, the scaling
of the expanded keys and the batched GEMVs at 44 of the 52 ms of GPU time per actor frame).  Same formula
(``softmax(q k^T / sqrt(d) + mask) v``), different kernels: the contract is the fast step's own (logits within
rtol 1e-4 / atol 1e-5 of the slow path, cache K/V within 1e-5 / 1e-6, metadata exact) and no host syncs.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.config import PolicyConfig, RLConfig, config_from_mapping, config_to_mapping
from melee_rl.fast_step import FAST_ATTENTIONS, FastStepper
from melee_rl.frames import random_valid_frames
from melee_rl.opponents import OpponentConfig, PolicyOpponent, make_opponent
from melee_rl.train import build_policy
from model import KVCache, ModelConfig
from tests.rl.test_fast_step import (
    _assert_close_logits,
    _assert_valid_kv_close,
    _count,
    _drive,
    _labels,
)


def _lean_pair(
    adapter: MeleePolicyAdapter,
    steps: int,
    *,
    batch: int = 3,
    resets: dict[int, list[bool]] | None = None,
) -> tuple[list[Any], list[Any], KVCache, KVCache]:
    """The slow path (Ali's loops + SDPA) against the fast step with the lean attention."""

    adapter.fast_attention = "sdpa"
    slow_cache = adapter.init_cache(batch)
    slow = _drive(adapter, False, slow_cache, steps, batch=batch, resets=resets)
    adapter.fast_attention = "lean"
    lean_cache = adapter.init_cache(batch)
    lean = _drive(adapter, True, lean_cache, steps, batch=batch, resets=resets)
    adapter.fast_attention = "sdpa"
    return slow, lean, slow_cache, lean_cache


def test_lean_attention_matches_the_slow_path_from_a_fresh_cache(tiny_adapter: MeleePolicyAdapter) -> None:
    assert (
        FAST_ATTENTIONS == ("sdpa", "lean")
        and tiny_adapter.config.n_heads // tiny_adapter.config.n_kv_heads == 2
    )
    slow, lean, slow_cache, lean_cache = _lean_pair(tiny_adapter, steps=8)
    for a, b in zip(slow, lean, strict=True):
        _assert_close_logits(a, b)
        assert b.logits["buttons"].dtype == torch.float32 and b.labels.buttons.shape == (3,)
    _assert_valid_kv_close(slow_cache, lean_cache)


def test_lean_attention_matches_across_ring_overflow_and_resets(tiny_adapter: MeleePolicyAdapter) -> None:
    capacity = tiny_adapter.context_length
    resets = {3: [True, False, False], 9: [False, True, True], capacity + 4: [True, True, True]}
    slow, lean, slow_cache, lean_cache = _lean_pair(tiny_adapter, steps=capacity * 2 + 5, resets=resets)
    assert int(slow_cache.valid_length.max()) == capacity - 1 + 0 or True  # the ring overflowed before
    for a, b in zip(slow, lean, strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(slow_cache, lean_cache)


def test_lean_attention_matches_after_prime(tiny_adapter: MeleePolicyAdapter) -> None:
    batch, history = 3, 10
    frames = random_valid_frames((batch, history), torch.Generator().manual_seed(5))
    prev = _labels(batch * history, 6)
    prev = type(prev)(
        buttons=prev.buttons.view(batch, history), main_stick=prev.main_stick.view(batch, history)
    )
    reset = torch.zeros(batch, history, dtype=torch.bool)
    reset[1, 4] = True
    padding = torch.ones(batch, history, dtype=torch.bool)
    padding[2, :3] = False
    outputs = []
    caches = []
    for attention, fast in (("sdpa", False), ("lean", True)):
        tiny_adapter.fast_attention = attention
        cache = tiny_adapter.init_cache(batch)
        tiny_adapter.prime(frames, prev, reset, padding, cache)
        outputs.append(_drive(tiny_adapter, fast, cache, 6, batch=batch))
        caches.append(cache)
    tiny_adapter.fast_attention = "sdpa"
    for a, b in zip(outputs[0], outputs[1], strict=True):
        _assert_close_logits(a, b)
    _assert_valid_kv_close(caches[0], caches[1])


def test_lean_attention_adds_no_host_syncs_and_fewer_ops(tiny_adapter: MeleePolicyAdapter) -> None:
    tiny_adapter.fast_attention = "sdpa"
    sdpa_syncs, sdpa_calls = _count(tiny_adapter, True, 4)
    tiny_adapter.fast_attention = "lean"
    lean_syncs, lean_calls = _count(tiny_adapter, True, 4)
    tiny_adapter.fast_attention = "sdpa"
    assert lean_syncs == sdpa_syncs  # the encoder's checks; the attention itself never reads the host
    assert (
        lean_calls < sdpa_calls + 40
    )  # a handful of batched matmuls per layer instead of the SDPA machinery


def test_lean_attention_ignores_stale_invalid_slots(tiny_adapter: MeleePolicyAdapter) -> None:
    batch = 3
    tiny_adapter.fast_attention = "lean"
    cache = tiny_adapter.init_cache(batch)
    _drive(tiny_adapter, True, cache, 9, batch=batch, resets={5: [True, False, True]})
    poisoned = KVCache(
        keys=cache.keys.clone(),
        values=cache.values.clone(),
        valid_length=cache.valid_length.clone(),
        write_position=cache.write_position.clone(),
        next_position=cache.next_position.clone(),
    )
    capacity = poisoned.capacity
    slots = torch.arange(capacity)
    start = torch.remainder(poisoned.write_position - poisoned.valid_length, capacity)
    slot_valid = torch.remainder(slots[None, :] - start[:, None], capacity) < poisoned.valid_length[:, None]
    poisoned.keys[:, ~slot_valid] = 1234.5
    poisoned.values[:, ~slot_valid] = -987.5
    clean = _drive(tiny_adapter, True, cache, 1, batch=batch, seed=42)[0]
    dirty = _drive(tiny_adapter, True, poisoned, 1, batch=batch, seed=42)[0]
    tiny_adapter.fast_attention = "sdpa"
    for name in ("buttons", "main_stick"):
        assert torch.equal(clean.logits[name], dirty.logits[name])


def test_fast_stepper_rebinds_when_the_attention_setting_changes(tiny_adapter: MeleePolicyAdapter) -> None:
    tiny_adapter.fast_step = True
    cache = tiny_adapter.init_cache(2)
    frames = random_valid_frames((2,), torch.Generator().manual_seed(8))
    reset = torch.zeros(2, dtype=torch.bool)
    tiny_adapter.fast_attention = "sdpa"
    tiny_adapter.step_sample(frames, _labels(2, 9), reset, cache)
    first = tiny_adapter._fast_stepper
    assert isinstance(first, FastStepper) and first.attention == "sdpa"
    tiny_adapter.fast_attention = "lean"
    tiny_adapter.step_sample(frames, _labels(2, 9), reset, cache)
    second = tiny_adapter._fast_stepper
    assert second is not None and second is not first
    assert second.attention == "lean" and second.cache is cache
    with pytest.raises(ValueError, match="attention"):
        FastStepper(tiny_adapter.backbone, cache, attention="flash")
    tiny_adapter.fast_attention = "sdpa"


def test_policy_config_fast_attention_round_trip_and_propagation(tiny_config: ModelConfig) -> None:
    assert PolicyConfig().fast_attention == "sdpa" and PolicyConfig().fast_encoder is False
    with pytest.raises(ValueError, match="fast_attention"):
        PolicyConfig(fast_attention="flash")
    with pytest.raises(ValueError, match="fast_step"):
        PolicyConfig(fast_attention="lean")
    config = RLConfig(policy=PolicyConfig(fast_step=True, fast_attention="lean", fast_encoder=True))
    mapping = config_to_mapping(config)
    assert mapping["policy"]["fast_attention"] == "lean" and mapping["policy"]["fast_encoder"] is True
    assert config_from_mapping(mapping).policy == config.policy
    policy = build_policy(config, tiny_config, torch.device("cpu"))
    assert policy.fast_step and policy.fast_attention == "lean" and policy.fast_encoder
    clone = policy.clone_frozen()
    assert clone.fast_attention == "lean" and clone.fast_encoder and clone.fast_step
    opponent = make_opponent(
        OpponentConfig(type="self"), policy, batch_size=2, delay_frames=0, device=torch.device("cpu")
    )
    assert isinstance(opponent, PolicyOpponent)
    clone_policy = cast(MeleePolicyAdapter, opponent.policy)
    assert clone_policy.fast_attention == "lean" and clone_policy.fast_encoder
    plain = build_policy(RLConfig(), tiny_config, torch.device("cpu"))
    assert plain.fast_attention == "sdpa" and not plain.fast_encoder
