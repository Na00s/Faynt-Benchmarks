"""``melee_rl.kv_cache``: snapshots of the actor's ring for ``context_mode = "prefix"`` (session 39)."""

from __future__ import annotations

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.frames import random_valid_frames
from melee_rl.kv_cache import CACHE_FIELDS, cache_cat, cache_index, cache_to, snapshot_cache
from model import KVCache

B = 3


def _drive(
    adapter: MeleePolicyAdapter,
    cache: KVCache,
    steps: int,
    *,
    fast: bool = False,
    first: int = 0,
    resets: dict[int, list[bool]] | None = None,
) -> None:
    """``steps`` cached steps of input-aligned random frames (step ``first + i`` draws its own seeds)."""

    adapter.fast_step = fast
    try:
        for step in range(first, first + steps):
            generator = torch.Generator().manual_seed(2000 + step)
            frames = random_valid_frames((B,), torch.Generator().manual_seed(1000 + step))
            prev = ControllerLabels(
                buttons=torch.randint(0, 728, (B,), generator=generator),
                main_stick=torch.randint(0, 85, (B,), generator=generator),
            )
            reset = torch.tensor((resets or {}).get(step, [False] * B), dtype=torch.bool)
            adapter.step_sample(frames, prev, reset, cache, generator=torch.Generator().manual_seed(step))
    finally:
        adapter.fast_step = False


def _assert_caches_equal(a: KVCache, b: KVCache) -> None:
    for name in CACHE_FIELDS:
        assert torch.equal(getattr(a, name), getattr(b, name)), name


def test_snapshot_is_a_detached_frozen_copy(tiny_adapter: MeleePolicyAdapter) -> None:
    cache = tiny_adapter.init_cache(B)
    _drive(tiny_adapter, cache, 5)
    snapshot = snapshot_cache(cache)
    _assert_caches_equal(snapshot, cache)
    for name in CACHE_FIELDS:
        copied, source = getattr(snapshot, name), getattr(cache, name)
        assert copied.data_ptr() != source.data_ptr() and copied.device.type == "cpu"
        assert not copied.requires_grad
    frozen = snapshot_cache(snapshot, device=None)
    _drive(tiny_adapter, cache, 3, first=5, resets={6: [False, True, False]})
    assert cache.next_position.tolist() == [8, 2, 8]
    _assert_caches_equal(snapshot, frozen)  # the source rolled on, the snapshot did not
    moved = cache_to(snapshot, "cpu")
    _assert_caches_equal(moved, snapshot)


def test_index_and_cat_round_trip_and_agree_with_chronological(tiny_adapter: MeleePolicyAdapter) -> None:
    capacity = tiny_adapter.context_length
    cache = tiny_adapter.init_cache(B)
    _drive(tiny_adapter, cache, capacity + 5, resets={7: [True, False, False], 20: [False, False, True]})
    assert cache.valid_length.tolist() == [22, capacity, 9]  # env 1 overflowed, envs 0 and 2 restarted
    assert cache.next_position.tolist() == [22, capacity + 5, 9]
    parts = [cache_index(cache, [0]), cache_index(cache, slice(1, 3))]
    assert parts[0].batch_size == 1 and parts[1].batch_size == 2
    _assert_caches_equal(cache_cat(parts), cache)
    single = cache_index(cache, torch.tensor([1]))
    for layer in range(cache.n_layers):
        keys, values, valid = single.chronological(layer)
        reference_keys, reference_values, reference_valid = cache.chronological(layer)
        assert torch.equal(keys, reference_keys[1:2]) and torch.equal(values, reference_values[1:2])
        assert torch.equal(valid, reference_valid[1:2])
    with pytest.raises(ValueError, match="at least one"):
        cache_cat([])


def test_fast_step_snapshot_reads_like_the_slow_path(tiny_adapter: MeleePolicyAdapter) -> None:
    """A fast-path ring keeps stale K/V in invalidated slots; ``chronological_all`` on its snapshot must
    still equal the slow path's history slot for slot, zeros included."""

    capacity = tiny_adapter.context_length
    steps = capacity + 5
    resets = {3: [True, False, False], capacity + 1: [False, True, False]}
    slow = tiny_adapter.init_cache(B)
    fast = tiny_adapter.init_cache(B)
    _drive(tiny_adapter, slow, steps, fast=False, resets=resets)
    _drive(tiny_adapter, fast, steps, fast=True, resets=resets)
    snapshot = snapshot_cache(fast)
    assert snapshot.valid_length.tolist() == [capacity, 4, capacity]
    assert snapshot.next_position.tolist() == [steps - 3, 4, steps]
    keys, values, valid, positions = snapshot.chronological_all()
    slow_keys, slow_values, slow_valid, slow_positions = slow.chronological_all()
    assert torch.equal(valid, slow_valid) and torch.equal(positions, slow_positions)
    torch.testing.assert_close(keys, slow_keys, rtol=1.0e-5, atol=1.0e-6)
    torch.testing.assert_close(values, slow_values, rtol=1.0e-5, atol=1.0e-6)
    assert torch.count_nonzero(keys[:, 1, 4:]) == 0 and torch.count_nonzero(values[:, 1, 4:]) == 0
