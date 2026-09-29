"""Packed frame transfers and the packed history ring (5 Sep 2026, ``/rl-speedup`` lever 3).

The actor moved every frame to the policy device leaf by leaf -- 78 pageable copies, each a host sync (the
bench of 5 Sep 2026: 79 sync points and ≈ 4 ms per frame in ``observe``) -- and wrote it into its history
ring leaf by leaf (88 index-puts).  ``melee_rl.packed`` lays the canonical tree out feature-major per
dtype, moves a frame in one pinned copy per dtype into static device buffers, and keeps the ring packed the
same way, so a frame is three copies and three writes.  Everything is a view of the same numbers: the
trees, the ring and the trajectories are bit-identical to the leaf-wise path.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.config import RLConfig, config_from_mapping, config_to_mapping
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.frames import (
    FRAME_SPEC,
    neutral_frame,
    random_valid_frames,
    tree_cat,
    tree_index,
    tree_leaves,
    tree_to,
)
from melee_rl.opponents import CpuOpponent, SelfOpponent
from melee_rl.packed import FrameLayout, FramePacker, PackedRing, StaticFrameTree
from melee_rl.trajectory import Trajectory

B = 3
C = 16
T = 8


def _tree_equal(left: dict, right: dict) -> bool:
    lefts, rights = tree_leaves(left), tree_leaves(right)
    return [path for path, _ in lefts] == [path for path, _ in rights] and all(
        a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        for (_, a), (_, b) in zip(lefts, rights, strict=True)
    )


def test_layout_covers_the_canonical_spec_feature_major() -> None:
    layout = FrameLayout()
    leaves = tree_leaves(FRAME_SPEC)
    assert [leaf.path for leaf in layout.leaves] == [tuple(path.split(".")) for path, _ in leaves]
    assert len(layout.leaves) == 78
    kinds = {leaf.kind for leaf in layout.leaves}
    assert kinds == {"float", "int", "bool"} and set(layout.rows) == kinds
    items = [leaf for leaf in layout.leaves if leaf.path[0] == "items"]
    assert len(items) == 5 and all(leaf.width == 15 for leaf in items)
    assert all(leaf.width == 1 for leaf in layout.leaves if leaf.path[0] != "items")
    for kind in kinds:
        assert layout.rows[kind] == sum(leaf.width for leaf in layout.leaves if leaf.kind == kind)
        offsets = [leaf.offset for leaf in layout.leaves if leaf.kind == kind]
        assert offsets == sorted(offsets) and offsets[0] == 0


@pytest.mark.parametrize("shape", [(B,), (B, 5)])
def test_pack_and_unpack_round_trip_bit_for_bit(shape: tuple[int, ...]) -> None:
    layout = FrameLayout()
    tree = random_valid_frames(shape, torch.Generator().manual_seed(3))
    buffers = layout.empty(shape, torch.device("cpu"))
    layout.pack([tree], buffers)
    assert _tree_equal(layout.unpack(buffers), tree)
    # the unpacked leaves are views of the packed buffers
    view = layout.unpack(buffers)
    buffers["float"].zero_()
    assert float(view["p0"]["x"].abs().sum()) == 0.0


def test_packer_uploads_one_copy_per_dtype_and_matches_tree_to() -> None:
    device = torch.device("cpu")
    packer = FramePacker(B, device)
    tree = random_valid_frames((B,), torch.Generator().manual_seed(4))
    uploaded = packer.upload([tree])
    assert uploaded is packer.tree  # the same static views every frame
    assert _tree_equal(uploaded, tree_to(tree, device))
    other = random_valid_frames((B,), torch.Generator().manual_seed(5))
    packer.upload([other])
    assert _tree_equal(uploaded, tree_to(other, device))  # overwritten in place
    # two ports, port-major rows: the packed tree equals tree_cat of the perspectives
    two = FramePacker(2 * B, device, ports=2)
    p0 = random_valid_frames((B,), torch.Generator().manual_seed(6))
    p1 = random_valid_frames((B,), torch.Generator().manual_seed(7))
    assert _tree_equal(two.upload([p0, p1]), tree_cat([p0, p1]))
    with pytest.raises(ValueError, match="port"):
        two.upload([p0])
    with pytest.raises(ValueError, match="ports"):
        FramePacker(B, device, ports=2)


def test_packed_ring_starts_neutral_and_writes_a_frame_in_three_puts() -> None:
    device = torch.device("cpu")
    capacity = 6
    ring = PackedRing(B, capacity, device)
    assert _tree_equal(ring.tree, neutral_frame((B, capacity)))
    packer = FramePacker(B, device)
    tree = random_valid_frames((B,), torch.Generator().manual_seed(8))
    packer.upload([tree])
    ring.write(4, packer.buffers)
    assert _tree_equal(tree_index(ring.tree, (slice(None), 4)), tree)
    assert _tree_equal(tree_index(ring.tree, (slice(None), 3)), neutral_frame((B,)))
    assert set(ring.buffers) == {"float", "int", "bool"}
    assert ring.buffers["float"].shape[1:] == (B, capacity)


def _worker(adapter: MeleePolicyAdapter, *, packed: bool, two_port: bool) -> RolloutWorker:
    players = ("policy", "policy") if two_port else ("policy", "cpu")
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, game_frames=21, players=players))
    config = ActorConfig(rollout_length=T, context_frames=C, packed_frames=packed, seed=7)
    if two_port:
        return RolloutWorker(adapter, env, None, config, ports=(0, 1))
    return RolloutWorker(adapter, env, CpuOpponent(), config)


def _same_trajectory(left: Trajectory, right: Trajectory) -> None:
    assert _tree_equal(left.frames, right.frames)
    for name in ("controller_exec", "controller_prev_sample", "sampled_labels"):
        for component in ("buttons", "main_stick"):
            assert torch.equal(
                getattr(getattr(left, name), component), getattr(getattr(right, name), component)
            )
    for name in ("buttons", "main_stick"):
        assert torch.equal(left.sampled_logits[name], right.sampled_logits[name])
    for name in ("rewards", "env_rewards", "is_resetting", "padding", "frame_index"):
        assert torch.equal(getattr(left, name), getattr(right, name)), name


@pytest.mark.parametrize("two_port", [False, True])
def test_packed_frames_give_bit_identical_rollouts(tiny_adapter: MeleePolicyAdapter, two_port: bool) -> None:
    tiny_adapter.fast_step = True
    plain = _worker(tiny_adapter, packed=False, two_port=two_port)
    packed = _worker(tiny_adapter, packed=True, two_port=two_port)
    assert packed._packer is not None and packed._ring is not None and plain._packer is None
    for _ in range(3):
        _same_trajectory(plain.rollout(), packed.rollout())
    assert packed.timings["policy_s"] > 0.0
    tiny_adapter.fast_step = False


def test_a_policy_opponent_reads_its_own_packer_s_static_tree(tiny_adapter: MeleePolicyAdapter) -> None:
    """17 Sep 2026: the league smoke's delay-0 arm (an ``other`` checkpoint on 12 Dolphins) spent 44 s of a
    137-s rollout in its opponent call. The adapter captures its CUDA graph only for a
    :class:`StaticFrameTree` (``adapter.py:253-262``), and the actor handed a policy opponent a leaf-wise
    tree, so that opponent stepped eagerly. With packed frames the opponent reads its own packer's static
    tree, and the rollouts are bit-identical to the leaf-wise path."""

    tiny_adapter.fast_step = True
    rollouts: dict[bool, list[Trajectory]] = {}
    seen: dict[bool, list[object]] = {}
    for packed in (False, True):
        env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, game_frames=21, players=("policy", "policy")))
        rival = SelfOpponent(tiny_adapter, B, port=1, delay_frames=2, seed=11)
        frames_in: list[object] = []
        act = rival.act

        def recording(
            frames: Any, needs_reset: torch.Tensor, act: Any = act, into: list[object] = frames_in
        ) -> Any:
            into.append(frames)
            return act(frames, needs_reset)

        rival.act = recording  # type: ignore[method-assign]
        config = ActorConfig(rollout_length=T, context_frames=C, packed_frames=packed, seed=7)
        worker = RolloutWorker(tiny_adapter, env, rival, config)
        rollouts[packed] = [worker.rollout() for _ in range(3)]
        seen[packed] = frames_in
    for plain, packed_rollout in zip(rollouts[False], rollouts[True], strict=True):
        _same_trajectory(plain, packed_rollout)
    assert seen[False] and not any(isinstance(frames, StaticFrameTree) for frames in seen[False])
    assert len(seen[True]) == len(seen[False])
    assert all(isinstance(frames, StaticFrameTree) for frames in seen[True])
    assert len({id(frames) for frames in seen[True]}) == 1  # one static tree, overwritten every frame
    tiny_adapter.fast_step = False


def test_packed_frames_flag_round_trips_through_the_config() -> None:
    assert ActorConfig().packed_frames is False
    config = RLConfig(actor=ActorConfig(packed_frames=True))
    mapping = config_to_mapping(config)
    assert mapping["actor"]["packed_frames"] is True
    assert config_from_mapping(mapping).actor == config.actor
    assert replace(config.actor, packed_frames=False) == ActorConfig()
