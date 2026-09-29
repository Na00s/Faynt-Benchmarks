"""Dummy env (PLAN.md §6 P1): determinism, reset semantics, game length, dynamics, perspectives, signal."""

from __future__ import annotations

import pytest
import torch

from controller_codec import ButtonState, ControllerLabels, ControllerState, CustomV1Codec, StickState
from melee_rl import reward as reward_lib
from melee_rl.env import dummy
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.env.protocol import (
    INITIAL_FRAME_INDEX,
    EnvOutput,
    EnvProtocol,
    canonical_frame_tree,
    swap_perspective,
)
from melee_rl.frames import FrameTree, neutral_labels, tree_leaves, tree_stack, validate_frame_tree

CODEC = CustomV1Codec()


def _controller(
    batch: int,
    *,
    stick_x: float = 0.5,
    stick_y: float = 0.5,
    a: bool = False,
    y: bool = False,
    left: bool = False,
) -> ControllerState:
    half = torch.full((batch,), 0.5)
    falses = torch.zeros(batch, dtype=torch.bool)
    return ControllerState(
        main_stick=StickState(x=torch.full((batch,), stick_x), y=torch.full((batch,), stick_y)),
        c_stick=StickState(x=half, y=half),
        shoulder=torch.zeros(batch),
        buttons=ButtonState(
            A=torch.full((batch,), a),
            B=falses,
            X=falses,
            Y=torch.full((batch,), y),
            Z=falses,
            L=torch.full((batch,), left),
            R=falses,
            D_UP=falses,
        ),
    )


def _neutral(batch: int) -> ControllerState:
    return CODEC.decode(neutral_labels((batch,)))


def _run(env: DummyMeleeEnv, steps: int, control: ControllerState | None = None) -> list[EnvOutput]:
    control = _neutral(env.num_envs) if control is None else control
    outputs = []
    for _ in range(steps):
        output = env.step({port: control for port in env.controlled_ports})
        output.validate(env.num_envs)
        outputs.append(output)
    return outputs


def _stack_port0(outputs: list[EnvOutput]) -> FrameTree:
    return tree_stack([output.frames[0] for output in outputs], dim=1)


def test_protocol_conformance_and_initial_state() -> None:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=2, seed=1))
    assert isinstance(env, EnvProtocol)
    assert env.num_envs == 2 and env.controlled_ports == (0,)
    output = env.current()
    output.validate(2)
    assert tuple(output.frames) == (0,) and tuple(output.rewards) == (0,)
    assert output.frame_index.tolist() == [INITIAL_FRAME_INDEX] * 2
    assert output.needs_reset.tolist() == [True, True]
    assert validate_frame_tree(output.frames[0]) == torch.Size((2,))
    tree = output.frames[0]
    assert bool((tree["p0"]["x"] < 0).all()) and bool((tree["p1"]["x"] > 0).all())
    assert tree["p0"]["facing"].tolist() == [True, True] and tree["p1"]["facing"].tolist() == [False, False]
    assert tree["p0"]["action"].tolist() == [dummy.ACTION_STANDING] * 2
    assert tree["p0"]["character"].tolist() == [2, 2] and tree["p1"]["character"].tolist() == [20, 20]
    assert tree["stage"].tolist() == [32, 32]
    assert tree["p0"]["controller"]["main_stick"]["x"].tolist() == [0.5, 0.5]
    assert not bool(tree["p0"]["controller"]["buttons"]["A"].any())
    assert not bool(tree["p0"]["nana"]["exists"].any()) and not bool(tree["items"]["exists"].any())
    # The neutral controller decoded from neutral labels is the neutral frame's controller.
    neutral = _neutral(2)
    assert neutral.main_stick.x.tolist() == [0.5, 0.5] and neutral.shoulder.tolist() == [0.0, 0.0]
    assert not any(bool(value.any()) for value in neutral.buttons.values())
    assert torch.equal(CODEC.encode(neutral).buttons, torch.zeros(2, dtype=torch.long))


def test_determinism_same_seed_identical_and_independent_of_batch() -> None:
    config = DummyEnvConfig(num_envs=2, seed=3, game_frames=80)
    first = _run(DummyMeleeEnv(config), 180)
    second = _run(DummyMeleeEnv(config), 180)
    wider = _run(DummyMeleeEnv(DummyEnvConfig(num_envs=3, seed=3, game_frames=80)), 180)
    for out_a, out_b, out_c in zip(first, second, wider, strict=True):
        assert torch.equal(out_a.frame_index, out_b.frame_index)
        assert torch.equal(out_a.frame_index, out_c.frame_index[:2])
        leaves = zip(
            tree_leaves(out_a.frames[0]),
            tree_leaves(out_b.frames[0]),
            tree_leaves(out_c.frames[0]),
            strict=True,
        )
        for (path, leaf_a), (_, leaf_b), (_, leaf_c) in leaves:
            assert torch.equal(leaf_a, leaf_b), path
            assert torch.equal(leaf_a, leaf_c[:2]), path
    other = DummyMeleeEnv(DummyEnvConfig(num_envs=2, seed=4, game_frames=80)).current()
    assert not torch.equal(other.frames[0]["p0"]["x"], DummyMeleeEnv(config).current().frames[0]["p0"]["x"])
    # Something actually happened in 180 frames: damage, and two game restarts (game_frames = 80).
    frames = _stack_port0(first)
    assert int(reward_lib.damages(frames["p0"]["percent"]).sum()) > 0
    assert sum(int(output.needs_reset[0]) for output in first) == 2


def test_reset_semantics_and_fixed_game_length() -> None:
    config = DummyEnvConfig(num_envs=2, seed=0, game_frames=40, cpu_attack_every=0, cpu_jump_every=0)
    env = DummyMeleeEnv(config)
    outputs = [env.current(), *_run(env, 100)]
    index = torch.stack([output.frame_index[0] for output in outputs])
    resets = torch.stack([output.needs_reset[0] for output in outputs])
    expected = torch.arange(101) % 40 + INITIAL_FRAME_INDEX
    assert torch.equal(index, expected)
    assert torch.equal(resets, index == INITIAL_FRAME_INDEX)
    assert resets.nonzero().flatten().tolist() == [0, 40, 80]
    # Nobody dies without attacks; percents stay 0 and the players keep their stocks (no dying states).
    frames = _stack_port0(outputs)
    assert int(frames["p0"]["percent"].sum()) == 0 and int(frames["p1"]["percent"].sum()) == 0
    assert not bool(reward_lib.is_dying(frames["p0"]["action"]).any())
    # A hard reset starts new games immediately.
    mid = env.reset()
    mid.validate(2)
    assert mid.needs_reset.tolist() == [True, True] and mid.frame_index.tolist() == [-123, -123]
    after = env.step({0: _neutral(2)})
    assert after.frame_index.tolist() == [-122, -122] and not bool(after.needs_reset.any())
    env.close()


def test_deaths_respawn_stocks_and_game_over() -> None:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=1, seed=2, stocks=2, game_frames=10_000))
    outputs = [env.current(), *_run(env, 600)]
    frames = _stack_port0(outputs)
    action = frames["p0"]["action"][0]
    percent = frames["p0"]["percent"][0]
    resets = torch.stack([output.needs_reset[0] for output in outputs])
    death_edges = reward_lib.deaths(action).nonzero().flatten().tolist()
    assert len(death_edges) >= 2, death_edges
    first, second = death_edges[:2]
    # The death starts the frame after the percent crossed the threshold, lasts DYING_FRAMES frames,
    # then RESPAWN_FRAMES halo frames at percent 0 with invulnerability, then play resumes.
    assert int(percent[first]) > 100 >= int(percent[first - 1])
    dying = action[first + 1 : first + 1 + dummy.DYING_FRAMES]
    assert dying.tolist() == [dummy.ACTION_DEAD_UP] * dummy.DYING_FRAMES
    respawn_row = first + 1 + dummy.DYING_FRAMES
    halo = action[respawn_row : respawn_row + dummy.RESPAWN_FRAMES]
    assert halo.tolist() == [dummy.ACTION_ON_HALO_DESCENT] * dummy.RESPAWN_FRAMES
    assert int(percent[respawn_row]) == 0
    assert bool(frames["p0"]["invulnerable"][0, respawn_row])
    assert float(frames["p0"]["x"][0, respawn_row]) == pytest.approx(-dummy.SPAWN_X)
    resumed = int(action[respawn_row + dummy.RESPAWN_FRAMES])
    assert resumed not in (dummy.ACTION_DEAD_UP, dummy.ACTION_ON_HALO_DESCENT)
    # Damage comes in ATTACK_DAMAGE steps and is counted by the reward.
    damage = reward_lib.damages(percent)
    assert set(damage[damage > 0].tolist()) == {float(dummy.ATTACK_DAMAGE)}
    # The second death was the last stock: after its dying frames the next output is a new game,
    # and any later death belongs to that new game.
    game_over_row = second + 1 + dummy.DYING_FRAMES
    assert not bool(resets[1:game_over_row].any())
    assert bool(resets[game_over_row])
    assert all(edge > game_over_row for edge in death_edges[2:])
    rewards = reward_lib.compute_rewards(frames)
    assert float(rewards[0, first]) == pytest.approx(-1.0)
    assert float(rewards[0, second]) == pytest.approx(-1.0)


def test_controls_move_jump_attack_and_shield() -> None:
    config = DummyEnvConfig(num_envs=1, seed=0, cpu_attack_every=0, cpu_jump_every=0, game_frames=10_000)
    env = DummyMeleeEnv(config)
    start = env.current().frames[0]
    # Walk right for 5 frames.
    walked = _run(env, 5, _controller(1, stick_x=1.0))
    x_after = float(walked[-1].frames[0]["p0"]["x"][0])
    assert x_after == pytest.approx(float(start["p0"]["x"][0]) + 5 * dummy.WALK_SPEED)
    assert walked[-1].frames[0]["p0"]["action"].tolist() == [dummy.ACTION_WALK_MIDDLE]
    assert bool(walked[-1].frames[0]["p0"]["facing"][0])
    # Walk left once: facing flips.
    left = _run(env, 1, _controller(1, stick_x=0.0))[0]
    assert not bool(left.frames[0]["p0"]["facing"][0])
    # Jump: one press leaves the ground with one jump left, then falls and lands with both jumps back.
    jumped = _run(env, 1, _controller(1, y=True))[0]
    player = jumped.frames[0]["p0"]
    assert not bool(player["on_ground"][0]) and int(player["jumps_left"][0]) == dummy.JUMPS - 1
    assert int(player["action"][0]) == dummy.ACTION_JUMPING_FORWARD
    airborne = _run(env, 40, _controller(1, y=True))  # holding Y does not re-jump (edge-triggered)
    actions = [int(output.frames[0]["p0"]["action"][0]) for output in airborne]
    assert dummy.ACTION_FALLING in actions and dummy.ACTION_LANDING in actions
    landed = airborne[actions.index(dummy.ACTION_LANDING)].frames[0]["p0"]
    assert bool(landed["on_ground"][0]) and int(landed["jumps_left"][0]) == dummy.JUMPS
    assert float(landed["y"][0]) == 0.0
    assert actions.count(dummy.ACTION_JUMPING_FORWARD) < 40
    # Attack when the cpu has walked into range: the cpu takes ATTACK_DAMAGE and is in hitstun.
    outputs = _run(env, 200)
    tree = outputs[-1].frames[0]
    assert float((tree["p1"]["x"] - tree["p0"]["x"]).abs()[0]) <= dummy.ATTACK_RANGE_X
    before = int(tree["p1"]["percent"][0])
    hit = _run(env, 1, _controller(1, a=True))[0].frames[0]
    assert int(hit["p1"]["percent"][0]) == before + dummy.ATTACK_DAMAGE
    assert int(hit["p1"]["action"][0]) == dummy.ACTION_DAMAGE_HIGH_1
    assert int(hit["p0"]["action"][0]) == dummy.ACTION_NEUTRAL_ATTACK_1
    assert bool(hit["p0"]["controller"]["buttons"]["A"][0])
    # Shield: the action changes and shield strength decays.
    shield_before = float(hit["p0"]["shield_strength"][0])
    shielded = _run(env, dummy.ATTACK_FRAMES + 2, _controller(1, left=True))[-1].frames[0]
    assert int(shielded["p0"]["action"][0]) == dummy.ACTION_SHIELD
    assert float(shielded["p0"]["shield_strength"][0]) < shield_before


def test_perspective_swap_and_two_controlled_ports() -> None:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=2, seed=5, players=("policy", "policy")))
    assert env.controlled_ports == (0, 1)
    output = env.current()
    assert tuple(output.frames) == (0, 1) and tuple(output.rewards) == (0, 1)
    for (path, leaf), (other_path, other) in zip(
        tree_leaves(output.frames[1]["p0"]), tree_leaves(output.frames[0]["p1"]), strict=True
    ):
        assert path == other_path and leaf is other
    assert output.frames[1]["stage"] is output.frames[0]["stage"]
    back = swap_perspective(swap_perspective(output.frames[0]))
    assert back["p0"] is output.frames[0]["p0"] and back["p1"] is output.frames[0]["p1"]
    canonical = canonical_frame_tree({**output.frames[0], "extra": torch.zeros(2)})
    assert "extra" not in canonical and canonical["p0"]["x"] is output.frames[0]["p0"]["x"]
    # Port 1's controller drives player 1 and is echoed in p1.controller (p0 of port 1's view).
    stepped = env.step({0: _neutral(2), 1: _controller(2, stick_x=0.0)})
    stepped.validate(2)
    assert bool((stepped.frames[0]["p1"]["x"] < output.frames[0]["p1"]["x"]).all())
    assert stepped.frames[1]["p0"]["controller"]["main_stick"]["x"].tolist() == [0.0, 0.0]
    assert stepped.frames[0]["p1"]["controller"]["main_stick"]["x"].tolist() == [0.0, 0.0]
    with pytest.raises(ValueError, match="needs controllers for ports"):
        env.step({0: _neutral(2)})
    with pytest.raises(ValueError, match="batch shape"):
        env.step({0: _neutral(3), 1: _neutral(3)})


def test_learnable_signal_bonus_and_reset_zeroing() -> None:
    target = 5
    config = DummyEnvConfig(num_envs=2, seed=1, learnable_signal=True, target_buttons=target, game_frames=6)
    env = DummyMeleeEnv(config)
    labels = ControllerLabels(buttons=torch.tensor([target, 0]), main_stick=torch.tensor([0, 0]))
    control = CODEC.decode(labels)
    output = env.step({0: control})
    assert output.rewards[0].tolist() == [1.0, 0.0]
    assert output.rewards[0].dtype == torch.float32
    assert torch.equal(CODEC.encode(output.frames[0]["p0"]["controller"]).buttons, labels.buttons)
    # The reset output carries no bonus even if the target label was sent.
    both = CODEC.decode(ControllerLabels(buttons=torch.tensor([target] * 2), main_stick=torch.tensor([0, 0])))
    outputs = [env.step({0: both}) for _ in range(5)]
    assert outputs[-1].needs_reset.tolist() == [True, True]
    assert outputs[-1].rewards[0].tolist() == [0.0, 0.0]
    assert outputs[-2].rewards[0].tolist() == [1.0, 1.0]
    plain = DummyMeleeEnv(DummyEnvConfig(num_envs=2, seed=1)).step({0: control})
    assert plain.rewards[0].tolist() == [0.0, 0.0]


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="players"):
        DummyEnvConfig(players=("policy", "human"))
    with pytest.raises(ValueError, match="stage"):
        DummyEnvConfig(stage=64)
    with pytest.raises(ValueError, match="characters"):
        DummyEnvConfig(characters=(2, 33))
    with pytest.raises(ValueError, match="target_buttons"):
        DummyEnvConfig(target_buttons=728)
    with pytest.raises(ValueError, match="num_envs"):
        DummyEnvConfig(num_envs=0)
    with pytest.raises(ValueError, match="game_frames"):
        DummyEnvConfig(game_frames=0)
    assert DummyEnvConfig(players=("cpu", "policy")).controlled_ports == (1,)
