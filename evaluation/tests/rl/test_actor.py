"""Rollout worker (PLAN.md §6 P1, P6a): delay queue D = 0/1/2, determinism, resets, parity on rollouts,
rewards, and two-port rollouts (both perspectives of one game in one batch)."""

from __future__ import annotations

import pytest
import torch

from controller_codec import CustomV1Codec
from melee_rl import distributions
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.env.protocol import swap_perspective
from melee_rl.frames import labels_index, tree_index, tree_leaves
from melee_rl.opponents import CpuOpponent, SelfOpponent
from melee_rl.trajectory import Trajectory

B = 3
C = 16
T = 8
CODEC = CustomV1Codec()


def _worker(
    adapter: MeleePolicyAdapter,
    *,
    delay: int = 0,
    mode: str = "reprime",
    game_frames: int = 10_000,
    seed: int = 7,
    env_seed: int = 5,
    learnable_signal: bool = False,
) -> RolloutWorker:
    env = DummyMeleeEnv(
        DummyEnvConfig(num_envs=B, seed=env_seed, game_frames=game_frames, learnable_signal=learnable_signal)
    )
    config = ActorConfig(rollout_length=T, context_frames=C, delay_frames=delay, context_mode=mode, seed=seed)
    return RolloutWorker(adapter, env, CpuOpponent(), config)


def _assert_delay_alignment(trajectory: Trajectory) -> None:
    """``sampled_labels[w] == controller_exec[w + 1 + D]`` wherever both rows are real frames."""

    delay = trajectory.delay_frames
    for row in range(trajectory.window_length - 1 - delay):
        both = trajectory.padding[:, row] & trajectory.padding[:, row + 1 + delay]
        sampled = labels_index(trajectory.sampled_labels, (slice(None), row))
        executed = labels_index(trajectory.controller_exec, (slice(None), row + 1 + delay))
        assert torch.equal(sampled.buttons[both], executed.buttons[both]), row
        assert torch.equal(sampled.main_stick[both], executed.main_stick[both]), row


def _assert_executed_matches_frames(trajectory: Trajectory) -> None:
    """The env echoes the executed controller, so re-encoding it gives ``controller_exec``."""

    encoded = CODEC.encode(trajectory.frames["p0"]["controller"])
    real = trajectory.padding
    assert torch.equal(encoded.buttons[real], trajectory.controller_exec.buttons[real])
    assert torch.equal(encoded.main_stick[real], trajectory.controller_exec.main_stick[real])


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_delay_queue_d0_d1_d2(tiny_adapter: MeleePolicyAdapter, delay: int) -> None:
    worker = _worker(tiny_adapter, delay=delay)
    assert worker.delay_frames == delay and worker.context_frames == C and worker.window_length == C + T + 1
    trajectories = [worker.rollout() for _ in range(3)]
    for rollout, trajectory in enumerate(trajectories):
        trajectory.validate()
        assert trajectory.delay_frames == delay and trajectory.context_mode == "reprime"
        _assert_delay_alignment(trajectory)
        _assert_executed_matches_frames(trajectory)
        # controller_prev_sample is the previous sample, neutral at reset rows and padding.
        for row in range(1, trajectory.window_length):
            prev = labels_index(trajectory.controller_prev_sample, (slice(None), row))
            last = labels_index(trajectory.sampled_labels, (slice(None), row - 1))
            real = trajectory.padding[:, row] & trajectory.padding[:, row - 1]
            keep = real & ~trajectory.is_resetting[:, row]
            assert torch.equal(prev.buttons[keep], last.buttons[keep]), row
            blank = ~keep
            assert int(prev.buttons[blank].abs().sum()) == 0 and int(prev.main_stick[blank].abs().sum()) == 0
        # Padding: the first rollout has no history, the second C - T rows, the third none.
        expected_padding = max(C - rollout * T, 0)
        assert trajectory.padding[:, :expected_padding].sum() == 0
        assert bool(trajectory.padding[:, expected_padding:].all())
        assert trajectory.delayed_actions.buttons.shape == (B, delay)
        assert torch.equal(
            trajectory.delayed_actions.buttons, trajectory.sampled_labels.buttons[:, C + T - delay : C + T]
        )
    first = trajectories[0]
    # The stream starts at a game start with D neutral pre-fills: exec rows C..C+D are neutral, then samples.
    assert bool(first.is_resetting[:, C].all()) and first.frame_index[:, C].tolist() == [-123] * B
    assert int(first.controller_exec.buttons[:, C : C + 1 + delay].abs().sum()) == 0
    assert torch.equal(first.controller_exec.buttons[:, C + 1 + delay], first.sampled_labels.buttons[:, C])
    assert int(first.sampled_labels.buttons[:, C : C + T].abs().sum()) > 0  # the policy did sample buttons
    assert worker.frames_seen == 3 * T * B and worker.rollouts_done == 3
    assert set(worker.timings) == {"rollout_s", "reprime_s", "policy_s", "env_s", "opponent_s"}
    assert worker.timings["reprime_s"] > 0.0


def test_rollouts_are_bitwise_deterministic(tiny_adapter: MeleePolicyAdapter) -> None:
    first = [_worker(tiny_adapter, delay=1, game_frames=20).rollout() for _ in range(1)]
    worker_a = _worker(tiny_adapter, delay=1, game_frames=20)
    worker_b = _worker(tiny_adapter, delay=1, game_frames=20)
    for _ in range(3):
        a, b = worker_a.rollout(), worker_b.rollout()
        for (path, leaf), (_, other) in zip(tree_leaves(a.frames), tree_leaves(b.frames), strict=True):
            assert torch.equal(leaf, other), path
        for name in ("buttons", "main_stick"):
            assert torch.equal(a.sampled_logits[name], b.sampled_logits[name])
            assert torch.equal(getattr(a.sampled_labels, name), getattr(b.sampled_labels, name))
            assert torch.equal(getattr(a.controller_exec, name), getattr(b.controller_exec, name))
        assert torch.equal(a.rewards, b.rewards) and torch.equal(a.is_resetting, b.is_resetting)
        assert torch.equal(a.frame_index, b.frame_index) and torch.equal(a.padding, b.padding)
    assert torch.equal(first[0].sampled_logits["buttons"], a.sampled_logits["buttons"]) is False
    # A different worker seed changes the samples.
    other = _worker(tiny_adapter, delay=1, game_frames=20, seed=8).rollout()
    assert not torch.equal(other.sampled_labels.buttons, first[0].sampled_labels.buttons)


@pytest.mark.parametrize("delay", [0, 2])
def test_actor_learner_parity_on_full_rollouts(tiny_adapter: MeleePolicyAdapter, delay: int) -> None:
    worker = _worker(tiny_adapter, delay=delay, game_frames=21)  # games restart inside the windows
    saw_reset_inside_rollout = False
    saw_padding = False
    for _ in range(4):
        trajectory = worker.rollout()
        trajectory.validate()
        saw_reset_inside_rollout |= bool(trajectory.is_resetting[:, C + 1 :].any())
        saw_padding |= bool((~trajectory.padding).any())
        window = trajectory.policy_window()
        assert window.targets is not None
        tiny_adapter.train()
        with torch.no_grad():
            learner = tiny_adapter.unroll_window(
                window.frames, window.controller_prev, window.targets, window.reset_mask, window.padding_mask
            )
        actor_logits = trajectory.actor_logits()
        for name in ("buttons", "main_stick"):
            learner_logits = learner.logits[name][:, C:]
            torch.testing.assert_close(learner_logits, actor_logits[name], rtol=1.0e-4, atol=1.0e-5)
        labels = trajectory.actor_labels()
        learner_log_prob = distributions.log_prob({n: v[:, C:] for n, v in learner.logits.items()}, labels)
        actor_log_prob = distributions.log_prob(actor_logits, labels)
        assert float((learner_log_prob - actor_log_prob).abs().max()) < 1.0e-5
        assert torch.isfinite(learner_log_prob).all()
    assert saw_reset_inside_rollout and saw_padding


def test_ring_mode_never_reprimes_and_mismatch_is_documented(tiny_adapter: MeleePolicyAdapter) -> None:
    """Ring mode: actor == learner iff a reset lies in window rows ``[0, C]`` (PLAN.md §3.2)."""

    worker = _worker(tiny_adapter, mode="ring")
    exact_pairs = mismatch_pairs = 0
    for _ in range(4):
        trajectory = worker.rollout()
        assert trajectory.context_mode == "ring" and worker.timings["reprime_s"] == 0.0
        window = trajectory.policy_window()
        assert window.targets is not None
        with torch.no_grad():
            learner = tiny_adapter.unroll_window(
                window.frames, window.controller_prev, window.targets, window.reset_mask, window.padding_mask
            )
        actor_logits = trajectory.actor_logits()
        for env in range(B):
            expected_exact = bool(trajectory.is_resetting[env, : C + 1].any())
            difference = max(
                float((learner.logits[name][env, C:] - actor_logits[name][env]).abs().max())
                for name in ("buttons", "main_stick")
            )
            if expected_exact:
                exact_pairs += 1
                assert difference < 1.0e-4, (env, difference)
            else:
                mismatch_pairs += 1
                assert difference > 1.0e-4, (env, difference)
    # Windows of rollouts 0-2 contain the game start (row 0 of the stream); rollout 3's does not.
    assert exact_pairs == 3 * B and mismatch_pairs == B


@pytest.mark.parametrize(
    ("delay", "game_frames"), [(0, 21), (2, 21), (0, 10_000)], ids=["d0", "d2", "d0-long-games"]
)
def test_prefix_mode_snapshots_the_ring_and_is_exact(
    tiny_adapter: MeleePolicyAdapter, delay: int, game_frames: int
) -> None:
    """Prefix mode: no re-prime, the trajectory carries the ring as it stood before the rollout's first
    sample, and the learner's prefix forward over the decision rows alone reproduces the actor exactly --
    with games restarting inside the windows (21 frames) and, in long games, for the rollout whose window
    holds no game start (the mismatch ``test_ring_mode_never_reprimes_and_mismatch_is_documented`` shows)."""

    worker = _worker(tiny_adapter, delay=delay, mode="prefix", game_frames=game_frames)
    saw_reset_inside_rollout = saw_padding = saw_full_window_without_reset = False
    for _ in range(4):
        trajectory = worker.rollout()
        trajectory.validate()
        assert trajectory.context_mode == "prefix" and worker.timings["reprime_s"] == 0.0
        cache = trajectory.initial_cache
        assert cache is not None and cache.batch_size == B and cache.keys.device.type == "cpu"
        # The snapshot precedes the first decision row: its positions count the frames since each env's
        # game start, exactly what the first decision frame's index says (unless that row restarts).
        first = ~trajectory.is_resetting[:, C]
        expected_positions = trajectory.frame_index[:, C] + 123
        assert torch.equal(cache.next_position[first], expected_positions[first])
        saw_reset_inside_rollout |= bool(trajectory.is_resetting[:, C + 1 :].any())
        saw_padding |= bool((~trajectory.padding).any())
        saw_full_window_without_reset |= bool((~trajectory.is_resetting[:, : C + 1].any(dim=1)).any())
        window = trajectory.prefix_window()
        assert window.targets is not None and window.length == T - delay
        tiny_adapter.train()
        with torch.no_grad():
            learner = tiny_adapter.unroll_window(
                window.frames,
                window.controller_prev,
                window.targets,
                window.reset_mask,
                window.padding_mask,
                prefix=cache,
            )
        actor_logits = trajectory.actor_logits()
        for name in ("buttons", "main_stick"):
            torch.testing.assert_close(learner.logits[name], actor_logits[name], rtol=1.0e-4, atol=1.0e-5)
        labels = trajectory.actor_labels()
        learner_log_prob = distributions.log_prob(learner.logits, labels)
        actor_log_prob = distributions.log_prob(actor_logits, labels)
        assert float((learner_log_prob - actor_log_prob).abs().max()) < 1.0e-5
        assert torch.isfinite(learner_log_prob).all()
    if game_frames == 21:
        assert saw_reset_inside_rollout and saw_padding
    else:
        assert saw_full_window_without_reset  # rollout 3: no reset in rows [0, C], a full ring


def test_reset_env_and_reset_history(tiny_adapter: MeleePolicyAdapter) -> None:
    worker = _worker(tiny_adapter, delay=1)
    worker.rollout()
    worker.rollout()
    worker.reset_env()
    trajectory = worker.rollout()
    trajectory.validate()
    assert int(trajectory.padding[:, :C].sum()) == 0 and bool(trajectory.padding[:, C:].all())
    assert bool(trajectory.is_resetting[:, C].all()) and trajectory.frame_index[:, C].tolist() == [-123] * B
    assert int(trajectory.controller_exec.buttons[:, C : C + 2].abs().sum()) == 0  # neutral queue again
    _assert_delay_alignment(trajectory)
    # Clearing the history of one env pads only that env (the others keep the T rows since the reset).
    worker.reset_history(torch.tensor([False, True, False]))
    trajectory = worker.rollout()
    assert int(trajectory.padding[1, :C].sum()) == 0 and bool(trajectory.padding[1, C:].all())
    assert int(trajectory.padding[0, : C - T].sum()) == 0 and bool(trajectory.padding[0, C - T :].all())
    assert bool(trajectory.padding[2, C - T :].all())


def test_rewards_combine_frame_reward_and_env_signal(tiny_adapter: MeleePolicyAdapter) -> None:
    worker = _worker(tiny_adapter, delay=1, game_frames=24, learnable_signal=True)
    target = worker.env.config.target_buttons if isinstance(worker.env, DummyMeleeEnv) else 0
    total_bonus = 0.0
    for _ in range(6):
        trajectory = worker.rollout()
        executed = trajectory.controller_exec.buttons[:, C + 1 :] == target
        resets = trajectory.is_resetting[:, C + 1 :]
        expected_bonus = torch.where(resets, 0.0, executed.to(torch.float32))
        torch.testing.assert_close(trajectory.env_rewards, expected_bonus)
        recomputed = trajectory.recompute_rewards()
        torch.testing.assert_close(recomputed.rewards, trajectory.rewards)
        frame_part = trajectory.rewards - trajectory.env_rewards
        assert bool((frame_part[resets] == 0).all())
        assert bool((frame_part > -2).all()) and bool((frame_part < 2).all())
        total_bonus += float(trajectory.env_rewards.sum())
    assert total_bonus >= 0.0


def test_worker_validation(tiny_adapter: MeleePolicyAdapter) -> None:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=1))
    good = ActorConfig(rollout_length=T, context_frames=C)
    with pytest.raises(ValueError, match="not controlled"):
        RolloutWorker(tiny_adapter, env, CpuOpponent(port=0), good, ports=(1,))
    with pytest.raises(ValueError, match="different port"):
        RolloutWorker(tiny_adapter, env, CpuOpponent(port=0), good)
    with pytest.raises(ValueError, match="controls_port"):
        RolloutWorker(tiny_adapter, env, SelfOpponent(tiny_adapter, B), good)
    with pytest.raises(ValueError, match="must fit in"):
        RolloutWorker(tiny_adapter, env, CpuOpponent(), ActorConfig(rollout_length=T, context_frames=17))
    with pytest.raises(ValueError, match="must fit in"):
        RolloutWorker(tiny_adapter, env, CpuOpponent(), ActorConfig(rollout_length=25))
    assert ActorConfig(rollout_length=T).resolve_context_frames(24) == 24 - T
    with pytest.raises(ValueError, match="delay_frames"):
        ActorConfig(rollout_length=T, delay_frames=T)
    with pytest.raises(ValueError, match="context_mode"):
        ActorConfig(context_mode="cache")
    with pytest.raises(ValueError, match="temperature"):
        ActorConfig(temperature=-1.0)
    two_port = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=1, players=("policy", "policy")))
    with pytest.raises(ValueError, match="controls_port"):
        RolloutWorker(tiny_adapter, two_port, CpuOpponent(), good)


# ---------------------------------------------------------------------------
# P6a: two-port rollouts (slippi-ai ``opponent.train = True``: both perspectives are trained)
# ---------------------------------------------------------------------------


def _two_port_worker(adapter: MeleePolicyAdapter, *, delay: int = 0, game_frames: int = 21) -> RolloutWorker:
    env = DummyMeleeEnv(
        DummyEnvConfig(num_envs=B, seed=5, players=("policy", "policy"), game_frames=game_frames)
    )
    config = ActorConfig(rollout_length=T, context_frames=C, delay_frames=delay, seed=7)
    return RolloutWorker(adapter, env, None, config, ports=(0, 1))


def test_two_port_rollout_rows_are_swapped_perspectives(tiny_adapter: MeleePolicyAdapter) -> None:
    worker = _two_port_worker(tiny_adapter, delay=1)
    assert worker.ports == (0, 1) and worker.envs == B and worker.batch_size == 2 * B
    assert worker.opponent is None and worker.port == 0
    trajectories = [worker.rollout() for _ in range(3)]
    for trajectory in trajectories:
        trajectory.validate()
        assert trajectory.batch_size == 2 * B
        # Rows [E, 2E) are the swapped perspective of rows [0, E): port-major, p0 = the row's own player.
        first = tree_index(trajectory.frames, slice(0, B))
        second = tree_index(trajectory.frames, slice(B, 2 * B))
        for (path, leaf), (_, other) in zip(
            tree_leaves(second), tree_leaves(swap_perspective(first)), strict=True
        ):
            assert torch.equal(leaf, other), path
        # The zero-sum reward negates between the two perspectives; no env signal in the dummy.
        torch.testing.assert_close(trajectory.rewards[B:], -trajectory.rewards[:B], rtol=0.0, atol=0.0)
        assert int(trajectory.env_rewards.abs().sum()) == 0
        for name in ("is_resetting", "padding", "frame_index"):
            value = getattr(trajectory, name)
            assert torch.equal(value[:B], value[B:]), name
        # Every row's executed controller is the one its own player saw (the dummy echoes both ports).
        _assert_executed_matches_frames(trajectory)
        _assert_delay_alignment(trajectory)
    assert bool(trajectories[0].is_resetting[:, C].all()) and trajectories[0].frame_index[:, C].tolist() == [
        -123
    ] * (2 * B)
    # Both perspectives sample independently (two rows per env); env frames are counted once.
    last = trajectories[-1]
    assert not torch.equal(last.sampled_labels.buttons[:B, C:], last.sampled_labels.buttons[B:, C:])
    assert worker.frames_seen == 3 * T * B and worker.rollouts_done == 3
    assert any(bool(t.is_resetting[:, C + 1 :].any()) for t in trajectories)  # games restart inside windows
    # The rewards of the p0 rows are the same as a one-port worker would compute on the same frames.
    recomputed = last.recompute_rewards()
    torch.testing.assert_close(recomputed.rewards, last.rewards)


@pytest.mark.parametrize("delay", [0, 2])
def test_two_port_actor_learner_parity(tiny_adapter: MeleePolicyAdapter, delay: int) -> None:
    worker = _two_port_worker(tiny_adapter, delay=delay)
    saw_reset_inside_rollout = saw_padding = False
    for _ in range(4):
        trajectory = worker.rollout()
        trajectory.validate()
        saw_reset_inside_rollout |= bool(trajectory.is_resetting[:, C + 1 :].any())
        saw_padding |= bool((~trajectory.padding).any())
        window = trajectory.policy_window()
        assert window.targets is not None and window.reset_mask.shape[0] == 2 * B
        tiny_adapter.train()
        with torch.no_grad():
            learner = tiny_adapter.unroll_window(
                window.frames, window.controller_prev, window.targets, window.reset_mask, window.padding_mask
            )
        actor_logits = trajectory.actor_logits()
        for name in ("buttons", "main_stick"):
            torch.testing.assert_close(
                learner.logits[name][:, C:], actor_logits[name], rtol=1.0e-4, atol=1.0e-5
            )
        labels = trajectory.actor_labels()
        learner_log_prob = distributions.log_prob({n: v[:, C:] for n, v in learner.logits.items()}, labels)
        actor_log_prob = distributions.log_prob(actor_logits, labels)
        assert float((learner_log_prob - actor_log_prob).abs().max()) < 1.0e-5
    assert saw_reset_inside_rollout and saw_padding


def test_two_port_reset_history_tiling_and_validation(tiny_adapter: MeleePolicyAdapter) -> None:
    worker = _two_port_worker(tiny_adapter, game_frames=10_000)
    worker.rollout()
    worker.rollout()
    # An [E] mask applies to both perspectives of that env; the other envs keep T rows of history.
    worker.reset_history(torch.tensor([False, True, False]))
    trajectory = worker.rollout()
    for row in (1, B + 1):
        assert int(trajectory.padding[row, :C].sum()) == 0 and bool(trajectory.padding[row, C:].all()), row
    for row in (0, 2, B, B + 2):  # the other envs keep their full history (24 rows since the start)
        assert bool(trajectory.padding[row].all()), row
    # A [P * E] mask is taken row by row.
    rows = torch.zeros(2 * B, dtype=torch.bool)
    rows[B] = True
    worker.reset_history(rows)
    trajectory = worker.rollout()
    assert int(trajectory.padding[B, :C].sum()) == 0 and bool(trajectory.padding[B, C:].all())
    assert bool(trajectory.padding[0, C - T :].all()) and bool(trajectory.padding[1, C - T :].all())
    with pytest.raises(ValueError, match="mask"):
        worker.reset_history(torch.ones(5, dtype=torch.bool))
    worker.reset_env()  # no opponent to reset; every row starts a new game
    trajectory = worker.rollout()
    assert bool(trajectory.is_resetting[:, C].all()) and int(trajectory.padding[:, :C].sum()) == 0
    # Validation of the port / opponent combinations.
    two_port = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=1, players=("policy", "policy")))
    one_port = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=1))
    good = ActorConfig(rollout_length=T, context_frames=C)
    with pytest.raises(ValueError, match="opponent"):
        RolloutWorker(tiny_adapter, two_port, None, good)  # ports (0,) leave port 1 without a player
    with pytest.raises(ValueError, match="trained port"):
        RolloutWorker(tiny_adapter, two_port, SelfOpponent(tiny_adapter, B), good, ports=(0, 1))
    with pytest.raises(ValueError, match="ports"):
        RolloutWorker(tiny_adapter, two_port, None, good, ports=(0, 0))
    with pytest.raises(ValueError, match="ports"):
        RolloutWorker(tiny_adapter, two_port, None, good, ports=())
    with pytest.raises(ValueError, match="not controlled"):
        RolloutWorker(tiny_adapter, two_port, None, good, ports=(0, 2))
    with pytest.raises(ValueError, match="not controlled"):
        RolloutWorker(tiny_adapter, one_port, None, good, ports=(1,))
    with pytest.raises(ValueError, match="different port"):
        RolloutWorker(tiny_adapter, two_port, SelfOpponent(tiny_adapter, B, port=0), good, ports=(0,))
    # A cpu env with ``opponent=None``: the environment drives port 1 itself (same as ``CpuOpponent``).
    implicit = RolloutWorker(tiny_adapter, one_port, None, good)
    assert implicit.opponent is None and implicit.ports == (0,) and implicit.batch_size == B
    implicit.rollout().validate()
    implicit.reset_env()
    assert implicit.rollout().batch_size == B


def test_decision_stride_holds_the_executed_controller(tiny_adapter: MeleePolicyAdapter) -> None:
    """Lever 8 check (ii), 5 Sep 2026: ``decision_stride = k`` executes the sample of every k-th frame and
    holds it for the k - 1 frames between (a reset row forces a decision); the model still sees every frame,
    and its previous-action input is the executed (held) controller.  ``k = 1`` is today's loop, bit for
    bit."""

    def build(stride: int) -> RolloutWorker:
        env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, game_frames=10_000))
        config = ActorConfig(rollout_length=T, context_frames=C, decision_stride=stride, seed=7)
        return RolloutWorker(tiny_adapter, env, CpuOpponent(), config)

    with pytest.raises(ValueError, match="decision_stride"):
        ActorConfig(decision_stride=0)
    assert ActorConfig().decision_stride == 1
    baseline = _worker(tiny_adapter).rollout()
    same = build(1).rollout()
    for name in ("controller_exec", "controller_prev_sample", "sampled_labels"):
        assert torch.equal(getattr(same, name).buttons, getattr(baseline, name).buttons), name
        assert torch.equal(getattr(same, name).main_stick, getattr(baseline, name).main_stick), name
    assert torch.equal(same.rewards, baseline.rewards)

    stride = 3
    worker = build(stride)
    trajectory = worker.rollout()
    _assert_executed_matches_frames(trajectory)
    exec_ = trajectory.controller_exec
    sampled = trajectory.sampled_labels
    prev = trajectory.controller_prev_sample
    resetting = trajectory.is_resetting
    decisions = 0
    holds = 0
    for t in range(T - 1):
        row = C + t
        into_next = labels_index(exec_, (slice(None), row + 1))
        if t % stride == 0:  # a decision frame: the sample of frame t executes into frame t + 1
            expected = labels_index(sampled, (slice(None), row))
            decisions += 1
        else:  # a held frame: whatever executed into frame t executes into t + 1 (unless t + 1 reset)
            expected = labels_index(exec_, (slice(None), row))
            holds += 1
        keep = ~resetting[:, row + 1]
        assert torch.equal(into_next.buttons[keep], expected.buttons[keep]), t
        assert torch.equal(into_next.main_stick[keep], expected.main_stick[keep]), t
        # The policy's previous-action input at frame t + 1 is what the game executed into it.
        prev_next = labels_index(prev, (slice(None), row + 1))
        assert torch.equal(prev_next.buttons[keep], into_next.buttons[keep]), t
        assert torch.equal(prev_next.main_stick[keep], into_next.main_stick[keep]), t
    assert decisions == 3 and holds == 4  # T = 8: decisions at frames 0, 3, 6
    # The stride's phase runs across rollouts (a per-worker frame counter), so the next rollout's first
    # frame (global frame 8) is a held frame, and a reset resets the counter with the held state.
    second = worker.rollout()
    first_row = labels_index(second.controller_exec, (slice(None), C + 1))
    held_from = labels_index(second.controller_exec, (slice(None), C))
    keep = ~second.is_resetting[:, C + 1]
    assert torch.equal(first_row.buttons[keep], held_from.buttons[keep])
    worker.reset_env()
    third = worker.rollout()
    again = labels_index(third.controller_exec, (slice(None), C + 1))
    assert torch.equal(again.buttons, labels_index(third.sampled_labels, (slice(None), C)).buttons)
