"""Opponents (PLAN.md §3.8, P6a): self-opponent refresh, perspective swap, delay queue, cpu opponent, the
``other`` opponent from a checkpoint (ours and Ali's format), ``make_opponent`` rules."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from controller_codec import ControllerLabels, CustomV1Codec
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import save_bc_checkpoint, sha256_file
from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
from melee_rl.frames import FrameTree, random_valid_frames, tree_index, tree_leaves
from melee_rl.opponents import (
    OPPONENT_TYPES,
    CpuOpponent,
    Opponent,
    OpponentConfig,
    OtherOpponent,
    PolicyOpponent,
    SelfOpponent,
    make_opponent,
)
from model import MeleePolicy, ModelConfig

B = 2
C = 16
T = 8
CODEC = CustomV1Codec()


class RecordingSelfOpponent(SelfOpponent):
    """Records the trees it is shown and the labels it returns."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[FrameTree] = []
        self.returned: list[ControllerLabels] = []

    def act(self, frames: Any, needs_reset: torch.Tensor) -> ControllerLabels:
        self.seen.append(frames)
        labels = super().act(frames, needs_reset)
        self.returned.append(labels)
        return labels


def _same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def tree_leaves_row(tree: FrameTree, row: int) -> FrameTree:
    return tree_index(tree, (slice(None), row))


def test_self_opponent_refresh_and_perspective_swap(tiny_adapter: MeleePolicyAdapter) -> None:
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, players=("policy", "policy"), game_frames=30))
    opponent = RecordingSelfOpponent(tiny_adapter, B, delay_frames=1, seed=3)
    assert isinstance(opponent, Opponent) and opponent.controls_port and opponent.port == 1
    assert isinstance(opponent, PolicyOpponent)
    policy = opponent.policy
    assert isinstance(policy, MeleePolicyAdapter)
    assert policy is not tiny_adapter and not policy.training
    worker = RolloutWorker(
        tiny_adapter,
        env,
        opponent,
        ActorConfig(rollout_length=T, context_frames=C, delay_frames=1, seed=7),
    )
    trajectory = worker.rollout()
    trajectory.validate()
    assert len(opponent.seen) == T
    # The opponent saw port 1's perspective: its p0 is the student's p1 of the same row.
    for step, seen in enumerate(opponent.seen):
        row = C + step
        pairs = ((seen["p0"], trajectory.frames["p1"]), (seen["p1"], trajectory.frames["p0"]))
        for mine, theirs in pairs:
            for (path, leaf), (_, other) in zip(tree_leaves(mine), tree_leaves(theirs), strict=True):
                assert torch.equal(leaf, other[:, row]), (step, path)
    # Its returned labels (delayed by one) were executed by player 1 on the next transition.
    executed = CODEC.encode(trajectory.frames["p1"]["controller"])
    for step, labels in enumerate(opponent.returned):
        row = C + step + 1
        assert torch.equal(executed.buttons[:, row], labels.buttons), step
        assert torch.equal(executed.main_stick[:, row], labels.main_stick), step
    assert int(opponent.returned[0].buttons.abs().sum()) == 0  # the neutral pre-fill (D = 1)
    assert any(int(labels.buttons.abs().sum()) > 0 for labels in opponent.returned[1:])

    # Refresh copies the student's parameters; before it the copies differ.
    with torch.no_grad():
        for parameter in tiny_adapter.parameters():
            parameter.add_(0.05)
    student = tiny_adapter.get_state()
    before = opponent.policy.get_state()
    assert any(not torch.equal(before[name], student[name]) for name in student)
    opponent.refresh(tiny_adapter)
    after = opponent.policy.get_state()
    assert all(torch.equal(after[name], student[name]) for name in student)
    assert not policy.training and all(not p.requires_grad for p in policy.parameters())
    # The student is unchanged by the opponent's rollouts and refresh.
    assert all(torch.equal(tiny_adapter.get_state()[name], student[name]) for name in student)


def test_self_opponent_deterministic_and_reset_state(tiny_adapter: MeleePolicyAdapter) -> None:
    frames = random_valid_frames((B, 4), torch.Generator().manual_seed(3))
    reset = torch.zeros(B, dtype=torch.bool)
    first = SelfOpponent(tiny_adapter, B, delay_frames=2, seed=11)
    second = SelfOpponent(tiny_adapter, B, delay_frames=2, seed=11)
    outputs_a = []
    outputs_b = []
    for step in range(4):
        frame = tree_leaves_row(frames, step)
        flag = torch.ones(B, dtype=torch.bool) if step == 0 else reset
        outputs_a.append(first.act(frame, flag))
        outputs_b.append(second.act(frame, flag))
    for a, b in zip(outputs_a, outputs_b, strict=True):
        assert torch.equal(a.buttons, b.buttons) and torch.equal(a.main_stick, b.main_stick)
    assert int(outputs_a[0].buttons.abs().sum()) == 0 and int(outputs_a[1].buttons.abs().sum()) == 0
    first.reset_state()
    again = first.act(tree_leaves_row(frames, 0), torch.ones(B, dtype=torch.bool))
    assert int(again.buttons.abs().sum()) == 0 and int(again.main_stick.abs().sum()) == 0
    with pytest.raises(ValueError, match="delay_frames"):
        SelfOpponent(tiny_adapter, B, delay_frames=-1)


def test_cpu_opponent_make_opponent_and_config(tiny_adapter: MeleePolicyAdapter) -> None:
    cpu = CpuOpponent(cpu_level=7)
    assert isinstance(cpu, Opponent) and not cpu.controls_port and cpu.port == 1 and cpu.cpu_level == 7
    assert cpu.act({}, torch.zeros(1, dtype=torch.bool)) is None
    cpu.refresh(tiny_adapter)
    cpu.reset_state()
    made = make_opponent(OpponentConfig(type="cpu", cpu_level=3), tiny_adapter, batch_size=B, delay_frames=0)
    assert isinstance(made, CpuOpponent) and made.cpu_level == 3
    made_self = make_opponent(
        OpponentConfig(type="self", temperature=0.5, seed=9), tiny_adapter, batch_size=B, delay_frames=2
    )
    assert isinstance(made_self, SelfOpponent)
    assert made_self.delay_frames == 2 and made_self.temperature == 0.5 and made_self.batch_size == B
    with pytest.raises(ValueError, match="opponent type"):
        OpponentConfig(type="human")
    with pytest.raises(ValueError, match="cpu_level"):
        OpponentConfig(cpu_level=10)
    with pytest.raises(ValueError, match="update_interval"):
        OpponentConfig(update_interval=0)


# ---------------------------------------------------------------------------
# P6a: the ``other`` opponent and the two-port rules
# ---------------------------------------------------------------------------


def test_other_opponent_from_our_and_ali_checkpoints(
    tiny_adapter: MeleePolicyAdapter, tiny_config: ModelConfig, tmp_path: Path
) -> None:
    ours = tmp_path / "ours.pt"
    save_bc_checkpoint(ours, tiny_adapter.get_state(), tiny_adapter.config)
    opponent = OtherOpponent(ours, B, delay_frames=1, seed=3)
    assert isinstance(opponent, Opponent) and isinstance(opponent, PolicyOpponent)
    assert opponent.controls_port and opponent.port == 1 and opponent.batch_size == B
    assert opponent.delay_frames == 1 and opponent.temperature == 1.0
    assert opponent.source == str(ours) and opponent.sha256 == sha256_file(ours)
    policy = opponent.policy
    assert isinstance(policy, MeleePolicyAdapter) and policy is not tiny_adapter
    assert not policy.training and all(not p.requires_grad for p in policy.parameters())
    assert policy.config.compute_dtype == "float32"
    before = policy.get_state()
    assert _same_state(before, tiny_adapter.get_state())
    # Deterministic: two opponents with the same seed return the same labels; D neutral pre-fills first.
    frames = random_valid_frames((B, 4), torch.Generator().manual_seed(3))
    twin = OtherOpponent(ours, B, delay_frames=1, seed=3)
    outputs: list[ControllerLabels] = []
    for step in range(4):
        frame = tree_leaves_row(frames, step)
        flag = torch.ones(B, dtype=torch.bool) if step == 0 else torch.zeros(B, dtype=torch.bool)
        mine, theirs = opponent.act(frame, flag), twin.act(frame, flag)
        assert torch.equal(mine.buttons, theirs.buttons) and torch.equal(mine.main_stick, theirs.main_stick)
        outputs.append(mine)
    assert int(outputs[0].buttons.abs().sum()) == 0 and int(outputs[0].main_stick.abs().sum()) == 0
    assert any(int(labels.buttons.abs().sum()) > 0 for labels in outputs[1:])
    # ``refresh`` is a no-op: the student may move, the frozen opponent does not follow.
    with torch.no_grad():
        for parameter in tiny_adapter.parameters():
            parameter.add_(0.05)
    opponent.refresh(tiny_adapter)
    assert _same_state(opponent.policy.get_state(), before)
    assert not _same_state(opponent.policy.get_state(), tiny_adapter.get_state())
    # It plays port 1 in a rollout on the two-port dummy env like the self-opponent does.
    env = DummyMeleeEnv(DummyEnvConfig(num_envs=B, seed=5, players=("policy", "policy"), game_frames=30))
    worker = RolloutWorker(
        tiny_adapter, env, opponent, ActorConfig(rollout_length=T, context_frames=C, delay_frames=1, seed=7)
    )
    trajectory = worker.rollout()
    trajectory.validate()
    assert int(CODEC.encode(trajectory.frames["p1"]["controller"]).buttons[:, C + 2 :].abs().sum()) > 0
    opponent.reset_state()
    # Ali's training checkpoint (format_version 1) works the same way.
    import optimizers as ali_optimizers
    import training as ali_training

    torch.manual_seed(9)
    ali_policy = MeleePolicy(tiny_config)
    ali_path = tmp_path / "ali.pt"
    ali_training.save_checkpoint(
        ali_path,
        ali_policy,
        ali_optimizers.build_adamw(ali_policy, learning_rate=1e-3),
        ali_training.TrainingState(),
        resolved_config={"model": asdict(tiny_config)},
        metadata={"run_name": "bc"},
    )
    from_ali = OtherOpponent(ali_path, B, seed=1)
    expected = {name: value.detach().to("cpu", copy=True) for name, value in ali_policy.state_dict().items()}
    assert _same_state(from_ali.policy.get_state(), expected)
    assert from_ali.source == str(ali_path) and from_ali.sha256 == sha256_file(ali_path)
    assert not _same_state(from_ali.policy.get_state(), before)
    with pytest.raises(ValueError, match="delay_frames"):
        OtherOpponent(ours, B, delay_frames=-1)


def test_make_opponent_rules_and_config_validation(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    assert OPPONENT_TYPES == ("cpu", "self", "other", "mix")
    assert OpponentConfig().train is False and OpponentConfig().checkpoint is None
    assert OpponentConfig().trained_ports == (0,) and OpponentConfig(type="self").trained_ports == (0,)
    assert OpponentConfig(type="self", train=True).trained_ports == (0, 1)
    # self + train: the worker plays both ports itself, there is no opponent object.
    both = make_opponent(OpponentConfig(type="self", train=True), tiny_adapter, batch_size=B, delay_frames=0)
    assert both is None
    plain = make_opponent(OpponentConfig(type="self"), tiny_adapter, batch_size=B, delay_frames=0)
    assert isinstance(plain, SelfOpponent)
    path = tmp_path / "weights.pt"
    save_bc_checkpoint(path, tiny_adapter.get_state(), tiny_adapter.config)
    config = OpponentConfig(type="other", checkpoint=str(path), temperature=0.5, seed=9)
    assert config.trained_ports == (0,) and config.delay_frames is None
    # 11 Sep 2026 (/delay-finetune): a frozen checkpoint plays at the delay its file was trained at (a BC
    # file: 0), whatever the run's delay is; ``opponent.delay_frames`` overrides it.
    other = make_opponent(config, tiny_adapter, batch_size=B, delay_frames=2, device="cpu")
    assert isinstance(other, OtherOpponent) and other.temperature == 0.5 and other.delay_frames == 0
    assert other.source == str(path) and other.batch_size == B and other.port == 1
    forced = make_opponent(replace(config, delay_frames=2), tiny_adapter, batch_size=B, delay_frames=0)
    assert isinstance(forced, OtherOpponent) and forced.delay_frames == 2
    with pytest.raises(ValueError, match="delay_frames"):
        OpponentConfig(type="cpu", delay_frames=2)
    with pytest.raises(ValueError, match="delay_frames"):
        OpponentConfig(type="other", checkpoint="x.pt", delay_frames=-1)
    with pytest.raises(ValueError, match="checkpoint"):
        OpponentConfig(type="other")
    with pytest.raises(ValueError, match="train"):
        OpponentConfig(type="cpu", train=True)
    with pytest.raises(ValueError, match="train"):
        OpponentConfig(type="other", checkpoint="x.pt", train=True)
    with pytest.raises(ValueError, match="checkpoint"):
        OpponentConfig(type="self", checkpoint="x.pt")
    with pytest.raises(ValueError, match="checkpoint"):
        OpponentConfig(type="cpu", checkpoint="x.pt")
    with pytest.raises(FileNotFoundError):
        make_opponent(
            OpponentConfig(type="other", checkpoint=str(tmp_path / "missing.pt")),
            tiny_adapter,
            batch_size=B,
            delay_frames=0,
        )
