"""The past-self pool (STRENGTH-PLAN.md §6, 11 Sep 2026): a ``pool*<n>`` mix arm plays the run's own
``step_<n>.pt`` snapshots on port 1 -- one snapshot per lockstep group, loaded in place every
``[opponent.pool] interval`` learner steps, drawn by prioritised fictitious self-play (AlphaStar's
``f_hard(x) = (1 - x)^2``) over the arm's KO share against each snapshot.  With no snapshot old enough it
plays the current self.  The table of KO shares and the draw generator survive a resume."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import load_checkpoint, save_bc_checkpoint
from melee_rl.config import CONFIG_DIR, RLConfig, finalize_config, load_config, resolve_model_config
from melee_rl.logging import format_summary, missing_metrics
from melee_rl.opponents import (
    MixGroup,
    OpponentConfig,
    PoolConfig,
    make_group_opponent,
    parse_mix_spec,
)
from melee_rl.pool import (
    PoolOpponent,
    ko_share,
    list_snapshots,
    pfsp_probabilities,
    pool_counts,
    thin_snapshots,
)
from melee_rl.train import RunOptions, Trainer, mix_env_configs

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
IGNORED_ON_COMPARE = ("timing/", "env/fps")


def _same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _comparable(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if not key.startswith(IGNORED_ON_COMPARE)}


def _mixed(base: RLConfig, mix: tuple[str, ...], *, pool: PoolConfig | None = None) -> RLConfig:
    """``base`` as a mixed run: ``[opponent] type = "mix"`` and the neutral two-policy player template."""

    opponent = replace(base.opponent, type="mix", mix=mix)
    if pool is not None:
        opponent = replace(opponent, pool=pool)
    return finalize_config(
        replace(
            base,
            opponent=opponent,
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
        ),
        resolve_model_config(base),
    )


def _shifted(adapter: MeleePolicyAdapter, shift: float) -> MeleePolicyAdapter:
    """A weight copy of ``adapter`` with every parameter moved by ``shift``."""

    copy = MeleePolicyAdapter.from_policy(adapter)
    with torch.no_grad():
        for parameter in copy.parameters():
            parameter.add_(shift)
    return copy


# ---------------------------------------------------------------------------
# the spec, the settings, the run-file rules
# ---------------------------------------------------------------------------


def test_pool_mix_spec_grammar() -> None:
    group = parse_mix_spec("pool*24")
    assert group == MixGroup(spec="pool*24", kind="pool", num_envs=24)
    assert group.name == "pool" and group.trained_ports == (0,) and group.players == ("policy", "policy")
    assert group.rows == 24
    assert parse_mix_spec("pool*4@old").name == "pool_old"
    for bad in ("pool", "pool*", "pool*0", "pool:x*2", "pools*2", "pool*2@not-an-identifier"):
        with pytest.raises(ValueError):
            parse_mix_spec(bad)


def test_pool_settings_validate_and_load_from_the_run_file(tmp_path: Path) -> None:
    assert PoolConfig() == PoolConfig(interval=8, min_age=32, max_candidates=0, floor=0.05, decay=0.5)
    assert OpponentConfig().pool == PoolConfig()
    bad: list[Callable[[], PoolConfig]] = [
        lambda: PoolConfig(interval=0),
        lambda: PoolConfig(min_age=-1),
        lambda: PoolConfig(max_candidates=-1),
        # max_candidates 0 keeps every snapshot; a limit keeps the oldest and the newest, so it is >= 2
        lambda: PoolConfig(max_candidates=1),
        lambda: PoolConfig(floor=0.0),
        lambda: PoolConfig(floor=1.5),
        lambda: PoolConfig(decay=-0.1),
        lambda: PoolConfig(decay=1.5),
    ]
    for make in bad:
        with pytest.raises(ValueError):
            make()

    text = SMOKE_TINY.read_text(encoding="utf-8")
    good = tmp_path / "pool.toml"
    good.write_text(text + "\n[opponent.pool]\ninterval = 2\nmin_age = 0\nfloor = 0.1\n", encoding="utf-8")
    assert load_config(good).opponent.pool == PoolConfig(interval=2, min_age=0, floor=0.1)
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(text + "\n[opponent.pool]\nsize = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(unknown)


def test_a_pool_arm_needs_the_snapshot_ladder() -> None:
    base = load_config(SMOKE_TINY)
    config = _mixed(base, ("self*1", "pool*2"))
    assert [group.kind for group in config.opponent.groups] == ["self", "pool"]
    envs = mix_env_configs(config)
    assert [sub.num_envs for sub in envs] == [1, 2] and envs[1].players == ("policy", "policy")
    no_snapshots = replace(base, runtime=replace(base.runtime, snapshot_interval_steps=0))
    with pytest.raises(ValueError, match="snapshot_interval_steps"):
        _mixed(no_snapshots, ("self*1", "pool*2"))
    _mixed(no_snapshots, ("self*1", "cpu9*2"))  # only a pool arm reads the ladder


# ---------------------------------------------------------------------------
# the pieces: the ladder, the KO share, the PFSP draw
# ---------------------------------------------------------------------------


def test_list_and_thin_snapshots(tmp_path: Path) -> None:
    for step in (0, 8, 16, 24, 32, 40):
        (tmp_path / f"step_{step:08d}.pt").write_bytes(b"x")
    for other in ("latest.pt", "init.pt", "best.pt", "best_16.pt", "step_00000048.pt.tmp", "step_x.pt"):
        (tmp_path / other).write_bytes(b"x")  # never a candidate: not a finished step_<n>.pt
    assert [step for step, _ in list_snapshots(tmp_path, 40, min_age=16)] == [0, 8, 16, 24]
    everything = list_snapshots(tmp_path, 40, min_age=0)
    assert [step for step, _ in everything] == [0, 8, 16, 24, 32, 40]
    assert everything[2][1] == tmp_path / "step_00000016.pt"
    assert list_snapshots(tmp_path, 7, min_age=0) == [(0, tmp_path / "step_00000000.pt")]
    assert list_snapshots(tmp_path / "missing", 40) == [] and list_snapshots(None, 40) == []

    assert thin_snapshots(everything, 0) == everything  # 0 keeps every snapshot
    assert thin_snapshots(everything, 6) == everything and thin_snapshots(everything, 9) == everything
    # evenly spaced, rounded half up
    assert [step for step, _ in thin_snapshots(everything, 3)] == [0, 24, 40]
    assert [step for step, _ in thin_snapshots(everything, 2)] == [0, 40]  # the oldest and the newest
    assert thin_snapshots([], 3) == []


def test_ko_share_and_pfsp_probabilities() -> None:
    assert ko_share(0.0, 0.0) == pytest.approx(0.5)
    assert ko_share(3.0, 1.0) == pytest.approx(4.0 / 6.0)
    # f_hard: (1 - 0.5)^2 = 0.25; (1 - 0.8)^2 = 0.04 floored to 0.05; never played = the highest weight.
    expected = [0.25 / 0.55, 0.05 / 0.55, 0.25 / 0.55]
    assert pfsp_probabilities([0.5, 0.8, None], floor=0.05) == pytest.approx(expected)
    assert pfsp_probabilities([None, None], floor=0.05) == pytest.approx([0.5, 0.5])
    assert pfsp_probabilities([0.0], floor=0.05) == pytest.approx([1.0])
    beaten_by, beats = pfsp_probabilities([0.2, 0.9], floor=0.05)
    assert beaten_by > beats  # the snapshot that takes our stocks is drawn most
    with pytest.raises(ValueError):
        pfsp_probabilities([], floor=0.05)


def test_pool_counts_read_the_stocks_off_the_rollout(tiny_adapter: MeleePolicyAdapter) -> None:
    from melee_rl.actor import ActorConfig, RolloutWorker
    from melee_rl.env.dummy import DummyEnvConfig, DummyMeleeEnv
    from melee_rl.reward import deaths

    env = DummyMeleeEnv(DummyEnvConfig(num_envs=2, seed=5, players=("policy", "policy"), game_frames=30))
    pool = PoolOpponent(tiny_adapter, 2, directory=None, config=PoolConfig(), seed=3)
    worker = RolloutWorker(tiny_adapter, env, pool, ActorConfig(rollout_length=8, context_frames=16, seed=7))
    trajectories = [worker.rollout() for _ in range(3)]
    kos, lost = pool_counts(trajectories)
    frames = [trajectory.rollout_frames() for trajectory in trajectories]
    assert kos == float(sum(int(deaths(tree["p1"]["action"]).sum()) for tree in frames))
    assert lost == float(sum(int(deaths(tree["p0"]["action"]).sum()) for tree in frames))


# ---------------------------------------------------------------------------
# the opponent: draws, in-place loads, the table, the state
# ---------------------------------------------------------------------------


def test_pool_opponent_draws_snapshots_and_loads_them_in_place(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    states: dict[int, dict[str, torch.Tensor]] = {}
    for step, shift in ((0, 0.01), (8, 0.02), (16, 0.03)):
        snapshot = _shifted(tiny_adapter, shift)
        save_bc_checkpoint(directory / f"step_{step:08d}.pt", snapshot.get_state(), snapshot.config)
        states[step] = snapshot.get_state()
    config = PoolConfig(interval=4, min_age=8, floor=0.05, decay=0.5)
    pool = PoolOpponent(tiny_adapter, 2, directory=directory, config=config, seed=3)
    assert pool.port == 1 and pool.batch_size == 2 and pool.controls_port and pool.directory == directory
    # Until the first draw the pool is a frozen copy of the student (the fast-path flags come with it).
    assert pool.snapshot_step is None and _same_state(pool.policy.get_state(), tiny_adapter.get_state())
    policy = pool.policy
    assert isinstance(policy, MeleePolicyAdapter) and policy is not tiny_adapter
    pointers = [parameter.data_ptr() for parameter in policy.parameters()]

    assert pool.maybe_draw(6, tiny_adapter) is False  # not a draw step (interval 4)
    assert pool.maybe_draw(8, tiny_adapter) is True  # step 8, min_age 8: only step 0 is old enough
    assert pool.snapshot_step == 0 and pool.candidates == 1 and pool.reloads == 1
    assert _same_state(policy.get_state(), states[0])
    assert [parameter.data_ptr() for parameter in policy.parameters()] == pointers  # loaded in place
    assert not policy.training and all(not parameter.requires_grad for parameter in policy.parameters())
    pool.record(6.0, 2.0)
    metrics = pool.metrics(8)
    assert metrics["snapshot_step"] == 0.0 and metrics["snapshot_age"] == 8.0
    assert metrics["candidates"] == 1.0 and metrics["scored"] == 1.0 and metrics["probability"] == 1.0
    assert metrics["ko_share"] == pytest.approx(7.0 / 10.0)
    assert metrics["min_ko_share"] == pytest.approx(7.0 / 10.0) == metrics["mean_ko_share"]

    # Drawn again: the old counts are decayed (x 0.5), the weights are already loaded (no file read).
    assert pool.maybe_draw(12, tiny_adapter) and pool.snapshot_step == 0 and pool.reloads == 1
    pool.record(2.0, 2.0)
    assert pool.table["step_00000000.pt"] == pytest.approx([3.0 + 2.0, 1.0 + 2.0, 2.0])
    assert pool.metrics(12)["min_ko_share"] == pytest.approx(ko_share(5.0, 3.0))

    # Two pools with one seed draw the same sequence; every draw's weights are the snapshot's.
    twin = PoolOpponent(tiny_adapter, 2, directory=directory, config=config, seed=3)
    for step, (kos, lost) in ((8, (6.0, 2.0)), (12, (2.0, 2.0))):
        twin.maybe_draw(step, tiny_adapter)
        twin.record(kos, lost)
    for step in (16, 20, 24, 28, 32):
        assert pool.maybe_draw(step, tiny_adapter) and twin.maybe_draw(step, tiny_adapter)
        drawn = pool.snapshot_step
        assert drawn is not None and drawn <= step - config.min_age and twin.snapshot_step == drawn
        assert _same_state(pool.policy.get_state(), states[drawn])
        pool.record(1.0, 3.0)
        twin.record(1.0, 3.0)
    assert pool.table == twin.table and set(pool.table) <= {f"step_{s:08d}.pt" for s in states}

    # refresh() is the resume path: it reloads the snapshot being played, not the student.
    with torch.no_grad():
        for parameter in policy.parameters():  # the pool's own policy object: loads never replace it
            parameter.zero_()
    pool.refresh(tiny_adapter)
    assert pool.snapshot_step is not None and _same_state(pool.policy.get_state(), states[pool.snapshot_step])

    # The bookkeeping round-trips through a checkpoint's plain containers, the draw generator included.
    fresh = PoolOpponent(tiny_adapter, 2, directory=directory, config=config, seed=99)
    fresh.restore(pool.pool_state())
    fresh.refresh(tiny_adapter)
    assert fresh.snapshot_step == pool.snapshot_step and fresh.table == pool.table
    assert _same_state(fresh.policy.get_state(), states[pool.snapshot_step])
    for step in (36, 40, 44):
        pool.maybe_draw(step, tiny_adapter)
        fresh.maybe_draw(step, tiny_adapter)
        assert fresh.snapshot_step == pool.snapshot_step


def test_pool_plays_the_current_self_until_a_snapshot_is_old_enough(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    student = _shifted(tiny_adapter, 0.05)
    for directory in (None, tmp_path / "empty"):
        pool = PoolOpponent(tiny_adapter, 2, directory=directory, config=PoolConfig(interval=2), seed=1)
        assert pool.maybe_draw(4, student) and pool.snapshot_step is None and pool.candidates == 0
        assert _same_state(pool.policy.get_state(), student.get_state())  # refreshed like a self-opponent
        pool.record(3.0, 1.0)
        assert pool.table == {}  # the current self is not a past self: nothing is scored
        metrics = pool.metrics(4)
        assert metrics["snapshot_step"] == -1.0 and metrics["snapshot_age"] == 0.0
        assert metrics["ko_share"] == pytest.approx(ko_share(3.0, 1.0)) and "min_ko_share" not in metrics
    # A snapshot that vanished after a resume falls back to the current self as well.
    gone = PoolOpponent(tiny_adapter, 2, directory=tmp_path / "empty", config=PoolConfig(), seed=1)
    gone.restore({"current": "step_00000008.pt", "table": {}, "draws": 1, "reloads": 1})
    gone.refresh(student)
    assert gone.snapshot_step is None and _same_state(gone.policy.get_state(), student.get_state())


def test_make_group_opponent_builds_the_pool(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    config = OpponentConfig(
        type="mix", mix=("self*1", "pool*2"), pool=PoolConfig(interval=2), temperature=0.5
    )
    pool = make_group_opponent(
        parse_mix_spec("pool*2"),
        config,
        tiny_adapter,
        delay_frames=1,
        seed=5,
        device=None,
        snapshot_dir=tmp_path,
    )
    assert isinstance(pool, PoolOpponent)
    assert pool.batch_size == 2 and pool.port == 1 and pool.delay_frames == 1 and pool.temperature == 0.5
    assert pool.directory == tmp_path and pool.config == PoolConfig(interval=2)


# ---------------------------------------------------------------------------
# the trainer: a self arm and a pool arm in one learner step, resume, determinism
# ---------------------------------------------------------------------------


def test_pool_arm_trains_with_the_self_arm_and_resumes(tmp_path: Path) -> None:
    base = load_config(SMOKE_TINY)
    base = replace(base, runtime=replace(base.runtime, snapshot_interval_steps=1))
    config = _mixed(base, ("self*1", "pool*2"), pool=PoolConfig(interval=1, min_age=1))
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert trainer.group_names == ("self", "pool")
    assert [worker.ports for worker in trainer.workers] == [(0, 1), (0,)]
    pool = trainer.opponents[1]
    assert isinstance(pool, PoolOpponent) and pool.directory == tmp_path / "run"
    result = trainer.run(4)
    batches, length = config.learner.ppo.num_batches, config.actor.rollout_length
    assert result.frames_seen == 4 * batches * (2 + 2) * length  # self*1: two rows, pool*2: one port each
    # Learner step s draws among the snapshots of steps <= s - 1 (min_age 1); step_<n>.pt is written after
    # step n, so steps 0 and 1 play the current self, step 2 plays step_1, step 3 plays step_1 or step_2.
    played = [row["arm/pool/snapshot_step"] for row in result.history]
    assert played[:3] == [-1.0, -1.0, 1.0] and played[3] in (1.0, 2.0)
    row = result.history[-1]
    assert missing_metrics(row) == []
    assert {
        "arm/pool/per_frame",
        "arm/pool/ko_diff_per_minute",
        "arm/pool/snapshot_age",
        "arm/pool/ko_share",
        "arm/pool/min_ko_share",
        "arm/pool/candidates",
        "arm/pool/probability",
    } <= set(row)
    assert row["arm/pool/candidates"] == 2.0
    assert "pool vs step" in format_summary(row)

    payload = load_checkpoint(tmp_path / "run" / "latest.pt")
    pools = payload["loop_state"]["pools"]
    assert pools[0] is None and pools[1]["current"] == pool.current and pools[1]["table"] == pool.table
    assert isinstance(payload["rng"]["opponents"][1], torch.Tensor)  # the pool's own sampling generator
    trainer.close()

    resumed = Trainer(config, options)
    again = resumed.opponents[1]
    assert isinstance(again, PoolOpponent) and resumed.resumed and resumed.step == 4
    assert again.current == pool.current and again.table == pool.table and again.draws == pool.draws
    assert pool.current is not None
    snapshot = load_checkpoint(tmp_path / "run" / pool.current)
    assert _same_state(again.policy.get_state(), dict(snapshot["state_dict"]))
    assert resumed.run(1).step == 5
    resumed.close()


def test_pool_runs_are_deterministic(tmp_path: Path) -> None:
    base = load_config(SMOKE_TINY)
    base = replace(base, runtime=replace(base.runtime, snapshot_interval_steps=1))
    config = _mixed(base, ("self*1", "pool*2"), pool=PoolConfig(interval=1, min_age=0))
    histories = []
    tables = []
    for name in ("first", "second"):
        trainer = Trainer(config, RunOptions(checkpoint_dir=tmp_path / name, wandb_mode="disabled"))
        histories.append(trainer.run(3).history)
        pool = trainer.opponents[1]
        assert isinstance(pool, PoolOpponent)
        tables.append(dict(pool.table))
        trainer.close()
    for left, right in zip(histories[0], histories[1], strict=True):
        assert _comparable(left) == _comparable(right)
    assert tables[0] == tables[1] and tables[0]
