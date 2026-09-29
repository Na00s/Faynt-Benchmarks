"""Mixed opponent curriculum (P10): per-group environments and workers feeding one learner step.

``[opponent] type = "mix"`` with specs like ``["self*32", "cpu9*64"]`` builds one environment and one
``RolloutWorker`` per group -- the ``self`` group is P6a's two-port path, a ``cpu<1-9>`` group the
vs-CPU path, an ``other:<path>`` group the frozen-checkpoint path -- and every group's trajectories
feed the same ``learner.step``.  Single-group mixes must be bit-identical to the plain opponent
types (golden tests); per-arm statistics land under ``arm/<name>/*``; ``policy.init_value`` warm
starts the value net from an RL checkpoint's value head.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import BC_FORMAT, INIT_NAME, load_checkpoint, load_value_net, sha256_file
from melee_rl.config import (
    CONFIG_DIR,
    RLConfig,
    apply_overrides,
    config_from_mapping,
    config_to_mapping,
    finalize_config,
    load_config,
    resolve_model_config,
    resolve_paths,
)
from melee_rl.env.dolphin import DolphinEnvConfig
from melee_rl.env.dolphin_mp import worker_configs
from melee_rl.env.dummy import DummyEnvConfig
from melee_rl.logging import format_summary, missing_metrics
from melee_rl.opponents import (
    ArmConfig,
    CpuOpponent,
    MixGroup,
    OpponentConfig,
    OtherOpponent,
    make_group_opponent,
    parse_mix_spec,
)
from melee_rl.train import RunOptions, Trainer, main, mix_env_configs
from melee_rl.trajectory import Trajectory

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
IGNORED_ON_COMPARE = ("timing/", "env/fps")


def _same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _comparable(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if not key.startswith(IGNORED_ON_COMPARE)}


def _without_arms(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if not key.startswith("arm/")}


def _mixed(base: RLConfig, mix: tuple[str, ...], **opponent_overrides: object) -> RLConfig:
    """``base`` as a mixed run: ``[opponent] type = "mix"`` and the neutral two-policy player template."""

    return finalize_config(
        replace(
            base,
            opponent=replace(base.opponent, type="mix", mix=mix, **opponent_overrides),  # type: ignore[arg-type]
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
        ),
        resolve_model_config(base),
    )


# ---------------------------------------------------------------------------
# spec grammar and config validation
# ---------------------------------------------------------------------------


def test_parse_mix_spec_grammar() -> None:
    group = parse_mix_spec("self*32")
    assert group == MixGroup(spec="self*32", kind="self", num_envs=32)
    assert group.trained_ports == (0, 1) and group.players == ("policy", "policy")
    assert group.rows == 64 and group.name == "self"
    cpu = parse_mix_spec("cpu9*64")
    assert cpu.kind == "cpu" and cpu.cpu_level == 9 and cpu.num_envs == 64
    assert cpu.trained_ports == (0,) and cpu.players == ("policy", "cpu")
    assert cpu.rows == 64 and cpu.name == "cpu9"
    other = parse_mix_spec("other:/runs/x/best.pt*4")
    assert other.kind == "other" and other.checkpoint == "/runs/x/best.pt" and other.num_envs == 4
    assert other.trained_ports == (0,) and other.players == ("policy", "policy") and other.name == "other"
    # 5 Sep 2026 (lever 1): an alias makes several arms of one kind distinct (smaller lockstep groups).
    aliased = parse_mix_spec("self*24@a")
    assert (
        aliased.kind == "self"
        and aliased.num_envs == 24
        and aliased.alias == "a"
        and aliased.name == "self_a"
    )
    assert parse_mix_spec("cpu9*8@b").name == "cpu9_b" and parse_mix_spec("other:/x.pt*2@c").name == "other_c"
    # 11 Sep 2026 (/delay-finetune): an other arm may name the delay its file plays at (unset: its own).
    delayed = parse_mix_spec("other:/runs/x/best.pt@d3*4@z")
    assert delayed.kind == "other" and delayed.checkpoint == "/runs/x/best.pt" and delayed.delay_frames == 3
    assert delayed.num_envs == 4 and delayed.alias == "z" and delayed.name == "other_z"
    assert other.delay_frames is None and parse_mix_spec("self*2").delay_frames is None
    with pytest.raises(ValueError):
        parse_mix_spec("other:@d3*2")
    with pytest.raises(ValueError):
        parse_mix_spec("self*24@not-an-identifier")
    for bad in (
        "self",
        "self*",
        "*2",
        "self*0",
        "self*-1",
        "self*two",
        "cpu0*2",
        "cpu10*2",
        "fox*2",
        "other:*2",
        "cpu9",
    ):
        with pytest.raises(ValueError):
            parse_mix_spec(bad)


def test_mix_opponent_config_validation() -> None:
    config = OpponentConfig(type="mix", mix=("self*2", "cpu9*2"))
    assert [group.name for group in config.groups] == ["self", "cpu9"]
    assert OpponentConfig().groups == ()
    with pytest.raises(ValueError, match="mix"):
        OpponentConfig(type="mix")  # a mix needs at least one group
    with pytest.raises(ValueError, match="mix"):
        OpponentConfig(type="cpu", mix=("self*2",))  # mix specs belong to type = "mix"
    with pytest.raises(ValueError, match="distinct"):
        OpponentConfig(type="mix", mix=("self*2", "self*2"))
    with pytest.raises(ValueError, match="checkpoint"):
        OpponentConfig(type="mix", mix=("self*2",), checkpoint="x.pt")
    with pytest.raises(ValueError, match="train"):
        OpponentConfig(type="mix", mix=("self*2",), train=True)
    with pytest.raises(ValueError, match="mix_workers"):
        OpponentConfig(type="cpu", mix_workers=(1,))
    with pytest.raises(ValueError, match="mix_workers"):
        OpponentConfig(type="mix", mix=("self*2", "cpu9*2"), mix_workers=(1,))  # one entry per group
    with pytest.raises(ValueError, match="mix_workers"):
        OpponentConfig(type="mix", mix=("self*2",), mix_workers=(0,))
    with pytest.raises(ValueError, match="divisible"):
        OpponentConfig(type="mix", mix=("self*3",), mix_workers=(2,))
    with pytest.raises(ValueError, match="per-group"):
        _ = OpponentConfig(type="mix", mix=("self*2",)).trained_ports


def test_mix_finalize_rules_and_env_configs() -> None:
    base = load_config(SMOKE_TINY)
    model = resolve_model_config(base)
    ok = _mixed(base, ("self*1", "cpu9*2"))
    assert [group.num_envs for group in ok.opponent.groups] == [1, 2]
    dummies = [sub for sub in mix_env_configs(ok) if isinstance(sub, DummyEnvConfig)]
    assert [config.num_envs for config in dummies] == [1, 2]
    assert dummies[0].players == ("policy", "policy") and dummies[1].players == ("policy", "cpu")
    assert [config.seed for config in dummies] == [base.env.dummy.seed, base.env.dummy.seed + 1]
    with pytest.raises(ValueError, match="cover"):
        _mixed(base, ("self*1", "cpu9*1"))  # 2 environments configured, 3 in [env.dummy]
    with pytest.raises(ValueError, match="players"):
        finalize_config(
            replace(base, opponent=replace(base.opponent, type="mix", mix=("self*1", "cpu9*2"))), model
        )  # smoke_tiny's derived players ("policy", "cpu") must stay the neutral template
    with pytest.raises(ValueError, match="mix_workers"):
        _mixed(base, ("self*1", "cpu9*2"), mix_workers=(1, 1))  # the dummy env has no worker processes
    with pytest.raises(ValueError, match="microbatch"):
        finalize_config(
            replace(_mixed(base, ("self*1", "cpu9*2")), learner=replace(base.learner, microbatch_envs=4)),
            model,
        )  # both groups have 2 rows; 4 fits neither
    finalize_config(
        replace(_mixed(base, ("self*1", "cpu9*2")), learner=replace(base.learner, microbatch_envs=2)), model
    )
    with pytest.raises(ValueError, match="pipelined mode"):
        finalize_config(
            replace(
                _mixed(base, ("self*1", "cpu9*2")),
                actor=replace(base.actor, batch_steps=2, delay_frames=2),
                delay=replace(base.delay, allow_mismatch=True),
            ),
            model,
        )

    dolphin = replace(
        base,
        env=replace(
            base.env,
            type="dolphin",
            dolphin=replace(base.env.dolphin, num_envs=6, worker_processes=3, players=("policy", "policy")),
        ),
        opponent=replace(base.opponent, type="mix", mix=("self*2", "cpu5*4"), mix_workers=(1, 2)),
    )
    final = finalize_config(dolphin, model)
    configs = [sub for sub in mix_env_configs(final) if isinstance(sub, DolphinEnvConfig)]
    assert [config.num_envs for config in configs] == [2, 4]
    assert configs[0].players == ("policy", "policy") and configs[1].players == ("policy", "cpu")
    assert [config.worker_processes for config in configs] == [1, 2]
    assert configs[1].slippi_port == configs[0].slippi_port + 2
    assert [config.env_index_offset for config in configs] == [0, 2]
    assert [group.cpu_level for group in final.opponent.groups] == [9, 5]
    with pytest.raises(ValueError, match="mix_workers"):
        finalize_config(replace(dolphin, opponent=replace(dolphin.opponent, mix_workers=())), model)
    with pytest.raises(ValueError, match="mix_workers"):
        finalize_config(replace(dolphin, opponent=replace(dolphin.opponent, mix_workers=(1, 1))), model)


def test_resolve_paths_resolves_other_mix_specs() -> None:
    base = load_config(SMOKE_TINY)
    config = replace(
        base, opponent=replace(base.opponent, type="mix", mix=("self*1", "other:ckpt/best.pt*2"))
    )
    resolved = resolve_paths(config, "/tmp/base")
    assert resolved.opponent.mix == ("self*1", "other:/tmp/base/ckpt/best.pt*2")
    absolute = replace(
        base, opponent=replace(base.opponent, type="mix", mix=("other:/abs/best.pt*2", "self*1"))
    )
    assert resolve_paths(absolute, "/tmp/base").opponent.mix == ("other:/abs/best.pt*2", "self*1")


def test_make_group_opponent_matrix(tiny_adapter: MeleePolicyAdapter) -> None:
    config = OpponentConfig(type="mix", mix=("self*2", "cpu5*2"))
    assert (
        make_group_opponent(
            parse_mix_spec("self*2"), config, tiny_adapter, delay_frames=0, seed=3, device=None
        )
        is None
    )
    cpu = make_group_opponent(
        parse_mix_spec("cpu5*2"), config, tiny_adapter, delay_frames=0, seed=3, device=None
    )
    assert isinstance(cpu, CpuOpponent) and cpu.cpu_level == 5 and cpu.port == 1 and not cpu.controls_port


def test_other_arms_play_at_their_file_s_delay_unless_the_spec_says_otherwise(
    tiny_adapter: MeleePolicyAdapter, tiny_config: Any, tiny_value_config: Any, tmp_path: Path
) -> None:
    """11 Sep 2026 (/delay-finetune): a frozen checkpoint arm plays at the delay its file was trained at (a
    delay-0 file in a D = 18 run keeps its 0), ``@d<n>`` in the spec overrides it; the run's delay passed to
    ``make_group_opponent`` is only the self / pool arms'."""

    from melee_rl.checkpoint import save_rl_checkpoint
    from melee_rl.learner import LearnerConfig, PPOLearner
    from melee_rl.value import ValueNet

    learner = PPOLearner(
        tiny_adapter, None, ValueNet(tiny_config, tiny_value_config), LearnerConfig(kl_teacher_weight=0.0)
    )
    delayed = tmp_path / "d2.pt"
    save_rl_checkpoint(delayed, learner, tiny_config, step=0, delay_frames=2, context_mode="ring")
    config = OpponentConfig(type="mix", mix=("self*2", f"other:{delayed}*2"))
    own = make_group_opponent(
        parse_mix_spec(f"other:{delayed}*2"), config, tiny_adapter, delay_frames=18, seed=3
    )
    assert isinstance(own, OtherOpponent) and own.delay_frames == 2
    forced = make_group_opponent(
        parse_mix_spec(f"other:{delayed}@d0*2"), config, tiny_adapter, delay_frames=18, seed=3
    )
    assert isinstance(forced, OtherOpponent) and forced.delay_frames == 0


# ---------------------------------------------------------------------------
# golden guard: a single-group mix is bit-identical to the plain opponent type
# ---------------------------------------------------------------------------


def test_single_group_mix_is_bit_identical_to_plain_types() -> None:
    base = load_config(SMOKE_TINY)
    model = resolve_model_config(base)

    plain_cpu = Trainer(base, RunOptions(wandb_mode="disabled"))
    a = plain_cpu.run(2)
    plain_cpu.close()
    mixed_cpu = Trainer(_mixed(base, ("cpu9*3",)), RunOptions(wandb_mode="disabled"))
    assert len(mixed_cpu.workers) == 1 and isinstance(mixed_cpu.opponents[0], CpuOpponent)
    b = mixed_cpu.run(2)
    mixed_cpu.close()
    for left, right in zip(a.history, b.history, strict=True):
        assert _comparable(left) == _without_arms(_comparable(right))
        assert {"arm/cpu9/per_frame", "arm/cpu9/resets"} <= set(right)
    assert _same_state(plain_cpu.policy.get_state(), mixed_cpu.policy.get_state())

    two_port = finalize_config(
        replace(
            base,
            opponent=replace(base.opponent, type="self", train=True),
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
        ),
        model,
    )
    plain_self = Trainer(two_port, RunOptions(wandb_mode="disabled"))
    c = plain_self.run(2)
    plain_self.close()
    mixed_self = Trainer(_mixed(base, ("self*3",)), RunOptions(wandb_mode="disabled"))
    assert mixed_self.workers[0].ports == (0, 1) and mixed_self.opponents == [None]
    d = mixed_self.run(2)
    mixed_self.close()
    for left, right in zip(c.history, d.history, strict=True):
        assert _comparable(left) == _without_arms(_comparable(right))
    assert _same_state(plain_self.policy.get_state(), mixed_self.policy.get_state())


# ---------------------------------------------------------------------------
# the real thing: self + cpu arms in one run, resume, determinism, other arm
# ---------------------------------------------------------------------------


def test_concat_arms_hands_the_learner_one_trajectory_per_rollout(tmp_path: Path) -> None:
    """17 Sep 2026: the league smoke cut its five arms into 21 learner chunks per rollout at microbatch 6, and
    the learner took 60 s of a 172-s step. ``[learner] concat_arms`` joins each rollout's arm trajectories
    into one learner trajectory, so ``microbatch_envs`` divides the joined rows instead of every arm's rows.
    The per-arm metrics and the frame counts are unchanged."""

    base = load_config(SMOKE_TINY)
    arms = ("self*1", "cpu9*2")  # 2 + 2 rows
    with pytest.raises(ValueError, match="every mix group's rows"):
        _mixed(replace(base, learner=replace(base.learner, microbatch_envs=4)), arms)
    joined = replace(base.learner, microbatch_envs=4, concat_arms=True, loss_weighting="row")
    config = _mixed(replace(base, learner=joined), arms)
    with pytest.raises(ValueError, match="joined rows"):
        _mixed(replace(base, learner=replace(joined, microbatch_envs=3)), arms)
    with pytest.raises(ValueError, match="concat_arms"):
        finalize_config(replace(base, learner=joined), resolve_model_config(base))  # no mix, nothing to join
    trainer = Trainer(config, RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled"))
    seen: list[list[int]] = []
    step = trainer.learner.step

    def spy(trajectories: Sequence[Trajectory]) -> dict[str, float]:
        seen.append([trajectory.batch_size for trajectory in trajectories])
        return step(trajectories)

    trainer.learner.step = spy  # type: ignore[method-assign]
    result = trainer.run(2)
    batches = config.learner.ppo.num_batches
    assert seen == [[4] * batches, [4] * batches]
    row = result.history[-1]
    assert missing_metrics(row) == [] and {"arm/self/per_frame", "arm/cpu9/per_frame"} <= set(row)
    assert result.frames_seen == 2 * batches * 4 * config.actor.rollout_length
    trainer.close()


def test_mixed_self_and_cpu_arms_train_in_one_step(tmp_path: Path) -> None:
    base = load_config(SMOKE_TINY)
    config = _mixed(base, ("self*1", "cpu9*2"))
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert trainer.group_names == ("self", "cpu9")
    assert [worker.ports for worker in trainer.workers] == [(0, 1), (0,)]
    assert [worker.batch_size for worker in trainer.workers] == [2, 2]
    assert trainer.opponents[0] is None and isinstance(trainer.opponents[1], CpuOpponent)
    assert trainer.envs[0].controlled_ports == (0, 1) and trainer.envs[1].controlled_ports == (0,)
    assert trainer.worker is trainer.workers[0] and trainer.env is trainer.envs[0]
    result = trainer.run(2)
    batches = config.learner.ppo.num_batches
    length = config.actor.rollout_length
    rows = 4  # self*1 trains both ports (2 rows), cpu9*2 trains one port each (2 rows)
    assert result.frames_seen == 2 * batches * rows * length
    row = result.history[-1]
    assert missing_metrics(row) == []
    assert row["env/frames_total"] == float(2 * batches * 3 * length)
    assert row["env/trained_ports"] == pytest.approx(rows / 3)
    assert {
        "arm/self/per_frame",
        "arm/self/resets",
        "arm/cpu9/per_frame",
        "arm/cpu9/resets",
        "arm/cpu9/damage_dealt_per_minute",
    } <= set(row)
    assert row["env/resets"] == row["arm/self/resets"] + row["arm/cpu9/resets"]
    assert row["arm/self/per_frame"] == pytest.approx(0.0, abs=1e-6)  # zero-sum over both perspectives
    summary = format_summary(row)
    assert "self" in summary and "cpu9" in summary

    payload = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert len(payload["rng"]["actors"]) == 2 and payload["rng"]["opponents"] == [None, None]
    states = [worker.generator.get_state() for worker in trainer.workers]
    trainer.close()
    resumed = Trainer(config, options)
    assert resumed.resumed and resumed.step == 2 and len(resumed.workers) == 2
    for worker, state in zip(resumed.workers, states, strict=True):
        assert worker.generator.get_state().equal(state)
    assert resumed.run(1).step == 3
    resumed.close()
    single = _mixed(base, ("cpu9*3",))
    with pytest.raises(ValueError, match="rollout workers"):
        Trainer(single, options)  # the checkpoint was written by a two-group run


def test_mixed_runs_are_deterministic() -> None:
    base = load_config(SMOKE_TINY)
    config = _mixed(base, ("self*1", "cpu9*2"))
    first = Trainer(config, RunOptions(wandb_mode="disabled"))
    a = first.run(2)
    first.close()
    second = Trainer(config, RunOptions(wandb_mode="disabled"))
    b = second.run(2)
    second.close()
    for left, right in zip(a.history, b.history, strict=True):
        assert _comparable(left) == _comparable(right)
    assert _same_state(first.policy.get_state(), second.policy.get_state())


def test_mix_other_arm_plays_a_frozen_checkpoint(tmp_path: Path) -> None:
    first = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(tmp_path / "first"),
            "--wandb-mode",
            "disabled",
            "--steps",
            "1",
        ]
    )
    assert first.checkpoint is not None
    base = load_config(SMOKE_TINY)
    config = _mixed(base, ("self*1", f"other:{first.checkpoint}*2"))
    trainer = Trainer(config, RunOptions(checkpoint_dir=tmp_path / "second", wandb_mode="disabled"))
    other = trainer.opponents[1]
    assert isinstance(other, OtherOpponent) and other.batch_size == 2 and other.port == 1
    assert other.sha256 == sha256_file(first.checkpoint)
    result = trainer.run(1)
    trainer.close()
    row = result.history[-1]
    assert "arm/other/per_frame" in row and missing_metrics(row) == []
    assert result.frames_seen == config.learner.ppo.num_batches * 4 * config.actor.rollout_length
    payload = load_checkpoint(tmp_path / "second" / "latest.pt")
    assert payload["rng"]["opponents"][0] is None
    assert isinstance(payload["rng"]["opponents"][1], torch.Tensor)


# ---------------------------------------------------------------------------
# policy.init_value: warm-starting the value net from an RL checkpoint
# ---------------------------------------------------------------------------


def test_init_value_warm_starts_the_value_net(tmp_path: Path) -> None:
    first = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(tmp_path / "run"),
            "--wandb-mode",
            "disabled",
            "--steps",
            "1",
        ]
    )
    latest = first.checkpoint
    assert latest is not None
    payload = load_checkpoint(latest)
    base = load_config(SMOKE_TINY)
    config = finalize_config(
        replace(
            base, policy=replace(base.policy, init="checkpoint", checkpoint=str(latest), init_value=True)
        ),
        resolve_model_config(base),
    )
    trainer = Trainer(config, RunOptions(wandb_mode="disabled"))
    saved = payload["value_state_dict"]
    current = trainer.value_net.state_dict()
    assert set(saved) == set(current)
    assert all(torch.equal(current[key], saved[key]) for key in saved)
    trainer.close()

    smaller = replace(config, learner=replace(config.learner, value=replace(config.learner.value, d_ff=32)))
    smaller = finalize_config(smaller, resolve_model_config(smaller))
    with pytest.raises(ValueError, match="value config"):
        Trainer(smaller, RunOptions(wandb_mode="disabled"))

    bc_path = tmp_path / "bc.pt"
    torch.save(
        {"format": BC_FORMAT, "model_config": payload["model_config"], "state_dict": payload["state_dict"]},
        bc_path,
    )
    bc_config = finalize_config(
        replace(
            base, policy=replace(base.policy, init="checkpoint", checkpoint=str(bc_path), init_value=True)
        ),
        resolve_model_config(base),
    )
    with pytest.raises(ValueError, match="init_value"):
        Trainer(bc_config, RunOptions(wandb_mode="disabled"))

    with pytest.raises(ValueError, match="init_value"):
        replace(base.policy, init="random", checkpoint=None, init_value=True)
    value_net = Trainer(base, RunOptions(wandb_mode="disabled", num_steps=0))
    with pytest.raises(ValueError, match="init_value"):
        load_value_net(bc_path, value_net.value_net)
    value_net.close()


def test_init_checkpoint_is_frozen_into_the_run_dir(tmp_path: Path) -> None:
    """The init path (``best.pt`` / ``latest.pt``) is mutable -- resuming the source run overwrites
    it -- so the first launch freezes the exact bytes as ``init.pt`` in its own directory, and every
    later construction (resumes included) reads the frozen copy; the ``source = "policy"`` teacher
    record pins the frozen file, so even deleting the source cannot break a resume."""

    source_dir = tmp_path / "source"
    source_argv = [
        "--config",
        str(SMOKE_TINY),
        "--checkpoint-dir",
        str(source_dir),
        "--wandb-mode",
        "disabled",
    ]
    first = main([*source_argv, "--steps", "1"])
    assert first.checkpoint is not None
    original = first.checkpoint.read_bytes()

    base = load_config(SMOKE_TINY)
    config = finalize_config(
        replace(
            base,
            policy=replace(base.policy, init="checkpoint", checkpoint=str(first.checkpoint), init_value=True),
        ),
        resolve_model_config(base),
    )
    # Without a checkpoint directory nothing is frozen: the source is read directly.
    loose = Trainer(config, RunOptions(wandb_mode="disabled", num_steps=0))
    assert loose.init_checkpoint == first.checkpoint
    assert loose.teacher_record == {"source": str(first.checkpoint), "sha256": sha256_file(first.checkpoint)}
    loose.close()

    run_dir = tmp_path / "child"
    options = RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled")
    child = Trainer(config, options)
    frozen = run_dir / INIT_NAME
    assert child.init_checkpoint == frozen
    assert frozen.exists() and frozen.read_bytes() == original
    assert not frozen.with_name(frozen.name + ".tmp").exists()
    assert child.teacher_record == {"source": str(frozen), "sha256": sha256_file(frozen)}
    payload = load_checkpoint(frozen)
    current = child.value_net.state_dict()
    assert all(torch.equal(current[key], payload["value_state_dict"][key]) for key in current)
    child.run(1)
    child.close()

    # The source run continues: the file the child started from changes in place.
    main([*source_argv, "--steps", "1"])
    assert first.checkpoint.read_bytes() != original
    resumed = Trainer(config, options)
    assert resumed.resumed and resumed.step == 1
    assert frozen.read_bytes() == original  # the frozen copy never follows the source
    assert resumed.run(1).step == 2
    resumed.close()

    # Even a deleted source cannot break a resume: everything reads the frozen copy.
    first.checkpoint.unlink()
    survivor = Trainer(config, options)
    assert survivor.resumed and survivor.step == 2 and survivor.init_checkpoint == frozen
    survivor.close()


# ---------------------------------------------------------------------------
# the league arms (17 Sep 2026, LEAGUE-PLAN.md §2 / §3.1)
# ---------------------------------------------------------------------------

LEAGUE_GM_CHARACTERS = (1, 22, 18, 7, 15, 2, 9, 14, 10, 17, 12, 13)
"""gm's twelve characters in its own ``rl_config.agent.char`` order (libmelee ids)."""
LEAGUE_STAGES = (25, 24, 18, 26, 8, 6)
LEAGUE_MIX = (
    "self*30",
    "slippi:gm*24@gm",
    "slippi:fox_d18_ditto_v4*24@foxd18",
    "other:/runs/selfplay/10m_leash/kl0p003/step_00000632.pt@d0*12@d0",
    "slippi:falco_d18_vs_fox_v4*6@falco",
)


def _league_arms() -> dict[str, ArmConfig]:
    return {
        "gm": ArmConfig(characters=LEAGUE_GM_CHARACTERS, character_repeat=2, names=("Master Player",)),
        "foxd18": ArmConfig(names=("Cody", "Hax", "Aklo", "SFAT"), name_repeat=6),
        "falco": ArmConfig(names=("Ginger", "KJH", "Frenzy", "BBB")),
    }


def test_slippi_arm_grammar_and_arm_tables() -> None:
    group = parse_mix_spec("slippi:gm*24@gm")
    assert group.kind == "slippi" and group.release == "gm" and group.num_envs == 24 and group.alias == "gm"
    assert group.name == "slippi_gm" and group.trained_ports == (0,) and group.rows == 24
    assert group.players == ("policy", "policy") and group.checkpoint is None
    assert parse_mix_spec("slippi:fox_d18_ditto_v4*6").name == "slippi"
    assert parse_mix_spec("self*2").release is None
    with pytest.raises(ValueError, match="release"):
        parse_mix_spec("slippi:not_a_release*2")
    with pytest.raises(ValueError):
        parse_mix_spec("slippi:*2")

    gm = ArmConfig(characters=(1, 22, 18), character_repeat=2, names=("Master Player",))
    assert [gm.character_for(j) for j in range(8)] == [1, 1, 22, 22, 18, 18, 1, 1]
    assert [gm.name_for(j) for j in range(3)] == ["Master Player"] * 3
    tags = ArmConfig(names=("Cody", "Hax"), name_repeat=3)
    assert [tags.name_for(j) for j in range(7)] == ["Cody"] * 3 + ["Hax"] * 3 + ["Cody"]
    assert ArmConfig().character_for(5) is None and ArmConfig().name_for(0) is None
    for bad in ({"character_repeat": 0}, {"name_repeat": 0}):
        with pytest.raises(ValueError, match="repeat"):
            ArmConfig(**bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="characters"):
        ArmConfig(characters=(33,))

    config = OpponentConfig(type="mix", mix=("self*2", "slippi:gm*24@gm"), arms={"gm": gm})
    assert config.arm(config.groups[1]) == gm and config.arm(config.groups[0]) == ArmConfig()
    with pytest.raises(ValueError, match="arms"):  # not an alias of this mix
        OpponentConfig(type="mix", mix=("self*2", "slippi:gm*24@gm"), arms={"zz": gm})
    with pytest.raises(ValueError, match="arms"):  # a self / pool arm's port 1 is the student itself
        OpponentConfig(type="mix", mix=("self*2@a",), arms={"a": ArmConfig(characters=(22,))})
    with pytest.raises(ValueError, match="arms"):  # arm tables belong to a mix
        OpponentConfig(type="cpu", arms={"a": gm})
    with pytest.raises(ValueError, match="gm"):  # gm plays twelve characters: its arm must say which
        OpponentConfig(type="mix", mix=("slippi:gm*2@gm",))
    with pytest.raises(ValueError, match="cannot play"):  # the Falco release plays Falco only
        OpponentConfig(
            type="mix", mix=("slippi:falco_d18_vs_fox_v4*2@f",), arms={"f": ArmConfig(characters=(1,))}
        )
    with pytest.raises(ValueError, match="trained with"):  # an RL release plays the tags it trained with
        OpponentConfig(
            type="mix", mix=("slippi:fox_d18_ditto_v4*2@f",), arms={"f": ArmConfig(names=("Mango",))}
        )
    with pytest.raises(ValueError, match="names"):  # a name tag is a slippi-ai input
        OpponentConfig(type="mix", mix=("cpu9*2@c",), arms={"c": ArmConfig(names=("Cody",))})
    # A CPU arm may pick its CPU's characters (Dolphin selects them like any other port).
    OpponentConfig(type="mix", mix=("cpu9*2@c",), arms={"c": ArmConfig(characters=(22,))})


def test_league_arms_give_every_dolphin_its_own_matchup() -> None:
    """The league split (LEAGUE-PLAN.md §2): 24 gm Dolphins play its twelve characters in consecutive pairs --
    two Dolphins each, on two different stages, every stage four times -- the Falco arm plays Falco, every
    other arm keeps the run's Fox (no table at all: the default path is unchanged), and the global table
    survives the worker slicing."""

    base = load_config(SMOKE_TINY)
    model = resolve_model_config(base)
    league = replace(
        base,
        env=replace(
            base.env,
            type="dolphin",
            dolphin=replace(
                base.env.dolphin,
                num_envs=96,
                worker_processes=16,
                players=("policy", "policy"),
                stages=LEAGUE_STAGES,
            ),
        ),
        opponent=replace(
            base.opponent, type="mix", mix=LEAGUE_MIX, mix_workers=(5, 4, 4, 2, 1), arms=_league_arms()
        ),
        eval=replace(base.eval, enabled=False),
        learner=replace(base.learner, microbatch_envs=6),
    )
    final = finalize_config(league, model)
    configs = [sub for sub in mix_env_configs(final) if isinstance(sub, DolphinEnvConfig)]
    self_arm, gm, foxd18, d0, falco = configs
    assert self_arm.opponent_characters == () and d0.opponent_characters == ()
    assert foxd18.opponent_characters == ()
    assert {self_arm.characters_for(j) for j in range(30)} == {(1, 1)}
    assert {foxd18.characters_for(j) for j in range(24)} == {(1, 1)}
    assert [falco.characters_for(j) for j in range(6)] == [(1, 22)] * 6
    table = [(gm.characters_for(j)[1], gm.stage_for(j)) for j in range(24)]
    for index, character in enumerate(LEAGUE_GM_CHARACTERS):
        assert table[2 * index][0] == table[2 * index + 1][0] == character
        assert table[2 * index][1] != table[2 * index + 1][1]  # never the same stage twice
    assert sorted(Counter(stage for _, stage in table).values()) == [4] * 6
    assert {gm.characters_for(j)[0] for j in range(24)} == {1}  # port 0 is our Fox everywhere
    sliced = worker_configs(gm)
    assert [c.characters_for(j)[1] for c in sliced for j in range(6)] == [row[0] for row in table]
    with pytest.raises(ValueError, match="microbatch"):
        finalize_config(replace(league, learner=replace(base.learner, microbatch_envs=12)), model)


def test_arm_tables_round_trip_through_toml_and_overrides() -> None:
    raw: dict[str, Any] = {
        "format": "melee_rl.rl_config.v1",
        "opponent": {
            "type": "mix",
            "mix": ["self*2", "slippi:gm*4@gm"],
            "arms": {"gm": {"characters": [1, 22], "character_repeat": 2, "names": ["Master Player"]}},
        },
    }
    config = config_from_mapping(raw)
    assert config.opponent.arms == {
        "gm": ArmConfig(characters=(1, 22), character_repeat=2, names=("Master Player",))
    }
    assert config_from_mapping({**config_to_mapping(config), "format": "melee_rl.rl_config.v1"}) == config
    changed = config_from_mapping(apply_overrides(raw, ["opponent.arms.gm.characters=[9, 10]"]))
    assert changed.opponent.arms["gm"].characters == (9, 10)
    with pytest.raises(ValueError, match="unknown key"):
        config_from_mapping(apply_overrides(raw, ["opponent.arms.gm.character=[9]"]))


def test_a_slippi_arm_trains_against_the_torch_port_and_resumes_its_generator(tmp_path: Path) -> None:
    from melee_rl.slippi_ai_torch import SlippiAITorchOpponent
    from tests.rl._slippi_release import write_release

    model_dir = tmp_path / "releases"
    write_release(model_dir / "gm", names=("Master Player",), delay=2)
    base = load_config(SMOKE_TINY)
    config = _mixed(base, ("self*1", "slippi:gm*2@gm"), arms={"gm": ArmConfig(characters=(1, 22))})
    config = replace(
        config, slippi_ai=replace(config.slippi_ai, model_dir=str(model_dir), verify_sha256=False)
    )
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    trainer = Trainer(config, options)
    opponent = trainer.opponents[1]
    assert isinstance(opponent, SlippiAITorchOpponent) and opponent.release == "gm" and opponent.port == 1
    assert opponent.agent.name_tags == ("Master Player", "Master Player") and opponent.characters == (1, 22)
    result = trainer.run(1)
    row = result.history[-1]
    assert missing_metrics(row) == [] and "arm/slippi_gm/per_frame" in row
    assert row["timing/slippi_gm/opponent_s"] > 0.0 and row["timing/self/opponent_s"] == 0.0
    assert {"arm/slippi_gm/fox/damage_dealt_per_minute", "arm/slippi_gm/falco/deaths_per_minute"} <= set(row)
    assert "arm/self/fox/per_frame" not in row  # only an arm with several characters is split
    # The console line names each arm once; the per-character rows stay in the metrics (the league smoke's
    # line repeated "slippi_gm r/f" twelve times).
    assert format_summary(row).count("slippi_gm r/f") == 1
    payload = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert payload["rng"]["opponents"][0] is None and isinstance(payload["rng"]["opponents"][1], torch.Tensor)
    state = opponent.generator.get_state()
    trainer.close()
    resumed = Trainer(config, options)
    rival = resumed.opponents[1]
    assert isinstance(rival, SlippiAITorchOpponent) and rival.generator.get_state().equal(state)
    resumed.close()
    # The release file is checked against the registry's sha256 unless the run file says otherwise.
    strict = replace(config, slippi_ai=replace(config.slippi_ai, verify_sha256=True))
    with pytest.raises(ValueError, match="sha256"):
        Trainer(strict, RunOptions(wandb_mode="disabled"))
