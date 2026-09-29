"""Training loop and launcher (PLAN.md §6 P3): entrypoint smoke, checkpoint/resume, determinism, self-play."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from melee_rl.checkpoint import checkpoint_paths, load_checkpoint, sha256_file
from melee_rl.config import (
    CONFIG_DIR,
    CONFIG_FORMAT,
    RLConfig,
    finalize_config,
    load_config,
    resolve_model_config,
)
from melee_rl.logging import EXTRA_METRIC_NAMES, missing_metrics
from melee_rl.opponents import CpuOpponent, OtherOpponent, SelfOpponent
from melee_rl.train import RunOptions, Trainer, TrainResult, main, parse_args, trajectory_metrics

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
IGNORED_ON_COMPARE = ("timing/", "env/fps")


def _same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _comparable(row: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in row.items() if not key.startswith(IGNORED_ON_COMPARE)}


def test_train_smoke_entrypoint_two_steps_checkpoint_and_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    argv = ["--config", str(SMOKE_TINY), "--checkpoint-dir", str(run_dir), "--wandb-mode", "disabled"]
    result = main([*argv, "--steps", "2"])
    assert isinstance(result, TrainResult) and result.step == 2 and result.steps_run == 2
    assert len(result.history) == 2 and result.history[-1]["step"] == 2.0 and result.wandb_run_id is None
    config = result.config
    assert isinstance(config, RLConfig)
    assert config.runtime.checkpoint_dir is None  # the CLI flag is not folded into the config
    batch = config.env.num_envs * config.actor.rollout_length * config.learner.ppo.num_batches
    assert result.frames_seen == 2 * batch and result.history[-1]["frames_seen"] == float(result.frames_seen)
    assert missing_metrics(result.history[-1]) == []
    latest, snapshot = checkpoint_paths(run_dir, 2)
    assert result.checkpoint == latest and latest.exists()
    assert snapshot.exists()  # snapshot_interval_steps = 2
    assert not checkpoint_paths(run_dir, 1)[1].exists()
    payload = load_checkpoint(latest)
    assert payload["step"] == 2 and payload["learner_step"] == 2
    assert payload["teacher"] == {"random_init": True, "seed": config.policy.seed}
    assert payload["loop_state"]["frames_seen"] == result.frames_seen
    assert payload["loop_state"]["rollouts_trained"] == 2 * config.learner.ppo.num_batches
    assert payload["delay_frames"] == 0 and payload["context_mode"] == "reprime"
    assert payload["wandb_run_id"] is None
    assert payload["rl_config"]["format"] == CONFIG_FORMAT
    assert payload["rl_config"]["env"]["dummy"]["num_envs"] == config.env.num_envs
    assert payload["model_config"]["d_model"] == 32 and payload["model_config"]["compute_dtype"] == "float32"
    assert isinstance(payload["rng"]["torch"], torch.Tensor)
    assert isinstance(payload["rng"]["actor"], torch.Tensor)
    assert payload["config_yaml_sha256"] is not None

    # Resume: one more step continues at 3 (the teacher is rebuilt from the record).
    resumed = main([*argv, "--steps", "1"])
    assert resumed.step == 3 and resumed.steps_run == 1 and len(resumed.history) == 1
    assert resumed.history[0]["step"] == 3.0 and resumed.frames_seen == 3 * batch
    again = load_checkpoint(latest)
    assert again["step"] == 3 and again["learner_step"] == 3
    assert again["loop_state"]["frames_seen"] == 3 * batch
    assert checkpoint_paths(run_dir, 2)[1].exists() and not checkpoint_paths(run_dir, 3)[1].exists()
    # --no-resume starts over and overwrites latest.pt.
    fresh = main([*argv, "--steps", "1", "--no-resume"])
    assert fresh.step == 1 and fresh.steps_run == 1 and load_checkpoint(latest)["step"] == 1


def test_trainer_resume_rebuilds_teacher_and_restores_state(tmp_path: Path) -> None:
    config = load_config(SMOKE_TINY)
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    first = Trainer(config, options)
    assert first.step == 0 and first.frames_seen == 0 and not first.resumed
    assert isinstance(first.opponent, CpuOpponent) and first.teacher is not None
    assert first.teacher_record == {"random_init": True, "seed": config.policy.seed}
    assert first.device == torch.device("cpu") and first.model_config.context_length == 24
    before = first.teacher.get_state()
    assert _same_state(before, first.policy.get_state())  # the teacher is the initial policy
    first.run(2)
    teacher_state, policy_state = first.teacher.get_state(), first.policy.get_state()
    value_state = {name: value.clone() for name, value in first.value_net.state_dict().items()}
    # The teacher never moves.  (The policy may not either: the toy games carry no reward in their
    # first 32 frames, so the PPO gradient of smoke_tiny is exactly zero -- the learning-signal test
    # is the one that checks movement.)
    assert _same_state(teacher_state, before)
    first.close()

    second = Trainer(config, options)  # resumes from latest.pt
    assert second.resumed and second.step == 2 and second.learner.steps_done == 2
    assert second.frames_seen == first.frames_seen and second.teacher is not None
    assert _same_state(second.teacher.get_state(), teacher_state)
    assert _same_state(second.policy.get_state(), policy_state)
    assert _same_state(dict(second.value_net.state_dict()), value_state)
    assert second.worker.generator.get_state().equal(first.worker.generator.get_state())
    result = second.run(1)
    assert result.step == 3 and result.steps_run == 1 and second.learner.steps_done == 3
    second.close()

    # A checkpoint of another model is refused; so is a resume with another teacher.
    overrides = {**config.policy.model_overrides, "d_model": 48, "n_heads": 6}
    other = finalize_config(
        replace(config, policy=replace(config.policy, model_overrides=overrides)),
        resolve_model_config(replace(config, policy=replace(config.policy, model_overrides=overrides))),
    )
    with pytest.raises(ValueError, match="model_config"):
        Trainer(other, options)
    different_teacher = finalize_config(
        replace(config, teacher=replace(config.teacher, source="none")), resolve_model_config(config)
    )
    with pytest.raises(ValueError, match="teacher"):
        Trainer(different_teacher, options)
    # 5 Sep 2026: gradient checkpointing may change on a resume (memory only, bit-identical numbers), so a
    # run can switch it off to take learner.compile mid-way; the model_config check ignores that one key.
    from melee_rl.train import RESUMABLE_MODEL_FIELDS

    assert frozenset({"gradient_checkpointing"}) == RESUMABLE_MODEL_FIELDS
    flipped = {
        **config.policy.model_overrides,
        "gradient_checkpointing": not config_model_checkpointing(config),
    }
    flipped_config = finalize_config(
        replace(config, policy=replace(config.policy, model_overrides=flipped)),
        resolve_model_config(replace(config, policy=replace(config.policy, model_overrides=flipped))),
    )
    third = Trainer(flipped_config, options)
    assert third.resumed and third.step == 3
    assert third.model_config.gradient_checkpointing is not config_model_checkpointing(config)
    assert _same_state(third.policy.get_state(), second.policy.get_state())
    third.close()


def config_model_checkpointing(config: RLConfig) -> bool:
    return bool(resolve_model_config(config).gradient_checkpointing)


def test_a_checkpoint_teacher_is_built_without_gradient_checkpointing(tmp_path: Path) -> None:
    """10 Sep 2026 (strength gate 1): the leash arms load their teacher from Ali's post-trained file, whose
    model config says ``gradient_checkpointing = true``; with ``learner.fast_forward`` + ``learner.compile``
    the learner wraps the teacher in the compiled lean forward, which refuses a checkpointed model (the
    first launch of all three arms died on it).  A frozen teacher never backpropagates, so both teacher
    builders -- the fresh run's and the resume's -- read the file with checkpointing off: the same
    weights, no refusal."""

    from melee_rl.adapter import MeleePolicyAdapter
    from melee_rl.checkpoint import save_bc_checkpoint
    from melee_rl.train import build_teacher, teacher_from_record

    config = load_config(SMOKE_TINY)
    model_config = resolve_model_config(config)
    torch.manual_seed(7)
    source = MeleePolicyAdapter(model_config)
    path = tmp_path / "teacher.pt"
    save_bc_checkpoint(path, source.get_state(), replace(model_config, gradient_checkpointing=True))
    with_file_teacher = replace(config, teacher=replace(config.teacher, source="checkpoint", path=str(path)))
    teacher, record = build_teacher(with_file_teacher, model_config, source, torch.device("cpu"))
    assert isinstance(teacher, MeleePolicyAdapter) and not teacher.config.gradient_checkpointing
    assert record == {"source": str(path), "sha256": sha256_file(path)}
    assert _same_state(teacher.get_state(), source.get_state())
    rebuilt = teacher_from_record(record, model_config, torch.device("cpu"))
    assert isinstance(rebuilt, MeleePolicyAdapter) and not rebuilt.config.gradient_checkpointing
    assert _same_state(rebuilt.get_state(), source.get_state())


def test_fresh_runs_are_bitwise_deterministic() -> None:
    config = load_config(SMOKE_TINY)
    assert config.runtime.deterministic
    first = Trainer(config, RunOptions(wandb_mode="disabled"))
    a = first.run(2)
    first.close()
    second = Trainer(config, RunOptions(wandb_mode="disabled"))
    b = second.run(2)
    second.close()
    assert len(a.history) == len(b.history) == 2
    for left, right in zip(a.history, b.history, strict=True):
        assert _comparable(left) == _comparable(right)
    assert _same_state(first.policy.get_state(), second.policy.get_state())
    # The deterministic-algorithms switch is restored after a run.
    assert not torch.are_deterministic_algorithms_enabled()


def test_self_play_reset_and_burnin() -> None:
    base = load_config(SMOKE_TINY)
    config = finalize_config(
        replace(
            base,
            opponent=replace(base.opponent, type="self", update_interval=1),
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
            runtime=replace(base.runtime, reset_every_n_steps=1, burnin_steps_after_reset=1, num_steps=2),
        ),
        resolve_model_config(base),
    )
    trainer = Trainer(config, RunOptions(wandb_mode="disabled"))
    assert isinstance(trainer.opponent, SelfOpponent) and trainer.env.controlled_ports == (0, 1)
    result = trainer.run()
    trainer.close()
    batches = config.learner.ppo.num_batches
    frames = config.env.num_envs * config.actor.rollout_length
    # The reset happens before step 2 (step > 0 and step % 1 == 0): one burn-in rollout is discarded.
    assert trainer.resets == 1 and trainer.worker.rollouts_done == 2 * batches + 1
    assert trainer.opponent_refreshes == 2  # update_interval 1: refreshed at the start of both steps
    assert result.step == 2 and result.frames_seen == 2 * batches * frames
    assert result.history[-1]["env/frames_total"] == float((2 * batches + 1) * frames)
    assert all(row["ppo/stage"] == 2.0 for row in result.history)
    assert all(missing_metrics(row) == [] for row in result.history)
    # Without the interval nothing is reset.
    quiet = Trainer(load_config(SMOKE_TINY), RunOptions(wandb_mode="disabled"))
    quiet.run(1)
    quiet.close()
    assert quiet.resets == 0 and quiet.opponent_refreshes == 0


def test_trajectory_metrics_and_cli_parsing(tmp_path: Path) -> None:
    trainer = Trainer(load_config(SMOKE_TINY), RunOptions(wandb_mode="disabled"))
    trajectories = [trainer.worker.rollout() for _ in range(2)]
    trainer.close()
    metrics = trajectory_metrics(trajectories)
    assert set(metrics) == {
        "reward/per_frame",
        "reward/env_per_frame",
        "reward/ko_diff_per_minute",
        "reward/kos_per_minute",
        "reward/damage_dealt_per_minute",
        "reward/damage_taken_per_minute",
        "reward/deaths_per_minute",
        "reward/ledge_grabs_per_minute",
        "reward/stalling_fraction",
        "reward/approaching_factor",
        "env/resets",
    }
    # Two trained ports stack both perspectives: game starts are counted per environment.
    doubled = trajectory_metrics(trajectories, ports=2)
    assert doubled["env/resets"] == metrics["env/resets"] / 2
    with pytest.raises(ValueError, match="ports"):
        trajectory_metrics(trajectories, ports=0)
    stacked = torch.cat([trajectory.rewards for trajectory in trajectories])
    assert metrics["reward/per_frame"] == pytest.approx(float(stacked.mean()))
    assert metrics["env/resets"] == float(
        sum(int(t.is_resetting[:, t.context_frames :].sum()) for t in trajectories)
    )
    assert metrics["reward/ko_diff_per_minute"] == pytest.approx(
        metrics["reward/kos_per_minute"] - metrics["reward/deaths_per_minute"], abs=1.0e-3
    )
    with pytest.raises(ValueError, match="at least one"):
        trajectory_metrics([])

    args = parse_args(["--config", "x.toml"])
    assert args.config == Path("x.toml") and args.steps is None and args.checkpoint_dir is None
    assert args.wandb_mode is None and args.device is None and args.resume is True
    assert args.wandb_name is None and args.wandb_tags == []
    args = parse_args(
        [
            "--config",
            "x.toml",
            "--steps",
            "3",
            "--wandb-mode",
            "offline",
            "--no-resume",
            "--device",
            "cpu",
            "--checkpoint-dir",
            "d",
            "--wandb-name",
            "n",
            "--wandb-tag",
            "a",
            "--wandb-tag",
            "b",
        ]
    )
    assert args.steps == 3 and args.wandb_mode == "offline" and not args.resume
    assert args.device == "cpu" and args.checkpoint_dir == Path("d")
    assert args.wandb_name == "n" and args.wandb_tags == ["a", "b"]
    # Run identity: RunOptions name/tags reach the logger; main() derives a name and a tag from the config.
    named = Trainer(load_config(SMOKE_TINY), RunOptions(wandb_mode="disabled", run_name="r", tags=("t",)))
    assert named.logger.name == "r" and named.logger.tags[:2] == ("rl", "melee_rl")
    assert "t" in named.logger.tags
    named.close()
    argv_base = ["--config", str(SMOKE_TINY), "--wandb-mode", "disabled"]
    auto = main([*argv_base, "--steps", "0"])
    assert auto.run_name is not None and auto.run_name.startswith("rl-smoke_tiny-")
    assert auto.tags[:2] == ("rl", "melee_rl") and "smoke_tiny" in auto.tags and "ppo" in auto.tags
    explicit = main([*argv_base, "--steps", "0", "--wandb-name", "x"])
    assert explicit.run_name == "x"
    with pytest.raises(SystemExit):
        parse_args(["--config", "x.toml", "--wandb-mode", "bogus"])
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(FileNotFoundError):
        main(["--config", str(tmp_path / "missing.toml")])
    options = RunOptions(num_steps=0, wandb_mode="disabled")
    empty = Trainer(load_config(SMOKE_TINY), options)
    result = empty.run()
    empty.close()
    assert result.steps_run == 0 and result.history == [] and result.checkpoint is None


# ---------------------------------------------------------------------------
# P6a: two-port self-play and the ``other`` opponent end to end on the dummy env
# ---------------------------------------------------------------------------


def _two_port_config() -> RLConfig:
    base = load_config(SMOKE_TINY)
    return finalize_config(
        replace(
            base,
            opponent=replace(base.opponent, type="self", train=True),
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
        ),
        resolve_model_config(base),
    )


def test_two_port_self_play_trains_both_perspectives_and_resumes(tmp_path: Path) -> None:
    config = _two_port_config()
    options = RunOptions(checkpoint_dir=tmp_path / "run", wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert (
        trainer.opponent is None and trainer.worker.ports == (0, 1) and trainer.env.controlled_ports == (0, 1)
    )
    assert trainer.worker.batch_size == 2 * config.env.num_envs and trainer.worker.envs == config.env.num_envs
    result = trainer.run(2)
    trainer.close()
    batches = config.learner.ppo.num_batches
    env_frames = config.env.num_envs * config.actor.rollout_length
    # Training frames count both perspectives; env frames count each environment once.
    assert result.step == 2 and result.frames_seen == 2 * batches * 2 * env_frames
    row = result.history[-1]
    assert missing_metrics(row) == [] and set(EXTRA_METRIC_NAMES) <= set(row)
    assert row["frames_seen"] == float(result.frames_seen)
    assert row["env/frames_total"] == float(2 * batches * env_frames)
    assert row["env/trained_ports"] == 2.0 and trainer.opponent_refreshes == 0
    # Game starts are counted per environment, not per perspective: smoke_tiny restarts every 21 frames, so
    # step 1 sees the initial game start and step 2 the restart at stream row 21 -- once per env each.
    assert [r["env/resets"] for r in result.history] == [float(config.env.num_envs)] * 2
    assert row["reward/per_frame"] == pytest.approx(0.0, abs=1e-6)  # zero-sum over both perspectives
    payload = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert payload["step"] == 2 and "opponent" not in payload["rng"]
    assert payload["rl_config"]["opponent"]["train"] is True
    resumed = Trainer(config, options)
    assert (
        resumed.resumed and resumed.step == 2 and resumed.opponent is None and resumed.worker.ports == (0, 1)
    )
    assert resumed.run(1).step == 3
    resumed.close()


def test_other_opponent_end_to_end_from_a_previous_run(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(first_dir),
            "--wandb-mode",
            "disabled",
            "--steps",
            "1",
        ]
    )
    assert first.checkpoint is not None and first.checkpoint.exists()
    base = load_config(SMOKE_TINY)
    config = finalize_config(
        replace(
            base,
            opponent=replace(base.opponent, type="other", checkpoint=str(first.checkpoint), seed=3),
            env=replace(base.env, dummy=replace(base.env.dummy, players=("policy", "policy"))),
        ),
        resolve_model_config(base),
    )
    options = RunOptions(checkpoint_dir=tmp_path / "second", wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert isinstance(trainer.opponent, OtherOpponent) and trainer.worker.ports == (0,)
    assert trainer.opponent.source == str(first.checkpoint)
    assert trainer.opponent.sha256 == sha256_file(first.checkpoint)
    assert trainer.env.controlled_ports == (0, 1) and trainer.worker.batch_size == config.env.num_envs
    result = trainer.run(1)
    trainer.close()
    batches = config.learner.ppo.num_batches
    env_frames = config.env.num_envs * config.actor.rollout_length
    row = result.history[-1]
    assert result.frames_seen == batches * env_frames and row["env/trained_ports"] == 1.0
    assert missing_metrics(row) == [] and trainer.opponent_refreshes == 0  # never refreshed
    payload = load_checkpoint(tmp_path / "second" / "latest.pt")
    assert isinstance(payload["rng"]["opponent"], torch.Tensor)
    assert payload["rl_config"]["opponent"]["checkpoint"] == str(first.checkpoint)
    state_after = trainer.opponent.generator.get_state()
    resumed = Trainer(config, options)
    assert resumed.resumed and isinstance(resumed.opponent, OtherOpponent)
    assert resumed.opponent.generator.get_state().equal(state_after)
    assert _same_state(resumed.opponent.policy.get_state(), trainer.opponent.policy.get_state())
    resumed.close()


# ---------------------------------------------------------------------------
# P6b item 0: the in-loop evaluator, best.pt and eval_history.json
# ---------------------------------------------------------------------------


def _eval_config(**eval_overrides: object) -> RLConfig:
    base = load_config(SMOKE_TINY)
    settings: dict[str, object] = {
        "enabled": True,
        "interval_seconds": None,
        "interval_steps": 1,
        "frames": 2 * 8 * 3,
        "num_envs": 2,
        "rollout_length": 8,
        "opponents": ("cpu9", "best"),
    }
    settings.update(eval_overrides)
    return finalize_config(
        replace(base, eval=replace(base.eval, **settings)),  # type: ignore[arg-type]
        resolve_model_config(base),
    )


def test_in_loop_evaluation_saves_best_and_resumes(tmp_path: Path) -> None:
    config = _eval_config()
    run_dir = tmp_path / "run"
    options = RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled")
    trainer = Trainer(config, options)
    assert trainer.evaluator is not None and trainer.best_eval is None and trainer.evals == []
    result = trainer.run(2)
    trainer.close()
    # Evaluated at the start (step 0, no best.pt yet), then after every learner step; at_end adds nothing
    # when the last step was already evaluated.
    assert [item.step for item in result.evals] == [0, 1, 2]
    assert (
        result.evals[0].skipped == ("best",)
        and result.evals[1].skipped == ()
        and result.evals[2].skipped == ()
    )
    best = run_dir / "best.pt"
    assert best.exists() and (run_dir / "latest.pt").exists()
    records = json.loads((run_dir / "eval_history.json").read_text())
    assert [record["step"] for record in records] == [0, 1, 2]
    assert trainer.best_eval is not None and trainer.best_eval["metric"] == "damage_dealt_per_minute"
    assert trainer.best_eval["value"] == max(
        item.selector for item in result.evals if item.selector is not None
    )
    assert trainer.best_eval["step"] == min(
        item.step for item in result.evals if item.selector == trainer.best_eval["value"]
    )  # ties keep the earlier best
    # The W&B history carries the eval rows at their learner step next to the two training rows.
    eval_rows = [row for row in result.history if "eval/step" in row]
    train_rows = [row for row in result.history if "loss/total" in row]
    assert len(train_rows) == 2 and [row["eval/step"] for row in eval_rows] == [0.0, 1.0, 2.0]
    assert "eval/cpu9/damage_dealt_per_minute" in eval_rows[0] and "eval/best/frames" not in eval_rows[0]
    assert "eval/best/damage_dealt_per_minute" in eval_rows[1] and eval_rows[1]["eval/skipped"] == 0.0
    assert eval_rows[0]["eval/best_value"] == eval_rows[0]["eval/cpu9/damage_dealt_per_minute"]
    assert eval_rows[0]["eval/improved"] == 1.0
    # best.pt is a full RL checkpoint of the best step and carries the eval record.
    payload = load_checkpoint(best)
    assert payload["step"] == trainer.best_eval["step"]
    assert payload["loop_state"]["best_eval"]["step"] == trainer.best_eval["step"]
    assert payload["loop_state"]["evals_done"] == payload["loop_state"]["best_eval"]["step"] + 1
    # Resume: the best and the eval count come back, the start eval is not repeated, the history file grows.
    resumed = Trainer(config, options)
    assert resumed.resumed and resumed.best_eval == trainer.best_eval and resumed.evals_done == 3
    more = resumed.run(1)
    resumed.close()
    assert [item.step for item in more.evals] == [3] and more.evals[0].skipped == ()
    assert len(json.loads((run_dir / "eval_history.json").read_text())) == 4
    # The eval time counts toward the run's wall budget: a tiny max_runtime_s stops before the first step
    # but the start and end evals still happen.
    tight = finalize_config(
        replace(config, runtime=replace(config.runtime, max_runtime_s=1e-6)), resolve_model_config(config)
    )
    short = Trainer(tight, RunOptions(wandb_mode="disabled"))
    brief = short.run(5)
    short.close()
    assert brief.steps_run == 0 and [item.step for item in brief.evals] == [0]
    # Without a checkpoint directory evaluation still runs, best.pt is never written, ``best`` stays skipped.
    nodir = Trainer(config, RunOptions(wandb_mode="disabled"))
    loose = nodir.run(1)
    nodir.close()
    assert [item.step for item in loose.evals] == [0, 1] and all(
        item.skipped == ("best",) for item in loose.evals
    )
    assert nodir.best_eval is not None and nodir.evals_done == 2
    # Disabled (the shipped smoke config): no evaluator, no eval rows.
    plain = Trainer(load_config(SMOKE_TINY), RunOptions(wandb_mode="disabled"))
    assert plain.evaluator is None
    quiet = plain.run(1)
    plain.close()
    assert quiet.evals == [] and all("eval/step" not in row for row in quiet.history)
    # interval_steps = 2 with at_start off: evaluated after step 2 only (and at_end adds nothing).
    every_two = Trainer(_eval_config(interval_steps=2, at_start=False), RunOptions(wandb_mode="disabled"))
    paced = every_two.run(3)
    every_two.close()
    assert [item.step for item in paced.evals] == [2, 3]  # step 3 = the end eval


class _SaveRecorder:
    """An ``on_save`` callback that records the trainer's step (and that ``latest.pt`` already exists)."""

    def __init__(self, run_dir: Path | None = None) -> None:
        self.trainer: Trainer | None = None
        self.run_dir = run_dir
        self.steps: list[int] = []

    def __call__(self) -> None:
        assert self.trainer is not None
        if self.run_dir is not None:
            assert (self.run_dir / "latest.pt").exists()
        self.steps.append(self.trainer.step)


def test_on_save_fires_once_per_checkpoint_write(tmp_path: Path) -> None:
    """``Trainer(..., on_save=...)`` runs after every ``save()``: the Modal functions hang the Volume
    commit on it, so a container restart loses at most one save interval (BC-INIT-PLAN.md step 2.2)."""

    config = load_config(SMOKE_TINY)  # save_interval_steps = 1, snapshot_interval_steps = 2
    run_dir = tmp_path / "run"
    recorder = _SaveRecorder(run_dir)
    trainer = Trainer(config, RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled"), on_save=recorder)
    recorder.trainer = trainer
    assert trainer.on_save is recorder and recorder.steps == []
    trainer.run(3)
    trainer.close()
    # One call per save() -- steps 1, 2, 3 -- not per file: step 2 also wrote step_00000002.pt.
    assert recorder.steps == [1, 2, 3] and checkpoint_paths(run_dir, 2)[1].exists()
    # A manual save fires it too, and a trainer without the hook is unaffected.
    resumed = Trainer(config, RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled"), on_save=recorder)
    recorder.trainer = resumed
    resumed.save()
    resumed.close()
    assert recorder.steps == [1, 2, 3, 3]
    silent = Trainer(config, RunOptions(checkpoint_dir=run_dir, wandb_mode="disabled"))
    assert silent.on_save is None
    silent.save()
    silent.close()
    assert recorder.steps == [1, 2, 3, 3]
    # Without a checkpoint directory nothing is ever written, so the hook never runs.
    idle = _SaveRecorder()
    nodir = Trainer(config, RunOptions(wandb_mode="disabled"), on_save=idle)
    idle.trainer = nodir
    nodir.run(1)
    nodir.close()
    assert idle.steps == [] and nodir.latest_checkpoint is None
    # The evaluator's bookkeeping save counts as well: the start evaluation, then per step the training
    # save and the evaluation's (best.pt is written inside the same evaluation, not a separate call).
    evaluated_dir = tmp_path / "eval"
    on_eval = _SaveRecorder(evaluated_dir)
    evaluated = Trainer(
        _eval_config(), RunOptions(checkpoint_dir=evaluated_dir, wandb_mode="disabled"), on_save=on_eval
    )
    on_eval.trainer = evaluated
    evaluated.run(2)
    evaluated.close()
    assert on_eval.steps == [0, 1, 1, 2, 2] and (evaluated_dir / "best.pt").exists()
    # main() threads it through for the launcher (the Modal functions call main).
    via_main: list[int] = []
    result = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(tmp_path / "cli"),
            "--wandb-mode",
            "disabled",
            "--steps",
            "2",
        ],
        on_save=lambda: via_main.append(1),
    )
    assert result.step == 2 and via_main == [1, 1]


def test_pipelined_trainer_end_to_end_runs_saves_and_resumes(tmp_path: Path) -> None:
    """P8: the pipeline TOML builds PipelinedRolloutWorker over AsyncEnvAdapter, runs and resumes."""

    from melee_rl.env.async_env import AsyncEnvAdapter
    from melee_rl.pipeline import PipelinedRolloutWorker

    run_dir = tmp_path / "run"
    toml = CONFIG_DIR / "smoke_tiny_pipeline.toml"
    argv = ["--config", str(toml), "--checkpoint-dir", str(run_dir), "--wandb-mode", "disabled"]
    result = main([*argv, "--steps", "2"])
    assert result.step == 2 and result.steps_run == 2
    assert result.config.actor.batch_steps == 2 and result.config.actor.delay_frames == 2
    latest, _ = checkpoint_paths(run_dir, 2)
    assert result.checkpoint == latest and latest.exists()
    payload = load_checkpoint(latest)
    assert payload["delay_frames"] == 2
    assert payload["rl_config"]["actor"]["batch_steps"] == 2
    resumed = main([*argv, "--steps", "1"])
    assert resumed.step == 3 and resumed.steps_run == 1

    # The builder selection: the pipelined knobs make a PipelinedRolloutWorker over the adapter;
    # the plain smoke config keeps today's RolloutWorker over the bare env.
    config = load_config(toml)
    trainer = Trainer(config, RunOptions(num_steps=0))
    try:
        assert isinstance(trainer.worker, PipelinedRolloutWorker)
        assert isinstance(trainer.worker.env, AsyncEnvAdapter)
        assert trainer.worker.env.chunk_frames == 1 and trainer.worker.batch_steps == 2
    finally:
        trainer.close()
    plain = Trainer(load_config(SMOKE_TINY), RunOptions(num_steps=0))
    try:
        assert type(plain.worker) is not PipelinedRolloutWorker
    finally:
        plain.close()


def test_max_total_runtime_s_accumulates_across_launches(tmp_path: Path) -> None:
    """The 36-hour cap (user decision 2 Sep 2026).  Modal caps a container at 24 h, so a longer run is
    several launches of the same command; ``[runtime] max_total_runtime_s`` counts every launch's wall
    time through the checkpoint's ``loop_state["runtime_s"]`` and stops the run once the total is spent
    -- ``max_runtime_s`` alone restarts its clock with every process."""

    config = load_config(SMOKE_TINY)
    run_dir = tmp_path / "run"

    def with_cap(cap: float) -> RLConfig:
        runtime = replace(config.runtime, checkpoint_dir=str(run_dir), max_total_runtime_s=cap)
        return finalize_config(replace(config, runtime=runtime), resolve_model_config(config))

    first = Trainer(with_cap(3600.0), RunOptions(wandb_mode="disabled"))
    assert first.runtime_s_before == 0.0
    result = first.run(1)
    first.close()
    assert result.steps_run == 1 and result.history[-1]["timing/runtime_total_s"] > 0.0
    saved = load_checkpoint(run_dir / "latest.pt")["loop_state"]["runtime_s"]
    assert saved > 0.0
    # The resume restores the spent time; a cap below it stops before the first step.
    second = Trainer(with_cap(saved / 2), RunOptions(wandb_mode="disabled"))
    assert second.runtime_s_before == saved and second.runtime_s >= saved
    again = second.run(3)
    second.close()
    assert again.steps_run == 0 and again.step == 1
    # With headroom the resumed launch trains on, and the total keeps growing across the launches.
    third = Trainer(with_cap(3600.0), RunOptions(wandb_mode="disabled"))
    more = third.run(1)
    third.close()
    assert more.steps_run == 1 and more.step == 2
    assert load_checkpoint(run_dir / "latest.pt")["loop_state"]["runtime_s"] > saved
    with pytest.raises(ValueError, match="max_total_runtime_s"):
        replace(config.runtime, max_total_runtime_s=0.0)


def test_cli_set_overrides_reach_the_config(tmp_path: Path) -> None:
    """5 Sep 2026 (/rl-speedup): ``--set table.key=value`` is applied before validation and recorded."""

    args = parse_args(
        ["--config", str(SMOKE_TINY), "--set", "policy.fast_step=true", "--set", "runtime.threads=2"]
    )
    assert args.overrides == ["policy.fast_step=true", "runtime.threads=2"]
    result = main(
        [
            "--config",
            str(SMOKE_TINY),
            "--checkpoint-dir",
            str(tmp_path / "run"),
            "--wandb-mode",
            "disabled",
            "--steps",
            "1",
            "--set",
            "policy.fast_step=true",
            "--set",
            "learner.ppo.num_batches=1",
        ]
    )
    assert result.config.policy.fast_step is True and result.config.learner.ppo.num_batches == 1
    assert result.checkpoint is not None
    payload = load_checkpoint(result.checkpoint)
    assert payload["rl_config"]["policy"]["fast_step"] is True
