"""In-loop evaluation (P6b item 0): ``EvalConfig`` / opponent specs, the ``Evaluator`` on the dummy env, the
``best`` opponent that appears once ``best.pt`` exists, flattened ``eval/*`` metrics, JSON records."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import save_bc_checkpoint
from melee_rl.config import CONFIG_DIR, RLConfig, finalize_config, load_config, resolve_model_config
from melee_rl.evaluate import (
    EVAL_PREFIX,
    EVAL_STATS,
    OPPONENT_KINDS,
    EvalConfig,
    EvalOpponent,
    EvalResult,
    Evaluator,
    format_eval_summary,
    parse_opponent,
    trajectory_metrics,
)
from melee_rl.train import trajectory_metrics as train_trajectory_metrics

SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
FRAMES = 2 * 8 * 3  # 2 envs x T = 8 x 3 rollouts


def _config(**eval_overrides: object) -> RLConfig:
    """smoke_tiny with an enabled evaluator: 2 eval envs, T = 8, 48 frames per opponent, cpu9 + best."""

    base = load_config(SMOKE_TINY)
    settings: dict[str, object] = {
        "enabled": True,
        "interval_seconds": None,
        "interval_steps": 1,
        "frames": FRAMES,
        "num_envs": 2,
        "rollout_length": 8,
        "opponents": ("cpu9", "best"),
    }
    settings.update(eval_overrides)
    eval_config = replace(base.eval, **settings)  # type: ignore[arg-type]
    return finalize_config(replace(base, eval=eval_config), resolve_model_config(base))


def _timeless(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        name: {k: v for k, v in values.items() if k not in ("seconds", "fps")}
        for name, values in stats.items()
    }


def test_eval_config_defaults_opponent_specs_and_validation() -> None:
    assert parse_opponent("cpu9") == EvalOpponent(name="cpu9", kind="cpu", cpu_level=9, path=None)
    assert parse_opponent("cpu3").cpu_level == 3 and parse_opponent("cpu3").name == "cpu3"
    assert parse_opponent("best") == EvalOpponent(name="best", kind="best", cpu_level=None, path=None)
    other = parse_opponent("other:/runs/x/latest.pt")
    assert other.kind == "other" and other.path == "/runs/x/latest.pt" and other.name == "other"
    assert other.delay_frames is None  # 11 Sep 2026: unset = the file's own trained delay
    tagged = parse_opponent("other:/runs/x/latest.pt@d3")
    assert tagged.kind == "other" and tagged.path == "/runs/x/latest.pt" and tagged.delay_frames == 3
    assert tagged.name == "other"
    for bad in ("cpu0", "cpu10", "cpu", "human", "other:", "best:x", "", "other", "mimic:x", "Mimic"):
        with pytest.raises(ValueError, match="opponent"):
            parse_opponent(bad)
    with pytest.raises(ValueError, match="opponent"):
        parse_opponent("other:@d3")
    # 2 Sep 2026: the released MIMIC Fox as an in-loop yardstick (P12 showed cpu9 mis-ranks runs).
    assert parse_opponent("mimic") == EvalOpponent(name="mimic", kind="mimic", cpu_level=None, path=None)
    assert OPPONENT_KINDS == ("cpu", "best", "other", "mimic", "slippi")
    assert EvalConfig(opponents=("cpu9", "mimic"), metric_opponent="mimic").metric_opponent == "mimic"
    default = EvalConfig()
    assert not default.enabled and default.interval_seconds == 3600.0 and default.interval_steps is None
    assert default.at_start and default.at_end and default.frames == 57_600 and default.num_envs == 8
    assert default.rollout_length == 128 and default.opponents == ("cpu9", "best")
    assert default.temperature == 1.0 and default.seed == 0
    assert default.metric == "damage_dealt_per_minute" and default.metric_opponent == "cpu9"
    assert [item.kind for item in default.parsed_opponents] == ["cpu", "best"]
    assert {
        "reward_per_frame",
        "ko_diff_per_minute",
        "kos_per_minute",
        "damage_dealt_per_minute",
        "damage_taken_per_minute",
        "deaths_per_minute",
        "ledge_grabs_per_minute",
        "stalling_fraction",
        "approaching_factor",
        "game_starts",
        "frames",
        "seconds",
        "fps",
    } == set(EVAL_STATS)
    assert EVAL_PREFIX == "eval/"
    with pytest.raises(ValueError, match="frames"):
        EvalConfig(frames=0)
    with pytest.raises(ValueError, match="frames"):
        EvalConfig(frames=100, num_envs=8, rollout_length=128)  # less than one rollout
    with pytest.raises(ValueError, match="num_envs"):
        EvalConfig(num_envs=0)
    with pytest.raises(ValueError, match="rollout_length"):
        EvalConfig(rollout_length=0)
    with pytest.raises(ValueError, match="opponents"):
        EvalConfig(opponents=())
    with pytest.raises(ValueError, match="opponents"):
        EvalConfig(opponents=("cpu9", "cpu9"))
    with pytest.raises(ValueError, match="metric"):
        EvalConfig(metric="wins")
    with pytest.raises(ValueError, match="metric_opponent"):
        EvalConfig(metric_opponent="best")  # selecting the best by beating the best is circular
    with pytest.raises(ValueError, match="metric_opponent"):
        EvalConfig(opponents=("cpu9",), metric_opponent="cpu7")
    with pytest.raises(ValueError, match="interval"):
        EvalConfig(interval_seconds=0.0)
    with pytest.raises(ValueError, match="interval"):
        EvalConfig(interval_steps=0)
    with pytest.raises(ValueError, match="interval"):
        EvalConfig(enabled=True, interval_seconds=None, interval_steps=None, at_start=False, at_end=False)
    with pytest.raises(ValueError, match="temperature"):
        EvalConfig(temperature=-1.0)
    assert EvalConfig().worker_processes is None  # inherit the training layout
    assert EvalConfig(num_envs=30, worker_processes=10).worker_processes == 10
    with pytest.raises(ValueError, match="worker_processes"):
        EvalConfig(worker_processes=-1)
    with pytest.raises(ValueError, match="worker_processes"):
        EvalConfig(num_envs=30, worker_processes=16)
    assert EvalConfig(enabled=True, interval_seconds=None, interval_steps=None).at_start  # start / end only
    assert EvalConfig(opponents=("cpu7",), metric_opponent="cpu7").parsed_opponents[0].cpu_level == 7


def test_evaluator_on_the_dummy_env_skips_best_until_it_exists(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    config = _config()
    evaluator = Evaluator(config.eval, config, resolve_model_config(config), torch.device("cpu"))
    before = tiny_adapter.get_state()
    result = evaluator.evaluate(tiny_adapter, step=0, best_path=None)
    assert isinstance(result, EvalResult) and result.step == 0 and result.skipped == ("best",)
    assert set(result.stats) == {"cpu9"} and set(result.stats["cpu9"]) == set(EVAL_STATS)
    cpu = result.stats["cpu9"]
    assert cpu["frames"] == float(FRAMES) and cpu["seconds"] > 0.0 and cpu["fps"] > 0.0
    assert cpu["game_starts"] >= 2.0  # the two environments' first frames
    assert result.selector == cpu["damage_dealt_per_minute"] and result.seconds >= cpu["seconds"]
    metrics = result.metrics()
    assert all(key.startswith(EVAL_PREFIX) for key in metrics)
    assert metrics["eval/step"] == 0.0 and metrics["eval/seconds"] > 0.0 and metrics["eval/skipped"] == 1.0
    assert metrics["eval/cpu9/damage_dealt_per_minute"] == cpu["damage_dealt_per_minute"]
    assert metrics["eval/cpu9/frames"] == float(FRAMES)
    assert "eval/best/frames" not in metrics
    # The student is untouched: a frozen clone played, the weights and the training mode are unchanged.
    assert tiny_adapter.training
    assert all(torch.equal(before[name], value) for name, value in tiny_adapter.get_state().items())
    # Deterministic for a fixed seed (the dummy env and the sampling are seeded).
    again = evaluator.evaluate(tiny_adapter, step=0, best_path=None)
    assert _timeless(again.stats) == _timeless(result.stats)
    # Once a best.pt exists the ``best`` opponent plays port 1 of a two-policy environment.
    best = tmp_path / "best.pt"
    save_bc_checkpoint(best, tiny_adapter.get_state(), tiny_adapter.config)
    with_best = evaluator.evaluate(tiny_adapter, step=3, best_path=best)
    assert with_best.skipped == () and set(with_best.stats) == {"cpu9", "best"}
    assert (
        with_best.stats["best"]["frames"] == float(FRAMES) and with_best.stats["best"]["game_starts"] >= 2.0
    )
    assert (
        with_best.metrics()["eval/skipped"] == 0.0
        and "eval/best/damage_dealt_per_minute" in with_best.metrics()
    )
    assert with_best.selector == with_best.stats["cpu9"]["damage_dealt_per_minute"]
    record = with_best.to_record()
    json.dumps(record)
    assert record["step"] == 3 and set(record["stats"]) == {"cpu9", "best"} and record["skipped"] == []
    assert record["metric"] == "damage_dealt_per_minute" and record["metric_opponent"] == "cpu9"
    summary = format_eval_summary(with_best)
    assert summary.startswith("eval step 3") and "cpu9" in summary and "best" in summary
    # An ``other:<path>`` opponent is the same machinery with an explicit checkpoint.
    other = Evaluator(
        replace(config.eval, opponents=("cpu9", f"other:{best}")), config, resolve_model_config(config), "cpu"
    ).evaluate(tiny_adapter, step=1)
    assert set(other.stats) == {"cpu9", "other"} and other.skipped == ()
    # A missing ``other`` file is an error (unlike ``best``, which is simply not there yet).
    with pytest.raises(FileNotFoundError):
        Evaluator(
            replace(config.eval, opponents=(f"other:{tmp_path / 'missing.pt'}",), metric_opponent="other"),
            config,
            resolve_model_config(config),
            "cpu",
        ).evaluate(tiny_adapter, step=1)
    assert train_trajectory_metrics is trajectory_metrics  # moved here from train.py, re-exported there


def test_other_rivals_play_at_their_own_delay_unless_told_otherwise(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: Any,
    tiny_value_config: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """11 Sep 2026 (/delay-finetune): an ``other`` (or ``best``) rival plays at the delay its file was trained
    at -- until now it inherited the run's, so a delay-0 anchor in a D = 18 run would have played at 18 --
    and ``other:<path>@d<n>`` overrides it (the naive fork point of a delayed run: the base at the run's
    D)."""

    from melee_rl import evaluate as evaluate_module
    from melee_rl.checkpoint import save_rl_checkpoint
    from melee_rl.learner import LearnerConfig, PPOLearner
    from melee_rl.opponents import OtherOpponent
    from melee_rl.value import ValueNet

    seen: list[int] = []

    class Recording(OtherOpponent):
        def __init__(self, path: str | Path, batch_size: int, **kwargs: Any) -> None:
            seen.append(int(kwargs.get("delay_frames", 0)))
            super().__init__(path, batch_size, **kwargs)

    monkeypatch.setattr(evaluate_module, "OtherOpponent", Recording)
    learner = PPOLearner(
        tiny_adapter, None, ValueNet(tiny_config, tiny_value_config), LearnerConfig(kl_teacher_weight=0.0)
    )
    delayed = tmp_path / "delayed.pt"
    save_rl_checkpoint(delayed, learner, tiny_config, step=0, delay_frames=2, context_mode="ring")
    own = _config(opponents=(f"other:{delayed}",), metric_opponent="other")
    Evaluator(own.eval, own, resolve_model_config(own), "cpu").evaluate(tiny_adapter, step=0)
    assert seen == [2]
    forced = _config(opponents=(f"other:{delayed}@d0",), metric_opponent="other")
    Evaluator(forced.eval, forced, resolve_model_config(forced), "cpu").evaluate(tiny_adapter, step=0)
    assert seen == [2, 0]
    # ``best`` is this run's own file: its trained delay is the run's.
    best = _config(opponents=("cpu9", "best"))
    Evaluator(best.eval, best, resolve_model_config(best), "cpu").evaluate(
        tiny_adapter, step=0, best_path=delayed
    )
    assert seen == [2, 0, 2]


def test_build_eval_env_replay_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """P9: the recorder keeps every clip's ``.slp``; the run file's training Dolphins are untouched."""

    from dataclasses import replace as dataclass_replace

    import melee_rl.env.dolphin_mp as dolphin_mp
    from melee_rl.env.dolphin import DolphinEnvConfig
    from melee_rl.evaluate import build_eval_env

    seen: list[DolphinEnvConfig] = []

    def fake_build(config: DolphinEnvConfig, *, cpu_level: int = 9) -> object:
        seen.append(config)
        return object()

    monkeypatch.setattr(dolphin_mp, "build_dolphin_env", fake_build)
    base = load_config(SMOKE_TINY)
    dolphin = DolphinEnvConfig(num_envs=64, slippi_port=51441, save_replays=False, replay_dir=None)
    config = dataclass_replace(base, env=dataclass_replace(base.env, type="dolphin", dolphin=dolphin))
    build_eval_env(config, num_envs=2, players=("policy", "cpu"), cpu_level=9, port_offset=72)
    assert seen[-1].save_replays is False and seen[-1].replay_dir is None  # unchanged by default
    assert seen[-1].num_envs == 2 and seen[-1].slippi_port == 51441 + 72
    assert seen[-1].worker_processes == dolphin.worker_processes  # the training layout by default
    # [eval] worker_processes gives the eval Dolphins their own layout; in_process (mimic) wins over both.
    own = dataclass_replace(config, eval=dataclass_replace(config.eval, num_envs=30, worker_processes=10))
    build_eval_env(own, num_envs=30, players=("policy", "policy"), cpu_level=9, port_offset=72)
    assert seen[-1].worker_processes == 10 and seen[-1].num_envs == 30
    build_eval_env(
        own, num_envs=30, players=("policy", "policy"), cpu_level=9, port_offset=72, in_process=True
    )
    assert seen[-1].worker_processes == 0
    build_eval_env(
        config,
        num_envs=2,
        players=("policy", "cpu"),
        cpu_level=9,
        port_offset=72,
        save_replays=True,
        replay_dir=tmp_path / "clips",
    )
    assert seen[-1].save_replays is True and seen[-1].replay_dir == str(tmp_path / "clips")
    assert config.env.dolphin.save_replays is False  # the run file itself is not mutated
    # The dummy toy ignores both (there is nothing to record).
    assert (
        build_eval_env(
            base, num_envs=2, players=("policy", "cpu"), cpu_level=9, port_offset=0, save_replays=True
        ).num_envs
        == 2
    )


def test_mimic_eval_opponent_plays_in_process_with_the_raw_gamestates(
    tiny_adapter: MeleePolicyAdapter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``mimic`` (2 Sep 2026): MIMIC builds its observation from the raw libmelee gamestates, which only
    exist in the trainer's process -- so its evaluation environment is built in-process
    (``worker_processes = 0``) and any environment without ``gamestates()`` is refused.  The released
    runtime loads once per evaluator (the model is the expensive part); every evaluation gets a fresh
    player (its context windows and sampling stream start over)."""

    from melee_rl import evaluate as evaluate_module
    from melee_rl.env.dummy import DummyMeleeEnv
    from tests.rl.test_mimic import FakeGamestate, make_runtime

    class GamestateDummy(DummyMeleeEnv):
        def gamestates(self) -> list[Any]:
            return [FakeGamestate(tag=index + 1) for index in range(self.num_envs)]

    built: list[dict[str, Any]] = []
    real_build = evaluate_module.build_eval_env

    def fake_build(config: RLConfig, **kwargs: Any) -> Any:
        built.append(dict(kwargs))
        dummy = replace(
            config.env.dummy,
            num_envs=kwargs["num_envs"],
            players=kwargs["players"],
            seed=config.env.dummy.seed,
        )
        return GamestateDummy(dummy)

    loads: list[Any] = []

    def fake_load(mimic_config: Any) -> Any:
        loads.append(mimic_config)
        return make_runtime()

    monkeypatch.setattr(evaluate_module, "build_eval_env", fake_build)
    monkeypatch.setattr(evaluate_module, "load_runtime", fake_load)
    config = _config(opponents=("cpu9", "mimic"), metric_opponent="mimic")
    evaluator = Evaluator(config.eval, config, resolve_model_config(config), "cpu")
    result = evaluator.evaluate(tiny_adapter, step=0)
    assert set(result.stats) == {"cpu9", "mimic"} and result.skipped == ()
    mimic = result.stats["mimic"]
    assert mimic["frames"] == float(FRAMES) and mimic["game_starts"] >= 2.0 and set(mimic) == set(EVAL_STATS)
    assert result.selector == mimic[config.eval.metric]
    cpu_build, mimic_build = built
    assert cpu_build["players"] == ("policy", "cpu") and "in_process" not in cpu_build
    assert mimic_build["players"] == ("policy", "policy") and mimic_build["in_process"] is True
    assert len(loads) == 1 and loads[0] == config.mimic
    again = evaluator.evaluate(tiny_adapter, step=1)
    assert len(loads) == 1  # the runtime is cached; a fresh player per evaluation
    assert _timeless(again.stats) == _timeless(result.stats)  # deterministic for a fixed seed
    # An environment without gamestates (the dummy toy through the real builder) is refused up front.
    monkeypatch.setattr(evaluate_module, "build_eval_env", real_build)
    with pytest.raises(ValueError, match="mimic"):
        Evaluator(config.eval, config, resolve_model_config(config), "cpu").evaluate(tiny_adapter, step=2)


def test_build_eval_env_in_process_drops_shared_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """S48: ``shared_memory`` is the worker mode's frame exchange; an in-process eval env (MIMIC, or
    ``[eval] worker_processes = 0``) must not inherit it -- the two post-trained day runs died at their
    first eval round on exactly this (5 Sep 2026).  A worker-based eval layout keeps the training flag."""

    from dataclasses import replace as dataclass_replace

    import melee_rl.env.dolphin_mp as dolphin_mp
    from melee_rl.env.dolphin import DolphinEnvConfig
    from melee_rl.evaluate import build_eval_env

    seen: list[DolphinEnvConfig] = []

    def fake_build(config: DolphinEnvConfig, *, cpu_level: int = 9) -> object:
        seen.append(config)
        return object()

    monkeypatch.setattr(dolphin_mp, "build_dolphin_env", fake_build)
    base = load_config(SMOKE_TINY)
    dolphin = DolphinEnvConfig(num_envs=96, worker_processes=16, shared_memory=True, slippi_port=51441)
    config = dataclass_replace(base, env=dataclass_replace(base.env, type="dolphin", dolphin=dolphin))
    players = ("policy", "policy")
    # The MIMIC path: in-process Dolphins, so no exchange.
    build_eval_env(config, num_envs=30, players=players, cpu_level=9, port_offset=96, in_process=True)
    assert seen[-1].worker_processes == 0 and seen[-1].shared_memory is False
    # [eval] worker_processes = 0 likewise.
    zero = dataclass_replace(config, eval=dataclass_replace(config.eval, num_envs=30, worker_processes=0))
    build_eval_env(zero, num_envs=30, players=players, cpu_level=9, port_offset=96)
    assert seen[-1].worker_processes == 0 and seen[-1].shared_memory is False
    # A worker layout keeps the training run's exchange.
    own = dataclass_replace(config, eval=dataclass_replace(config.eval, num_envs=30, worker_processes=10))
    build_eval_env(own, num_envs=30, players=players, cpu_level=9, port_offset=96)
    assert seen[-1].worker_processes == 10 and seen[-1].shared_memory is True
    assert config.env.dolphin.shared_memory is True  # the run file itself is not mutated


def test_slippi_eval_opponents_play_the_torch_port_with_their_own_character(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``slippi:<release>`` (17 Sep 2026, the league run): a released slippi-ai agent through the PyTorch
    port, on worker envs (it needs frames only), booted with the release's own character, logged as
    ``eval/<release>/*`` and selectable as ``metric_opponent``. A release of several characters needs a mix
    arm to say which."""

    from dataclasses import replace as dataclass_replace

    import melee_rl.env.dolphin_mp as dolphin_mp
    from melee_rl.env.dolphin import DolphinEnvConfig
    from melee_rl.evaluate import build_eval_env
    from tests.rl._slippi_release import write_release

    assert parse_opponent("slippi:falco_d18_vs_fox_v4") == EvalOpponent(
        name="falco_d18_vs_fox_v4", kind="slippi", release="falco_d18_vs_fox_v4"
    )
    assert "slippi" in OPPONENT_KINDS
    with pytest.raises(ValueError, match="release"):
        parse_opponent("slippi:not_a_release")
    with pytest.raises(ValueError, match="characters"):
        parse_opponent("slippi:gm")
    EvalConfig(opponents=("slippi:fox_d18_ditto_v4", "mimic"), metric_opponent="fox_d18_ditto_v4")

    seen: list[DolphinEnvConfig] = []

    def fake_build(config: DolphinEnvConfig, *, cpu_level: int = 9) -> object:
        seen.append(config)
        return object()

    monkeypatch.setattr(dolphin_mp, "build_dolphin_env", fake_build)
    base = load_config(SMOKE_TINY)
    dolphin = DolphinEnvConfig(num_envs=6, opponent_characters=(22,) * 6, slippi_port=51441)
    dolphin_config = dataclass_replace(base, env=dataclass_replace(base.env, type="dolphin", dolphin=dolphin))
    build_eval_env(
        dolphin_config,
        num_envs=2,
        players=("policy", "policy"),
        cpu_level=9,
        port_offset=8,
        characters=(1, 22),
    )
    assert seen[-1].characters == (1, 22) and seen[-1].opponent_characters == ()

    model_dir = tmp_path / "releases"
    write_release(model_dir / "fox_d18_ditto_v4", version=4, max_names=16, names=("Cody", "Hax"), delay=2)
    config = _config(opponents=("slippi:fox_d18_ditto_v4",), metric_opponent="fox_d18_ditto_v4")
    config = replace(
        config, slippi_ai=replace(config.slippi_ai, model_dir=str(model_dir), verify_sha256=False)
    )
    evaluator = Evaluator(config.eval, config, resolve_model_config(config), "cpu")
    result = evaluator.evaluate(tiny_adapter, step=3)
    assert set(result.stats) == {"fox_d18_ditto_v4"} and result.selector is not None
    assert result.stats["fox_d18_ditto_v4"]["frames"] == FRAMES
    assert f"{EVAL_PREFIX}fox_d18_ditto_v4/damage_dealt_per_minute" in result.metrics()
    again = evaluator.evaluate(tiny_adapter, step=3)  # a fresh agent per round: the same numbers
    assert _timeless(again.stats) == _timeless(result.stats)
