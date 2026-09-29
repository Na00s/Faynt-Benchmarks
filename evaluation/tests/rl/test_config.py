"""TOML config (PLAN.md §3.4, §6 P3): parsing, round trip, validation (delay, context, microbatch, dtype)."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from melee_rl.config import (
    CONFIG_DIR,
    CONFIG_FORMAT,
    EnvConfig,
    PolicyConfig,
    RLConfig,
    RuntimeConfig,
    TeacherConfig,
    config_from_mapping,
    config_to_mapping,
    default_config_yaml,
    finalize_config,
    load_config,
    resolve_model_config,
)
from melee_rl.env.dolphin import DolphinEnvConfig
from melee_rl.env.dummy import DummyEnvConfig
from melee_rl.evaluate import EvalConfig
from melee_rl.logging import LoggingConfig
from melee_rl.video import expand_matches
from model import ModelConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SMOKE_TINY = CONFIG_DIR / "smoke_tiny.toml"
SMOKE_3M = CONFIG_DIR / "smoke_3m.toml"
LEARN_TINY = CONFIG_DIR / "learn_tiny.toml"

TINY_OVERRIDES: dict[str, Any] = {
    "d_model": 32,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 8,
    "d_ff": 64,
    "context_length": 24,
    "encoder_hidden_size": 8,
    "item_mlp_layers": 1,
    "gradient_checkpointing": False,
}


def _raw(**tables: Any) -> dict[str, Any]:
    """A minimal valid mapping (tiny random-init policy, no teacher KL) with ``tables`` merged in."""

    raw: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "policy": {"profile": "3m", "model_overrides": dict(TINY_OVERRIDES)},
        "env": {"type": "dummy", "dummy": {"num_envs": 3}},
        "actor": {"rollout_length": 8, "context_frames": 16},
        "learner": {"kl_teacher_weight": 0.0},
    }
    for name, table in tables.items():
        if isinstance(table, Mapping) and isinstance(raw.get(name), dict):
            merged = copy.deepcopy(raw[name])
            for key, value in table.items():
                if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value
            raw[name] = merged
        else:
            raw[name] = table
    return raw


def _finalized(**tables: Any) -> RLConfig:
    config = config_from_mapping(_raw(**tables))
    return finalize_config(config, resolve_model_config(config))


def test_shipped_configs_load_and_round_trip() -> None:
    assert CONFIG_DIR == PACKAGE_ROOT / "configs" / "rl"
    assert default_config_yaml() == PACKAGE_ROOT / "config.yaml"

    tiny = load_config(SMOKE_TINY)
    assert tiny.format == CONFIG_FORMAT
    assert tiny.policy.init == "random" and tiny.teacher.source == "policy"
    assert tiny.policy.compute_dtype == "float32" and tiny.policy.cache_dtype == "float32"
    assert tiny.env.type == "dummy" and tiny.env.num_envs == tiny.env.dummy.num_envs
    assert tiny.env.dummy.players == ("policy", "cpu") and tiny.opponent.type == "cpu"
    assert tiny.learner.kl_teacher_weight == 0.0  # random teacher: no anchoring to noise (PLAN.md §5, §9.4)
    assert tiny.learner.ppo.num_batches >= 1 and tiny.runtime.num_steps >= 2
    assert tiny.logging.mode == "disabled"
    model = resolve_model_config(tiny)
    assert model.d_model == 32 and model.n_layers == 2 and model.context_length == 24
    assert model.compute_dtype == "float32" and model.cache_dtype == "float32" and model.profile == "3m"
    assert tiny.actor.resolve_context_frames(model.context_length) + tiny.actor.rollout_length <= 24
    assert Path(tiny.policy.config_yaml or "") == PACKAGE_ROOT / "config.yaml"

    learn = load_config(LEARN_TINY)
    assert learn.env.dummy.learnable_signal and learn.env.dummy.signal_reward > 0.0
    assert learn.learner.ppo.epsilon == pytest.approx(0.2) and learn.learner.ppo.num_epochs == 2
    assert learn.learner.learning_rate == pytest.approx(3.0e-3)
    assert learn.teacher.source == "none" and learn.learner.kl_teacher_weight == 0.0
    assert learn.learner.ppo.max_mean_actor_kl >= 1.0  # the guard would revert every lr 3e-3 step
    learn_model = resolve_model_config(learn)
    assert learn_model.context_length == 32 and learn_model.d_model == 32 and learn_model.n_layers == 2
    assert learn.env.num_envs == 8 and learn.actor.rollout_length == 16
    assert learn.actor.resolve_context_frames(32) == 16

    three = load_config(SMOKE_3M)
    three_model = resolve_model_config(three)
    assert three.policy.profile == "3m" and three_model.d_model == 128 and three_model.context_length == 256
    assert three_model.compute_dtype == "float32"  # RL precision applied to Ali's bf16 default
    assert three.learner.kl_teacher_weight == 0.0

    for config in (tiny, learn, three):
        # Shared W&B project (frisson-labs/Faynt): our runs are separable by group/tags/job type.
        assert config.logging.entity == "frisson-labs" and config.logging.project == "Faynt"
        assert config.logging.group == "rl-post-training" and config.logging.job_type == "rl"
        assert "ppo" in config.logging.tags and "dummy-env" in config.logging.tags
        assert config.logging.mode == "disabled"  # tests never talk to W&B; the cloud runs pass --wandb-mode
        mapping = config_to_mapping(config)
        assert mapping["format"] == CONFIG_FORMAT and set(mapping) == {
            "format",
            "runtime",
            "policy",
            "teacher",
            "env",
            "actor",
            "opponent",
            "learner",
            "delay",
            "logging",
            "eval",
            "video",
            "mimic",
            "smashbot",
            "slippi_ai",
            "phillip",
            "profiling",
        }
        assert config_from_mapping(mapping) == config
        assert isinstance(mapping["env"]["dummy"]["players"], tuple)
        assert not config.eval.enabled  # the shipped smoke configs do not evaluate


def test_defaults_and_nested_tables() -> None:
    config = _finalized()
    assert isinstance(config.runtime, RuntimeConfig) and isinstance(config.policy, PolicyConfig)
    assert isinstance(config.teacher, TeacherConfig) and isinstance(config.env, EnvConfig)
    assert isinstance(config.env.dummy, DummyEnvConfig) and isinstance(config.logging, LoggingConfig)
    assert config.runtime.device == "cpu" and config.runtime.deterministic
    assert config.policy.config_yaml is None and config.policy.seed == 0
    assert config.actor.context_frames == 16 and config.actor.delay_frames == 0
    assert config.learner.ppo.num_epochs == 2 and config.learner.returns.reward_halflife == 4.0
    assert config.learner.value.context_length is None
    # ``players`` is derived from the opponent type when the env table omits it.
    assert config.env.dummy.players == ("policy", "cpu")
    self_play = _finalized(opponent={"type": "self"})
    assert self_play.env.dummy.players == ("policy", "policy")
    # Nested tables and TOML arrays map onto the existing frozen dataclasses.
    nested = _finalized(
        learner={
            "returns": {"reward_halflife": 8.0, "discount_on_death": 0.5},
            "value": {"d_model": 16, "n_layers": 1, "n_heads": 2, "n_kv_heads": 1, "head_dim": 8, "d_ff": 32},
            "ppo": {"num_epochs": 1, "num_batches": 4},
            "reward": {"damage_ratio": 0.02},
        },
        actor={"reward": {"nana_ratio": 0.0}},
        env={"dummy": {"characters": [1, 2], "stocks": 2}},
        logging={"tags": ["ppo", "tiny"], "mode": "offline"},
    )
    assert nested.learner.returns.discount_on_death == 0.5 and nested.learner.value.d_model == 16
    assert nested.learner.ppo.num_batches == 4 and nested.learner.reward is not None
    assert nested.learner.reward.damage_ratio == pytest.approx(0.02)
    assert nested.actor.reward.nana_ratio == 0.0
    assert nested.env.dummy.characters == (1, 2) and nested.env.dummy.stocks == 2
    assert nested.logging.tags == ("ppo", "tiny") and nested.logging.mode == "offline"
    # Integers are accepted where floats are declared (TOML ``1`` vs ``1.0``) and stored as floats.
    ints = _finalized(learner={"learning_rate": 1, "ppo": {"beta": 0}})
    assert ints.learner.learning_rate == 1.0 and isinstance(ints.learner.learning_rate, float)
    assert ints.learner.ppo.beta == 0.0 and isinstance(ints.learner.ppo.beta, float)


def test_delay_validation_d_positive_rejected_without_override() -> None:
    with pytest.raises(ValueError, match=r"delay_frames 1 .* action_offset_frames"):
        _finalized(actor={"delay_frames": 1})
    allowed = _finalized(actor={"delay_frames": 1}, delay={"allow_mismatch": True})
    assert allowed.actor.delay_frames == 1 and allowed.delay.allow_mismatch
    assert _finalized(actor={"delay_frames": 0}).actor.delay_frames == 0


def test_context_and_microbatch_validation() -> None:
    too_long = r"context_frames 20 \+ rollout_length 8 must fit in context_length 24"
    with pytest.raises(ValueError, match=too_long):
        _finalized(actor={"context_frames": 20})
    with pytest.raises(ValueError, match="must fit in context_length"):
        _finalized(actor={"rollout_length": 25, "context_frames": 0})
    # ``context_frames`` omitted -> ``context_length - T``.
    assert _finalized(actor={"context_frames": None}).actor.context_frames is None
    with pytest.raises(ValueError, match="microbatch_envs 2 must divide num_envs 3"):
        _finalized(learner={"microbatch_envs": 2})
    with pytest.raises(ValueError, match="microbatch_envs 4 must divide num_envs 3"):
        _finalized(learner={"microbatch_envs": 4})
    assert _finalized(learner={"microbatch_envs": 3}).learner.microbatch_envs == 3
    assert _finalized(learner={"microbatch_envs": 1}).learner.microbatch_envs == 1
    with pytest.raises(ValueError, match=r"value.context_length 20 .* C \+ T \+ 1 = 25"):
        _finalized(learner={"value": {"context_length": 20}})


def test_precision_and_teacher_validation() -> None:
    with pytest.raises(ValueError, match=r"behaviour_logits = \"actor\".*bfloat16"):
        _finalized(policy={"compute_dtype": "bfloat16"}, learner={"ppo": {"behaviour_logits": "actor"}})
    forced = _finalized(policy={"compute_dtype": "bfloat16"})
    assert forced.learner.ppo.behaviour_logits == "learner"  # "auto" is forced to "learner" under bf16
    assert resolve_model_config(forced).compute_dtype == "bfloat16"
    assert _finalized().learner.ppo.behaviour_logits == "auto"
    with pytest.raises(ValueError, match=r"random-init teacher.*kl_teacher_weight"):
        _finalized(learner={"kl_teacher_weight": 0.1})
    with pytest.raises(ValueError, match=r"random-init teacher.*reverse_kl_teacher_weight"):
        _finalized(learner={"reverse_kl_teacher_weight": 0.1})
    with pytest.raises(ValueError, match=r"teacher.source = \"none\".*kl_teacher_weight"):
        _finalized(teacher={"source": "none"}, learner={"kl_teacher_weight": 0.1})
    assert _finalized(teacher={"source": "none"}).teacher.source == "none"
    # The teacher may attend to the stored prefix only where there is one (context_mode = "prefix").
    with pytest.raises(ValueError, match=r"teacher_prefix.*context_mode"):
        _finalized(learner={"teacher_prefix": True})
    assert _finalized().learner.teacher_prefix is False
    with_prefix = _finalized(actor={"context_mode": "prefix"}, learner={"teacher_prefix": True})
    assert with_prefix.learner.teacher_prefix is True and with_prefix.actor.context_mode == "prefix"
    with pytest.raises(ValueError, match=r"teacher.path"):
        _finalized(teacher={"source": "checkpoint"})
    with pytest.raises(ValueError, match=r"teacher.source"):
        _finalized(teacher={"source": "oracle"})
    with pytest.raises(ValueError, match=r"policy.checkpoint"):
        _finalized(policy={"init": "checkpoint"})
    with pytest.raises(ValueError, match=r"policy.init"):
        _finalized(policy={"init": "zeros"})
    # A policy initialised from a checkpoint may anchor to it with any weight.
    checkpoint = config_from_mapping(
        _raw(policy={"init": "checkpoint", "checkpoint": "weights.pt"}, learner={"kl_teacher_weight": 0.1})
    )
    assert finalize_config(checkpoint, resolve_model_config(checkpoint)).learner.kl_teacher_weight == 0.1
    with pytest.raises(ValueError, match="model_overrides"):
        _finalized(policy={"model_overrides": {"not_a_field": 1}})
    with pytest.raises(ValueError, match=r"d_model must equal n_heads \* head_dim"):
        _finalized(policy={"model_overrides": {**TINY_OVERRIDES, "d_model": 48}})
    with pytest.raises(KeyError, match="unknown scaling profile"):
        resolve_model_config(config_from_mapping(_raw(policy={"profile": "7m"})))


def test_env_opponent_consistency_unknown_keys_and_types() -> None:
    with pytest.raises(ValueError, match=r"players .* opponent.type"):
        _finalized(env={"dummy": {"players": ["policy", "policy"]}}, opponent={"type": "cpu"})
    with pytest.raises(ValueError, match=r"players .* opponent.type"):
        _finalized(env={"dummy": {"players": ["policy", "cpu"]}}, opponent={"type": "self"})
    with pytest.raises(ValueError, match=r"env.type .*'wii'"):
        _finalized(env={"type": "wii"})
    dolphin = _finalized(env={"type": "dolphin", "dolphin": {"num_envs": 3}})
    assert dolphin.env.type == "dolphin" and dolphin.env.num_envs == 3
    assert dolphin.env.players == ("policy", "cpu") and dolphin.env.active is dolphin.env.dolphin
    with pytest.raises(ValueError, match=r"env.dolphin.players .* opponent.type"):
        _finalized(
            env={"type": "dolphin", "dolphin": {"players": ["policy", "policy"]}}, opponent={"type": "cpu"}
        )
    with pytest.raises(ValueError, match=r"\[env.dolphin\] unknown key 'iso'"):
        config_from_mapping(_raw(env={"type": "dolphin", "dolphin": {"iso": "x"}}))
    with pytest.raises(ValueError, match=r"\[actor\] unknown key 'num_envs'"):
        config_from_mapping(_raw(actor={"num_envs": 4}))
    with pytest.raises(ValueError, match=r"\[learner.ppo\] unknown key 'clip'"):
        config_from_mapping(_raw(learner={"ppo": {"clip": 0.2}}))
    with pytest.raises(ValueError, match="unknown table 'actors'"):
        config_from_mapping(_raw(actors={}))
    with pytest.raises(ValueError, match="format"):
        config_from_mapping({**_raw(), "format": "melee_rl.rl_config.v0"})
    with pytest.raises(ValueError, match="format"):
        raw = _raw()
        del raw["format"]
        config_from_mapping(raw)
    with pytest.raises(ValueError, match=r"\[actor\] rollout_length must be int"):
        config_from_mapping(_raw(actor={"rollout_length": "8"}))
    with pytest.raises(ValueError, match=r"\[runtime\] deterministic must be bool"):
        config_from_mapping(_raw(runtime={"deterministic": 1}))
    with pytest.raises(ValueError, match=r"\[logging\] tags must be"):
        config_from_mapping(_raw(logging={"tags": "ppo"}))
    with pytest.raises(ValueError, match=r"\[env.dummy\] characters must be"):
        config_from_mapping(_raw(env={"dummy": {"characters": [1, 2, 3]}}))
    with pytest.raises(ValueError, match=r"\[runtime\] device"):
        _finalized(runtime={"device": "tpu:0"})
    with pytest.raises(ValueError, match="save_interval_steps"):
        _finalized(runtime={"save_interval_steps": 0})
    with pytest.raises(ValueError, match="num_steps"):
        _finalized(runtime={"num_steps": -1})
    with pytest.raises(ValueError, match="reset_every_n_steps"):
        _finalized(runtime={"reset_every_n_steps": 0})
    with pytest.raises(ValueError, match="burnin_steps_after_reset"):
        _finalized(runtime={"burnin_steps_after_reset": -1})
    with pytest.raises(ValueError, match=r"logging.mode"):
        _finalized(logging={"mode": "sometimes"})
    # The existing dataclass validations surface unchanged.
    with pytest.raises(ValueError, match="rollout_length must be >= 1"):
        config_from_mapping(_raw(actor={"rollout_length": 0}))
    with pytest.raises(ValueError, match="opponent type must be one of"):
        config_from_mapping(_raw(opponent={"type": "human"}))


def test_paths_resolve_relative_to_the_toml(tmp_path: Path) -> None:
    toml = tmp_path / "run.toml"
    toml.write_text(
        "\n".join(
            [
                f'format = "{CONFIG_FORMAT}"',
                "[runtime]",
                'checkpoint_dir = "ckpt"',
                "[policy]",
                'profile = "3m"',
                f'config_yaml = "{(PACKAGE_ROOT / "config.yaml").as_posix()}"',
                "[policy.model_overrides]",
                *[
                    f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in TINY_OVERRIDES.items()
                ],
                "[teacher]",
                'source = "checkpoint"',
                'path = "teacher/latest.pt"',
                "[env.dummy]",
                "num_envs = 2",
                "[actor]",
                "rollout_length = 8",
                "context_frames = 16",
                "[learner]",
                "kl_teacher_weight = 0.1",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(toml)
    assert config.runtime.checkpoint_dir == str(tmp_path / "ckpt")
    assert config.teacher.path == str(tmp_path / "teacher" / "latest.pt")
    assert config.policy.config_yaml == str(PACKAGE_ROOT / "config.yaml")
    assert resolve_model_config(config).d_model == 32
    # Absolute paths and the default YAML are left alone; a missing TOML is an error.
    plain = load_config(SMOKE_TINY)
    assert plain.runtime.checkpoint_dir is None
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.toml")
    with pytest.raises(ValueError, match="config_yaml"):
        resolve_model_config(config_from_mapping(_raw(policy={"config_yaml": str(tmp_path / "nope.yaml")})))
    assert isinstance(resolve_model_config(plain), ModelConfig)


def test_dolphin_run_files_round_trip_and_resolve_their_paths(tmp_path: Path) -> None:
    for name in ("dolphin_cpu.toml", "dolphin_self.toml"):
        config = load_config(CONFIG_DIR / name)
        assert config.env.type == "dolphin" and isinstance(config.env.dolphin, DolphinEnvConfig)
        assert config.env.num_envs == 2 and config.env.players == config.env.dolphin.players
        assert config.env.dolphin.iso_path == "/iso/melee.iso"
        assert config.env.dolphin.dolphin_path.startswith("/opt/dolphin/")
        mapping = config_to_mapping(config)
        assert set(mapping["env"]) == {"type", "dummy", "dolphin"}
        assert isinstance(mapping["env"]["dolphin"]["players"], tuple)
        assert config_from_mapping(mapping) == config
        assert "dolphin-env" in config.logging.tags and config.logging.mode == "disabled"
    cpu = load_config(CONFIG_DIR / "dolphin_cpu.toml")
    assert cpu.env.players == ("policy", "cpu") and cpu.opponent.type == "cpu" and cpu.opponent.cpu_level == 9
    self_play = load_config(CONFIG_DIR / "dolphin_self.toml")
    assert self_play.env.players == ("policy", "policy") and self_play.opponent.type == "self"
    # Relative Dolphin paths resolve against the TOML's directory; the dummy table is untouched.
    toml = tmp_path / "local.toml"
    toml.write_text(
        "\n".join(
            [
                f'format = "{CONFIG_FORMAT}"',
                "[policy]",
                'profile = "3m"',
                "[policy.model_overrides]",
                *[
                    f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in TINY_OVERRIDES.items()
                ],
                "[env]",
                'type = "dolphin"',
                "[env.dolphin]",
                "num_envs = 1",
                'dolphin_path = "dolphin/squashfs-root/usr/bin/dolphin-emu"',
                'iso_path = "iso/melee.iso"',
                'replay_dir = "replays"',
                "[actor]",
                "rollout_length = 8",
                "context_frames = 16",
                "[learner]",
                "kl_teacher_weight = 0.0",
            ]
        ),
        encoding="utf-8",
    )
    local = load_config(toml)
    assert local.env.dolphin.dolphin_path == str(
        tmp_path / "dolphin" / "squashfs-root" / "usr" / "bin" / "dolphin-emu"
    )
    assert local.env.dolphin.iso_path == str(tmp_path / "iso" / "melee.iso")
    assert local.env.dolphin.replay_dir == str(tmp_path / "replays")
    assert local.env.dummy == DummyEnvConfig(players=("policy", "cpu"))
    assert local.env.dolphin.players == ("policy", "cpu") and local.env.num_envs == 1


# ---------------------------------------------------------------------------
# P6a: the 8-Dolphin run files, two-port microbatching, ``other`` / ``train`` rules
# ---------------------------------------------------------------------------


def test_p6a_run_files_load_resolve_and_round_trip(tmp_path: Path) -> None:
    cpu8 = load_config(CONFIG_DIR / "dolphin_cpu8.toml")
    self8 = load_config(CONFIG_DIR / "dolphin_self8.toml")
    other8 = load_config(CONFIG_DIR / "dolphin_other8.toml")
    for config in (cpu8, self8, other8):
        assert config.env.type == "dolphin" and config.env.num_envs == 8
        assert config.env.dolphin.step_threads == 0 and config.env.dolphin.boot == "two_phase"
        assert config.actor.rollout_length == 128 and config.actor.context_frames == 128
        assert config.actor.reward.ledge_grab_penalty == pytest.approx(0.02)
        assert config.actor.reward.stalling_penalty == pytest.approx(0.1)
        assert config.actor.reward.stalling_threshold == pytest.approx(50.0)
        assert config.actor.reward.approaching_factor == pytest.approx(1e-3)
        assert config.runtime.max_runtime_s is not None and config.runtime.max_runtime_s <= 600.0
        assert config.runtime.num_steps >= 100 and config.runtime.save_interval_steps == 5
        assert config.runtime.threads == 4 and config.learner.ppo.beta == pytest.approx(0.3)
        assert "p6a" in config.logging.tags and "dolphin-env" in config.logging.tags
        assert config.logging.mode == "disabled"
        mapping = config_to_mapping(config)
        assert config_from_mapping(mapping) == config
        model = resolve_model_config(config)
        assert model.context_length >= 256 and model.profile == "3m"
    assert (
        cpu8.opponent.type == "cpu" and cpu8.opponent.cpu_level == 9 and cpu8.opponent.trained_ports == (0,)
    )
    assert cpu8.env.players == ("policy", "cpu")
    assert self8.opponent.type == "self" and self8.opponent.train and self8.opponent.trained_ports == (0, 1)
    assert self8.env.players == ("policy", "policy")
    assert other8.opponent.type == "other" and not other8.opponent.train
    assert other8.opponent.checkpoint == "/runs/p6a/cpu8/latest.pt"  # absolute: left alone
    assert other8.opponent.trained_ports == (0,) and other8.env.players == ("policy", "policy")
    # A relative ``opponent.checkpoint`` resolves against the TOML's directory.
    toml = tmp_path / "other.toml"
    toml.write_text(
        "\n".join(
            [
                f'format = "{CONFIG_FORMAT}"',
                "[policy]",
                'profile = "3m"',
                "[policy.model_overrides]",
                *[
                    f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in TINY_OVERRIDES.items()
                ],
                "[env.dummy]",
                "num_envs = 2",
                "[actor]",
                "rollout_length = 8",
                "context_frames = 16",
                "[opponent]",
                'type = "other"',
                'checkpoint = "opponents/latest.pt"',
                "[learner]",
                "kl_teacher_weight = 0.0",
            ]
        ),
        encoding="utf-8",
    )
    relative = load_config(toml)
    assert relative.opponent.checkpoint == str(tmp_path / "opponents" / "latest.pt")
    assert relative.env.dummy.players == ("policy", "policy")  # derived from the opponent type


def test_two_port_microbatch_and_opponent_rules() -> None:
    # self + train: the learner batch is num_envs x 2 rows; microbatch_envs must divide that.
    two_port = _finalized(opponent={"type": "self", "train": True}, learner={"microbatch_envs": 2})
    assert two_port.opponent.trained_ports == (0, 1) and two_port.learner.microbatch_envs == 2
    assert two_port.env.dummy.players == ("policy", "policy")
    assert (
        _finalized(
            opponent={"type": "self", "train": True}, learner={"microbatch_envs": 6}
        ).learner.microbatch_envs
        == 6
    )
    assert (
        _finalized(
            opponent={"type": "self", "train": True}, learner={"microbatch_envs": 3}
        ).learner.microbatch_envs
        == 3
    )
    with pytest.raises(ValueError, match=r"microbatch_envs 4 must divide num_envs 3 x trained ports 2 = 6"):
        _finalized(opponent={"type": "self", "train": True}, learner={"microbatch_envs": 4})
    with pytest.raises(ValueError, match=r"microbatch_envs 2 must divide num_envs 3"):
        _finalized(opponent={"type": "self"}, learner={"microbatch_envs": 2})
    # ``train`` is a self-play switch; ``other`` needs a checkpoint; both go through the dataclass.
    with pytest.raises(ValueError, match="train"):
        config_from_mapping(_raw(opponent={"type": "cpu", "train": True}))
    with pytest.raises(ValueError, match="checkpoint"):
        config_from_mapping(_raw(opponent={"type": "other"}))
    other = _finalized(opponent={"type": "other", "checkpoint": "weights.pt"})
    assert other.opponent.trained_ports == (0,) and other.env.dummy.players == ("policy", "policy")
    with pytest.raises(ValueError, match=r"players .* opponent.type"):
        _finalized(
            opponent={"type": "other", "checkpoint": "w.pt"}, env={"dummy": {"players": ["policy", "cpu"]}}
        )
    with pytest.raises(ValueError, match=r"\[opponent\] unknown key 'ports'"):
        config_from_mapping(_raw(opponent={"ports": 2}))


def test_eval_table_parses_validates_and_resolves_paths(tmp_path: Path) -> None:
    assert _finalized().eval == EvalConfig() and not _finalized().eval.enabled
    config = _finalized(
        eval={
            "enabled": True,
            "interval_seconds": 120.0,
            "interval_steps": 50,
            "frames": 48,
            "num_envs": 2,
            "rollout_length": 8,
            "opponents": ["cpu9", "best"],
            "metric": "ko_diff_per_minute",
        }
    )
    assert config.eval.enabled and config.eval.interval_seconds == 120.0 and config.eval.interval_steps == 50
    assert config.eval.frames == 48 and config.eval.num_envs == 2 and config.eval.rollout_length == 8
    assert config.eval.opponents == ("cpu9", "best") and config.eval.metric == "ko_diff_per_minute"
    mapping = config_to_mapping(config)
    assert "eval" in mapping and config_from_mapping(mapping) == config
    assert isinstance(mapping["eval"]["opponents"], tuple)
    with pytest.raises(ValueError, match=r"\[eval\] unknown key 'every'"):
        config_from_mapping(_raw(eval={"every": 1}))
    with pytest.raises(ValueError, match="opponent"):
        config_from_mapping(_raw(eval={"opponents": ["cpu10"]}))
    # The eval rollout must fit the model's context (ring mode, no history window): 32 > 24 here.
    with pytest.raises(ValueError, match=r"eval.rollout_length 32 .* context_length 24"):
        _finalized(eval={"enabled": True, "frames": 64, "num_envs": 2, "rollout_length": 32})
    # Relative ``other:<path>`` opponents resolve against the TOML's directory; absolute ones are left alone.
    toml = tmp_path / "eval.toml"
    toml.write_text(
        "\n".join(
            [
                f'format = "{CONFIG_FORMAT}"',
                "[policy]",
                'profile = "3m"',
                "[policy.model_overrides]",
                *[
                    f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
                    for key, value in TINY_OVERRIDES.items()
                ],
                "[env.dummy]",
                "num_envs = 2",
                "[actor]",
                "rollout_length = 8",
                "context_frames = 16",
                "[learner]",
                "kl_teacher_weight = 0.0",
                "[eval]",
                "enabled = true",
                "frames = 32",
                "num_envs = 2",
                "rollout_length = 8",
                'opponents = ["cpu9", "other:opponents/best.pt"]',
            ]
        ),
        encoding="utf-8",
    )
    resolved = load_config(toml)
    assert resolved.eval.opponents == ("cpu9", f"other:{tmp_path / 'opponents' / 'best.pt'}")
    assert [item.kind for item in resolved.eval.parsed_opponents] == ["cpu", "other"]
    absolute = _finalized(eval={"opponents": ["other:/abs/other.pt"], "metric_opponent": "other"})
    assert absolute.eval.opponents == ("other:/abs/other.pt",)  # absolute paths are left alone
    with pytest.raises(ValueError, match="distinct"):  # one ``other`` per config (names must differ)
        EvalConfig(opponents=("other:/a.pt", "other:/b.pt"), metric_opponent="other")
    # The shipped P6b eval smoke file.
    smoke = load_config(CONFIG_DIR / "dolphin_cpu8_eval.toml")
    assert smoke.eval.enabled and smoke.eval.at_start and smoke.eval.opponents == ("cpu9", "best")
    assert smoke.eval.interval_seconds == 150.0 and smoke.eval.frames == 7200 and smoke.eval.num_envs == 8
    assert smoke.eval.metric == "damage_dealt_per_minute" and smoke.env.type == "dolphin"
    assert smoke.runtime.max_runtime_s is not None and smoke.runtime.max_runtime_s <= 400.0
    assert "eval" in smoke.logging.tags
    # The P6b one-day run file: from scratch, no teacher, hourly evals, 23.5 h per launch, the 1e-2 guard.
    day = load_config(CONFIG_DIR / "dolphin_cpu8_day.toml")
    assert (
        day.policy.init == "random" and day.teacher.source == "none" and day.learner.kl_teacher_weight == 0.0
    )
    assert day.runtime.max_runtime_s == 84600.0 and day.runtime.save_interval_steps == 20
    assert day.eval.enabled and day.eval.interval_seconds == 3600.0 and day.eval.frames == 57600
    assert day.eval.opponents == ("cpu9", "best") and day.learner.ppo.max_mean_actor_kl == pytest.approx(1e-2)
    assert day.env.num_envs == 8 and day.opponent.type == "cpu" and day.opponent.cpu_level == 9
    assert resolve_model_config(day).profile == "3m"
    # The 20m variant (user decision 22 Aug 2026: Ali's profile): the same run on the L4 function (--gpu).
    day_20m = load_config(CONFIG_DIR / "dolphin_20m_day.toml")
    assert resolve_model_config(day_20m).profile == "20m" and day_20m.runtime.device == "cuda"
    for field in ("init", "seed", "compute_dtype", "cache_dtype"):
        assert getattr(day_20m.policy, field) == getattr(day.policy, field)
    assert day_20m.teacher == day.teacher and day_20m.actor == day.actor and day_20m.opponent == day.opponent
    assert day_20m.env == day.env and day_20m.runtime == replace(day.runtime, device="cuda")
    # Relaunch 1 (22 Aug 2026): lr 3e-5 (1e-4 gave actor KL 0.02-0.04 per update, every update reverted) and a
    # shorter eval (the 20m eval worker runs at ~130 fps on the L4).
    assert day_20m.learner == replace(day.learner, learning_rate=3e-5)
    assert day_20m.eval == replace(day.eval, frames=28800) and day_20m.eval.frames == 28800
    assert "20m" in day_20m.logging.tags and "3m" not in day_20m.logging.tags
    # The wide variant (user decision 22 Aug 2026, session 11): 64 Dolphins on L4 + 32 cores / 64 GiB, ring
    # mode (no re-prime), a 32-Dolphin eval; everything else as the 20m day file (it resumes its checkpoint).
    wide = load_config(CONFIG_DIR / "dolphin_20m_wide_day.toml")
    assert wide.env.num_envs == 64 and wide.env.dolphin.num_envs == 64
    assert wide.env.dolphin.boot_timeout_s == 300.0
    assert wide.actor.context_mode == "ring" and wide.actor == replace(day_20m.actor, context_mode="ring")
    assert wide.eval == replace(day_20m.eval, num_envs=32, frames=57600)
    assert wide.env.dolphin == replace(day_20m.env.dolphin, num_envs=64, boot_timeout_s=300.0)
    assert wide.runtime == day_20m.runtime and wide.policy == day_20m.policy
    assert wide.teacher == day_20m.teacher
    assert wide.learner == day_20m.learner and wide.opponent == day_20m.opponent
    assert resolve_model_config(wide).profile == "20m" and {"wide", "ring"} <= set(wide.logging.tags)


def test_self_play_day_run_file_is_the_from_scratch_experiment() -> None:
    """``dolphin_20m_self_wide_day``: two-port self-play from scratch (session 13, 23 Aug 2026).

    The 20m-vs-CPU-9 run went flat over 15.7 M frames with the policy still at 97.7 % of maximum
    entropy: per-update actor KL was 3.5e-5 against a 1e-2 guard, so the learning rate, not the trust
    region, was the constraint.  This run answers that with a bigger step (lr 3e-4 behind both burn-in
    ladders, ``beta`` back to slippi-ai's 0) and a symmetric opponent -- the student plays both ports,
    so the reward has mean 0 and both signs instead of CPU 9's uniformly negative -4.8/min.
    """

    wide = load_config(CONFIG_DIR / "dolphin_20m_wide_day.toml")
    run = load_config(CONFIG_DIR / "dolphin_20m_self_wide_day.toml")
    # Two-port self-play: no opponent object, both ports are the student, so the learner batch is 2 x E.
    assert run.opponent.type == "self" and run.opponent.train
    assert run.opponent.trained_ports == (0, 1)
    assert run.env.players == ("policy", "policy") and run.env.num_envs == 64
    # 128 rows would not fit the L4 in one learner pass (the 64-row wide run measured 14.0 GB).
    assert run.learner.microbatch_envs == 64
    # Step size: the 1e-2 guard, not the lr, must bind (KL/update scales about as lr^2: 3.5e-5 -> 3.5e-3).
    assert run.learner.learning_rate == pytest.approx(3e-4)
    assert run.learner.value_burnin_steps == 50 and run.learner.optimizer_burnin_steps == 50
    assert run.learner.ppo.beta == 0.0
    assert run.learner.ppo.epsilon == pytest.approx(0.05)  # at 1e-2 the second epoch would be all clipped
    assert run.learner.ppo.max_mean_actor_kl == pytest.approx(1e-2)
    assert run.learner.returns == wide.learner.returns and run.learner.kl_teacher_weight == 0.0
    # Monitoring: in self-play every zero-sum training metric cancels (reward/per_frame == 0, dealt ==
    # taken, kos == deaths), so progress is read from the evaluator against fixed references -- the CPU
    # ladder at both ends plus a frozen random-init anchor that cannot cycle the way ``best`` can.
    assert run.eval.enabled and run.eval.interval_seconds == 3600.0 and run.eval.at_start
    assert run.eval.opponents == ("cpu1", "cpu9", "other:/runs/p6b/20m_day/step_00000000.pt")
    assert [item.kind for item in run.eval.parsed_opponents] == ["cpu", "cpu", "other"]
    assert run.eval.metric == "reward_per_frame" and run.eval.metric_opponent == "cpu1"
    assert run.eval.num_envs == 32 and run.eval.frames == 57600
    # Everything else is the wide run: 20m from scratch, ring mode, T = C = 128, Fox vs Fox on FD.
    assert run.policy == wide.policy and run.actor == wide.actor and run.teacher == wide.teacher
    assert run.env.dolphin == replace(wide.env.dolphin, players=("policy", "policy"))
    assert resolve_model_config(run).profile == "20m"
    assert run.runtime.max_runtime_s == 84600.0 and run.runtime.device == "cuda"
    assert {"self-play", "two-port", "20m", "wide", "ring"} <= set(run.logging.tags)
    # The 20-minute shake-out: the same run with the burn-in ladders and the evaluation cut down, so a
    # launch reaches the real PPO stage -- and the 128-row GPU-memory peak -- in minutes, not 83 of them.
    smoke = load_config(CONFIG_DIR / "dolphin_20m_self_wide_smoke.toml")
    assert smoke.runtime.max_runtime_s == 1200.0
    assert smoke.learner == replace(run.learner, value_burnin_steps=3, optimizer_burnin_steps=3)
    assert smoke.eval == replace(run.eval, frames=8192)
    assert smoke.env == run.env and smoke.policy == run.policy and smoke.actor == run.actor
    assert smoke.opponent == run.opponent and smoke.teacher == run.teacher
    assert "smoke" in smoke.logging.tags and "day" not in smoke.logging.tags

    # Take two (user decision 23 Aug 2026): the 5m profile.  Same experiment, a quarter of the parameters,
    # so it cannot resume the 20m checkpoint and its step size has to be measured again -- the 20m file's
    # lr 3e-4 reverted 520 of 520 updates, and its predecessor's 3e-5 never moved.
    five = load_config(CONFIG_DIR / "dolphin_5m_self_wide_day.toml")
    assert resolve_model_config(five).profile == "5m"
    assert five.opponent == run.opponent and five.env.num_envs == 64
    assert five.learner.microbatch_envs is None  # 5m at 128 rows fits the L4 in one pass
    assert five.learner.learning_rate == pytest.approx(2e-4)  # seeded guess; the smoke run measures it
    assert five.learner.ppo.max_mean_actor_kl == pytest.approx(2e-2)  # a backstop, not the throttle
    assert five.learner.ppo.epsilon == pytest.approx(0.1)  # 0.05 clipped 85 % of the second epoch
    assert five.learner.value_burnin_steps == 50 and five.learner.optimizer_burnin_steps == 50
    # Two-hourly evaluations (user decision 23 Aug 2026): three opponents cost ~27 % of wall clock
    # hourly, and the CPU ladder's measured noise floor makes that sampling rate poor value.
    assert five.eval == replace(run.eval, interval_seconds=7200.0)
    assert five.actor == run.actor and five.teacher == run.teacher
    assert "5m" in five.logging.tags and "20m" not in five.logging.tags
    # A snapshot every ~50 min: a past-self ladder, and any two diff offline into a drift number
    # (the same quantity weights/drift_total now logs every step).
    assert five.runtime.snapshot_interval_steps == 100
    five_smoke = load_config(CONFIG_DIR / "dolphin_5m_self_wide_smoke.toml")
    assert five_smoke.runtime.max_runtime_s == 1200.0
    assert five_smoke.learner == replace(five.learner, value_burnin_steps=3, optimizer_burnin_steps=3)
    assert five_smoke.eval == replace(five.eval, frames=8192)


def test_bc_init_run_files_start_from_ali_s_5m_checkpoint() -> None:
    """``dolphin_5m_bc_self_wide_{day,smoke}``: the 5m self-play run re-based on Ali's best 5m BC
    checkpoint with a frozen BC teacher (BC-INIT-PLAN.md, approved 24 Aug 2026).

    The from-scratch run cycled: entropy 10.94 -> 3.5 and CPU evaluations swinging far outside their
    noise floors while the zero-noise anchor evaluation stayed flat for 800 steps.  A competent init
    plus ``KL(pi || teacher)`` at 0.1 starts from a real policy and pins the distribution;
    ``entropy_weight`` stays 0 -- one change at a time.
    """

    scratch = load_config(CONFIG_DIR / "dolphin_5m_self_wide_day.toml")
    day = load_config(CONFIG_DIR / "dolphin_5m_bc_self_wide_day.toml")
    bc_path = "/runs/bc/5m-muon-aux-lr2x.pt"  # the stable Volume path: the teacher record pins its sha256
    assert day.policy == replace(scratch.policy, init="checkpoint", checkpoint=bc_path, fast_step=True)
    assert day.policy.fast_step is True  # P7b (25 Aug 2026): the vectorised ring step, parity-tested
    assert day.policy.profile == "5m" and resolve_model_config(day).profile == "5m"
    assert day.policy.compute_dtype == "float32" and day.policy.cache_dtype == "float32"
    # The teacher is a frozen clone of the just-loaded BC weights: no second file read.
    assert day.teacher == TeacherConfig(source="policy")
    assert day.learner.kl_teacher_weight == pytest.approx(0.1)
    assert day.learner.reverse_kl_teacher_weight == 0.0 and day.learner.entropy_weight == 0.0
    # Post-training step sizes: the smoke measures the actor KL at 3e-5 and the day rate is set from that
    # measurement, capped at 1e-4 (BC-INIT-PLAN.md section 6); the guard is a real backstop with headroom.
    assert 0.0 < day.learner.learning_rate <= 1e-4
    assert day.learner.value_burnin_steps == 100 and day.learner.optimizer_burnin_steps == 50
    assert day.learner.ppo.epsilon == pytest.approx(1e-2) and day.learner.ppo.beta == 0.0
    assert day.learner.ppo.max_mean_actor_kl == pytest.approx(1e-3)
    assert day.learner.ppo.num_epochs == 2 and day.learner.ppo.num_batches == 1
    assert day.learner.microbatch_envs is None and day.learner.returns == scratch.learner.returns
    assert day.learner.value == scratch.learner.value
    # The evaluation anchor is the BC checkpoint itself -- zero-noise (policy-vs-policy consumes no Melee
    # RNG) and literally "am I better than what I started as" -- and it selects best.pt.
    assert day.eval.opponents == ("cpu1", "cpu9", f"other:{bc_path}")
    assert [item.kind for item in day.eval.parsed_opponents] == ["cpu", "cpu", "other"]
    assert day.eval.parsed_opponents[2].path == day.policy.checkpoint
    assert day.eval.metric == "reward_per_frame" and day.eval.metric_opponent == "other"
    assert day.eval == replace(scratch.eval, opponents=day.eval.opponents, metric_opponent="other")
    # Everything else is the from-scratch run: 64 Dolphins x 2 trained ports, ring mode, T = C = 128 --
    # except the P7a worker processes (25 Aug 2026): 16 spawned workers of 4 Dolphins each.
    assert day.env.dolphin.worker_processes == 16 and day.env.dolphin.mp_start_method == "spawn"
    assert day.env == replace(scratch.env, dolphin=replace(scratch.env.dolphin, worker_processes=16))
    assert day.actor == scratch.actor and day.opponent == scratch.opponent
    assert day.runtime == scratch.runtime and day.delay == scratch.delay
    expected_tags = {"bc-init", "teacher-kl", "self-play", "two-port", "5m", "wide", "ring", "day"}
    assert expected_tags <= set(day.logging.tags) and "from-scratch" not in day.logging.tags
    # The 20-minute shake-out (not optional): it measures the actor KL at the seeded 3e-5 in a regime
    # nothing here has been measured in -- the policy is peaked instead of near-uniform.
    smoke = load_config(CONFIG_DIR / "dolphin_5m_bc_self_wide_smoke.toml")
    assert smoke.runtime.max_runtime_s == 1200.0
    assert smoke.learner == replace(
        day.learner, learning_rate=3e-5, value_burnin_steps=3, optimizer_burnin_steps=3
    )
    assert smoke.eval == replace(day.eval, frames=8192)
    assert smoke.policy == day.policy and smoke.teacher == day.teacher and smoke.env == day.env
    assert smoke.actor == day.actor and smoke.opponent == day.opponent
    assert "smoke" in smoke.logging.tags and "day" not in smoke.logging.tags


def test_pipeline_knobs_slack_and_opponent_rules() -> None:
    """P8: batch_steps / chunk_frames round-trip; the slack and opponent restrictions fail at load."""

    from melee_rl.config import pipeline_enabled

    def build(
        *,
        batch_steps: int = 1,
        chunk_frames: int = 1,
        delay: int = 5,
        env_type: str = "dummy",
        opponent: dict[str, object] | None = None,
    ) -> RLConfig:
        tables: dict[str, object] = {
            "format": CONFIG_FORMAT,
            "actor": {
                "rollout_length": 8,
                "context_frames": 8,
                "delay_frames": delay,
                "batch_steps": batch_steps,
            },
            "env": {"type": env_type, "dolphin": {"chunk_frames": chunk_frames, "check_iso": False}},
            "teacher": {"source": "none"},
            "learner": {"kl_teacher_weight": 0.0},
            "delay": {"allow_mismatch": True},
        }
        if opponent is not None:
            tables["opponent"] = opponent
        return config_from_mapping(tables)

    plain = build(batch_steps=4, chunk_frames=3)
    assert plain.actor.batch_steps == 4 and plain.env.dolphin.chunk_frames == 3
    assert config_to_mapping(plain)["actor"]["batch_steps"] == 4
    assert config_to_mapping(plain)["env"]["dolphin"]["chunk_frames"] == 3
    # On the dummy env the (inactive) dolphin chunk does not switch the pipeline on; batch_steps does.
    assert pipeline_enabled(build(chunk_frames=4)) is False
    assert pipeline_enabled(build(batch_steps=2)) is True
    assert pipeline_enabled(build(chunk_frames=4, env_type="dolphin")) is True
    assert pipeline_enabled(build()) is False

    model = resolve_model_config(build())
    # b = 4, k = 4 at D = 5 is the canonical illegal point: (4 - 1) + (4 - 1) = 6 > 5.
    with pytest.raises(ValueError, match=r"\(4 - 1\) \+ \(4 - 1\) = 6 > 5"):
        finalize_config(build(batch_steps=4, chunk_frames=4, env_type="dolphin"), model)
    # b = 4, k = 3 and b = 2, k = 4 use the slack legally (the P8 bench points).
    finalize_config(build(batch_steps=4, chunk_frames=3, env_type="dolphin"), model)
    finalize_config(build(batch_steps=2, chunk_frames=4, env_type="dolphin"), model)
    # b = 2 at D = 0 violates the slack even on the dummy env (k = 1).
    with pytest.raises(ValueError, match="delay pipeline"):
        finalize_config(build(batch_steps=2, delay=0), model)
    # b must divide T at ActorConfig construction time.
    with pytest.raises(ValueError, match="batch_steps 3 must divide"):
        build(batch_steps=3)
    # Opponent restriction: cpu and self + train pass, self without train and other are errors.
    finalize_config(build(batch_steps=2, opponent={"type": "cpu"}), model)
    two_port = build(batch_steps=2, opponent={"type": "self", "train": True})
    finalize_config(two_port, model)
    with pytest.raises(ValueError, match="pipelined mode"):
        finalize_config(build(batch_steps=2, opponent={"type": "self"}), model)
    with pytest.raises(ValueError, match="pipelined mode"):
        finalize_config(build(batch_steps=2, opponent={"type": "other", "checkpoint": "x.pt"}), model)
    # With both knobs at 1 the opponent rules do not apply (today's loop).
    finalize_config(build(opponent={"type": "self"}), model)


def test_p8_delay_run_files_pipeline_from_scratch_at_d5() -> None:
    """The P8 run files: the canary and the two (b, k) bench points at D = 5 (P8-PLAN.md 3.8).

    All three are from-scratch (user decision 25 Aug 2026): a random init has no trained-in delay,
    so D = 5 training is in-distribution by construction; ``allow_mismatch`` only acknowledges the
    model's declared ``action_offset_frames = 1``.  The benches keep the wide P7 box (64 Dolphins x
    2 ports, 16 workers, fast_step, ring) and the from-scratch learner of the 5m self-play smoke.
    """

    from melee_rl.config import pipeline_enabled

    canary = load_config(CONFIG_DIR / "dolphin_cpu8_delay.toml")
    assert pipeline_enabled(canary)
    assert canary.actor.delay_frames == 5 and canary.actor.batch_steps == 2
    assert canary.env.dolphin.chunk_frames == 2 and canary.env.dolphin.worker_processes == 2
    assert canary.env.dolphin.num_envs == 8 and canary.opponent.type == "cpu"
    assert canary.delay.allow_mismatch is True and canary.runtime.max_runtime_s == 300.0
    assert canary.eval.enabled is False

    scratch = load_config(CONFIG_DIR / "dolphin_5m_self_wide_smoke.toml")
    for name, batch_steps, chunk in (
        ("dolphin_5m_delay_b4k3_smoke", 4, 3),
        ("dolphin_5m_delay_b2k4_smoke", 2, 4),
    ):
        bench = load_config(CONFIG_DIR / f"{name}.toml")
        assert pipeline_enabled(bench)
        assert bench.actor.batch_steps == batch_steps and bench.env.dolphin.chunk_frames == chunk
        assert bench.actor.delay_frames == 5 and (batch_steps - 1) + (chunk - 1) <= 5
        assert bench.actor.rollout_length % batch_steps == 0
        assert bench.policy.init == "random" and bench.policy.profile == "5m"
        assert bench.policy.fast_step is True and bench.teacher.source == "none"
        assert bench.env.dolphin.num_envs == 64 and bench.env.dolphin.worker_processes == 16
        assert bench.opponent.type == "self" and bench.opponent.train is True
        assert bench.actor.context_mode == "ring" and bench.delay.allow_mismatch is True
        assert bench.eval.enabled is False and bench.runtime.max_runtime_s == 900.0
        # The from-scratch learner settings of the measured 5m self-play smoke (session 13).
        assert bench.learner.learning_rate == scratch.learner.learning_rate == pytest.approx(2e-4)
        assert bench.learner.ppo.epsilon == scratch.learner.ppo.epsilon == pytest.approx(0.1)
        assert bench.learner.ppo.max_mean_actor_kl == pytest.approx(2e-2)
        assert bench.learner.kl_teacher_weight == 0.0

    # The day file: the same from-scratch delayed regime with the batch regime matched to slippi-ai's
    # production recipes (num_batches 16, their rl_example.sh:32 / launch_rl.py:49) and 96 Dolphins.
    day = load_config(CONFIG_DIR / "dolphin_5m_delay_day.toml")
    assert pipeline_enabled(day)
    assert day.actor.delay_frames == 5 and day.actor.batch_steps > 1
    assert (day.actor.batch_steps - 1) + (day.env.dolphin.chunk_frames - 1) <= 5
    assert day.actor.rollout_length % day.actor.batch_steps == 0
    assert day.policy.init == "random" and day.policy.fast_step is True and day.teacher.source == "none"
    assert day.env.dolphin.num_envs == 96 and day.env.dolphin.worker_processes == 16
    assert day.env.dolphin.num_envs % day.env.dolphin.worker_processes == 0
    assert day.learner.ppo.num_batches == 16
    # Per update: 16 x 96 x 128 = 196 608 env frames = 393 216 training rows (both ports train),
    # matching the reference recipes' 256-369k rows per update (TF 368 640, JAX 256 000).
    frames_per_update = 16 * 96 * 128
    rows_per_update = 2 * frames_per_update
    assert frames_per_update == 196_608 and rows_per_update >= 256_000
    assert day.opponent.type == "self" and day.opponent.train is True
    assert day.runtime.max_runtime_s == 84600.0 and day.runtime.snapshot_interval_steps == 8
    assert day.runtime.save_interval_steps == 2
    assert day.learner.value_burnin_steps == 4 and day.learner.optimizer_burnin_steps == 3
    assert day.eval.enabled and day.eval.metric_opponent == "cpu1"
    assert day.eval.opponents == ("cpu1", "cpu9", "other:/runs/p6b/20m_day/step_00000000.pt")
    assert day.delay.allow_mismatch is True

    # The matched-batch smoke: the day file's shape with fast burn-ins, eval off, a 30-min cap.
    smoke = load_config(CONFIG_DIR / "dolphin_5m_delay_nb16_smoke.toml")
    assert smoke.learner.ppo.num_batches == 16 and smoke.env.dolphin.num_envs == 96
    assert smoke.actor.batch_steps == day.actor.batch_steps
    assert smoke.env.dolphin.chunk_frames == day.env.dolphin.chunk_frames
    assert smoke.learner.value_burnin_steps == 1 and smoke.learner.optimizer_burnin_steps == 1
    assert smoke.eval.enabled is False and smoke.runtime.max_runtime_s == 1800.0
    assert smoke.learner.learning_rate == day.learner.learning_rate


def test_20m_bc_run_files_start_from_ali_s_c3_checkpoint() -> None:
    """The 20m BC-init pair (user decisions 25 Aug 2026): Ali's best 20m file (muon-scaling c3,
    val NLL 0.8409 at 137.4M frames), teacher KL 0.1, D = 0, lr 4e-4, the matched batch regime."""

    bc_path = "/runs/bc/20m-muon-scaling-c3.pt"  # the stable Volume path; the teacher record pins its sha256
    day = load_config(CONFIG_DIR / "dolphin_20m_bc_self_wide_day.toml")
    assert day.policy.profile == "20m" and day.policy.init == "checkpoint"
    assert day.policy.checkpoint == bc_path and day.policy.fast_step is True
    assert day.teacher == TeacherConfig(source="policy")
    assert day.learner.kl_teacher_weight == pytest.approx(0.1)
    assert day.learner.learning_rate == pytest.approx(4e-5)  # ladder: 4e-4 and 1e-4 reverted, 4e-5 clean
    # Above the first-step value transient: a guard under it freezes the run (launch 1 at lr 1e-4
    # reverted 58/58 steps, and a revert restores Adam's moments, so step 1 repeats forever).
    assert day.learner.ppo.max_mean_actor_kl == pytest.approx(5e-3)
    assert day.actor.delay_frames == 0 and day.actor.batch_steps == 1
    assert day.env.dolphin.chunk_frames == 1 and day.delay.allow_mismatch is False
    from melee_rl.config import pipeline_enabled

    assert not pipeline_enabled(day)  # D = 0: the P8 pipeline stays dormant by construction
    assert day.learner.ppo.num_batches == 16 and day.env.dolphin.num_envs == 96
    assert day.learner.microbatch_envs == 48  # 192-row trajectories in 4 chunks on the L4
    assert (2 * day.env.dolphin.num_envs) % 48 == 0
    assert day.learner.value_burnin_steps == 6 and day.learner.optimizer_burnin_steps == 3
    assert day.eval.enabled and day.eval.metric_opponent == "other"
    assert day.eval.opponents == ("cpu1", "cpu9", f"other:{bc_path}")
    assert day.runtime.max_runtime_s == 84600.0

    smoke = load_config(CONFIG_DIR / "dolphin_20m_bc_self_wide_smoke.toml")
    assert smoke.policy.checkpoint == bc_path and smoke.policy.profile == "20m"
    assert smoke.learner.learning_rate == pytest.approx(4e-5)  # the take-two measurement point
    assert smoke.learner.ppo.num_batches == 16 and smoke.env.dolphin.num_envs == 96
    assert smoke.learner.ppo.max_mean_actor_kl == pytest.approx(5e-3)  # loose: it measures
    assert smoke.eval.enabled is False and smoke.runtime.max_runtime_s == 3600.0


def test_20m_bc_mix_run_files_train_the_cpu9_curriculum_from_best() -> None:
    """P10 (user decisions 26 Aug 2026): 50/50 rows self vs cpu9 from the 20m BC run's best.pt,
    teacher re-anchored to the original BC file, best.pt selected by the cpu9 eval."""

    bc_path = "/runs/bc/20m-muon-scaling-c3.pt"
    day = load_config(CONFIG_DIR / "dolphin_20m_bc_mix_day.toml")
    assert day.opponent.type == "mix" and day.opponent.mix == ("self*32", "cpu9*64")
    assert day.opponent.mix_workers == (8, 8)
    groups = day.opponent.groups
    assert [group.rows for group in groups] == [64, 64]  # the 50/50-by-rows decision
    assert [group.num_envs for group in groups] == [32, 64]
    assert day.policy.init == "checkpoint" and day.policy.checkpoint == "/runs/selfplay/20m_bc/best.pt"
    assert day.policy.init_value is True and day.policy.profile == "20m" and day.policy.fast_step
    assert day.teacher.source == "checkpoint" and day.teacher.path == bc_path
    assert day.learner.kl_teacher_weight == pytest.approx(0.1)
    assert day.learner.learning_rate == pytest.approx(4e-5)  # unchanged: the curriculum is the variable
    assert day.learner.ppo.max_mean_actor_kl == pytest.approx(5e-3)
    assert day.env.dolphin.num_envs == 96 and day.env.dolphin.worker_processes == 16
    assert day.env.dolphin.players == ("policy", "policy")  # the neutral template; groups derive theirs
    assert day.learner.microbatch_envs == 64  # divides both groups' 64 rows
    assert day.learner.ppo.num_batches == 16 and day.learner.value_burnin_steps == 6
    assert day.eval.metric == "reward_per_frame" and day.eval.metric_opponent == "cpu9"
    assert day.eval.opponents == ("cpu1", "cpu9", f"other:{bc_path}")
    assert day.runtime.max_runtime_s == 84600.0 and day.runtime.snapshot_interval_steps == 8
    assert day.actor.delay_frames == 0 and day.actor.batch_steps == 1

    smoke = load_config(CONFIG_DIR / "dolphin_20m_bc_mix_smoke.toml")
    assert smoke.opponent.mix == day.opponent.mix and smoke.opponent.mix_workers == day.opponent.mix_workers
    assert smoke.policy.checkpoint == day.policy.checkpoint and smoke.policy.init_value is True
    assert smoke.teacher == day.teacher
    assert smoke.learner.learning_rate == day.learner.learning_rate
    assert smoke.learner.value_burnin_steps == 1 and smoke.learner.optimizer_burnin_steps == 1
    assert smoke.eval.enabled and smoke.eval.at_start is False and smoke.eval.at_end is True
    assert smoke.eval.opponents == ("cpu9",) and smoke.eval.metric_opponent == "cpu9"
    assert smoke.eval.frames == 28800 and smoke.runtime.max_runtime_s == 2400.0


def test_video_run_file_loads_and_resolves() -> None:
    """P9: ``configs/rl/video_20m_bc.toml`` -- the on-demand recorder for the 20m BC run."""

    config = load_config(CONFIG_DIR / "video_20m_bc.toml")
    video = config.video
    assert video.matchups == ("policy:cpu1", "policy:cpu9", "policy:bc", "bc:cpu1", "bc:cpu9")
    assert video.clips == 2 and video.seconds == 120.0  # 10 clips of 2 minutes (user decision)
    assert video.bc == "/runs/bc/20m-muon-scaling-c3.pt"  # the 20m BC run's own start, sha-keyed
    assert video.output_dir == "/runs/videos" and video.baseline_subdir == "baseline"
    assert video.rollout_length == 128 and video.temperature == 1.0 and video.seed == 0
    assert not video.wandb and video.wandb_seconds == 30.0  # the uploader stays off (26 Aug 2026)
    assert video.needs_bc and len(expand_matches(video)) == 5
    assert sum(spec.clips for spec in expand_matches(video)) == 10
    render = video.render
    assert render.backend == "OGL" and not render.platform and render.xvfb  # wx build under Xvfb
    assert render.internal_resolution == 1 and render.bitrate_kbps == 3000  # 1x native 640x480
    assert render.caption and render.workers == 6 and render.dump_format == "avi"
    assert render.crf == 26  # ~25 MB per 2-minute clip: a ten-clip job stays shareable
    # P9b: the render is ended by the --cout markers, and 1x native means EFBScale = SCALE_1X.
    assert render.cout and render.hide_seekbar and render.stop_grace_s == 10.0
    assert render.iso_path == "/iso/melee.iso"
    # The pinned Slippi PLAYBACK build, in the video image only: the only binary that accepted -i.
    assert render.dolphin_path.endswith("ishiiruka-playback-3.5.2/squashfs-root/usr/bin/dolphin-emu")
    assert render.speed_factor == 20.0 and render.min_timeout_s == 120.0
    # Stage 1 keeps every replay and shuts Dolphin down gracefully so the .slp gets its metadata block.
    assert config.env.type == "dolphin" and config.env.dolphin.save_replays
    assert config.env.dolphin.stop_timeout_s == 20.0 and config.env.dolphin.num_envs == 2
    assert config.env.dolphin.worker_processes == 0 and config.runtime.device == "cpu"
    assert config.actor.context_mode == "ring" and config.actor.context_frames == 0
    assert config.actor.delay_frames == 0 and config.runtime.num_steps == 0  # nothing trains
    assert config.teacher.source == "none" and config.logging.mode == "disabled"
    assert config_from_mapping(config_to_mapping(config)) == config


def test_20m_bc_mix_dmg005_changes_only_the_reward_weighting() -> None:
    """27 Aug 2026 (user decision): the parent mixed run came out flat vs cpu9 over 23.5 h -- damage
    flowed, KOs did not move -- so the pre-registered lever fires: ``damage_ratio`` 0.01 -> 0.005.
    A NEW run directory from the parent's final ``latest.pt`` (the reward changed, so it is not one
    curve), and that reward weight is the ONLY research variable that differs from the parent."""

    parent = load_config(CONFIG_DIR / "dolphin_20m_bc_mix_day.toml")
    run = load_config(CONFIG_DIR / "dolphin_20m_bc_mix_dmg005_day.toml")

    assert parent.actor.reward.damage_ratio == pytest.approx(0.01)
    assert run.actor.reward.damage_ratio == pytest.approx(0.005)  # KOs weigh 2x more, relatively
    assert run.policy.init == "checkpoint" and run.policy.init_value is True
    assert run.policy.checkpoint == "/runs/selfplay/20m_bc_mix/latest.pt"  # the parent's final state

    # Everything else is the parent's, field by field: the reward weight and the init file apart,
    # the two runs are the same experiment (lr 4e-5, guard 5e-3, teacher 0.1 on the ORIGINAL BC
    # anchor, 32-self + 64-cpu9, nb16 x 96 envs, the same evaluation).
    assert replace(run.actor.reward, damage_ratio=parent.actor.reward.damage_ratio) == parent.actor.reward
    assert replace(run.actor, reward=parent.actor.reward) == parent.actor
    assert replace(run.policy, checkpoint=parent.policy.checkpoint) == parent.policy
    assert run.teacher == parent.teacher and run.opponent == parent.opponent
    assert run.learner == parent.learner and run.env == parent.env
    assert run.eval == parent.eval and run.runtime == parent.runtime

    # The tags carry the change into W&B, and the run keeps its own name/dir there.
    assert "dmg005" in run.logging.tags and "ko-weighted" in run.logging.tags
    assert "from-mix-latest" in run.logging.tags and "from-20m-bc-best" not in run.logging.tags
    assert replace(run.logging, tags=parent.logging.tags) == parent.logging


def test_p11_probe_run_files_change_only_the_rules_and_the_seed_knobs() -> None:
    """28 Aug 2026: two copies of ``video_20m_bc.toml``, each moving exactly one thing.

    Arm A turns the endless no-stock mode off so a real four-stock match can be recorded; arms B-D
    set the two seed knobs over six Dolphins.  Everything else must stay byte-for-byte the recorder
    we already measured.
    """

    base = load_config(CONFIG_DIR / "video_20m_bc.toml")
    match = load_config(CONFIG_DIR / "video_match_probe.toml")
    seed = load_config(CONFIG_DIR / "video_seed_probe.toml")

    assert base.env.dolphin.infinite_time and not match.env.dolphin.infinite_time
    assert match.env.dolphin.rtc_values == () and match.env.dolphin.menu_idle_frames == ()
    assert replace(match.env.dolphin, infinite_time=True) == base.env.dolphin
    assert match.video.matchups == ("policy:cpu9",)
    assert match.video.clips == 2 and match.video.seconds == 360.0

    dolphin = seed.env.dolphin
    assert dolphin.infinite_time  # the seed question is independent of the rules
    assert dolphin.rtc_values == (1600000000, 1600000000, 1600003600, 1600086400, 1600000000, 1600000000)
    assert dolphin.menu_idle_frames == (0, 0, 0, 0, 45, 90)
    assert dolphin.num_envs == seed.video.clips == 6  # one Dolphin per probe cell
    assert [dolphin.rtc_value(index) for index in range(6)] == list(dolphin.rtc_values)
    assert [dolphin.menu_idle(index) for index in range(6)] == [0, 0, 0, 0, 45, 90]
    assert seed.video.matchups == ("policy:cpu9",) and seed.video.seconds == 10.0
    assert replace(dolphin, num_envs=2, rtc_values=(), menu_idle_frames=()) == base.env.dolphin


def test_only_the_probe_run_files_pin_melees_randomness() -> None:
    """User constraint (28 Aug 2026): training must keep a random start.

    The seed knobs and the stock rules may only be set by a recording probe -- every training and
    evaluation run file leaves Melee's own randomness alone.
    """

    real_matches = {
        "video_seed_probe.toml",
        "video_match_probe.toml",
        "video_match.toml",
        "video_mimic.toml",
        "video_match_stages.toml",
        "video_mimic_stages.toml",
        # P25: SmashBot's four files all record real four-stock matches (the two calibration arms
        # drive melee_rl.external_match, the two recorder files drive melee_rl.video).
        "smashbot_cpu9.toml",
        "smashbot_mimic.toml",
        "video_smashbot.toml",
        "video_smashbot_stages.toml",
        # P27: the same four shapes for the released slippi-ai agents.
        "slippi_ai_cpu9.toml",
        "slippi_ai_mimic.toml",
        "video_slippi_ai.toml",
        "video_slippi_ai_stages.toml",
        # P32: the same four shapes for the released vladfi1/phillip agents.
        "phillip_cpu9.toml",
        "phillip_mimic.toml",
        "video_phillip.toml",
        "video_phillip_stages.toml",
    }
    for path in sorted(CONFIG_DIR.glob("*.toml")):
        config = load_config(path)
        if config.env.type != "dolphin":
            continue
        dolphin = config.env.dolphin
        if path.name == "video_seed_probe.toml":
            assert dolphin.rtc_values and dolphin.menu_idle_frames
            continue
        assert dolphin.rtc_values == (), f"{path.name} pins the emulated console clock"
        assert dolphin.menu_idle_frames == (), f"{path.name} pins the frame START is pressed on"
        if path.name not in real_matches:
            assert dolphin.infinite_time, f"{path.name} leaves the endless training mode"


def test_video_match_run_file_records_real_matches() -> None:
    """P11: the run file the deliverable is recorded with (28 Aug 2026).

    Two changes from the P9 recorder and nothing else: the endless no-stock mode is off, and the
    recorder plays one real match per Dolphin instead of a fixed-length clip.
    """

    base = load_config(CONFIG_DIR / "video_20m_bc.toml")
    config = load_config(CONFIG_DIR / "video_match.toml")

    assert not config.env.dolphin.infinite_time  # stocks fall, the game ends
    assert config.video.mode == "match" and base.video.mode == "clip"
    assert config.video.matchups == ("policy:cpu9",)
    assert config.video.clips == config.env.dolphin.num_envs == 16
    assert config.video.seconds == 360.0  # the per-match cap, not the clip length
    assert config.env.dolphin.save_replays and config.env.dolphin.stop_timeout_s == 20.0
    assert config.env.dolphin.rtc_values == () and config.env.dolphin.menu_idle_frames == ()
    assert replace(config.env.dolphin, infinite_time=True, num_envs=2) == base.env.dolphin
    assert config_from_mapping(config_to_mapping(config)) == config


def test_video_mimic_run_file_plays_the_released_mimic(tmp_path: Path) -> None:
    """P12 (29 Aug 2026): the run file the MIMIC benchmark is recorded with.

    It is ``video_match.toml`` with the opponent swapped, plus the two things MIMIC needs: the raw
    gamestates (so ``worker_processes`` must be 0) and a GPU (it has no KV cache).
    """

    config = load_config(CONFIG_DIR / "video_mimic.toml")
    assert config.video.matchups == ("policy:mimic",) and config.video.needs_mimic
    assert config.video.mode == "match" and config.video.clips == 16
    # Past the 8-minute timer on purpose: the rollout starts at frame -123, so a 480 s budget would
    # stop 123 frames before a timed-out game writes its GAME_END.
    assert config.video.seconds == 520.0
    assert config.video.seconds * 60 > 28800 + 123
    dolphin = config.env.dolphin
    assert not dolphin.infinite_time  # stocks fall and the game ends
    assert dolphin.worker_processes == 0  # MIMIC reads gamestates that only exist in this process
    assert dolphin.stage == 25 and tuple(dolphin.characters) == (1, 1)  # Fox vs Fox on FD
    assert dolphin.save_replays
    assert config.runtime.device == "cuda" and config.mimic.device == "cuda"
    assert config.mimic.temperature == 1.0 and config.mimic.top_k == 0 and config.mimic.top_p == 0.0
    assert config.mimic.player == 1  # environment player 1 = Dolphin port 2, as in Ali's launcher
    assert config.mimic.resolved_checkpoint == "/opt/mimic/fox-master/model.pt"
    assert config.runtime.num_steps == 0 and config.logging.mode == "disabled"


def test_six_stage_match_run_files_cover_every_legal_stage() -> None:
    """P14 (2 Sep 2026): the six-stage twins of ``video_match.toml`` and ``video_mimic.toml``.

    Ali's 10m checkpoint is trained and RL-tuned on the six legal stages (Final Destination is the
    rarest stage in its data), so its matches are recorded twice: 16 on FD with the P11/P12 files and
    18 across the six stages with these.  ``stages`` cycles over the environment index, so 18 Dolphins
    are exactly three per stage.  Everything else is the FD file: real matches, the first replay kept,
    the cap past Melee's timer, the same device and MIMIC table.
    """

    for name, base_name, matchup in (
        ("video_match_stages.toml", "video_match.toml", "policy:cpu9"),
        ("video_mimic_stages.toml", "video_mimic.toml", "policy:mimic"),
    ):
        config = load_config(CONFIG_DIR / name)
        base = load_config(CONFIG_DIR / base_name)
        dolphin = config.env.dolphin
        assert dolphin.stages == (25, 24, 18, 26, 8, 6), name  # FD, BF, PS, DL, FoD, YS
        assert config.video.clips == dolphin.num_envs == 18, name
        counts = Counter(dolphin.stage_for(index) for index in range(config.video.clips))
        assert counts == {stage: 3 for stage in dolphin.stages}, name
        assert config.video.matchups == (matchup,) and config.video.mode == "match", name
        assert config.video.seconds == 520.0 and config.video.seconds * 60 > 28800 + 123, name
        assert not dolphin.infinite_time and dolphin.save_replays, name
        assert dolphin.worker_processes == 0 and tuple(dolphin.characters) == (1, 1), name
        assert config.runtime.device == base.runtime.device and config.mimic == base.mimic, name
        assert config.policy.profile == "10m", name
        # The Dolphin table differs from the FD file's in the stage list and the match count only.
        assert replace(dolphin, stages=(), num_envs=base.env.dolphin.num_envs) == base.env.dolphin, name
    match = load_config(CONFIG_DIR / "video_match_stages.toml")
    assert match.video.bc is not None and match.video.bc.endswith("frisson-melee-10m-v1-best-val.pt")
    assert match.video.output_dir == "/runs/matches"


def test_10m_bc_stage_run_files_start_from_ali_s_family_checkpoint() -> None:
    """The 10m run pair (user decisions 2 Sep 2026): Ali's family-pretraining 10m checkpoint (7.0 B
    frames, step 106806) as init, frozen teacher (KL 0.1) and zero-noise anchor; the RL-run-1 recipe
    (two-port self-play, nb16 x 96 Dolphins, D = 0); the six legal stages cycled over the Dolphins (16
    each); evals on 30 Dolphins (5 per stage) vs cpu9, MIMIC and the anchor; a 22-h launch cap inside
    a 36-h total."""

    bc_path = "/runs/bc/frisson-melee-10m-v1-best-val.pt"  # the stable Volume path (sha256 in the record)
    stages = (25, 24, 18, 26, 8, 6)  # FD, BF, PS, DL, FoD, YS (libmelee internal ids)
    day = load_config(CONFIG_DIR / "dolphin_10m_bc_stages_day.toml")
    assert day.policy.profile == "10m" and day.policy.init == "checkpoint"
    assert day.policy.checkpoint == bc_path and day.policy.fast_step is True
    assert day.teacher == TeacherConfig(source="policy")
    assert day.learner.kl_teacher_weight == pytest.approx(0.1)
    assert day.learner.ppo.max_mean_actor_kl == pytest.approx(5e-3)
    assert day.actor.delay_frames == 0 and day.actor.batch_steps == 1
    assert day.env.dolphin.chunk_frames == 1 and day.delay.allow_mismatch is False
    from melee_rl.config import pipeline_enabled

    assert not pipeline_enabled(day)
    assert day.learner.ppo.num_batches == 16 and day.env.dolphin.num_envs == 96
    # 85/15 self-play vs cpu9 by trained rows (user decision 2 Sep 2026): 72 self Dolphins = 144 rows.
    assert day.opponent.type == "mix" and day.opponent.mix == ("self*72", "cpu9*24")
    assert day.opponent.mix_workers == (12, 4)
    self_rows, cpu_rows = 2 * 72, 24
    assert 0.85 <= self_rows / (self_rows + cpu_rows) <= 0.87
    assert day.env.dolphin.stages == stages and day.env.dolphin.num_envs % len(stages) == 0
    assert all(group.num_envs % len(stages) == 0 for group in day.opponent.groups)  # balanced per arm
    assert day.env.dolphin.stage_for(95) == 6 and day.env.dolphin.stage_ids == tuple(sorted(stages))
    assert day.learner.microbatch_envs == 24  # must divide every arm's rows: 144 (self) and 24 (cpu9)
    assert day.learner.value_burnin_steps == 6 and day.learner.optimizer_burnin_steps == 3
    assert day.eval.enabled and day.eval.num_envs == 30 and day.eval.num_envs % len(stages) == 0
    assert day.eval.worker_processes == 10  # 3 per worker: 30 does not divide by the 16 training workers
    assert day.eval.opponents == ("cpu9", "mimic", f"other:{bc_path}")
    assert day.eval.metric_opponent == "other" and day.eval.metric == "reward_per_frame"
    assert day.eval.frames >= day.eval.num_envs * day.eval.rollout_length
    assert day.mimic.device == "cuda" and day.mimic.player == 1
    assert day.runtime.max_runtime_s == 79200.0 and day.runtime.max_total_runtime_s == 172800.0
    assert "10m" in day.logging.tags and "stages" in day.logging.tags and "mimic-eval" in day.logging.tags

    smoke = load_config(CONFIG_DIR / "dolphin_10m_bc_stages_smoke.toml")
    assert smoke.policy.checkpoint == bc_path and smoke.policy.profile == "10m"
    assert smoke.env.dolphin.stages == stages and smoke.env.dolphin.num_envs == 96
    assert smoke.opponent == day.opponent and smoke.learner.microbatch_envs == day.learner.microbatch_envs
    assert smoke.eval.enabled and smoke.eval.at_start and smoke.eval.opponents == day.eval.opponents
    assert smoke.eval.num_envs == day.eval.num_envs and smoke.eval.frames < day.eval.frames
    assert smoke.eval.worker_processes == day.eval.worker_processes
    # The smoke measured lr 4e-5 (KL ~8e-6/update, no transient); the day file runs 2.5x that (session 34).
    assert smoke.learner.learning_rate == pytest.approx(4e-5)
    assert day.learner.learning_rate == pytest.approx(1e-4)
    assert day.eval.interval_seconds == 10800.0 and day.eval.frames == 46080  # three-hourly (user decision)
    assert smoke.learner.ppo.num_batches == 16 and smoke.runtime.max_runtime_s == 2400.0
    assert smoke.runtime.max_total_runtime_s is None
    assert smoke.learner.value_burnin_steps == 2 and smoke.learner.optimizer_burnin_steps == 1


def test_10m_prefix_smoke_is_the_ring_smoke_in_prefix_mode() -> None:
    """Session 39 (INTERFACE.md patch #1): the like-for-like smoke is the session-34 smoke with
    ``context_mode = "prefix"`` as the only training change, a 25-min cap and one lean end eval (cpu9 and
    the BC anchor, no MIMIC -> the standard GPU image; user decision 3 Sep 2026)."""

    from dataclasses import replace

    from melee_rl.modal_app import config_needs_mimic

    ring_path = CONFIG_DIR / "dolphin_10m_bc_stages_smoke.toml"
    prefix_path = CONFIG_DIR / "dolphin_10m_bc_stages_prefix_smoke.toml"
    ring, prefix = load_config(ring_path), load_config(prefix_path)
    assert ring.actor.context_mode == "ring" and prefix.actor.context_mode == "prefix"
    assert replace(prefix.actor, context_mode="ring") == ring.actor
    assert prefix.policy == ring.policy and prefix.teacher == ring.teacher and prefix.delay == ring.delay
    assert prefix.env == ring.env and prefix.opponent == ring.opponent and prefix.learner == ring.learner
    assert prefix.runtime == replace(ring.runtime, max_runtime_s=1500.0)
    assert prefix.eval.enabled and not prefix.eval.at_start and prefix.eval.at_end
    assert prefix.eval.frames == ring.eval.frames and prefix.eval.num_envs == ring.eval.num_envs
    assert prefix.eval.worker_processes == ring.eval.worker_processes
    assert prefix.eval.opponents == ("cpu9", f"other:{ring.policy.checkpoint}")
    assert prefix.eval.metric_opponent == "other"
    assert "prefix" in prefix.logging.tags and "ring" not in prefix.logging.tags
    assert config_needs_mimic(ring_path) and not config_needs_mimic(prefix_path)


def test_10m_prefix_day_file_continues_the_day_run_in_prefix_mode() -> None:
    """Launch 3 of the 10m run (user decision 3 Sep 2026): the day file resumed in prefix mode with the prefix
    teacher and a 72-h cross-launch backstop; everything else byte-for-byte the day run's recipe."""

    from dataclasses import replace

    from melee_rl.modal_app import config_needs_mimic

    day = load_config(CONFIG_DIR / "dolphin_10m_bc_stages_day.toml")
    path = CONFIG_DIR / "dolphin_10m_bc_stages_prefix_day.toml"
    prefix = load_config(path)
    assert prefix.actor.context_mode == "prefix" and prefix.learner.teacher_prefix is True
    assert replace(prefix.actor, context_mode="ring") == day.actor
    assert replace(prefix.learner, teacher_prefix=False) == day.learner
    assert prefix.runtime == replace(day.runtime, max_total_runtime_s=259200.0)
    assert prefix.policy == day.policy and prefix.teacher == day.teacher and prefix.delay == day.delay
    assert prefix.env == day.env and prefix.opponent == day.opponent and prefix.eval == day.eval
    assert prefix.mimic == day.mimic and config_needs_mimic(path)
    assert set(prefix.logging.tags) - {"prefix", "teacher-prefix"} == set(day.logging.tags) - {"ring"}
    assert "ring" not in prefix.logging.tags
    finalized = finalize_config(prefix, resolve_model_config(prefix))
    assert finalized.learner.teacher_prefix is True and finalized.actor.context_mode == "prefix"


def test_75m_bc_stage_run_files_are_the_10m_prefix_recipe_on_ali_s_75m_checkpoint() -> None:
    """The 75m pair (user request 4 Sep 2026): the 10m six-stage prefix recipe (launch 3's day file) on Ali's
    75m family checkpoint (``75m-muon-low``, step 86016, 5.6 B target frames), with the three things the
    model size forces -- the stored prefixes kept on the host across the epochs (10.8 GiB per update), env
    microbatches of 12 (the 75m activations on the L4) and the 96 GiB box (``modal_app``) -- and ``best.pt``
    selected on the MIMIC eval (the P15 finding).  The smoke measures the day file's shape at lr 4e-5."""

    from melee_rl.modal_app import config_needs_mimic

    bc_path = "/runs/bc/frisson-melee-75m.pt"  # the stable Volume path (sha256 in the ledger)
    ten = load_config(CONFIG_DIR / "dolphin_10m_bc_stages_prefix_day.toml")
    day_path = CONFIG_DIR / "dolphin_75m_bc_stages_day.toml"
    smoke_path = CONFIG_DIR / "dolphin_75m_bc_stages_smoke.toml"
    day, smoke = load_config(day_path), load_config(smoke_path)
    for run in (day, smoke):
        assert run.policy.profile == "75m" and run.policy.init == "checkpoint"
        assert run.policy.checkpoint == bc_path and run.policy.fast_step is True
        assert run.actor.context_mode == "prefix"
        assert run.learner.teacher_prefix is True and run.learner.prefix_on_host is True
        assert run.learner.microbatch_envs == 12  # divides every arm's rows: 144 (self, both ports) and 24
        assert all(group.num_envs * len(group.trained_ports) % 12 == 0 for group in run.opponent.groups)
        assert run.env == ten.env and run.opponent == ten.opponent and run.teacher == ten.teacher
        assert run.delay == ten.delay and run.mimic == ten.mimic and run.actor == ten.actor
        model = resolve_model_config(run)  # the EXTRA_PROFILES fallback, reached through the run file
        assert model.profile == "75m" and model.d_model == 768 and model.n_layers == 11
        assert model.n_kv_heads == 3 and model.context_length == 256 and model.compute_dtype == "float32"
        finalized = finalize_config(run, model)
        assert finalized.learner.prefix_on_host is True and finalized.actor.context_mode == "prefix"
        assert run.eval.opponents == ("cpu9", "mimic", f"other:{bc_path}")
        assert run.eval.metric_opponent == "mimic" and run.eval.metric == "reward_per_frame"
        assert "75m" in run.logging.tags and "10m" not in run.logging.tags
        assert {"prefix", "teacher-prefix", "host-prefix", "mb12", "xwide"} <= set(run.logging.tags)
    assert config_needs_mimic(day_path) and config_needs_mimic(smoke_path)
    # The day file is launch 3's day file with the 75m checkpoint and the three forced changes.
    assert replace(day.policy, profile="10m", checkpoint=ten.policy.checkpoint) == ten.policy
    assert day.runtime == ten.runtime  # 22 h per launch (Modal refuses longer timeouts), 72 h in all
    assert replace(day.eval, opponents=ten.eval.opponents, metric_opponent="other") == ten.eval
    assert (
        replace(
            day.learner, microbatch_envs=24, prefix_on_host=False, learning_rate=ten.learner.learning_rate
        )
        == ten.learner
    )
    assert set(day.logging.tags) - {"75m", "host-prefix", "mb12", "xwide"} == set(ten.logging.tags) - {"10m"}
    # The smoke is the day file with fast burn-ins, a 70-min cap and a small start + end eval.
    assert smoke.runtime == replace(
        day.runtime, max_runtime_s=4200.0, max_total_runtime_s=None, snapshot_interval_steps=0
    )
    assert smoke.learner.learning_rate == pytest.approx(
        4e-5
    )  # the 10m smoke's rate; the smoke measures the 75m's KL
    assert 0.0 < day.learner.learning_rate <= 2e-4  # set from the smoke before the day launch
    assert (
        replace(
            smoke.learner,
            value_burnin_steps=6,
            optimizer_burnin_steps=3,
            learning_rate=day.learner.learning_rate,
        )
        == day.learner
    )
    assert smoke.learner.value_burnin_steps == 2 and smoke.learner.optimizer_burnin_steps == 1
    assert smoke.eval == replace(day.eval, interval_seconds=86400.0, frames=15360)
    assert smoke.eval.at_start and smoke.eval.at_end and smoke.eval.num_envs == 30
    assert set(smoke.logging.tags) - {"smoke"} == set(day.logging.tags) - {"day"}


def test_rl_speedup_run_files_profile_the_75m_shapes() -> None:
    """5 Sep 2026 (/rl-speedup): ``dolphin_75m_profile_smoke`` is the 75m smoke with a 1 + 1 burn-in ladder,
    no start eval, a 35-min cap and ``[profiling]`` on for its two PPO steps; ``dummy_75m_bench`` is the same
    policy / actor / learner shape (168 rows: 72 two-port self envs + 24 cpu9, prefix mode, mb 12) in the
    dummy env on the L4 box with ``num_batches`` 4, the same profiling table and no evals."""

    from melee_rl.profiling import ProfilingConfig

    smoke = load_config(CONFIG_DIR / "dolphin_75m_bc_stages_smoke.toml")
    assert smoke.profiling == ProfilingConfig()  # the training files stay unprofiled
    profile = load_config(CONFIG_DIR / "dolphin_75m_profile_smoke.toml")
    expected = ProfilingConfig(enabled=True, steps=(2, 3), actor_frames=32, learner_chunks=4)
    assert profile.profiling == expected
    assert replace(profile.runtime, max_runtime_s=4200.0) == smoke.runtime
    assert replace(profile.learner, value_burnin_steps=2) == smoke.learner
    assert replace(profile.eval, at_start=True) == smoke.eval
    for name in ("policy", "teacher", "env", "actor", "opponent", "delay", "mimic"):
        assert getattr(profile, name) == getattr(smoke, name), name
    assert set(profile.logging.tags) == set(smoke.logging.tags) | {"profile"}
    bench = load_config(CONFIG_DIR / "dummy_75m_bench.toml")
    assert bench.profiling == expected and not bench.eval.enabled
    assert bench.env.type == "dummy" and bench.env.dummy.num_envs == 96
    assert bench.env.players == ("policy", "policy")  # the mix groups derive their own players
    assert bench.opponent.type == "mix" and bench.opponent.mix == ("self*72", "cpu9*24")
    assert [group.rows for group in bench.opponent.groups] == [144, 24]
    assert bench.policy == smoke.policy and bench.actor == smoke.actor and bench.teacher == smoke.teacher
    assert replace(bench.learner, ppo=smoke.learner.ppo, value_burnin_steps=2) == smoke.learner
    assert bench.learner.ppo == replace(smoke.learner.ppo, num_batches=4)
    assert bench.runtime.device == "cuda" and bench.runtime.num_steps == 6
    assert bench.runtime.max_runtime_s == 1200.0 and bench.runtime.save_interval_steps == 100
    assert bench.logging.mode == "disabled"
    model = resolve_model_config(bench)
    assert model.profile == "75m" and finalize_config(bench, model).learner.microbatch_envs == 12


def test_rl_speedup_ab_bench_files_differ_from_the_baseline_in_one_knob_each() -> None:
    """5 Sep 2026 (/rl-speedup): the A/B files are ``dummy_75m_bench`` with profiling off and
    ``metrics.jsonl`` on (the baseline), and exactly one knob each: microbatch 24, checkpointing off, TF32
    for the policy."""

    from melee_rl.profiling import ProfilingConfig

    bench = load_config(CONFIG_DIR / "dummy_75m_bench.toml")
    base = load_config(CONFIG_DIR / "dummy_75m_bench_base.toml")
    assert not base.profiling.enabled and base.logging.jsonl and not bench.logging.jsonl
    assert ProfilingConfig().dir == base.profiling.dir  # only the switch differs from the traced bench
    assert replace(base, profiling=bench.profiling, logging=bench.logging) == bench
    mb24 = load_config(CONFIG_DIR / "dummy_75m_bench_mb24.toml")
    assert mb24.learner.microbatch_envs == 24 and replace(mb24.learner, microbatch_envs=12) == base.learner
    assert replace(mb24, learner=base.learner) == base
    nockpt = load_config(CONFIG_DIR / "dummy_75m_bench_nockpt.toml")
    assert nockpt.policy.model_overrides == {"gradient_checkpointing": False}
    assert replace(nockpt, policy=base.policy) == base
    assert (
        resolve_model_config(base).gradient_checkpointing
        and not resolve_model_config(nockpt).gradient_checkpointing
    )
    tf32 = load_config(CONFIG_DIR / "dummy_75m_bench_tf32.toml")
    assert tf32.learner.matmul_precision == "high" and base.learner.matmul_precision == "highest"
    assert replace(tf32.learner, matmul_precision="highest") == base.learner
    assert replace(tf32, learner=base.learner) == base
    for run in (base, mb24, nockpt, tf32):
        finalize_config(run, resolve_model_config(run))


def test_overrides_apply_toml_literals_and_bare_strings(tmp_path: Path) -> None:
    """5 Sep 2026 (/rl-speedup): ``--set table.key=value`` on the launcher and ``--overrides`` on Modal."""

    from melee_rl.config import apply_overrides, parse_override

    assert parse_override("learner.ppo.num_batches=4") == (["learner", "ppo", "num_batches"], 4)
    assert parse_override("policy.fast_encoder=true") == (["policy", "fast_encoder"], True)
    assert parse_override("learner.learning_rate=4e-5") == (["learner", "learning_rate"], 4e-5)
    assert parse_override("policy.fast_attention=lean") == (["policy", "fast_attention"], "lean")
    assert parse_override('policy.fast_attention="lean"') == (["policy", "fast_attention"], "lean")
    assert parse_override('logging.tags=["a", "b"]') == (["logging", "tags"], ["a", "b"])
    for bad in ("nokey", "=3", "a..b=1", "a.=1"):
        with pytest.raises(ValueError, match="override"):
            parse_override(bad)
    raw: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "policy": {"fast_step": True},
        "learner": {"ppo": {"num_batches": 16}},
    }
    applied = apply_overrides(
        raw, ["learner.ppo.num_batches=4", "policy.fast_attention=lean", "runtime.threads=2"]
    )
    assert applied["learner"]["ppo"]["num_batches"] == 4 and applied["policy"]["fast_attention"] == "lean"
    assert applied["runtime"] == {"threads": 2} and raw["learner"]["ppo"]["num_batches"] == 16  # a copy
    with pytest.raises(ValueError, match="table"):
        apply_overrides(raw, ["learner.ppo=1"])
    with pytest.raises(ValueError, match="value, not a table"):
        apply_overrides(raw, ["policy.fast_step.x=1"])
    config = load_config(SMOKE_TINY, overrides=["policy.fast_step=true", "policy.fast_attention=lean"])
    assert config.policy.fast_step and config.policy.fast_attention == "lean"
    assert load_config(SMOKE_TINY).policy.fast_attention == "sdpa"
    with pytest.raises(ValueError, match="unknown key"):
        load_config(SMOKE_TINY, overrides=["policy.no_such_key=1"])


def test_fast_forward_needs_the_prefix_context_and_round_trips() -> None:
    """5 Sep 2026 (/rl-learner-speedup): ``learner.fast_forward`` covers the prefix window only."""

    with pytest.raises(ValueError, match="fast_forward"):
        load_config(SMOKE_TINY, overrides=["learner.fast_forward=true", "actor.context_mode=reprime"])
    config = load_config(SMOKE_TINY, overrides=["learner.fast_forward=true", "actor.context_mode=prefix"])
    assert config.learner.fast_forward and config.learner.fast_forward_attention == "sdpa"
    assert config_from_mapping(config_to_mapping(config)) == config
    bmm = load_config(
        SMOKE_TINY,
        overrides=[
            "learner.fast_forward=true",
            "actor.context_mode=prefix",
            "learner.fast_forward_attention=bmm",
        ],
    )
    assert bmm.learner.fast_forward_attention == "bmm"
    with pytest.raises(ValueError, match="fast_forward_attention"):
        load_config(SMOKE_TINY, overrides=["learner.fast_forward_attention=flash"])
    assert not load_config(SMOKE_TINY).learner.fast_forward


def test_compile_needs_fast_forward_and_no_checkpointing() -> None:
    """5 Sep 2026 (/rl-learner-speedup, lever 2)."""

    prefix = ["actor.context_mode=prefix", "learner.fast_forward=true"]
    with pytest.raises(ValueError, match="compile"):
        load_config(SMOKE_TINY, overrides=["actor.context_mode=prefix", "learner.compile=true"])
    tiny = load_config(SMOKE_TINY, overrides=prefix)
    if resolve_model_config(tiny).gradient_checkpointing:
        with pytest.raises(ValueError, match="gradient_checkpointing"):
            load_config(SMOKE_TINY, overrides=[*prefix, "learner.compile=true"])
    config = load_config(
        SMOKE_TINY,
        overrides=[*prefix, "learner.compile=true", "policy.model_overrides.gradient_checkpointing=false"],
    )
    assert config.learner.compile and config.learner.compile_backend == "inductor"
    assert not resolve_model_config(config).gradient_checkpointing
    assert config_from_mapping(config_to_mapping(config)) == config
    with pytest.raises(ValueError, match="compile_backend"):
        load_config(SMOKE_TINY, overrides=[*prefix, "learner.compile_backend=cudagraphs"])


def test_post_trained_stage_run_files_are_the_bc_recipes_on_ali_s_curriculum_checkpoints() -> None:
    """The post-trained pair (user request 5 Sep 2026): the 10m prefix day recipe (launch 3's file) and the
    75m day recipe, unchanged, on Ali's curriculum checkpoints -- ``cur-6-kd/10m/B-mix10`` step 195248
    (distilled from the 75m below; weighted validation NLL 0.75746) and ``cur-3/75m/B-mix10`` step 127214
    (NLL 0.72489) -- each in a fresh run directory with the new file as init, frozen teacher and zero-noise
    anchor, and ``best.pt`` selected on the MIMIC eval for both (the 10m BC file still selects on the
    anchor)."""

    from melee_rl.modal_app import config_needs_mimic

    pairs = (
        (
            "dolphin_10m_pt_stages_day.toml",
            "dolphin_10m_bc_stages_prefix_day.toml",
            "/runs/bc/frisson-melee-10m-posttrained-best-val.pt",
            {"cur6kd"},
        ),
        (
            "dolphin_75m_pt_stages_day.toml",
            "dolphin_75m_bc_stages_day.toml",
            "/runs/bc/frisson-melee-75m-posttrained-best-val.pt",
            {"cur3"},
        ),
    )
    for name, base_name, checkpoint, tags in pairs:
        path = CONFIG_DIR / name
        run, base = load_config(path), load_config(CONFIG_DIR / base_name)
        assert run.policy.init == "checkpoint" and run.policy.checkpoint == checkpoint, name
        assert replace(run.policy, checkpoint=base.policy.checkpoint) == base.policy, name
        assert run.eval.opponents == ("cpu9", "mimic", f"other:{checkpoint}"), name
        assert run.eval.metric_opponent == "mimic" and run.eval.metric == "reward_per_frame", name
        assert (
            replace(run.eval, opponents=base.eval.opponents, metric_opponent=base.eval.metric_opponent)
            == base.eval
        ), name
        assert run.runtime == base.runtime and run.learner == base.learner and run.actor == base.actor, name
        assert run.env == base.env and run.opponent == base.opponent and run.teacher == base.teacher, name
        assert run.delay == base.delay and run.mimic == base.mimic and run.logging.mode == "disabled", name
        assert run.runtime.max_runtime_s == 79200.0 and run.runtime.max_total_runtime_s == 259200.0, name
        expected_tags = {"pt-init", "bc-init", "prefix", "teacher-prefix", "mimic-eval"} | tags
        assert expected_tags <= set(run.logging.tags), name
        assert set(run.logging.tags) - {"pt-init"} - tags == set(base.logging.tags), name
        assert config_needs_mimic(path), name
        model = resolve_model_config(run)
        assert model.profile == run.policy.profile and model.compute_dtype == "float32", name
        finalized = finalize_config(run, model)
        assert finalized.actor.context_mode == "prefix" and finalized.learner.teacher_prefix is True, name


def test_leash_gate_run_files_fork_step_1318_with_slippi_ai_s_recipe() -> None:
    """Gate 1 of the strength plan (user decision 10 Sep 2026, session 57): three 22-h arms fork the 10m
    post-trained run's step 1318 -- policy and value net -- into fresh directories and loosen the teacher
    leash into the range slippi-ai's releases were trained with (read out of the files: forward and reverse
    teacher KL 0.05 -> 0.02 -> 0.01 -> 0.005 at policy-gradient weight 3 up their ladder, 0.003 for the Fox
    dittos), with the rest of that recipe: PPO beta 0.3, an 8-s reward half-life and 100 % self-play (the
    CPU-9 arm dropped).  The teacher stays Ali's post-trained file, the in-loop anchor is step 1318 itself,
    every speed lever is on, and everything else -- lr, delay 0, the prefix learner, the six stages, the
    burn-ins -- is the post-trained day recipe.  The arms differ only in the leash and one tag."""

    from melee_rl.modal_app import config_needs_mimic

    base = load_config(CONFIG_DIR / "dolphin_10m_pt_stages_day.toml")
    teacher = "/runs/bc/frisson-melee-10m-posttrained-best-val.pt"
    fork = "/runs/selfplay/10m_pt_stages/best_1318.pt"
    for tag, weight in (("kl0p03", 0.03), ("kl0p01", 0.01), ("kl0p003", 0.003)):
        name = f"dolphin_10m_leash_{tag}_day.toml"
        path = CONFIG_DIR / name
        run = load_config(path)
        assert run.policy == replace(
            base.policy,
            checkpoint=fork,
            init_value=True,
            fast_attention="lean",
            fast_encoder=True,
            graph_step=True,
            model_overrides={"gradient_checkpointing": False},
        ), name
        assert run.teacher == TeacherConfig(source="checkpoint", path=teacher), name
        assert run.learner == replace(
            base.learner,
            policy_gradient_weight=3.0,
            kl_teacher_weight=weight,
            reverse_kl_teacher_weight=weight,
            fast_forward=True,
            compile=True,
            returns=replace(base.learner.returns, reward_halflife=8.0),
            ppo=replace(base.learner.ppo, beta=0.3),
        ), name
        assert run.learner.learning_rate == 1e-4 and run.learner.microbatch_envs == 24, name
        assert run.opponent == replace(base.opponent, mix=("self*48@a", "self*48@b"), mix_workers=(8, 8)), (
            name
        )
        assert run.env == replace(base.env, dolphin=replace(base.env.dolphin, shared_memory=True)), name
        assert run.actor == replace(base.actor, packed_frames=True, interleave_arms=True), name
        assert run.eval == replace(
            base.eval, opponents=("cpu9", "mimic", f"other:{fork}"), metric_opponent="other"
        ), name
        assert run.runtime == base.runtime and run.delay == base.delay and run.mimic == base.mimic, name
        assert run.logging == replace(base.logging, tags=run.logging.tags, jsonl=True), name
        assert {"leash", "gate1", "fork1318", "slippi-recipe", "self100", tag} <= set(run.logging.tags), name
        assert not {"cpu9-arm", "self85", "bc-init"} & set(run.logging.tags), name
        assert config_needs_mimic(path), name
        model = resolve_model_config(run)
        finalized = finalize_config(run, model)
        assert finalized.actor.context_mode == "prefix" and finalized.learner.teacher_prefix is True, name


def test_decision_stride_is_a_play_time_flag() -> None:
    """Lever 8 (5 Sep 2026): ``actor.decision_stride > 1`` is accepted by a recording file (``num_steps = 0``)
    and refused by a training run until the learner has a contract for held rows."""

    recorder = _finalized(actor={"decision_stride": 3}, runtime={"num_steps": 0})
    assert recorder.actor.decision_stride == 3
    assert _finalized(actor={"decision_stride": 1}).actor.decision_stride == 1
    with pytest.raises(ValueError, match="decision_stride"):
        _finalized(actor={"decision_stride": 2}, runtime={"num_steps": 5})


def test_teacher_delay_frames_rules_and_rival_delay_specs() -> None:
    """11 Sep 2026 (/delay-finetune): ``[teacher] delay_frames`` is the teacher's own delay ``D_T`` -- the
    run's ``actor.delay_frames`` when unset (today's rows bit for bit), 0 the hindsight teacher.  It cannot
    exceed the run's delay, has no meaning without a teacher, and rules out ``teacher_prefix`` unless it
    equals the run's delay (the stored keys hold the student's pairing).  Rival specs may carry the delay a
    file plays at (``other:<path>@d<n>``), and path resolution keeps the suffix."""

    from melee_rl.config import resolve_paths

    assert TeacherConfig().delay_frames is None
    assert _finalized(teacher={"delay_frames": 0}).teacher.delay_frames == 0
    with pytest.raises(ValueError, match="delay_frames"):
        TeacherConfig(delay_frames=-1)
    with pytest.raises(ValueError, match="delay_frames"):
        TeacherConfig(source="none", delay_frames=0)
    with pytest.raises(ValueError, match=r"teacher\.delay_frames"):
        _finalized(teacher={"delay_frames": 1})  # above the run's delay of 0
    hindsight = _finalized(
        actor={"delay_frames": 2}, delay={"allow_mismatch": True}, teacher={"delay_frames": 0}
    )
    assert hindsight.teacher.delay_frames == 0 and hindsight.actor.delay_frames == 2
    same = _finalized(
        actor={"delay_frames": 2, "context_mode": "prefix"},
        delay={"allow_mismatch": True},
        teacher={"delay_frames": 2},
        learner={"teacher_prefix": True},
    )
    assert same.learner.teacher_prefix and same.teacher.delay_frames == 2
    with pytest.raises(ValueError, match="teacher_prefix"):
        _finalized(
            actor={"delay_frames": 2, "context_mode": "prefix"},
            delay={"allow_mismatch": True},
            teacher={"delay_frames": 0},
            learner={"teacher_prefix": True},
        )
    mapping = config_to_mapping(hindsight)
    assert mapping["teacher"]["delay_frames"] == 0 and config_from_mapping(mapping) == hindsight
    base = load_config(SMOKE_TINY)
    spec = replace(
        base,
        eval=replace(base.eval, opponents=("cpu9", "other:ckpt/best.pt@d3")),
        opponent=replace(base.opponent, type="mix", mix=("self*1", "other:ckpt/best.pt@d0*2@x")),
    )
    resolved = resolve_paths(spec, "/tmp/base")
    assert resolved.eval.opponents == ("cpu9", "other:/tmp/base/ckpt/best.pt@d3")
    assert resolved.opponent.mix == ("self*1", "other:/tmp/base/ckpt/best.pt@d0*2@x")


def test_delay_run_files_fork_kl0p01_to_delay_18() -> None:
    """11 Sep 2026 (/delay-finetune, user decisions: D = 18 directly, the hindsight teacher + RL, the base is
    kl0p01's final snapshot).  Stage 1 is on-policy distillation at D = 18: the frozen base as the hindsight
    teacher (``delay_frames = 0``), KL(T || pi) alone (PG 0, beta 0, the guard out of the way), 8x the Adam
    steps per rollout (4 batches x 4 epochs), 100 % 18-vs-18 self-play, the in-loop anchor the base at the
    run's delay (the naive-18 fork point), an 8-h cap; the smoke is the same file at 35 min with a shorter
    eval window.  Stage 2 is kl0p01's PPO recipe anchored to the Stage-1 student at its own delay
    (``teacher_prefix`` allowed again).  Everything else is the kl0p01 recipe."""

    from melee_rl.modal_app import config_needs_mimic

    base = load_config(CONFIG_DIR / "dolphin_10m_leash_kl0p01_day.toml")
    fork = "/runs/selfplay/10m_delay/base/kl0p01_final.pt"
    day = load_config(CONFIG_DIR / "dolphin_10m_delay_distill_day.toml")
    smoke = load_config(CONFIG_DIR / "dolphin_10m_delay_distill_smoke.toml")
    for name, run in (("day", day), ("smoke", smoke)):
        assert run.policy == replace(base.policy, checkpoint=fork), name
        assert run.teacher == TeacherConfig(source="policy", delay_frames=0), name
        assert run.actor == replace(base.actor, delay_frames=18), name
        assert run.delay.allow_mismatch is True, name
        assert run.learner == replace(
            base.learner,
            policy_gradient_weight=0.0,
            kl_teacher_weight=0.0,
            reverse_kl_teacher_weight=1.0,
            teacher_prefix=False,
            value_burnin_steps=8,
            optimizer_burnin_steps=4,
            ppo=replace(base.learner.ppo, num_epochs=4, num_batches=4, beta=0.0, max_mean_actor_kl=1.0e3),
        ), name
        assert run.opponent == base.opponent and run.env == base.env and run.mimic == base.mimic, name
        assert run.eval.opponents == ("cpu9", "mimic", f"other:{fork}@d18"), name
        assert run.eval.metric_opponent == "other" and run.eval.at_start and run.eval.at_end, name
        assert {"delay", "d18", "distill", "hindsight", "self100"} <= set(run.logging.tags), name
        assert run.logging.jsonl is True and config_needs_mimic(
            CONFIG_DIR / f"dolphin_10m_delay_distill_{name}.toml"
        )
        finalized = finalize_config(run, resolve_model_config(run))
        assert finalized.actor.context_mode == "prefix" and finalized.learner.fast_forward, name
    assert day.runtime == replace(
        base.runtime,
        max_runtime_s=28800.0,
        max_total_runtime_s=28800.0,
        save_interval_steps=8,
        snapshot_interval_steps=32,
    )
    assert day.eval == replace(
        base.eval, interval_seconds=7200.0, opponents=day.eval.opponents, metric_opponent="other"
    )
    assert smoke.runtime == replace(day.runtime, max_runtime_s=3300.0, max_total_runtime_s=3300.0)
    assert smoke.eval == replace(day.eval, interval_seconds=3600.0, frames=30720)  # never due in 35 min
    assert "smoke" in smoke.logging.tags and "smoke" not in day.logging.tags
    student = "/runs/selfplay/10m_delay/d18_distill/latest.pt"
    ppo = load_config(CONFIG_DIR / "dolphin_10m_delay_ppo_day.toml")
    assert ppo.policy == replace(base.policy, checkpoint=student)
    assert ppo.teacher == TeacherConfig(source="policy", delay_frames=18)
    assert ppo.actor == replace(base.actor, delay_frames=18) and ppo.delay.allow_mismatch is True
    assert ppo.learner == base.learner and ppo.opponent == base.opponent and ppo.env == base.env
    assert ppo.runtime == base.runtime and ppo.mimic == base.mimic
    assert ppo.eval == replace(base.eval, opponents=("cpu9", "mimic", f"other:{student}@d18"))
    assert {"delay", "d18", "ppo", "stage2", "self100"} <= set(ppo.logging.tags)
    assert config_needs_mimic(CONFIG_DIR / "dolphin_10m_delay_ppo_day.toml")
    finalized = finalize_config(ppo, resolve_model_config(ppo))
    assert finalized.learner.teacher_prefix is True and finalized.actor.delay_frames == 18


def test_10m_delay_ema_day_file_is_the_stage2_recipe_with_the_adaptive_levers() -> None:
    """15 Sep 2026 (session 69, DELAY-PLAN.md §17): hop 2 forks hop 1's best snapshot with the EMA anchor,
    the entropy floor and the KL-targeted learning rate, the fork logged as the weight-0 reference, a hard
    22-h directory cap and step-interval evals; everything else is the Stage-2 PPO file's."""

    from melee_rl.modal_app import config_needs_mimic

    ppo = load_config(CONFIG_DIR / "dolphin_10m_delay_ppo_day.toml")
    ema = load_config(CONFIG_DIR / "dolphin_10m_delay_ema_day.toml")
    fork = "/runs/selfplay/10m_delay/d18_ppo_ra1/step_00001224.pt"
    assert ema.policy == replace(ppo.policy, checkpoint=fork)
    assert ema.teacher == TeacherConfig(source="ema", ema_tau=0.01, delay_frames=18, reference_path=fork)
    assert ema.learner == replace(
        ppo.learner,
        kl_teacher_weight=0.03,
        reverse_kl_teacher_weight=0.0,
        entropy_target=0.45,
        entropy_gain=2.0e-3,
        entropy_weight_max=0.02,
        actor_kl_target=1.0e-4,
        lr_min=2.5e-5,
        lr_max=4.0e-4,
    )
    assert ema.learner.teacher_prefix is True and ema.learner.learning_rate == 1.0e-4
    assert ema.runtime == replace(ppo.runtime, max_total_runtime_s=79200.0)
    assert ema.eval == replace(
        ppo.eval,
        interval_seconds=1.0e7,
        interval_steps=120,
        opponents=("cpu9", "mimic", f"other:{fork}@d18"),
    )
    assert ema.actor == ppo.actor and ema.opponent == ppo.opponent and ema.env == ppo.env
    assert ema.delay == ppo.delay and ema.mimic == ppo.mimic
    assert {"delay", "d18", "ema-anchor", "entropy-target", "kl-lr", "hop2"} <= set(ema.logging.tags)
    assert "stage2" not in ema.logging.tags and "kl0p01" not in ema.logging.tags
    assert config_needs_mimic(CONFIG_DIR / "dolphin_10m_delay_ema_day.toml")
    finalized = finalize_config(ema, resolve_model_config(ema))
    assert finalized.teacher.source == "ema" and finalized.learner.effective_lr_min == 2.5e-5


def test_10m_delay_league_files_are_hop_2s_recipe_against_the_league_mix() -> None:
    """17 Sep 2026 (the league run, LEAGUE-PLAN.md §2 / §3.5 / §3.6): the day file forks hop 2's step 760 with
    hop 2's own recipe -- the EMA anchor, the entropy floor, the KL-targeted lr, the fork as the weight-0
    reference -- so the opponents are the one change: 30 self / 24 gm as its twelve characters / 24 fox_d18
    (four tags) / 12 of our delay-0 best / 6 Falco Fox-killers, fox_d18 through the port as the in-loop
    selector in cpu9's place. After smoke 2 (user decisions, 17 Sep 2026) the learner weighs rows, not arms
    (``loss_weighting = "row"``: the planned 48 / 19 / 19 / 10 / 5 %) and joins each rollout's arms
    (``concat_arms``, microbatch 21 of the 126 joined rows). The smoke is the day file for an hour with a
    fox_d18-only start eval."""

    from melee_rl.modal_app import config_needs_mimic
    from melee_rl.opponents import ArmConfig

    ema = load_config(CONFIG_DIR / "dolphin_10m_delay_ema_day.toml")
    day = load_config(CONFIG_DIR / "dolphin_10m_delay_league_day.toml")
    smoke = load_config(CONFIG_DIR / "dolphin_10m_delay_league_smoke.toml")
    fork = "/runs/selfplay/10m_delay/d18_ema1/step_00000760.pt"
    d0 = "/runs/selfplay/10m_leash/kl0p003/step_00000632.pt"
    assert day.policy == replace(ema.policy, checkpoint=fork)
    assert day.teacher == replace(ema.teacher, reference_path=fork)
    assert day.learner == replace(ema.learner, microbatch_envs=21, loss_weighting="row", concat_arms=True)
    assert day.actor == ema.actor and day.env == ema.env and day.runtime == ema.runtime
    assert day.delay == ema.delay and day.mimic == ema.mimic
    assert day.opponent.mix == (
        "self*30",
        "slippi:gm*24@gm",
        "slippi:fox_d18_ditto_v4*24@foxd18",
        f"other:{d0}@d0*12@d0",
        "slippi:falco_d18_vs_fox_v4*6@falco",
    )
    assert day.opponent.mix_workers == (5, 4, 4, 2, 1)
    assert day.opponent.arms == {
        "gm": ArmConfig(
            characters=(1, 22, 18, 7, 15, 2, 9, 14, 10, 17, 12, 13),
            character_repeat=2,
            names=("Master Player",),
        ),
        "foxd18": ArmConfig(names=("Cody", "Hax", "Aklo", "SFAT"), name_repeat=6),
        "falco": ArmConfig(names=("Ginger", "KJH", "Frenzy", "BBB")),
    }
    assert [group.name for group in day.opponent.groups] == [
        "self",
        "slippi_gm",
        "slippi_foxd18",
        "other_d0",
        "slippi_falco",
    ]
    assert day.eval == replace(
        ema.eval,
        opponents=("slippi:fox_d18_ditto_v4", "mimic", f"other:{fork}@d18"),
        metric_opponent="fox_d18_ditto_v4",
    )
    assert day.slippi_ai.model_dir == "/runs/external/slippi-ai" and day.slippi_ai.verify_sha256
    assert day.slippi_ai.torch_graph and day.slippi_ai.sample_temperature == 1.0
    assert {"league", "league1", "d18", "ema-anchor", "slippi-port"} <= set(day.logging.tags)
    assert "hop2" not in day.logging.tags and "self100" not in day.logging.tags
    assert config_needs_mimic(CONFIG_DIR / "dolphin_10m_delay_league_day.toml")
    finalized = finalize_config(day, resolve_model_config(day))
    assert finalized.learner.microbatch_envs == 21 and finalized.actor.delay_frames == 18
    assert sum(group.rows for group in finalized.opponent.groups) == 126

    assert smoke.runtime == replace(
        day.runtime, max_runtime_s=3600.0, max_total_runtime_s=3600.0, snapshot_interval_steps=4
    )
    assert smoke.eval == replace(day.eval, opponents=("slippi:fox_d18_ditto_v4",), frames=11520, at_end=False)
    for table in ("policy", "teacher", "learner", "actor", "env", "opponent", "delay", "slippi_ai"):
        assert getattr(smoke, table) == getattr(day, table), table
    assert "smoke" in smoke.logging.tags and "day" not in smoke.logging.tags
    assert not config_needs_mimic(CONFIG_DIR / "dolphin_10m_delay_league_smoke.toml")
