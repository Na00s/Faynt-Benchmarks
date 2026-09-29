"""``DolphinEnv`` wiring that needs no libmelee (PLAN.md §6 P5): import guard, run files, shared constants.

Runs everywhere (with or without ``melee`` installed): the config and training modules import
``melee_rl.env.dolphin`` without touching libmelee; only ``DolphinEnv(...)`` itself does.
"""

from __future__ import annotations

import sys
import types

import pytest

from melee_rl.config import CONFIG_DIR, ENV_TYPES, load_config
from melee_rl.env import dolphin as dolphin_lib
from melee_rl.env.dolphin import DEFAULT_DOLPHIN_PATH, DEFAULT_ISO_PATH, DolphinEnvConfig
from melee_rl.env.libmelee_frames import MELEE_INSTALL_HINT, MELEE_VERSION, import_melee
from melee_rl.train import build_env

DOLPHIN_CPU = CONFIG_DIR / "dolphin_cpu.toml"
DOLPHIN_SELF = CONFIG_DIR / "dolphin_self.toml"


def test_build_env_without_libmelee_raises_a_guided_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ENV_TYPES == ("dummy", "dolphin")
    config = load_config(DOLPHIN_CPU)
    assert config.env.type == "dolphin"
    monkeypatch.setitem(sys.modules, "melee", None)  # ``import melee`` now raises ImportError
    with pytest.raises(ImportError, match=r"melee==0\.47\.3") as info:
        build_env(config)
    assert str(info.value) == MELEE_INSTALL_HINT
    with pytest.raises(ImportError, match=r"dolphin"):
        import_melee()
    # The module never binds libmelee at import time (the dummy path must work without it).
    bound = [name for name, value in vars(dolphin_lib).items() if isinstance(value, types.ModuleType)]
    assert "melee" not in bound and "melee" not in vars(dolphin_lib)
    assert MELEE_VERSION == "0.47.3"


def test_dolphin_run_files_load_and_point_at_the_modal_image_paths() -> None:
    cpu = load_config(DOLPHIN_CPU)
    self_play = load_config(DOLPHIN_SELF)
    for config in (cpu, self_play):
        dolphin = config.env.dolphin
        assert isinstance(dolphin, DolphinEnvConfig)
        assert config.env.num_envs == dolphin.num_envs == 2
        assert dolphin.dolphin_path == DEFAULT_DOLPHIN_PATH and dolphin.iso_path == DEFAULT_ISO_PATH
        assert dolphin.characters == (1, 1) and dolphin.costumes == (0, 1) and dolphin.stage == 25
        assert dolphin.console_timeout_s == 30.0 and dolphin.keepalive_seconds == 0.0
        assert dolphin.infinite_time and dolphin.instant_match_restart and not dolphin.save_replays
        assert config.policy.profile == "3m" and config.actor.rollout_length == 16
        assert config.actor.context_frames == 16 and config.runtime.threads == 4
        assert config.runtime.reset_every_n_steps is None  # a hard reset relaunches every Dolphin
        assert config.learner.ppo.beta == pytest.approx(0.3) and config.learner.kl_teacher_weight == 0.0
        assert config.logging.mode == "disabled" and "dolphin-env" in config.logging.tags
        assert config.logging.project == "Faynt" and config.logging.group == "rl-post-training"
    assert cpu.opponent.type == "cpu" and cpu.opponent.cpu_level == 9
    assert cpu.env.players == ("policy", "cpu") and cpu.env.dolphin.controlled_ports == (0,)
    assert self_play.opponent.type == "self"
    assert self_play.env.players == ("policy", "policy") and self_play.env.dolphin.controlled_ports == (0, 1)
    assert "self-play" in self_play.logging.tags and "smoke" in cpu.logging.tags


def test_modal_harness_and_dolphin_env_agree_on_paths() -> None:
    pytest.importorskip("modal")
    from melee_rl import modal_app

    default = modal_app.DOLPHIN_BUILDS[modal_app.DEFAULT_DOLPHIN_BUILD]
    assert default.exe == DEFAULT_DOLPHIN_PATH
    assert modal_app.ISO_PATH == DEFAULT_ISO_PATH
    assert f"melee=={MELEE_VERSION}" == modal_app.LIBMELEE_PIN
    exes = {build.exe for build in modal_app.DOLPHIN_BUILDS.values()}
    for path in (DOLPHIN_CPU, DOLPHIN_SELF):
        config = load_config(path)
        assert config.env.dolphin.dolphin_path in exes
        assert config.env.dolphin.iso_path == modal_app.ISO_PATH
        assert modal_app.config_env_type(path) == "dolphin"
