"""P27: vladfi1/slippi-ai's released agents as evaluation opponents.

The upstream runtime is faked wholesale here -- the real one needs TensorFlow, which only the Modal
image carries -- so what these tests pin is *our* side of the boundary: the released-model registry,
the per-environment parser and observation filter, the one batched ``agent.step`` per frame, the
per-environment controller decode, the hold-the-controller rule for an environment that is not in a
game, and the health counters a campaign is read through.

The one test that uses the real world is the analog audit at the bottom: slippi-ai's stick values are
already on Melee's 160-unit grid, so libmelee 0.47.3's ``fix_analog_inputs`` remap -- the bug that
cost SmashBot 1.9 stocks a game (P26) -- has to be a no-op for them.  That is measured against the
real libmelee rather than argued from the arithmetic.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pytest
import torch

from melee_rl.env.dolphin import PORTS, DolphinEnvConfig, neutral_row
from melee_rl.slippi_ai_agent import (
    CHARACTER_IDS,
    DEFAULT_MODEL_DIR,
    DEFAULT_SOURCE_DIR,
    INITIAL_FRAME,
    RELEASED_MODELS,
    SLIPPI_AI_D0_REVISION,
    SLIPPI_AI_LICENSE,
    SLIPPI_AI_REVISION,
    SlippiAIApi,
    SlippiAIConfig,
    SlippiAIPlayer,
    checkout_dir,
    load_slippi_ai_api,
    released_model,
)
from tensor_batch import BUTTON_ORDER

# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------


def test_the_pin_is_the_repositorys_own_slippi_ai_revision() -> None:
    """``third_party/VERSIONS.txt`` and E010 both pin this SHA; the checkpoints were read at it."""

    assert SLIPPI_AI_REVISION == "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
    assert SLIPPI_AI_LICENSE == "MIT"


DROPBOX_RELEASES: frozenset[str] = frozenset(
    {
        "medium-v2",
        "diamond",
        "master",
        "gm",
        "fox_d18_imitation_v3",
        "fox_d18_ditto_v4",
        "fox_d21_ditto_v4",
        "fox_d24_ditto_v4",
        "falco_d18_vs_fox_v4",
        "falco_d18_imitation_v3",
        "sheik_d18_imitation_v4",
        "ics_d21_imitation_v5",
    }
)
"""The twelve files downloaded from the author's public ``deployed_models`` folder (9-10 Sep 2026)."""
SHARED_D0_FOX: str = "fox_d0_tx_like_3x512"
"""The delay-0 Fox the author shared with the team privately on 22 Sep 2026 (not in the folder)."""


def test_the_registry_holds_the_twelve_measured_releases_and_the_shared_delay_zero_fox() -> None:
    assert set(RELEASED_MODELS) == DROPBOX_RELEASES | {SHARED_D0_FOX}


def test_the_shared_delay_zero_fox_is_a_jax_imitation_release() -> None:
    """Every field was read out of the file on 22 Sep 2026 with the stub unpickler, not from its name.

    It carries ``counters`` / ``dataset_metrics`` and no ``rl_config`` -- the JAX *imitation* trainer's
    save (``slippi_ai/jax/train_lib.py:414-435``), so it is behaviour cloning only; ``platform: 'jax'``
    with flax's nested parameter dict, so neither the TensorFlow loader nor our PyTorch port reads it.
    """

    model = released_model(SHARED_D0_FOX)
    assert model.byte_length == 50_676_427
    assert model.sha256 == "2e085fca35ae2405595577d15b30fefcc13289ffab7c863951d75fbb6ad703b7"
    assert model.delay == 0
    assert model.characters == ("fox",)
    assert model.kind == "imitation"
    assert model.platform == "jax"
    assert (model.hidden_size, model.num_layers) == (512, 3)
    assert model.rl_steps is None and model.teacher is None and model.names == ()
    assert model.imitation_steps == 532_176
    assert model.imitation_frames == 43_595_857_920
    assert model.default_name == "Master Player"
    # Fox-ditto data only (``allowed_opponents: 'fox'``): the campaign is a Fox mirror.
    assert model.opponent == "fox"
    assert "22 Sep 2026" in model.note
    # The file was trained on a main snapshot of March 2026 (the bracket 1f0a791..576c8ab^ of the
    # 22 May rebase; the checkout is its last commit before the top-level webdataset import): its
    # `_item_linear` embed parameter was
    # removed upstream on 23 Mar (`3885040`, a SEMANTIC change: sum(linear(items)) became
    # sum(where(exists, mlp(items), 0))) and its `counters` predate `train_epoch` (`576c8ab`, 16 Mar).
    # The pinned tree cannot restore it (measured 22 Sep 2026: "key in pure_dict not available in
    # state: ('network', '_embed_module', '_item_linear', 'bias')"), so it plays on its own checkout.
    assert model.revision == SLIPPI_AI_D0_REVISION == "9eca7479a9555f4ee4b7e6800c656a7f5bb55902"
    assert {RELEASED_MODELS[name].revision for name in DROPBOX_RELEASES} == {SLIPPI_AI_REVISION}


def test_every_dropbox_release_is_a_tensorflow_file() -> None:
    assert {RELEASED_MODELS[name].platform for name in DROPBOX_RELEASES} == {"tf"}
    assert {model.platform for model in RELEASED_MODELS.values()} == {"tf", "jax"}
    counted = [name for name, model in RELEASED_MODELS.items() if model.imitation_steps is not None]
    assert counted == [SHARED_D0_FOX], "the Dropbox files record no imitation step count"


@pytest.mark.parametrize(
    ("name", "byte_length", "sha256", "delay"),
    [
        (
            "falco_d18_imitation_v3",
            42_330_281,
            "9d48e885d5591a630002116eca038bb1c6574988d5897f748c8bac614457524e",
            18,
        ),
        (
            "sheik_d18_imitation_v4",
            42_330_241,
            "4e032eecc54ae94b0918d147764c987b68b7604a993c4805146122aa480c1311",
            18,
        ),
        (
            "ics_d21_imitation_v5",
            45_534_828,
            "845930e86babe16b297dd80ff58799056dd5e06438d34aa8de7d0043c012329d",
            21,
        ),
    ],
)
def test_the_three_imitation_releases_added_for_bc_versus_bc(
    name: str, byte_length: int, sha256: str, delay: int
) -> None:
    """Read out of the files on 10 Sep 2026; each carries "Master Player" in its name map."""

    model = released_model(name)
    assert (model.byte_length, model.sha256, model.delay) == (byte_length, sha256, delay)
    assert (model.kind, model.rl_steps, model.default_name) == ("imitation", None, "Master Player")


def test_the_imitation_releases_start_as_their_own_character() -> None:
    assert SlippiAIConfig(model="falco_d18_imitation_v3").resolved_character == 22  # melee FALCO
    assert SlippiAIConfig(model="sheik_d18_imitation_v4").resolved_character == 7  # melee SHEIK
    assert SlippiAIConfig(model="ics_d21_imitation_v5").resolved_character == 10  # melee POPO


@pytest.mark.parametrize(
    ("name", "byte_length", "sha256", "delay"),
    [
        ("medium-v2", 95_559_707, "48dcfd87c52fde9899fb37b0293ce2a597fb2384b7b40baf7fb49c760152a96c", 21),
        ("diamond", 95_559_704, "7650af84ab02a55f46f6278ec70030f2637c64e3b1538f35691c748455792312", 21),
        ("master", 95_559_704, "0c326f2ab772a926e9b13adac9ec78fe166a8d7b9350f75c5b8ed9e2170cc249", 21),
        ("gm", 95_559_733, "f03b1ae18a0cce1ed7b7a5b6db6f8a1e49a35344d72e39463780adbba43b8e27", 21),
        (
            "fox_d21_ditto_v4",
            42_332_034,
            "f4854fc783246b42d4bfc9309d6964c174a388f3425bb21f3f79355a4562e727",
            21,
        ),
        (
            "fox_d18_ditto_v4",
            42_332_052,
            "fdb96e96fed9f4320c28649351bd7318e9e7f2d1ab8a376af19595e9efd76f32",
            18,
        ),
        (
            "fox_d24_ditto_v4",
            42_332_052,
            "38ff0851e920f30d15e9974b621aa730e4a13f0e117132596afa3ecd3161c4bb",
            24,
        ),
        (
            "fox_d18_imitation_v3",
            42_078_960,
            "d0ec155079b55795b97dcec5e17bfa695bcbac9d5f9d533e11d8b3cdde260684",
            18,
        ),
        (
            "falco_d18_vs_fox_v4",
            42_330_517,
            "87983edf03d1e7133ef863821845217a29a489cc3fe0b3820c3e7982e87244f6",
            18,
        ),
    ],
)
def test_each_release_carries_the_identity_we_measured(
    name: str, byte_length: int, sha256: str, delay: int
) -> None:
    model = released_model(name)
    assert (model.byte_length, model.sha256, model.delay) == (byte_length, sha256, delay)


def test_the_tier_files_are_one_rl_ladder_at_a_constant_delay() -> None:
    """``medium-v2`` -> ``diamond`` -> ``master`` -> ``gm``: more RL, looser leash, same teacher."""

    ladder = [released_model(name) for name in ("medium-v2", "diamond", "master", "gm")]
    assert [model.rl_steps for model in ladder] == [6829, 14374, 26765, 32344]
    assert [model.kl_teacher_weight for model in ladder] == [0.05, 0.02, 0.01, 0.005]
    assert {model.teacher for model in ladder} == {"pickled_models/top12_d21_imitation_3x768_v5"}
    assert {(model.hidden_size, model.num_layers, model.delay) for model in ladder} == {(768, 3, 21)}
    assert all(len(model.characters) == 12 for model in ladder)


def test_the_fox_ditto_ladder_confounds_delay_with_training() -> None:
    """The report has to say this: their steps rise with the delay, so the slope is not pure delay."""

    ladder = [released_model(f"fox_d{delay}_ditto_v4") for delay in (18, 21, 24)]
    assert [model.delay for model in ladder] == [18, 21, 24]
    assert [model.rl_steps for model in ladder] == [19551, 26078, 37693]


def test_the_five_imitation_releases_are_the_only_ones_without_rl() -> None:
    imitation = [name for name, model in RELEASED_MODELS.items() if model.kind != "rl"]
    assert imitation == [
        "fox_d18_imitation_v3",
        "falco_d18_imitation_v3",
        "sheik_d18_imitation_v4",
        "ics_d21_imitation_v5",
        SHARED_D0_FOX,
    ]
    assert all(RELEASED_MODELS[name].kind == "imitation" for name in imitation)
    assert all(RELEASED_MODELS[name].rl_steps is None for name in imitation)


def test_the_falco_specialist_names_the_character_it_was_trained_against() -> None:
    model = released_model("falco_d18_vs_fox_v4")
    assert model.characters == ("falco",)
    assert model.opponent == "fox"
    assert model.default_name == "Ginger"


def test_the_fox_rl_releases_default_to_codys_nametag() -> None:
    """``build_delayed_agent`` takes ``rl_name[0]`` when no name is given; order is load-bearing."""

    assert released_model("fox_d21_ditto_v4").names == ("Cody", "Aklo", "Hax", "SFAT")
    assert released_model("fox_d21_ditto_v4").default_name == "Cody"
    assert released_model("gm").default_name == "Master Player"


def test_an_unknown_model_is_refused_by_name() -> None:
    with pytest.raises(KeyError, match="fox_d0_ditto_v9"):
        released_model("fox_d0_ditto_v9")


def test_no_dropbox_release_has_a_low_delay_and_the_shared_fox_is_the_only_delay_zero() -> None:
    """Probed every delay 0-24 on 9 Sep 2026: the deployed folder is 18/21/24 and nothing else.  The
    delay-0 file came from the author directly, so it is the first symmetric fight for our next-frame
    policy and every table has to say which delay each side played."""

    assert min(RELEASED_MODELS[name].delay for name in DROPBOX_RELEASES) == 18
    assert [name for name, model in RELEASED_MODELS.items() if model.delay == 0] == [SHARED_D0_FOX]
    with pytest.raises(ValueError, match="console_delay"):
        SlippiAIConfig(model=SHARED_D0_FOX, console_delay=1)
    assert SlippiAIConfig(model=SHARED_D0_FOX).effective_delay == 0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_the_default_config_is_the_fox_ditto_on_port_two_at_zero_console_delay() -> None:
    config = SlippiAIConfig()
    assert config.model == "fox_d21_ditto_v4"
    assert config.player == 1
    assert config.libmelee_port == PORTS[1] == 2
    assert config.libmelee_opponent_port == PORTS[0] == 1
    assert config.console_delay == 0
    assert config.model_dir == DEFAULT_MODEL_DIR


def test_the_effective_delay_is_the_checkpoints_delay_less_the_console_delay() -> None:
    """``DelayedAgent.delay = policy.delay - console_delay``; our env has no console lag, so 0."""

    assert SlippiAIConfig(model="fox_d18_ditto_v4").effective_delay == 18
    assert SlippiAIConfig(model="fox_d18_ditto_v4", console_delay=2).effective_delay == 16


def test_the_model_path_is_the_model_dir_plus_the_release_name() -> None:
    config = SlippiAIConfig(model="gm", model_dir="/runs/external/slippi-ai")
    assert config.model_path == "/runs/external/slippi-ai/gm"


def test_an_explicit_model_path_wins_over_the_directory() -> None:
    config = SlippiAIConfig(model="gm", model_path_override="/tmp/gm-copy")
    assert config.model_path == "/tmp/gm-copy"


def test_the_character_defaults_to_the_only_one_the_release_plays() -> None:
    assert SlippiAIConfig(model="fox_d21_ditto_v4").resolved_character == 1  # melee FOX
    assert SlippiAIConfig(model="falco_d18_vs_fox_v4").resolved_character == 22  # melee FALCO


def test_a_twelve_character_release_needs_the_character_named() -> None:
    """``gm`` plays twelve; the campaign plays the Fox ditto, and silence must not pick for us."""

    with pytest.raises(ValueError, match="character"):
        _ = SlippiAIConfig(model="gm").resolved_character
    assert SlippiAIConfig(model="gm", character=1).resolved_character == 1


def test_the_character_table_matches_the_installed_libmelee() -> None:
    """The table exists so importing this module does not need libmelee; a test keeps it honest."""

    melee = pytest.importorskip("melee")
    for name, value in CHARACTER_IDS.items():
        assert melee.Character[name.upper()].value == value, name
    assert set(CHARACTER_IDS) >= set(released_model("gm").characters)


def test_a_character_the_release_cannot_play_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot play"):
        SlippiAIConfig(model="fox_d21_ditto_v4", character=22)


@pytest.mark.parametrize("player", [-1, 2])
def test_the_player_index_must_be_an_environment_player(player: int) -> None:
    with pytest.raises(ValueError, match="player"):
        SlippiAIConfig(player=player)


def test_a_negative_console_delay_is_refused() -> None:
    with pytest.raises(ValueError, match="console_delay"):
        SlippiAIConfig(console_delay=-1)


def test_a_console_delay_past_the_policys_own_delay_is_refused() -> None:
    """Upstream raises inside the agent; it is cheaper to refuse the run file."""

    with pytest.raises(ValueError, match="console_delay"):
        SlippiAIConfig(model="fox_d18_ditto_v4", console_delay=19)


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_a_non_positive_sample_temperature_is_refused(temperature: float) -> None:
    with pytest.raises(ValueError, match="sample_temperature"):
        SlippiAIConfig(sample_temperature=temperature)


def test_an_unknown_model_is_refused_at_config_time() -> None:
    with pytest.raises(ValueError, match="model"):
        SlippiAIConfig(model="fox_d0_ditto_v9")


# ---------------------------------------------------------------------------
# the upstream boundary
# ---------------------------------------------------------------------------


def test_loading_the_api_reports_a_checkout_without_the_entry_module(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="eval_lib"):
        load_slippi_ai_api(tmp_path)


# ---------------------------------------------------------------------------
# fakes for the upstream runtime
# ---------------------------------------------------------------------------


class _Stick(NamedTuple):
    x: Any
    y: Any


class _Game(NamedTuple):
    """Stands in for ``slippi_ai.types.Game``: a nest of numpy leaves."""

    frame: Any
    main_stick: _Stick


def _map_nt(function: Any, first: Any, *rest: Any) -> Any:
    """``slippi_ai.utils.map_nt``, copied so the fake behaves like the real tree walker."""

    kind = type(first)
    if kind is tuple:
        return tuple(_map_nt(function, *values) for values in zip(first, *rest, strict=True))
    if issubclass(kind, tuple):
        return kind(*(_map_nt(function, *values) for values in zip(first, *rest, strict=True)))
    if issubclass(kind, dict):
        return {key: _map_nt(function, first[key], *(other[key] for other in rest)) for key in first}
    return function(first, *rest)


class _FakeParser:
    """One per environment, like upstream's ``slippi_db.parse_libmelee.Parser``."""

    instances: ClassVar[list[_FakeParser]] = []

    def __init__(self, ports: tuple[int, int]) -> None:
        self.ports = ports
        self.frames: list[int] = []
        _FakeParser.instances.append(self)

    def get_game(self, gamestate: Any) -> _Game:
        self.frames.append(int(gamestate.frame))
        return _Game(
            frame=np.int32(gamestate.frame),
            main_stick=_Stick(x=np.float32(gamestate.frame / 1000.0), y=np.float32(0.25)),
        )


class _FakeFilter:
    """Upstream's sequential observation filter: stateful, reset at a new game."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.resets = 0
        self.filtered = 0

    def reset(self) -> None:
        self.resets += 1

    def filter(self, game: _Game) -> _Game:
        self.filtered += 1
        return game


class _FakeController(NamedTuple):
    """What ``decode_controller`` hands ``send_controller``."""

    row: int
    press_a: bool


class _FakeAgent:
    """A batched ``DelayedAgent``: one ``step`` per frame, one row per environment."""

    def __init__(self, batch_size: int, delay: int) -> None:
        self.batch_size = batch_size
        self.delay = delay
        self.steps: list[tuple[_Game, np.ndarray]] = []
        self.warmups = 0
        self.started = 0
        self.stopped = 0
        self.raises = False

    def warmup(self) -> None:
        self.warmups += 1

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def step(self, game: _Game, needs_reset: np.ndarray) -> Any:
        if self.raises:
            raise RuntimeError("upstream blew up")
        self.steps.append((game, np.asarray(needs_reset).copy()))

        class _Sample(NamedTuple):
            controller_state: Any

        rows = np.arange(self.batch_size, dtype=np.int32) + len(self.steps) * 100
        return _Sample(controller_state=rows)

    def decode_controller(self, controller: Any) -> _FakeController:
        return _FakeController(row=int(controller), press_a=bool(int(controller) % 2))


class _Recorded(NamedTuple):
    row: int
    press_a: bool


def _fake_api(*, delay: int = 21, state: dict[str, Any] | None = None) -> tuple[SlippiAIApi, dict[str, Any]]:
    """An API whose every call is recorded in the returned ``log``."""

    log: dict[str, Any] = {
        "agents": [],
        "filters": [],
        "sent": [],
        "disable_gpus": 0,
        "loaded": [],
        "device_get": 0,
    }
    _FakeParser.instances = []

    def load_state(path: str) -> dict[str, Any]:
        log["loaded"].append(path)
        return state if state is not None else {"config": {"policy": {"delay": delay}}}

    def device_get(value: Any) -> Any:
        log["device_get"] += 1
        return value

    def build_agent(**kwargs: Any) -> _FakeAgent:
        # Upstream's rule (``eval_lib.build_delayed_agent``): a checkpoint with no RL tags needs a name.
        if kwargs.get("name") is None and not kwargs["state"].get("rl_config"):
            raise ValueError("Must specify an agent name.")
        log["agents"].append(kwargs)
        agent = _FakeAgent(batch_size=kwargs["batch_size"], delay=delay - kwargs["console_delay"])
        agent.raises = bool(log.get("raises", False))
        log["agent"] = agent
        return agent

    def observation_filter_factory(config: Any) -> _FakeFilter:
        made = _FakeFilter(config)
        log["filters"].append(made)
        return made

    def send_controller(recorder: Any, controller: _FakeController) -> None:
        log["sent"].append(controller)
        recorder.tilt_analog("MAIN", 0.5 + controller.row / 10_000.0, 0.5)
        if controller.press_a:
            recorder.press_button("A")
        else:
            recorder.release_button("A")

    def disable_gpus() -> None:
        log["disable_gpus"] += 1

    api = SlippiAIApi(
        load_state=load_state,
        build_agent=build_agent,
        parser_factory=_FakeParser,
        observation_filter_factory=observation_filter_factory,
        send_controller=send_controller,
        disable_gpus=disable_gpus,
        map_nt=_map_nt,
        device_get=device_get,
    )
    return api, log


def _jax_state(delay: int = 0) -> dict[str, Any]:
    """What the shared delay-0 file's config says about itself (``platform`` beside the policy delay)."""

    return {"config": {"policy": {"delay": delay}, "platform": "jax"}}


def _jax_player(
    num_envs: int = 2, *, config: SlippiAIConfig | None = None
) -> tuple[SlippiAIPlayer, dict[str, Any]]:
    api, log = _fake_api(delay=0, state=_jax_state())
    made = SlippiAIPlayer(config or SlippiAIConfig(model=SHARED_D0_FOX), num_envs, api=api)
    return made, log


class _FakePlayerState:
    def __init__(self, character: int = 1) -> None:
        self.character = character


class _FakeGamestate:
    """Just enough libmelee ``GameState`` for the guards."""

    def __init__(self, frame: int, *, menu: str = "IN_GAME", ports: tuple[int, ...] = (1, 2)) -> None:
        self.frame = frame
        self.menu_state = menu
        self.players = {port: _FakePlayerState() for port in ports}


def _player(
    num_envs: int = 2,
    *,
    config: SlippiAIConfig | None = None,
    delay: int = 21,
) -> tuple[SlippiAIPlayer, dict[str, Any]]:
    api, log = _fake_api(delay=delay)
    made = SlippiAIPlayer(config or SlippiAIConfig(), num_envs, api=api)
    return made, log


def _states(frames: list[int], **kwargs: Any) -> list[_FakeGamestate]:
    return [_FakeGamestate(frame, **kwargs) for frame in frames]


def _resets(*flags: bool) -> torch.Tensor:
    return torch.tensor(list(flags), dtype=torch.bool)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_the_player_builds_one_batched_agent_for_every_environment() -> None:
    player, log = _player(num_envs=4)
    assert len(log["agents"]) == 1
    assert log["agents"][0]["batch_size"] == 4
    assert log["agents"][0]["console_delay"] == 0
    assert player.num_envs == 4


def test_the_agent_is_built_from_the_state_on_disk_and_warmed_up() -> None:
    player, log = _player()
    assert log["loaded"] == [player.config.model_path]
    assert log["agent"].warmups == 1


def test_the_default_name_is_upstreams_own_cli_default() -> None:
    """``nametags.DEFAULT_NAME``: an RL release overrides it with its trained tag, as upstream does."""

    _, log = _player()
    assert log["agents"][0]["name"] == "Master Player"
    _, named = _player(config=SlippiAIConfig(name="Cody"))
    assert named["agents"][0]["name"] == "Cody"


def test_an_imitation_release_is_built_with_a_name() -> None:
    """The regression: ``fox_d18_imitation_v3`` has no RL tags, and ``name=None`` makes upstream raise."""

    config = SlippiAIConfig(model="fox_d18_imitation_v3")
    player, log = _player(config=config, delay=18)
    assert log["agents"][0]["name"] == "Master Player"
    assert player.stats()["name"] == "Master Player"


def test_the_sample_temperature_reaches_the_agent() -> None:
    _, log = _player(config=SlippiAIConfig(sample_temperature=0.8))
    assert log["agents"][0]["sample_temperature"] == 0.8


def test_xla_is_off_by_default_and_travels_in_the_framework_dict() -> None:
    """``build_basic_agent`` picks ``tf=`` or ``jax=`` by the policy's platform (E010 ran XLA off)."""

    _, log = _player()
    assert log["agents"][0]["tf"] == {"jit_compile": False}
    _, jitted = _player(config=SlippiAIConfig(jit_compile=True))
    assert jitted["agents"][0]["tf"] == {"jit_compile": True}


def test_tensorflow_is_kept_off_the_gpu_by_default() -> None:
    _, log = _player()
    assert log["disable_gpus"] == 1
    _, on_gpu = _player(config=SlippiAIConfig(use_gpu=True))
    assert on_gpu["disable_gpus"] == 0


def test_a_checkpoint_whose_delay_is_not_the_registrys_is_refused() -> None:
    """The one guard that catches a swapped or re-saved file before a campaign runs."""

    api, _ = _fake_api(delay=12)
    with pytest.raises(RuntimeError, match="delay"):
        SlippiAIPlayer(SlippiAIConfig(model="fox_d21_ditto_v4"), 2, api=api)


def test_the_effective_delay_is_reported_by_the_player() -> None:
    player, _ = _player(config=SlippiAIConfig(model="fox_d18_ditto_v4"), delay=18)
    assert player.delay == 18
    assert player.model.delay == 18


# ---------------------------------------------------------------------------
# the JAX release (the shared delay-0 Fox)
# ---------------------------------------------------------------------------


def test_a_file_whose_platform_is_not_the_registrys_is_refused() -> None:
    """Upstream reads a missing ``platform`` as TensorFlow (``saving.get_platform``), so a JAX file that
    lost its key, or a TensorFlow file registered as JAX, would build the wrong stack: refuse both."""

    api, _ = _fake_api(delay=21, state={"config": {"policy": {"delay": 21}, "platform": "jax"}})
    with pytest.raises(RuntimeError, match="platform"):
        SlippiAIPlayer(SlippiAIConfig(model="fox_d21_ditto_v4"), 2, api=api)
    api, _ = _fake_api(delay=0, state={"config": {"policy": {"delay": 0}}})
    with pytest.raises(RuntimeError, match="platform"):
        SlippiAIPlayer(SlippiAIConfig(model=SHARED_D0_FOX), 2, api=api)
    api, _ = _fake_api(delay=0, state=_jax_state())
    SlippiAIPlayer(SlippiAIConfig(model=SHARED_D0_FOX), 2, api=api)


def test_a_jax_release_is_stepped_with_its_outputs_copied_to_the_host() -> None:
    """Upstream's ``Agent.step`` runs ``jax.device_get`` on a JAX policy's outputs before decoding
    (``eval_lib.py:539-541``); the batched player bypasses ``Agent`` and has to do the same.  A
    TensorFlow release returns host arrays already and must not pay for the copy."""

    player, log = _jax_player(num_envs=2)
    player.rows(_states([INITIAL_FRAME, INITIAL_FRAME]), _resets(True, True))
    player.rows(_states([INITIAL_FRAME + 1, INITIAL_FRAME + 1]), _resets(False, False))
    assert log["device_get"] == 2
    assert len(log["agent"].steps) == 2
    tf_player, tf_log = _player(num_envs=2)
    tf_player.rows(_states([INITIAL_FRAME, INITIAL_FRAME]), _resets(True, True))
    assert tf_log["device_get"] == 0


def test_a_jax_release_keeps_jax_on_the_cpu_unless_the_run_file_asks_for_the_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorder's L4 belongs to our policy.  ``disable_gpus`` is TensorFlow's switch; JAX reads
    ``JAX_PLATFORMS`` at its first backend query, which is the agent build, so the player sets it first."""

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    _, log = _player()
    assert "JAX_PLATFORMS" not in os.environ, "a TensorFlow release does not touch it"
    _, log = _jax_player()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert log["disable_gpus"] == 1, "TensorFlow is imported by the same runtime and stays off the GPU too"
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    _jax_player(config=SlippiAIConfig(model=SHARED_D0_FOX, use_gpu=True))
    assert "JAX_PLATFORMS" not in os.environ
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    _jax_player()
    assert os.environ["JAX_PLATFORMS"] == "cuda", "an operator's explicit choice is kept"


def test_the_platform_and_the_imitation_provenance_are_in_stats() -> None:
    player, _ = _jax_player()
    stats = player.stats()
    assert stats["platform"] == "jax"
    assert stats["kind"] == "imitation"
    assert stats["policy_delay"] == 0 and stats["effective_delay"] == 0
    assert stats["imitation_steps"] == 532_176
    assert stats["name"] == "Master Player"
    tf_player, _ = _player()
    assert tf_player.stats()["platform"] == "tf"
    assert tf_player.stats()["imitation_steps"] is None


def test_a_release_plays_on_the_checkout_of_its_own_revision() -> None:
    """Two upstream trees in the image: the pin for the Dropbox files, the March snapshot for the shared
    Fox.  The run file's ``source_dir`` keeps its meaning as an operator override; at its default the
    registry's revision decides."""

    assert checkout_dir(SLIPPI_AI_REVISION) == DEFAULT_SOURCE_DIR == "/opt/slippi-ai/source"
    assert checkout_dir(SLIPPI_AI_D0_REVISION) == "/opt/slippi-ai/source-9eca7479a955"
    assert SlippiAIConfig(model="fox_d21_ditto_v4").resolved_source_dir == DEFAULT_SOURCE_DIR
    assert SlippiAIConfig(model=SHARED_D0_FOX).resolved_source_dir == "/opt/slippi-ai/source-9eca7479a955"
    assert SlippiAIConfig(model=SHARED_D0_FOX, source_dir="/elsewhere").resolved_source_dir == "/elsewhere"
    player, _ = _jax_player()
    assert player.stats()["revision"] == SLIPPI_AI_D0_REVISION
    tf_player, _ = _player()
    assert tf_player.stats()["revision"] == SLIPPI_AI_REVISION


def test_loading_a_second_upstream_tree_into_one_process_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``import slippi_ai`` is cached per process: a checkout put first on ``sys.path`` after another
    was imported would silently keep the other's modules.  One release per process, or an error."""

    import types

    other = tmp_path / "other"
    (other / "slippi_ai").mkdir(parents=True)
    module = types.ModuleType("slippi_ai")
    module.__file__ = str(other / "slippi_ai" / "__init__.py")
    monkeypatch.setitem(sys.modules, "slippi_ai", module)
    checkout = tmp_path / "checkout"
    (checkout / "slippi_ai").mkdir(parents=True)
    (checkout / "slippi_ai" / "eval_lib.py").write_text("")
    with pytest.raises(RuntimeError, match="already imported"):
        load_slippi_ai_api(checkout)
    # Upstream's package is a NAMESPACE package: no __init__.py, __file__ None, roots in __path__.
    namespace = types.ModuleType("slippi_ai")
    namespace.__file__ = None
    namespace.__path__ = [str(other / "slippi_ai")]
    monkeypatch.setitem(sys.modules, "slippi_ai", namespace)
    with pytest.raises(RuntimeError, match="already imported"):
        load_slippi_ai_api(checkout)
    # The same checkout again is not a conflict (a second player in one process, one release).
    namespace.__path__ = [str(checkout / "slippi_ai")]
    monkeypatch.setattr(sys, "path", list(sys.path))
    with pytest.raises(ImportError):  # past the guard: the fake tree has no modules to import
        load_slippi_ai_api(checkout)


def test_the_shared_fox_plays_under_any_tag_it_was_trained_with() -> None:
    """An imitation file has no RL tags to override the name, so the run file's choice is what plays:
    the campaign runs ``Master Player`` (the anonymised top-tier tag, like every other imitation release)
    and ``Cody`` (the strongest named Fox in its map) as two arms."""

    _, log = _jax_player(config=SlippiAIConfig(model=SHARED_D0_FOX, name="Cody"))
    assert log["agents"][0]["name"] == "Cody"
    _, log = _jax_player()
    assert log["agents"][0]["name"] == "Master Player"


# ---------------------------------------------------------------------------
# one frame
# ---------------------------------------------------------------------------


def test_one_frame_is_one_batched_step_and_one_row_per_environment() -> None:
    player, log = _player(num_envs=2)
    rows = player.rows(_states([INITIAL_FRAME, INITIAL_FRAME]), _resets(True, True))
    assert len(rows) == 2
    assert len(log["agent"].steps) == 1
    game, flags = log["agent"].steps[0]
    assert game.frame.shape == (2,)
    assert flags.tolist() == [True, True]


def test_each_environment_gets_its_own_parser_with_the_two_ports_in_order() -> None:
    config = SlippiAIConfig(player=1)
    player, _ = _player(num_envs=3, config=config)
    player.rows(_states([INITIAL_FRAME] * 3), _resets(True, True, True))
    assert len(_FakeParser.instances) == 3
    assert {parser.ports for parser in _FakeParser.instances} == {(2, 1)}


def test_the_ports_follow_the_player_index() -> None:
    player, _ = _player(num_envs=1, config=SlippiAIConfig(player=0))
    player.rows(_states([INITIAL_FRAME]), _resets(True))
    assert _FakeParser.instances[-1].ports == (1, 2)


def test_a_reset_flag_rebuilds_the_parser_and_resets_the_filter() -> None:
    player, log = _player(num_envs=2)
    player.rows(_states([10, 10]), _resets(True, True))
    built = len(_FakeParser.instances)
    player.rows(_states([11, 11]), _resets(False, False))
    assert len(_FakeParser.instances) == built
    player.rows(_states([INITIAL_FRAME, 12]), _resets(True, False))
    assert len(_FakeParser.instances) == built + 1
    assert [made.resets for made in log["filters"]][:2] == [2, 1]


def test_melees_first_frame_is_a_reset_even_without_the_flag() -> None:
    """Upstream keys on ``gamestate.frame == -123``; a recorder's second game must not inherit state."""

    player, _ = _player(num_envs=1)
    player.rows(_states([5]), _resets(True))
    built = len(_FakeParser.instances)
    player.rows(_states([INITIAL_FRAME]), _resets(False))
    assert len(_FakeParser.instances) == built + 1


def test_the_reset_flag_is_forwarded_to_the_agent_so_it_clears_its_hidden_state() -> None:
    player, log = _player(num_envs=2)
    player.rows(_states([7, 7]), _resets(False, True))
    _, flags = log["agent"].steps[-1]
    assert flags.tolist() == [False, True]


def test_every_environments_frame_goes_through_the_filter_before_the_agent() -> None:
    player, log = _player(num_envs=2)
    player.rows(_states([1, 1]), _resets(True, True))
    player.rows(_states([2, 2]), _resets(False, False))
    assert [made.filtered for made in log["filters"]] == [2, 2]


def test_the_decoded_controller_is_sent_once_per_environment_per_frame() -> None:
    player, log = _player(num_envs=3)
    player.rows(_states([1, 1, 1]), _resets(True, True, True))
    assert [controller.row for controller in log["sent"]] == [100, 101, 102]


def test_the_rows_carry_what_was_sent() -> None:
    player, _ = _player(num_envs=2)
    rows = player.rows(_states([1, 1]), _resets(True, True))
    assert rows[0] != rows[1]
    assert rows[0].main_x == pytest.approx(0.5 + 100 / 10_000.0)
    assert rows[1].main_x == pytest.approx(0.5 + 101 / 10_000.0)


def test_a_controller_is_durable_across_frames() -> None:
    """libmelee's ``Controller.current`` holds; upstream's ``send_controller`` writes every field."""

    player, log = _player(num_envs=1)
    player.rows(_states([1]), _resets(True))
    first = log["sent"][-1]
    rows = player.rows(_states([2]), _resets(False))
    assert rows[0].buttons[BUTTON_ORDER.index("A")] is bool(log["sent"][-1].press_a)
    assert log["sent"][-1].row != first.row


# ---------------------------------------------------------------------------
# environments that are not in a game
# ---------------------------------------------------------------------------


def test_a_menu_frame_holds_the_controller_and_is_counted() -> None:
    player, log = _player(num_envs=2)
    player.rows(_states([1, 1]), _resets(True, True))
    held = player.rows([_FakeGamestate(2), _FakeGamestate(2, menu="POSTGAME_SCORES")], _resets(False, False))
    assert held[1] == player.last_rows[1]
    assert player.stats()["held_frames"] == 1
    # the batch still advances every row: the LSTM and the delay queue stay aligned
    assert len(log["agent"].steps) == 2


def test_a_missing_port_holds_the_controller() -> None:
    player, _ = _player(num_envs=1)
    player.rows(_states([1]), _resets(True))
    before = player.last_rows[0]
    held = player.rows([_FakeGamestate(2, ports=(1,))], _resets(False))
    assert held[0] == before
    assert player.stats()["held_frames"] == 1


def test_a_held_environment_repeats_its_last_observation_in_the_batch() -> None:
    player, log = _player(num_envs=1)
    player.rows(_states([4]), _resets(True))
    player.rows([_FakeGamestate(5, menu="POSTGAME_SCORES")], _resets(False))
    first, second = (step[0].frame.tolist() for step in log["agent"].steps)
    assert first == second == [4]


def test_the_first_frame_of_a_held_environment_is_neutral_and_does_not_crash() -> None:
    player, _ = _player(num_envs=1)
    rows = player.rows([_FakeGamestate(-100, menu="POSTGAME_SCORES")], _resets(True))
    assert rows == [neutral_row()]


def test_a_none_gamestate_is_held() -> None:
    player, _ = _player(num_envs=2)
    player.rows(_states([1, 1]), _resets(True, True))
    rows = player.rows([_FakeGamestate(2), None], _resets(False, False))
    assert rows[1] == player.last_rows[1]


# ---------------------------------------------------------------------------
# resets and shape guards
# ---------------------------------------------------------------------------


def test_reset_all_forgets_every_parser_and_neutralises_every_row() -> None:
    player, _ = _player(num_envs=2)
    player.rows(_states([1, 1]), _resets(True, True))
    player.reset_all()
    assert player.last_rows == [neutral_row(), neutral_row()]
    assert player.stats()["resets"] == 4  # two at the first frame, two from reset_all


def test_a_wrong_number_of_gamestates_is_refused() -> None:
    player, _ = _player(num_envs=2)
    with pytest.raises(ValueError, match="gamestates"):
        player.rows(_states([1]), _resets(True, True))


def test_a_wrong_number_of_reset_flags_is_refused() -> None:
    player, _ = _player(num_envs=2)
    with pytest.raises(ValueError, match="reset flags"):
        player.rows(_states([1, 1]), _resets(True))


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_an_upstream_exception_propagates_under_strict() -> None:
    player, log = _player(num_envs=1)
    player.rows(_states([1]), _resets(True))
    log["agent"].raises = True
    with pytest.raises(RuntimeError, match="upstream blew up"):
        player.rows(_states([2]), _resets(False))
    assert player.stats()["exceptions"] == 1
    assert "upstream blew up" in player.stats()["first_error"]


def test_without_strict_an_exception_neutralises_the_frame_and_is_counted() -> None:
    player, log = _player(num_envs=2, config=SlippiAIConfig(strict=False))
    player.rows(_states([1, 1]), _resets(True, True))
    log["agent"].raises = True
    rows = player.rows(_states([2, 2]), _resets(False, False))
    assert rows == [neutral_row(), neutral_row()]
    assert player.stats()["exceptions"] == 1


def test_the_stats_name_the_release_and_the_delay_the_campaign_ran_at() -> None:
    player, _ = _player(num_envs=2, config=SlippiAIConfig(model="fox_d18_ditto_v4"), delay=18)
    player.rows(_states([1, 1]), _resets(True, True))
    stats = player.stats()
    assert stats["model"] == "fox_d18_ditto_v4"
    assert stats["sha256"] == released_model("fox_d18_ditto_v4").sha256
    assert stats["policy_delay"] == 18
    assert stats["console_delay"] == 0
    assert stats["effective_delay"] == 18
    assert stats["frames_seen"] == 2
    assert stats["name"] == "Cody"
    assert stats["revision"] == SLIPPI_AI_REVISION


def test_the_neutral_share_is_counted_so_a_silent_agent_cannot_hide() -> None:
    """P25's lesson: an opponent that presses nothing looks like a bad opponent, not a broken one."""

    player, _ = _player(num_envs=1)
    player.rows(_states([1]), _resets(True))
    assert player.stats()["neutral_frames"] == 0

    def neutral(recorder: Any, controller: Any) -> None:
        del recorder, controller

    object.__setattr__(player.api, "send_controller", neutral)
    player.rows(_states([2]), _resets(False))
    assert player.stats()["neutral_frames"] == 1


# ---------------------------------------------------------------------------
# the analog audit, against the real libmelee
# ---------------------------------------------------------------------------

AXIS_SPACING = 160
"""``slippi_ai.controller_lib.AXIS_SPACING``: their sticks live on a 161-point observation grid."""
PIPE_SPACING = 254
"""``127 * 2``: the scale of the value libmelee writes to Dolphin's pipe (``controller.py:11-19``)."""


def _raw_to_unit(raw: int) -> float:
    """``slippi_ai.controller_lib.from_raw_axis``: raw [-80, 80] -> observed [0, 1]."""

    return raw / AXIS_SPACING + 0.5


def _observed_raw(pipe_value: float) -> int:
    """What Melee reads back from a pipe value: ``floor((x - 0.5) * 254)``.

    libmelee's own comment for the formula it inverts (``controller.py:13``).
    """

    return math.floor((pipe_value - 0.5) * PIPE_SPACING)


class _FakeConsole:
    is_dolphin = True
    logger = None

    def __init__(self, path: Path) -> None:
        self._path = path

    def get_dolphin_pipes_path(self, port: int) -> str:
        return str(self._path)

    def setup_dolphin_controller(self, port: int, kind: Any) -> None:
        return None


def _written(tmp_path: Path, *, fix: bool, x: float, y: float) -> list[float]:
    melee = pytest.importorskip("melee")
    pipe = tmp_path / f"pipe-{fix}-{x}-{y}"
    controller = melee.Controller(_FakeConsole(pipe), 1, melee.ControllerType.STANDARD, fix_analog_inputs=fix)
    with pipe.open("w") as handle:
        controller.pipe = handle
        controller.tilt_analog(melee.Button.BUTTON_MAIN, x, y)
        controller.pipe = None
    return [float(token) for token in pipe.read_text().split()[2:4]]


@pytest.mark.parametrize("raw", [-80, -53, -24, -23, -1, 0, 1, 23, 24, 53, 80])
def test_the_remap_lands_slippi_ais_stick_values_exactly(tmp_path: Path, raw: int) -> None:
    """``fix_analog_inputs`` is not a rounding: it is the 160 -> 254 conversion they depend on.

    slippi-ai's action space is the *observed* grid (``from_raw_axis`` is ``raw / 160 + 0.5``), while
    libmelee's ``tilt_analog`` takes a *pipe* value, where Melee's +/-80 spans +/-80/254.  The flag
    converts between them, which is why vladfi1 added it and why its docstring promises "the stick
    values from Console.step are consistent with the values you send as inputs".  So unlike SmashBot,
    which wrote pipe values and was wrecked by the conversion (P26), slippi-ai *needs* it on.
    """

    value = _raw_to_unit(raw)
    written = _written(tmp_path, fix=True, x=value, y=value)
    assert [_observed_raw(component) for component in written] == [raw, raw]


def test_the_whole_grid_lands_exactly(tmp_path: Path) -> None:
    """All 161 axis values, so no corner of their action space is assumed from a sample of eleven."""

    for raw in range(-AXIS_SPACING // 2, AXIS_SPACING // 2 + 1):
        written = _written(tmp_path, fix=True, x=_raw_to_unit(raw), y=0.5)
        assert _observed_raw(written[0]) == raw, f"raw={raw} did not land"
        assert _observed_raw(written[1]) == 0, f"raw={raw} moved the neutral axis"


@pytest.mark.parametrize("raw", [-80, -53, -24, 24, 53, 80])
def test_turning_the_remap_off_would_wreck_them(tmp_path: Path, raw: int) -> None:
    """The mirror image of P26: ``legacy_analog_ports`` must stay empty for a slippi-ai port.

    Without the conversion their values are read as ``raw * 254 / 160`` -- 59 % too far in every
    direction, so every tilt, every wavedash angle and every DI would be wrong.
    """

    written = _written(tmp_path, fix=False, x=_raw_to_unit(raw), y=0.5)
    landed = _observed_raw(written[0])
    assert landed != raw
    assert abs(landed) > abs(raw)


def test_a_slippi_ai_campaign_leaves_every_port_remapped() -> None:
    """Our default; stated as a test so a future ``legacy_analog_ports`` edit cannot silently include them."""

    assert DolphinEnvConfig().legacy_analog_ports == ()


# ---------------------------------------------------------------------------
# the preflight (22 Sep 2026): the agent built and stepped before any Dolphin is paid for
# ---------------------------------------------------------------------------


def test_the_preflight_builds_the_agent_and_plays_synthetic_frames_through_the_real_parser_path() -> None:
    """P25's lesson, P32's shape: a bad pip layer, a missing checkout or a file that will not restore is
    a ten-second error here and a forty-minute one after a pool of Dolphins has booted.  The frames are
    real libmelee ``GameState`` objects, so upstream's parser reads exactly what the recorder feeds it."""

    from melee_rl.slippi_ai_agent import format_preflight, preflight

    api, log = _fake_api(delay=0, state=_jax_state())
    report = preflight(SlippiAIConfig(model=SHARED_D0_FOX), frames=30, num_envs=2, api=api)
    assert report["model"] == SHARED_D0_FOX
    assert report["platform"] == "jax" and report["kind"] == "imitation"
    assert report["frames"] == 30 and report["steps"] == 30 and report["num_envs"] == 2
    assert report["exceptions"] == 0 and report["first_error"] == ""
    assert report["delay"] == 0
    assert report["name"] == "Master Player"
    assert report["sha256_verified"] is False  # no file behind the fake API
    assert report["step_ms"] > 0.0 and report["seconds"] > 0.0
    assert report["neutral_share"] == 0.0, "the fake agent tilts the stick every frame"
    assert report["rows_changed"] > 0, "a stuck controller would read as 'playing' otherwise"
    assert log["device_get"] == 30 and log["agent"].warmups == 1
    # The first synthetic frame is Melee's first frame, so every environment resets once.
    assert report["resets"] == 2
    line = format_preflight(report)
    assert SHARED_D0_FOX in line and "jax" in line and "0 exceptions" in line


def test_the_preflight_reports_an_agent_that_raises_rather_than_hiding_it() -> None:
    from melee_rl.slippi_ai_agent import preflight

    api, log = _fake_api(delay=0, state=_jax_state())
    config = SlippiAIConfig(model=SHARED_D0_FOX, strict=False)
    report = preflight(config, frames=3, num_envs=1, api=api)
    assert report["exceptions"] == 0
    log["raises"] = True  # the next agent the API builds raises on every step
    report = preflight(config, frames=3, num_envs=1, api=api)
    assert report["exceptions"] == 3 and "upstream blew up" in report["first_error"]
    assert report["rows_changed"] == 0
