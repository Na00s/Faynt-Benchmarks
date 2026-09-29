"""vladfi1's Phillip as a live opponent (P32, 18 Sep 2026).

`vladfi1/phillip <https://github.com/vladfi1/phillip>`_ (GPL-3.0, pinned at ``114aea4c``) is the 2017-2019
deep-RL Melee agent that preceded slippi-ai: a small actor-critic over a hand-built game-state embedding
that picks one of 30-54 discrete controller actions every ``act_every`` frames.  Its code needs
``tensorflow-cpu==2.13`` and therefore Python 3.11, so it runs in a **sidecar process** with its own
interpreter (``melee_rl/phillip_sidecar.py``) and talks to the recorder over a line protocol; our side
converts the libmelee gamestate into upstream's ``GameMemory`` fields and applies the controller upstream
sends back.  Nothing of upstream is copied: the sidecar uses four public names of the package.

What these tests pin is *our* half: the config, the params reader (upstream's ``load_params`` merge), the
field conversion (character ids, jumps used, the derived smash-charge flag, the clamps), the line
protocol end to end against the real sidecar script driven by a stub ``phillip`` package
(``tests/rl/_phillip_stub``), the per-frame gating, the error policy and the health counters.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from melee_rl.env.dolphin import ControllerRow, neutral_row
from melee_rl.phillip import (
    ACT_FROM_FRAME,
    DELAY0_AGENTS,
    IN_REPO_AGENTS,
    MAX_ACTION_STATE,
    PHILLIP_CHARACTERS,
    PHILLIP_LICENSE,
    PHILLIP_REVISION,
    PROTOCOL_VERSION,
    SIDECAR_SCRIPT,
    SMASH_ATTACK_ACTIONS,
    STAGE_IDS,
    UNSELECTED_CHARACTER,
    PhillipConfig,
    PhillipPlayer,
    PhillipSidecar,
    SidecarError,
    game_message,
    load_max_jumps,
    player_message,
    preflight,
    read_agent_info,
    row_from_controller,
)
from tensor_batch import BUTTON_ORDER

melee = pytest.importorskip("melee", reason="the conversion reads libmelee enums and its character table")

STUB_PACKAGE = Path(__file__).with_name("_phillip_stub") / "phillip"


# ---------------------------------------------------------------------------
# fakes: a libmelee gamestate, and a source tree with the stub package + agents
# ---------------------------------------------------------------------------


class FakePlayerState:
    """The libmelee ``PlayerState`` fields the conversion reads, at upstream-like values."""

    def __init__(self, **overrides: Any) -> None:
        self.character = melee.Character.FOX
        self.position = type("P", (), {"x": 0.0, "y": 0.0})()
        self.percent = 0.0
        self.stock = 4
        self.facing = True
        self.action = melee.Action.STANDING
        self.action_frame = 1
        self.invulnerable = False
        self.hitlag_left = 0
        self.hitstun_frames_left = 0
        self.jumps_left = 2
        self.on_ground = True
        self.speed_air_x_self = 0.0
        self.speed_ground_x_self = 0.0
        self.speed_y_self = 0.0
        self.speed_x_attack = 0.0
        self.speed_y_attack = 0.0
        self.shield_strength = 60.0
        for name, value in overrides.items():
            if name in ("x", "y"):
                setattr(self.position, name, value)
            else:
                setattr(self, name, value)


class FakeGamestate:
    def __init__(
        self,
        *,
        frame: int = 100,
        menu_state: Any = None,
        ports: tuple[int, ...] = (1, 2),
        players: dict[int, FakePlayerState] | None = None,
    ) -> None:
        self.frame = frame
        self.menu_state = melee.Menu.IN_GAME if menu_state is None else menu_state
        self.players = players if players is not None else {port: FakePlayerState() for port in ports}
        self.stage = melee.Stage.FINAL_DESTINATION


def write_params(root: Path, name: str, **params: Any) -> Path:
    """An ``agents/<name>/params`` file in upstream's newer flat shape unless ``params`` says otherwise."""

    directory = root / "agents" / name
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"char": "fox", "act_every": 2, "action_type": "custom", "fix_scopes": True}
    payload.update(params)
    (directory / "params").write_text(json.dumps(payload))
    (directory / "snapshot").write_bytes(b"not a checkpoint")
    return directory


@pytest.fixture
def source_dir(tmp_path: Path) -> Path:
    """A source tree the sidecar can import ``phillip`` from: the stub package plus one delay-0 agent."""

    shutil.copytree(STUB_PACKAGE, tmp_path / "phillip")
    write_params(tmp_path, "delay0/FoxFD")
    return tmp_path


def make_config(source_dir: Path, **overrides: Any) -> PhillipConfig:
    settings: dict[str, Any] = {
        "source_dir": str(source_dir),
        "python": sys.executable,
        "agent": "delay0/FoxFD",
        "call_timeout_s": 30.0,
        "load_timeout_s": 30.0,
    }
    settings.update(overrides)
    return PhillipConfig(**settings)


def step(player: PhillipPlayer, states: list[Any], resets: list[bool] | None = None) -> list[ControllerRow]:
    flags = torch.tensor([False] * len(states) if resets is None else resets, dtype=torch.bool)
    return player.rows(states, flags)


def in_game(frame: int, x_port2: float, x_port1: float = -20.0) -> FakeGamestate:
    return FakeGamestate(
        frame=frame,
        players={1: FakePlayerState(x=x_port1), 2: FakePlayerState(x=x_port2)},
    )


# ---------------------------------------------------------------------------
# pins and tables
# ---------------------------------------------------------------------------


def test_pins_and_character_table() -> None:
    assert len(PHILLIP_REVISION) == 40 and PHILLIP_LICENSE == "GPL-3.0"
    assert PROTOCOL_VERSION == "melee_rl.phillip_sidecar.v1"
    assert SIDECAR_SCRIPT.is_file() and SIDECAR_SCRIPT.name == "phillip_sidecar.py"
    assert len(DELAY0_AGENTS) == 7 and set(DELAY0_AGENTS) <= set(IN_REPO_AGENTS) and len(IN_REPO_AGENTS) == 14
    assert "delay0/FoxFD" in DELAY0_AGENTS and "FalconFalconBF" in DELAY0_AGENTS
    expected = {
        "fox": "FOX",
        "falco": "FALCO",
        "marth": "MARTH",
        "peach": "PEACH",
        "sheik": "SHEIK",
        "falcon": "CPTFALCON",
        "puff": "JIGGLYPUFF",
        "kirby": "KIRBY",
        "zelda": "ZELDA",
        "roy": "ROY",
        "mewtwo": "MEWTWO",
        "luigi": "LUIGI",
        "ganon": "GANONDORF",
        "samus": "SAMUS",
        "bowser": "BOWSER",
        "yoshi": "YOSHI",
        "dk": "DK",
    }
    assert set(PHILLIP_CHARACTERS) == set(expected)
    for name, enum_name in expected.items():
        assert melee.Character(PHILLIP_CHARACTERS[name]).name == enum_name, name
    # upstream ``state_manager.py:185``: the smash-charge byte, read as a flag, is up for the whole
    # forward / up / down smash animation, which libmelee spells FSMASH_HIGH ... DOWNSMASH.
    smash = {getattr(melee.Action, n).value for n in ("FSMASH_HIGH", "FSMASH_LOW", "UPSMASH", "DOWNSMASH")}
    assert smash <= SMASH_ATTACK_ACTIONS and melee.Action.NAIR.value not in SMASH_ATTACK_ACTIONS
    assert MAX_ACTION_STATE == 0x17E and UNSELECTED_CHARACTER == 25 and ACT_FROM_FRAME == -3
    assert STAGE_IDS == {"final_destination": 25, "battlefield": 24}


def test_config_defaults_and_validation(tmp_path: Path) -> None:
    config = PhillipConfig()
    assert config.source_dir == "/opt/phillip/source" and config.python == "/opt/phillip/venv/bin/python"
    assert config.agent == "delay0/FoxFD" and config.player == 1 and config.epsilon == 0.0
    assert config.strict and config.save_cpu and config.act_from_frame == ACT_FROM_FRAME
    assert config.libmelee_port == 2 and config.libmelee_opponent_port == 1
    assert PhillipConfig(player=0).libmelee_port == 1
    assert config.agent_dir == Path("/opt/phillip/source/agents/delay0/FoxFD")
    with pytest.raises(ValueError, match="player"):
        PhillipConfig(player=2)
    with pytest.raises(ValueError, match="epsilon"):
        PhillipConfig(epsilon=1.5)
    for bad in ("", "/abs/agent", "../escape", "delay0/../FoxFD"):
        with pytest.raises(ValueError, match="agent"):
            PhillipConfig(agent=bad)
    with pytest.raises(ValueError, match="timeout"):
        PhillipConfig(call_timeout_s=0.0)
    with pytest.raises(ValueError, match="source_dir"):
        PhillipConfig(source_dir="")


# ---------------------------------------------------------------------------
# the params reader
# ---------------------------------------------------------------------------


def test_agent_info_merges_the_agent_subtable_like_upstream(tmp_path: Path) -> None:
    """Upstream's ``util.load_params(path, 'agent')`` folds the old-style ``agent`` table into the top level;
    the defaults are the ``Option`` defaults of ``RLConfig`` / ``RL`` / ``CPU``."""

    write_params(
        tmp_path,
        "FoxFD0",
        char=None,
        act_every=3,
        memory=1,
        action_type=None,
        fix_scopes=False,
        agent={"epsilon": 0.02, "char": "fox", "enemy_reload": 600},
    )
    # ``write_params`` seeds a flat shape; strip the keys this test wants absent.
    path = tmp_path / "agents" / "FoxFD0" / "params"
    payload = {k: v for k, v in json.loads(path.read_text()).items() if v is not None}
    path.write_text(json.dumps(payload))
    info = read_agent_info(tmp_path / "agents" / "FoxFD0", "FoxFD0")
    assert info.name == "FoxFD0" and info.char == "fox" and info.act_every == 3 and info.memory == 1
    assert info.delay == 0 and info.delay_frames == 0
    assert info.action_type == "diagonal"  # RL.action_type's default
    assert info.stage == "final_destination" and info.stage_id == 25
    assert info.libmelee_character == melee.Character.FOX.value == 1
    assert not info.recurrent and not info.predict


def test_agent_info_reads_the_newer_flat_shape_and_the_delay_in_frames(tmp_path: Path) -> None:
    write_params(
        tmp_path,
        "delay18/FalcoBF",
        char="falco",
        act_every=3,
        delay=6,
        memory=0,
        stage="battlefield",
        recurrent=True,
        predict=True,
        predict_steps=6,
    )
    info = read_agent_info(tmp_path / "agents" / "delay18/FalcoBF", "delay18/FalcoBF")
    assert info.char == "falco" and info.libmelee_character == melee.Character.FALCO.value
    assert info.delay == 6 and info.act_every == 3 and info.delay_frames == 18
    assert info.stage == "battlefield" and info.stage_id == 24
    assert info.recurrent and info.predict


def test_agent_info_names_a_missing_file_and_an_unknown_character(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="params"):
        read_agent_info(tmp_path / "agents" / "nope", "nope")
    write_params(tmp_path, "weird", char="pichu")
    with pytest.raises(ValueError, match="pichu"):
        read_agent_info(tmp_path / "agents" / "weird", "weird")


# ---------------------------------------------------------------------------
# the field conversion
# ---------------------------------------------------------------------------


def test_max_jumps_come_from_libmelees_character_table() -> None:
    jumps = load_max_jumps(melee)
    # libmelee's ``jumps_left`` counts the ground jump too: 2 for Fox on the ground, 6 for Puff.
    assert jumps[melee.Character.FOX] == 2 and jumps[melee.Character.JIGGLYPUFF] == 6
    assert jumps[melee.Character.KIRBY] == 6 and jumps[melee.Character.MARTH] == 2


def test_player_message_maps_every_field_upstream_embeds() -> None:
    jumps = load_max_jumps(melee)
    state = FakePlayerState(
        x=-12.5,
        y=3.0,
        percent=42.7,
        stock=3,
        facing=False,
        action=melee.Action.FSMASH_MID,
        action_frame=7,
        invulnerable=True,
        hitlag_left=3,
        hitstun_frames_left=11,
        jumps_left=1,
        on_ground=False,
        speed_air_x_self=0.5,
        speed_ground_x_self=-0.25,
        speed_y_self=-1.5,
        speed_x_attack=2.0,
        speed_y_attack=-0.75,
        shield_strength=43.5,
    )
    message = player_message(state, melee=melee, max_jumps=jumps)
    assert message["percent"] == 42 and isinstance(message["percent"], int)
    assert message["stock"] == 3 and message["facing"] == -1.0
    assert message["x"] == -12.5 and message["y"] == 3.0 and message["z"] == 0.0
    assert message["action_state"] == melee.Action.FSMASH_MID.value == 0x3C
    assert message["action_counter"] == 0 and message["action_frame"] == 7.0
    assert message["character"] == 10  # upstream's Character.Fox: the CSS id, not libmelee's internal one
    assert message["invulnerable"] is True
    assert message["hitlag_frames_left"] == 3.0 and message["hitstun_frames_left"] == 11.0
    assert message["jumps_used"] == 1  # Fox: 2 jumps, one left
    assert message["charging_smash"] is True and message["in_air"] is True
    assert message["speed_air_x_self"] == 0.5 and message["speed_ground_x_self"] == -0.25
    assert message["speed_y_self"] == -1.5 and message["speed_x_attack"] == 2.0
    assert message["speed_y_attack"] == -0.75 and message["shield_size"] == 43.5
    assert message["cursor_x"] == 0.0 and message["cursor_y"] == 0.0
    # Every key is a field of upstream's PlayerMemory struct the sidecar assigns; nothing else.
    assert set(message) == {
        "percent",
        "stock",
        "facing",
        "x",
        "y",
        "z",
        "action_state",
        "action_counter",
        "action_frame",
        "character",
        "invulnerable",
        "hitlag_frames_left",
        "hitstun_frames_left",
        "jumps_used",
        "charging_smash",
        "in_air",
        "speed_air_x_self",
        "speed_ground_x_self",
        "speed_y_self",
        "speed_x_attack",
        "speed_y_attack",
        "shield_size",
        "cursor_x",
        "cursor_y",
    }


def test_player_message_clamps_and_falls_back() -> None:
    jumps = load_max_jumps(melee)
    from melee.gamestate import UnknownAnimation

    odd = FakePlayerState(action=UnknownAnimation(0x999), character=melee.Character.SANDBAG, jumps_left=9)
    message = player_message(odd, melee=melee, max_jumps=jumps)
    assert message["action_state"] == MAX_ACTION_STATE
    assert message["character"] == UNSELECTED_CHARACTER  # from_internal has no CSS id for the sandbag
    assert message["jumps_used"] == 0  # never negative
    assert message["charging_smash"] is False
    puff = FakePlayerState(character=melee.Character.JIGGLYPUFF, jumps_left=6, action=melee.Action.NAIR)
    assert player_message(puff, melee=melee, max_jumps=jumps)["jumps_used"] == 0
    assert player_message(puff, melee=melee, max_jumps=jumps)["character"] == 20  # upstream's Jiggs
    assert player_message(puff, melee=melee, max_jumps=jumps)["charging_smash"] is False


def test_game_message_orders_the_players_by_port() -> None:
    jumps = load_max_jumps(melee)
    gamestate = FakeGamestate(frame=-5, players={1: FakePlayerState(x=-30.0), 2: FakePlayerState(x=30.0)})
    message = game_message(gamestate, melee=melee, max_jumps=jumps)
    assert message["frame"] == -5 and message["menu"] == 2 and message["stage"] == 0
    assert [p["x"] for p in message["players"]] == [-30.0, 30.0]


def test_row_from_controller_follows_the_button_order() -> None:
    row = row_from_controller(
        {"buttons": {"A": True, "L": True, "START": True}, "main": [0.04, 0.31], "c": [0.5, 0.5]}
    )
    assert row.buttons == tuple(name in ("A", "L") for name in BUTTON_ORDER)
    assert (row.main_x, row.main_y) == (0.04, 0.31) and (row.c_x, row.c_y) == (0.5, 0.5)
    assert row.shoulder == 0.0  # upstream's L is a digital press; the analog shoulder is never set
    with pytest.raises(ValueError, match="buttons"):
        row_from_controller({"buttons": {"Q": True}, "main": [0.5, 0.5], "c": [0.5, 0.5]})


# ---------------------------------------------------------------------------
# the line protocol, end to end against the real sidecar script
# ---------------------------------------------------------------------------


def test_the_sidecar_answers_hello_with_its_versions(source_dir: Path) -> None:
    sidecar = PhillipSidecar([sys.executable, str(SIDECAR_SCRIPT)], source_dir=str(source_dir))
    try:
        hello = sidecar.call("hello")
        assert hello["ok"] is True and hello["protocol"] == PROTOCOL_VERSION
        assert hello["python"].startswith(f"{sys.version_info.major}.{sys.version_info.minor}")
        assert hello["phillip"].startswith(str(source_dir))  # the stub package, from PYTHONPATH
        assert hello["tensorflow"] is None or isinstance(hello["tensorflow"], str)
    finally:
        sidecar.close()
    assert sidecar.returncode is not None


def test_the_player_loads_and_acts_through_the_sidecar(source_dir: Path) -> None:
    player = PhillipPlayer(make_config(source_dir), 2, melee=melee)
    try:
        assert player.info.char == "fox" and player.info.act_every == 2
        assert player.load["restore"]["missing"] == [] and player.load["restore"]["padded"] == []
        assert player.load["config"]["act_every"] == 2 and player.load["config"]["delay"] == 0
        assert any("Restoring from" in line for line in player.load["restore"]["lines"])
        # Frame 1: env 0's Phillip (port 2) stands at x = +10 -> A; env 1's at x = -10 -> stick left.
        rows = step(player, [in_game(0, 10.0), in_game(0, -10.0)])
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True and rows[0].main_x == 0.5
        assert rows[1].buttons[BUTTON_ORDER.index("A")] is False and rows[1].main_x == 0.0
        # Frame 2 is inside the two-frame action chain: the controller repeats although the state flipped.
        rows = step(player, [in_game(1, -10.0), in_game(1, 10.0)])
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True and rows[1].main_x == 0.0
        # Frame 3 decides again on the flipped state.
        rows = step(player, [in_game(2, -10.0), in_game(2, 10.0)])
        assert rows[0].main_x == 0.0 and rows[1].buttons[BUTTON_ORDER.index("A")] is True
        stats = player.stats()
        assert stats["frames"] == 6 and stats["live_frames"] == 6 and stats["decisions"] == 4
        assert stats["exceptions"] == 0 and stats["neutral_frames"] == 0
        assert stats["agent"] == "delay0/FoxFD" and stats["char"] == "fox" and stats["delay_frames"] == 0
        assert stats["revision"] == PHILLIP_REVISION and stats["epsilon"] == 0.0
        assert stats["sidecar"]["envs"][0]["frames"] == 3 and stats["sidecar"]["envs"][0]["decisions"] == 2
    finally:
        player.close()


def test_menu_frames_early_frames_and_missing_ports_are_neutral_and_never_reach_the_agent(
    source_dir: Path,
) -> None:
    player = PhillipPlayer(make_config(source_dir), 1, melee=melee)
    try:
        menu = FakeGamestate(menu_state=melee.Menu.CHARACTER_SELECT)
        early = in_game(ACT_FROM_FRAME - 1, 10.0)
        one_port = FakeGamestate(ports=(1,))
        for state in (menu, early, one_port, None):
            assert step(player, [state]) == [neutral_row()]
        assert player.stats()["frames"] == 4 and player.stats()["live_frames"] == 0
        assert player.stats()["sidecar"]["envs"][0]["frames"] == 0
        # The first frame upstream acts on is its 121st in-game frame: Slippi's frame -3.
        rows = step(player, [in_game(ACT_FROM_FRAME, 10.0)])
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True
    finally:
        player.close()


def test_a_reset_rebuilds_the_agent_and_neutralises_its_controller(source_dir: Path) -> None:
    player = PhillipPlayer(make_config(source_dir), 2, melee=melee)
    try:
        step(player, [in_game(0, 10.0), in_game(0, 10.0)])
        assert player.stats()["sidecar"]["envs"] == [
            {"frames": 1, "decisions": 1, "sends": 1, "actions": {"1": 1}},
            {"frames": 1, "decisions": 1, "sends": 1, "actions": {"1": 1}},
        ]
        rows = step(player, [in_game(1, 10.0), menu_state_frame()], resets=[True, False])
        # Env 0 was reset and acted on this frame with a fresh agent (its chain restarted: a decision);
        # env 1 is in a menu and neutral.
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True and rows[1] == neutral_row()
        envs = player.stats()["sidecar"]["envs"]
        assert envs[0] == {"frames": 1, "decisions": 1, "sends": 1, "actions": {"1": 1}}
        assert envs[1]["frames"] == 1
        player.reset_all()
        assert all(env["frames"] == 0 for env in player.stats()["sidecar"]["envs"])
    finally:
        player.close()


def menu_state_frame() -> FakeGamestate:
    return FakeGamestate(menu_state=melee.Menu.STAGE_SELECT)


def test_a_restore_that_initialises_variables_is_refused(source_dir: Path) -> None:
    """``tfl.restore`` prints ``<var> not in checkpoint, initializing`` and carries on with random weights;
    an agent loaded that way is not the released agent, so the player refuses it whatever ``strict`` says."""

    write_params(source_dir, "delay0/FoxFD", stub_missing=["actor/layer_0/weight:0"])
    with pytest.raises(SidecarError, match=r"actor/layer_0/weight:0"):
        PhillipPlayer(make_config(source_dir, strict=False), 1, melee=melee)
    write_params(source_dir, "delay0/FoxFD", stub_padded=["actor/layer_1/weight:0"])
    with pytest.raises(SidecarError, match="padded"):
        preflight(make_config(source_dir))


def test_preflight_reports_the_load_and_closes_the_sidecar(source_dir: Path) -> None:
    report = preflight(make_config(source_dir))
    assert report["agent"] == "delay0/FoxFD" and report["restore"]["missing"] == []
    assert report["hello"]["protocol"] == PROTOCOL_VERSION and report["info"]["char"] == "fox"
    assert report["seconds"] >= 0.0 and report["decision_ms"] >= 0.0


def test_strict_lets_a_sidecar_exception_out_and_lenient_counts_it(source_dir: Path) -> None:
    write_params(source_dir, "delay0/FoxFD", stub_raise_at=2)
    strict = PhillipPlayer(make_config(source_dir), 1, melee=melee)
    try:
        step(strict, [in_game(0, 10.0)])
        with pytest.raises(SidecarError, match="told to fail"):
            step(strict, [in_game(1, 10.0)])
    finally:
        strict.close()
    lenient = PhillipPlayer(make_config(source_dir, strict=False), 1, melee=melee)
    try:
        step(lenient, [in_game(0, 10.0)])
        rows = step(lenient, [in_game(1, 10.0)])
        assert rows == [neutral_row()]
        stats = lenient.stats()
        assert stats["exceptions"] == 1 and "told to fail" in stats["first_error"]
        assert stats["neutral_frames"] == 1 and stats["exception_envs"] == [0]
        # The sidecar survived its exception: the next frame plays on.
        rows = step(lenient, [in_game(2, 10.0)])
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True
    finally:
        lenient.close()


def test_a_load_that_overruns_its_timeout_is_an_error_and_the_process_is_killed(source_dir: Path) -> None:
    write_params(source_dir, "delay0/FoxFD", stub_sleep_s=5.0)
    config = make_config(source_dir, load_timeout_s=0.5)
    started = time.perf_counter()
    with pytest.raises(SidecarError, match="timed out"):
        PhillipPlayer(config, 1, melee=melee)
    assert time.perf_counter() - started < 4.0


def test_a_sidecar_that_dies_is_reported_with_its_stderr(source_dir: Path) -> None:
    sidecar = PhillipSidecar(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"],
        source_dir=str(source_dir),
    )
    with pytest.raises(SidecarError, match="boom"):
        sidecar.call("hello")
    sidecar.close()


def test_the_player_checks_the_agent_directory_before_starting_anything(source_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="params"):
        PhillipPlayer(make_config(source_dir, agent="delay0/Nope"), 1, melee=melee)


def test_sticks_are_sent_with_upstreams_two_decimal_pipe_formatting(source_dir: Path) -> None:
    """``Pad.tilt_stick`` writes ``SET MAIN {:.2f} {:.2f}``: the game upstream trained against saw two
    decimals, so the sidecar rounds the same way before the value reaches libmelee."""

    write_params(source_dir, "delay0/FoxFD", stub_stick=[0.0380602, 0.3086583])
    player = PhillipPlayer(make_config(source_dir), 1, melee=melee)
    try:
        rows = step(player, [in_game(0, -10.0)])
        assert (rows[0].main_x, rows[0].main_y) == (0.04, 0.31)
    finally:
        player.close()


# ---------------------------------------------------------------------------
# the opponent the rollout worker sees
# ---------------------------------------------------------------------------


def test_opponent_port_is_the_environment_player_index_and_act_state_carries_the_rows(
    source_dir: Path,
) -> None:
    from melee_rl.opponents import PhillipOpponent

    player = PhillipPlayer(make_config(source_dir, player=1), 2, melee=melee)
    states: list[Any] = [in_game(0, 10.0), in_game(0, -10.0)]
    opponent = PhillipOpponent(player, lambda: states, port=player.config.player)
    try:
        assert opponent.port == 1 and opponent.controls_port is True
        assert player.config.libmelee_port == 2
        control = opponent.act_state(None, torch.zeros(2, dtype=torch.bool))
        assert control.buttons.A.tolist() == [True, False]
        assert control.main_stick.x.tolist() == pytest.approx([0.5, 0.0])
        with pytest.raises(NotImplementedError, match="act_state"):
            opponent.act(None, torch.zeros(2, dtype=torch.bool))
        opponent.refresh(object())  # type: ignore[arg-type]  # never trained: a no-op
        opponent.reset_state()
        assert all(env["frames"] == 0 for env in player.stats()["sidecar"]["envs"])
    finally:
        opponent.close()
    assert player.sidecar.returncode is not None


def test_player_zero_swaps_upstreams_perspective(source_dir: Path) -> None:
    """Upstream's ``Agent`` plays player index 1 unless ``swap``; on port 1 the sidecar passes ``swap = 1``
    and the agent reads itself at index 0."""

    player = PhillipPlayer(make_config(source_dir, player=0), 1, melee=melee)
    try:
        assert player.load["swap"] == 1
        rows = step(player, [in_game(0, x_port2=-10.0, x_port1=10.0)])
        assert rows[0].buttons[BUTTON_ORDER.index("A")] is True  # its own player (port 1) is on the right
    finally:
        player.close()
