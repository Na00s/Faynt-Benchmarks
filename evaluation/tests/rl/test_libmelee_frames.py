"""libmelee ``GameState`` -> frame tree converter (PLAN.md §6 P5): exact leaves, Nana, items, stages, guards.

Builds real libmelee 0.47.3 dataclasses (``melee.GameState`` / ``PlayerState`` / ``Projectile``) and never
launches Dolphin; skips when libmelee is not installed (the Modal image has it).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from melee_rl.env import libmelee_frames as frames_lib  # noqa: E402
from melee_rl.env.libmelee_frames import (  # noqa: E402
    MELEE_VERSION,
    FrameBatch,
    FrameConversionError,
    FrameConverter,
    ItemSlots,
    import_melee,
)
from melee_rl.frames import tree_leaves, tree_stack, validate_frame_tree  # noqa: E402
from model import SlippiEncoder  # noqa: E402
from tensor_batch import BUTTON_ORDER, MAX_ITEMS  # noqa: E402

FD = melee.Stage.FINAL_DESTINATION


def _controller_state(
    *,
    main: tuple[float, float] = (0.5, 0.5),
    c: tuple[float, float] = (0.5, 0.5),
    shoulder: float = 0.0,
    processed: Sequence[str] = (),
    physical: Sequence[str] = (),
) -> Any:
    state = melee.ControllerState()
    state.main_stick = main
    state.c_stick = c
    state.l_shoulder = shoulder
    state.r_shoulder = shoulder
    for name in processed:
        state.processed_button[melee.Button(name)] = True
    for name in physical:
        state.button[melee.Button(name)] = True
    return state


def _player(
    *,
    character: Any = melee.Character.FOX,
    x: float = 0.0,
    y: float = 0.0,
    percent: float = 0.0,
    action: Any = melee.Action.STANDING,
    facing: bool = True,
    invulnerable: bool = False,
    jumps_left: int = 2,
    shield: float = 60.0,
    on_ground: bool = True,
    controller: Any | None = None,
    nana: Any | None = None,
) -> Any:
    player = melee.PlayerState()
    player.character = character
    player.position = melee.Position(np.float32(x), np.float32(y))
    player.percent = percent
    player.action = action
    player.facing = facing
    player.invulnerable = invulnerable
    player.jumps_left = jumps_left
    player.shield_strength = shield
    player.on_ground = on_ground
    player.controller_state = _controller_state() if controller is None else controller
    player.nana = nana
    return player


def _gamestate(
    frame: int,
    players: dict[int, Any],
    *,
    stage: Any = FD,
    projectiles: Sequence[Any] = (),
    fod: Any | None = None,
) -> Any:
    state = melee.GameState()
    state.frame = frame
    state.stage = stage
    state.menu_state = melee.Menu.IN_GAME
    state.players = dict(players)
    state.projectiles = list(projectiles)
    state.fod_platforms = fod
    return state


def _projectile(spawn_id: int, *, kind: Any = melee.ProjectileType.FOX_LASER, subtype: int = 0) -> Any:
    return melee.Projectile(
        position=melee.Position(np.float32(spawn_id * 1.5), np.float32(-spawn_id)),
        owner=1,
        type=kind,
        subtype=subtype,
        spawn_id=np.uint32(spawn_id),
    )


def _converter() -> FrameConverter:
    return FrameConverter(melee)


def test_player_and_controller_leaves_are_exact_and_typed() -> None:
    one = _player(
        character=melee.Character.FOX,
        x=-12.5,
        y=3.25,
        percent=12.7,
        action=melee.Action.DAMAGE_HIGH_1,
        facing=False,
        invulnerable=True,
        jumps_left=1,
        shield=43.5,
        on_ground=False,
        controller=_controller_state(main=(0.25, 0.75), c=(1.0, 0.0), shoulder=0.35, processed=("A", "Z")),
    )
    two = _player(
        character=melee.Character.FALCO,
        x=40.0,
        y=0.0,
        percent=0.0,
        action=melee.Action.STANDING,
        controller=_controller_state(physical=("B",)),  # physical-only presses are not processed presses
    )
    tree = _converter().convert(_gamestate(-120, {1: one, 2: two}))
    assert validate_frame_tree(tree) == torch.Size(())
    p0, p1 = tree["p0"], tree["p1"]
    assert int(p0["percent"]) == 12 and p0["percent"].dtype == torch.int64  # truncated like np.uint16
    assert bool(p0["facing"]) is False and p0["facing"].dtype == torch.bool
    assert float(p0["x"]) == -12.5 and float(p0["y"]) == 3.25 and p0["x"].dtype == torch.float32
    assert int(p0["action"]) == 0x4B and int(p0["character"]) == 1 and int(p0["jumps_left"]) == 1
    assert bool(p0["invulnerable"]) and not bool(p0["on_ground"])
    assert float(p0["shield_strength"]) == 43.5
    controller = p0["controller"]
    assert float(controller["main_stick"]["x"]) == 0.25 and float(controller["main_stick"]["y"]) == 0.75
    assert float(controller["c_stick"]["x"]) == 1.0 and float(controller["c_stick"]["y"]) == 0.0
    assert float(controller["shoulder"]) == pytest.approx(0.35)
    pressed = {name: bool(controller["buttons"][name]) for name in BUTTON_ORDER}
    assert pressed == {name: name in ("A", "Z") for name in BUTTON_ORDER}
    assert int(p1["character"]) == 22 and int(p1["percent"]) == 0 and bool(p1["facing"])
    assert not any(bool(p1["controller"]["buttons"][name]) for name in BUTTON_ORDER)
    assert float(p1["controller"]["main_stick"]["x"]) == 0.5 and float(p1["controller"]["shoulder"]) == 0.0
    assert int(tree["stage"]) == 25  # libmelee internal id of Final Destination
    assert float(tree["randall"]["x"]) == 0.0 and float(tree["randall"]["y"]) == 0.0
    assert float(tree["fod_platforms"]["left"]) == 0.0 and float(tree["fod_platforms"]["right"]) == 0.0
    assert tree["items"]["exists"].shape == (MAX_ITEMS,) and not bool(tree["items"]["exists"].any())
    assert tree["items"]["type"].dtype == torch.int64 and tree["items"]["x"].dtype == torch.float32
    for path, leaf in tree_leaves(tree):
        assert isinstance(leaf, torch.Tensor), path
        assert leaf.dtype in (torch.bool, torch.int64, torch.float32), path


def test_nana_present_and_absent() -> None:
    nana = _player(character=melee.Character.NANA, x=5.0, y=1.0, percent=33.9, jumps_left=0, on_ground=False)
    popo = _player(character=melee.Character.POPO, x=4.0, nana=nana)
    fox = _player()
    tree = _converter().convert(_gamestate(10, {1: popo, 2: fox}))
    assert bool(tree["p0"]["nana"]["exists"]) and int(tree["p0"]["nana"]["character"]) == 11
    assert int(tree["p0"]["nana"]["percent"]) == 33 and float(tree["p0"]["nana"]["x"]) == 5.0
    assert int(tree["p0"]["nana"]["jumps_left"]) == 0 and not bool(tree["p0"]["nana"]["on_ground"])
    assert int(tree["p0"]["character"]) == 10 and float(tree["p0"]["x"]) == 4.0
    absent = tree["p1"]["nana"]
    assert not bool(absent["exists"])
    assert all(
        float(absent[name]) == 0.0 for name in ("percent", "x", "y", "action", "character", "jumps_left")
    )
    assert not bool(absent["facing"]) and not bool(absent["invulnerable"]) and not bool(absent["on_ground"])
    assert float(absent["shield_strength"]) == 0.0


def test_item_slots_follow_slippi_ai_assignment_overflow_and_reset() -> None:
    slots = ItemSlots(num_slots=3)
    assert slots.assign([10, 11]) == [0, 1]
    assert slots.assign([11, 12]) == [1, 0]  # 10 vanished: its slot is reused first (LIFO free list)
    assert slots.assign([13, 11, 12]) == [2, 1, 0]
    assert slots.assign([13, 11, 12, 14]) == [2, 1, 0, None] and slots.dropped == 1
    assert (
        slots.assign([14, 13]) == [0, 2] and slots.dropped == 1
    )  # 11 then 12 freed: 12's slot 0 reused first
    slots.reset()
    assert slots.assign([11]) == [0] and slots.dropped == 1
    assert slots.assign([]) == []
    assert ItemSlots().num_slots == MAX_ITEMS
    with pytest.raises(ValueError, match="num_slots"):
        ItemSlots(num_slots=0)

    converter = _converter()
    players = {1: _player(), 2: _player(character=melee.Character.FALCO)}
    first = converter.convert(_gamestate(0, players, projectiles=[_projectile(7), _projectile(8, subtype=3)]))
    assert first["items"]["exists"].tolist()[:3] == [True, True, False]
    assert first["items"]["type"].tolist()[:2] == [0x36, 0x36] and first["items"]["state"].tolist()[:2] == [
        0,
        3,
    ]
    assert first["items"]["x"].tolist()[:2] == [pytest.approx(10.5), pytest.approx(12.0)]
    assert first["items"]["y"].tolist()[:2] == [-7.0, -8.0]
    unknown = melee.UnknownProjectileType(value=np.uint16(300))
    second = converter.convert(
        _gamestate(1, players, projectiles=[_projectile(8, subtype=3), _projectile(9, kind=unknown)])
    )
    assert second["items"]["exists"].tolist()[:3] == [True, True, False]  # 8 keeps slot 1, 9 takes slot 0
    assert second["items"]["type"].tolist()[:2] == [
        300,
        0x36,
    ]  # OOV types pass through (the encoder maps them)
    assert second["items"]["state"].tolist()[:2] == [0, 3]
    many = converter.convert(_gamestate(2, players, projectiles=[_projectile(100 + i) for i in range(16)]))
    assert int(many["items"]["exists"].sum()) == MAX_ITEMS and converter.dropped_items == 1
    converter.reset()
    again = converter.convert(_gamestate(-123, players, projectiles=[_projectile(200)]))
    assert again["items"]["exists"].tolist()[0] is True and int(again["items"]["exists"].sum()) == 1


def test_randall_fod_platforms_and_stage_ids() -> None:
    players = {1: _player(), 2: _player()}
    yoshis = _converter().convert(_gamestate(500, players, stage=melee.Stage.YOSHIS_STORY))
    height, left, right = melee.randall_position(500)
    assert int(yoshis["stage"]) == 6
    assert float(yoshis["randall"]["y"]) == pytest.approx(height)
    assert float(yoshis["randall"]["x"]) == pytest.approx((left + right) / 2)
    platforms = melee.FoDPlatforms(left=np.float32(21.5), right=np.float32(30.0))
    fod = _converter().convert(_gamestate(3, players, stage=melee.Stage.FOUNTAIN_OF_DREAMS, fod=platforms))
    assert int(fod["stage"]) == 8
    assert float(fod["fod_platforms"]["left"]) == 21.5 and float(fod["fod_platforms"]["right"]) == 30.0
    assert float(fod["randall"]["x"]) == 0.0
    for stage in (
        melee.Stage.BATTLEFIELD,
        melee.Stage.DREAMLAND,
        melee.Stage.POKEMON_STADIUM,
        melee.Stage.NO_STAGE,
    ):
        tree = _converter().convert(_gamestate(0, players, stage=stage))
        assert int(tree["stage"]) == stage.value < SlippiEncoder.STAGE_SIZE
        assert float(tree["fod_platforms"]["left"]) == 0.0 and float(tree["randall"]["y"]) == 0.0


def test_unknown_actions_pass_through_and_out_of_range_values_raise() -> None:
    players = {1: _player(action=melee.UnknownAnimation(400)), 2: _player()}
    tree = _converter().convert(_gamestate(0, players))
    assert int(tree["p0"]["action"]) == 400  # raw id; the encoder clamps actions, it does not reject them
    with pytest.raises(FrameConversionError, match="action"):
        _converter().convert(_gamestate(0, {1: _player(action=melee.UnknownAnimation(-1)), 2: _player()}))
    with pytest.raises(FrameConversionError, match="character"):
        _converter().convert(
            _gamestate(0, {1: _player(character=melee.Character.UNKNOWN_CHARACTER), 2: _player()})
        )
    with pytest.raises(FrameConversionError, match="jumps_left"):
        _converter().convert(_gamestate(0, {1: _player(), 2: _player(jumps_left=SlippiEncoder.JUMPS_SIZE)}))
    with pytest.raises(FrameConversionError, match="character"):
        bad_nana = _player(character=melee.Character.UNKNOWN_CHARACTER)
        _converter().convert(_gamestate(0, {1: _player(nana=bad_nana), 2: _player()}))
    with pytest.raises(FrameConversionError, match="ports"):
        _converter().convert(_gamestate(0, {1: _player()}))
    with pytest.raises(FrameConversionError, match="ports"):
        _converter().convert(_gamestate(0, {1: _player(), 2: _player(), 3: _player()}))
    with pytest.raises(FrameConversionError, match="ports"):
        FrameConverter(melee, ports=(1, 3)).convert(_gamestate(0, {1: _player(), 2: _player()}))
    # Other port pairs are honoured: p0 is always the first configured port.
    swapped = FrameConverter(melee, ports=(2, 1)).convert(
        _gamestate(0, {1: _player(x=-1.0), 2: _player(x=1.0)})
    )
    assert float(swapped["p0"]["x"]) == 1.0 and float(swapped["p1"]["x"]) == -1.0


def test_frame_batch_rows_match_stacked_convert_and_are_fresh() -> None:
    converter = _converter()
    states = [
        _gamestate(i - 123, {1: _player(x=float(i), percent=float(i * 10)), 2: _player(x=-float(i))})
        for i in range(3)
    ]
    states[1].projectiles = [_projectile(1)]
    batch = FrameBatch(3)
    assert batch.batch == 3
    for row, state in enumerate(states):
        converter.write(batch, row, state)
    tree = batch.tree()
    assert validate_frame_tree(tree, (3,)) == torch.Size((3,))
    converter.reset()
    singles = [converter.convert(state) for state in states]
    stacked = tree_stack(singles)
    for (path, leaf), (_, expected) in zip(tree_leaves(tree), tree_leaves(stacked), strict=True):
        assert torch.equal(leaf, expected), path
    assert tree["p0"]["x"].tolist() == [0.0, 1.0, 2.0] and tree["p0"]["percent"].tolist() == [0, 10, 20]
    assert tree["items"]["exists"][:, 0].tolist() == [False, True, False]
    # Leaves are torch views of the batch's numpy arrays (no copies); unwritten rows hold the neutral frame.
    partial = FrameBatch(2)
    converter.write(partial, 1, states[2])
    partial_tree = partial.tree()
    assert partial_tree["p0"]["controller"]["main_stick"]["x"].tolist() == [0.5, 0.5]
    assert partial_tree["p0"]["x"].tolist() == [0.0, 2.0]
    assert partial_tree["p0"]["x"].numpy().ctypes.data == partial.leaves["p0.x"].ctypes.data
    assert set(partial.leaves) == {path for path, _ in tree_leaves(tree)}


def test_import_guard_and_version_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    assert import_melee() is melee
    assert MELEE_VERSION == "0.47.3"
    assert frames_lib.installed_melee_version() == MELEE_VERSION
    monkeypatch.setitem(sys.modules, "melee", None)
    with pytest.raises(ImportError, match=r"melee==0\.47\.3"):
        import_melee()
