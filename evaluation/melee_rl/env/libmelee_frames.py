"""libmelee ``GameState`` -> frame tree: the observation side of ``DolphinEnv`` (PLAN.md §6 P5).

# Ported from slippi-ai slippi_db/parse_libmelee.py@577965a7731dc53e3472ea63d9e9853a4e9d65fa (MIT)
# (``get_base_player`` / ``get_player`` / ``get_controller`` / ``get_item`` / ``Parser.get_game``) and
# slippi_db/parsing_utils.py (``ItemAssigner``), re-expressed over batch-major numpy leaves that become
# torch tensors (``melee_rl.frames.FRAME_SPEC``).

The field semantics are slippi-ai's: ``percent`` truncated to an integer (their ``np.uint16``), raw
``action.value`` ids (``UnknownAnimation`` included), ``character.value`` / ``jumps_left`` /
``shield_strength`` / ``on_ground`` / ``invulnerable`` / ``facing`` / ``position`` from the
``PlayerState``, the controller from ``controller_state`` (``processed_button`` -- never the physical
``button`` --, ``main_stick``, ``c_stick``, ``l_shoulder``), Nana from ``player.nana`` with an ``exists``
flag, items from ``projectiles`` in 15 stable slots keyed by ``spawn_id`` (``ItemSlots``: a slot is kept
while its item lives, freed slots are reused most-recent-first, the 16th item of a frame is dropped and
counted), Randall's midpoint on Yoshi's Story only (``melee.randall_position``), Fountain platform
heights from ``gamestate.fod_platforms``, and ``stage = gamestate.stage.value`` -- the libmelee internal
id (Final Destination = 25), which is what ``SlippiEncoder`` expects of a frame tree without ``stage_id``.

Ranges: :class:`FrameConverter` raises :class:`FrameConversionError` only for values the encoder would
reject -- ``character >= 33``, ``jumps_left >= 7``, ``stage >= 64`` (``model.py:755-756, 765-766,
903-904``), a negative action id, players missing from the configured ports; everything else passes
through raw (unknown action / item ids are clamped or mapped to OOV by the encoder).

Nothing here imports libmelee at module level: :func:`import_melee` loads it on demand with a guided
``ImportError`` (the dummy environment never needs it) and :class:`FrameConverter` receives the module
explicitly, so the tests can build real libmelee dataclasses without a Dolphin.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import torch

from melee_rl.frames import FRAME_SPEC, NANA_FIELDS, PLAYER_FIELDS, FrameTree, tree_leaves, tree_map
from model import SlippiEncoder
from tensor_batch import BUTTON_ORDER, MAX_ITEMS

LOG = logging.getLogger("melee_rl.env.libmelee_frames")

MELEE_VERSION: Final[str] = "0.47.3"
"""The libmelee release the converter and ``DolphinEnv`` were written against (ENV.md §6.2)."""
MELEE_INSTALL_HINT: Final[str] = (
    "libmelee is not installed: pip install 'melee==0.47.3' (the 'dolphin' extra: "
    "pip install -e '.[dolphin]'); "
    "DolphinEnv needs it, the dummy environment does not"
)

Leaves = dict[str, npt.NDArray[Any]]
_LEAVES: Final[tuple[tuple[str, str], ...]] = tuple(tree_leaves(FRAME_SPEC))
"""``(dotted path, kind)`` of every :data:`FRAME_SPEC` leaf, in spec order."""


def installed_melee_version() -> str | None:
    """The installed ``melee`` distribution version, or ``None`` when it is not installed."""

    try:
        return importlib.metadata.version("melee")
    except importlib.metadata.PackageNotFoundError:
        return None


def import_melee() -> Any:
    """``import melee`` with a guided ``ImportError``; warns once per process about another version."""

    try:
        import melee
    except ImportError as error:
        raise ImportError(MELEE_INSTALL_HINT) from error
    version = installed_melee_version()
    if version is not None and version != MELEE_VERSION:
        LOG.warning("libmelee %s is installed; melee_rl was written against %s", version, MELEE_VERSION)
    return melee


class FrameConversionError(ValueError):
    """A libmelee gamestate carries a value the encoder would reject (or lacks a configured port)."""


class ItemSlots:
    """Stable item slots across frames (slippi-ai ``ItemAssigner`` semantics, bounded).

    ``assign(spawn_ids)`` frees the slots of items that vanished, keeps the slot of every item that is
    still there and gives new items the most recently freed slot (initially slot 0 first); when no
    slot is free the item gets ``None`` and :attr:`dropped` grows by one.
    """

    def __init__(self, num_slots: int = MAX_ITEMS) -> None:
        if num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {num_slots}")
        self.num_slots = num_slots
        self.dropped = 0
        self._assigned: dict[int, int] = {}
        self._free: list[int] = []
        self.reset()

    def reset(self) -> None:
        """Forget every assignment (a new game starts); the drop counter is kept."""

        self._assigned = {}
        self._free = list(range(self.num_slots))[::-1]

    def assign(self, spawn_ids: Sequence[int]) -> list[int | None]:
        wanted = set(spawn_ids)
        for spawn_id in list(self._assigned):
            if spawn_id not in wanted:
                self._free.append(self._assigned.pop(spawn_id))
        slots: list[int | None] = []
        for spawn_id in spawn_ids:
            slot = self._assigned.get(spawn_id)
            if slot is None:
                if not self._free:
                    self.dropped += 1
                    slots.append(None)
                    continue
                slot = self._free.pop()
                self._assigned[spawn_id] = slot
            slots.append(slot)
        return slots


def _is_stick_axis(path: str) -> bool:
    return ".main_stick." in path or ".c_stick." in path


_KINDS: Final[dict[str, str]] = dict(_LEAVES)


def leaf_paths() -> tuple[str, ...]:
    """The dotted leaf paths in :class:`FrameBatch` order."""

    return tuple(path for path, _ in _LEAVES)


def leaf_dtype(path: str) -> np.dtype[Any]:
    kind = _KINDS[path]
    if kind == "float":
        return np.dtype(np.float32)
    if kind == "int":
        return np.dtype(np.int64)
    return np.dtype(np.bool_)


def leaf_shape(path: str, batch: int) -> tuple[int, ...]:
    return (batch, MAX_ITEMS) if path.startswith("items.") else (batch,)


def leaf_neutral(path: str) -> float | int | bool:
    """The neutral value a fresh :class:`FrameBatch` row carries (sticks 0.5, everything else 0 / False)."""

    kind = _KINDS[path]
    if kind == "float":
        return 0.5 if _is_stick_axis(path) else 0.0
    if kind == "int":
        return 0
    return False


class FrameBatch:
    """Pre-allocated numpy leaves for ``batch`` environments, keyed by dotted :data:`FRAME_SPEC` path.

    Rows start as the neutral frame (zeros / ``False``, sticks at 0.5); :meth:`tree` wraps the arrays as
    torch tensors without copying (``torch.from_numpy``), so build a fresh batch per environment step
    rather than mutating one that was handed out.
    """

    def __init__(self, batch: int) -> None:
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        self.batch = batch
        self.leaves: Leaves = {}
        for path, kind in _LEAVES:
            shape = (batch, MAX_ITEMS) if path.startswith("items.") else (batch,)
            if kind == "float":
                self.leaves[path] = np.full(shape, 0.5 if _is_stick_axis(path) else 0.0, dtype=np.float32)
            elif kind == "int":
                self.leaves[path] = np.zeros(shape, dtype=np.int64)
            else:
                self.leaves[path] = np.zeros(shape, dtype=np.bool_)

    def clear(self) -> None:
        """Reset every row to the neutral frame in place (adopted arrays included): the converter writes
        only what a frame carries (Randall on Yoshi's, present items, an existing Nana), so a reused batch
        must start neutral like a fresh one."""

        for path, array in self.leaves.items():
            array.fill(leaf_neutral(path))

    @classmethod
    def from_leaves(cls, leaves: Mapping[str, npt.NDArray[Any]]) -> FrameBatch:
        """A batch adopting existing arrays (the worker mode's re-assembly; no copy, light checks).

        ``leaves`` must carry exactly the :data:`FRAME_SPEC` paths with a consistent leading
        dimension; the arrays are adopted as-is (their dtypes come from :class:`FrameBatch` in the
        worker that produced them).
        """

        expected = [path for path, _ in _LEAVES]
        if sorted(leaves) != sorted(expected):
            missing = sorted(set(expected) - set(leaves))
            extra = sorted(set(leaves) - set(expected))
            raise ValueError(f"leaves do not match FRAME_SPEC (missing {missing}, extra {extra})")
        batch_size = int(leaves["stage"].shape[0])
        adopted = cls.__new__(cls)
        adopted.batch = batch_size
        adopted.leaves = {}
        for path in expected:
            array = leaves[path]
            shape = (batch_size, MAX_ITEMS) if path.startswith("items.") else (batch_size,)
            if tuple(array.shape) != shape:
                raise ValueError(f"leaf {path} has shape {tuple(array.shape)}, expected {shape}")
            adopted.leaves[path] = array
        return adopted

    def tree(self) -> FrameTree:
        """The nested frame tree (leaves ``[B]`` / ``[B, 15]``) viewing this batch's arrays."""

        def build(spec: Mapping[str, Any], prefix: str) -> FrameTree:
            result: FrameTree = {}
            for key, kind in spec.items():
                path = f"{prefix}.{key}" if prefix else key
                result[key] = (
                    build(kind, path) if isinstance(kind, Mapping) else torch.from_numpy(self.leaves[path])
                )
            return result

        return build(FRAME_SPEC, "")


def _player_keys(prefix: str) -> dict[str, Any]:
    """Dotted leaf paths of one player (``p0`` / ``p1``), computed once."""

    return {
        "base": {name: f"{prefix}.{name}" for name, _ in PLAYER_FIELDS},
        "nana": {name: f"{prefix}.nana.{name}" for name, _ in NANA_FIELDS if name != "exists"},
        "nana_exists": f"{prefix}.nana.exists",
        "main_x": f"{prefix}.controller.main_stick.x",
        "main_y": f"{prefix}.controller.main_stick.y",
        "c_x": f"{prefix}.controller.c_stick.x",
        "c_y": f"{prefix}.controller.c_stick.y",
        "shoulder": f"{prefix}.controller.shoulder",
        "buttons": {name: f"{prefix}.controller.buttons.{name}" for name in BUTTON_ORDER},
    }


class FrameConverter:
    """Writes libmelee gamestates into :class:`FrameBatch` rows; one per environment (item slots are state).

    ``ports`` are the 1-based Dolphin controller ports that become ``p0`` and ``p1``.  Call
    :meth:`reset` at every game start (frame ``-123``) so that item slots start afresh, as slippi-ai
    rebuilds its ``Parser`` per game.
    """

    def __init__(self, melee: Any, ports: Sequence[int] = (1, 2), *, num_item_slots: int = MAX_ITEMS) -> None:
        if len(ports) != 2 or len(set(ports)) != 2:
            raise ValueError(f"ports must be two distinct 1-based controller ports, got {tuple(ports)}")
        self.ports = (int(ports[0]), int(ports[1]))
        self._melee = melee
        self._buttons = [(name, melee.Button(name)) for name in BUTTON_ORDER]
        self._yoshis = melee.Stage.YOSHIS_STORY
        self._randall_position = melee.randall_position
        self._items = ItemSlots(num_item_slots)
        self._keys = {prefix: _player_keys(prefix) for prefix in ("p0", "p1")}

    @property
    def dropped_items(self) -> int:
        """Items beyond the 15 slots seen so far (counted, never an error)."""

        return self._items.dropped

    def reset(self) -> None:
        self._items.reset()

    def write(self, batch: FrameBatch, row: int, gamestate: Any) -> None:
        """Write ``gamestate`` into row ``row`` of ``batch`` (every leaf of the row is set)."""

        leaves = batch.leaves
        frame = int(gamestate.frame)
        players = gamestate.players
        present = sorted(players)
        if present != sorted(self.ports):
            raise FrameConversionError(
                f"frame {frame}: expected players on ports {list(self.ports)}, got {present}"
            )
        for prefix, port in zip(("p0", "p1"), self.ports, strict=True):
            self._write_player(leaves, self._keys[prefix], row, players[port], frame)
        stage = int(gamestate.stage.value)
        if not 0 <= stage < SlippiEncoder.STAGE_SIZE:
            raise FrameConversionError(
                f"frame {frame}: stage id {stage} is outside [0, {SlippiEncoder.STAGE_SIZE})"
            )
        leaves["stage"][row] = stage
        if gamestate.stage is self._yoshis:
            height, left, right = self._randall_position(frame)
            leaves["randall.x"][row] = (float(left) + float(right)) / 2.0
            leaves["randall.y"][row] = float(height)
        platforms = gamestate.fod_platforms
        if platforms is not None:
            leaves["fod_platforms.left"][row] = float(platforms.left)
            leaves["fod_platforms.right"][row] = float(platforms.right)
        projectiles = gamestate.projectiles
        slots = self._items.assign([int(projectile.spawn_id) for projectile in projectiles])
        for slot, projectile in zip(slots, projectiles, strict=True):
            if slot is None:
                continue
            leaves["items.exists"][row, slot] = True
            leaves["items.type"][row, slot] = int(projectile.type.value)
            leaves["items.state"][row, slot] = int(projectile.subtype)
            leaves["items.x"][row, slot] = float(projectile.position.x)
            leaves["items.y"][row, slot] = float(projectile.position.y)

    def convert(self, gamestate: Any) -> FrameTree:
        """One gamestate as a batch-less frame tree (leaves ``[]`` / ``[15]``; tests and diagnostics)."""

        batch = FrameBatch(1)
        self.write(batch, 0, gamestate)
        return tree_map(lambda leaf: leaf[0], batch.tree())

    # -- players --------------------------------------------------------------------------

    def _write_player(self, leaves: Leaves, keys: dict[str, Any], row: int, player: Any, frame: int) -> None:
        self._write_base(leaves, keys["base"], row, player, frame, "player")
        state = player.controller_state
        main_x, main_y = state.main_stick
        c_x, c_y = state.c_stick
        leaves[keys["main_x"]][row] = float(main_x)
        leaves[keys["main_y"]][row] = float(main_y)
        leaves[keys["c_x"]][row] = float(c_x)
        leaves[keys["c_y"]][row] = float(c_y)
        leaves[keys["shoulder"]][row] = float(state.l_shoulder)
        processed = state.processed_button
        button_keys = keys["buttons"]
        for name, button in self._buttons:
            leaves[button_keys[name]][row] = bool(processed[button])
        nana = player.nana
        if nana is not None:
            leaves[keys["nana_exists"]][row] = True
            self._write_base(leaves, keys["nana"], row, nana, frame, "nana")

    @staticmethod
    def _write_base(
        leaves: Leaves, keys: dict[str, str], row: int, player: Any, frame: int, what: str
    ) -> None:
        action = int(player.action.value)
        if action < 0:
            raise FrameConversionError(f"frame {frame}: {what} action id {action} is negative")
        character = int(player.character.value)
        if not 0 <= character < SlippiEncoder.CHARACTER_SIZE:
            raise FrameConversionError(
                f"frame {frame}: {what} character id {character} is outside "
                f"[0, {SlippiEncoder.CHARACTER_SIZE})"
            )
        jumps_left = int(player.jumps_left)
        if not 0 <= jumps_left < SlippiEncoder.JUMPS_SIZE:
            raise FrameConversionError(
                f"frame {frame}: {what} jumps_left {jumps_left} is outside [0, {SlippiEncoder.JUMPS_SIZE})"
            )
        leaves[keys["percent"]][row] = int(player.percent)
        leaves[keys["facing"]][row] = bool(player.facing)
        leaves[keys["x"]][row] = float(player.position.x)
        leaves[keys["y"]][row] = float(player.position.y)
        leaves[keys["action"]][row] = action
        leaves[keys["invulnerable"]][row] = bool(player.invulnerable)
        leaves[keys["character"]][row] = character
        leaves[keys["jumps_left"]][row] = jumps_left
        leaves[keys["shield_strength"]][row] = float(player.shield_strength)
        leaves[keys["on_ground"]][row] = bool(player.on_ground)


__all__ = [
    "MELEE_INSTALL_HINT",
    "MELEE_VERSION",
    "FrameBatch",
    "FrameConversionError",
    "FrameConverter",
    "ItemSlots",
    "import_melee",
    "installed_melee_version",
    "leaf_dtype",
    "leaf_neutral",
    "leaf_paths",
    "leaf_shape",
]
