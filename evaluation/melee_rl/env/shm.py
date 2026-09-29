"""Shared-memory frame exchange between the parent and the Dolphin workers (5 Sep 2026, lever 0 / IPC).

The pickled pipe protocol of ``dolphin_mp`` costs the parent, per frame-step, one pickle of 13 controller
arrays per worker on the way out and one unpickle of the 78-leaf ``FrameBatch.leaves`` per worker on the
way back, plus a ``np.concatenate`` per leaf -- all serial in the parent and all on the critical path (the
bench of 5 Sep 2026: 6.0 + 8.8 + 1.6 ms of a 50 ms frame-step at 16 workers, and every later reply's arrival
is noticed only after the earlier ones are unpickled).  With ``DolphinEnvConfig.shared_memory = true`` the
arrays live in ``multiprocessing.shared_memory`` segments instead:

* one **controller block** ``[num_envs]`` per port and field (the parent writes, the workers read a slice);
* two **frame slots**, each the 78 leaves ``[num_envs]`` / ``[num_envs, 15]`` plus ``frame_index`` (the
  workers write their slice in place through ``DolphinEnv.step_rows(into=)``, the parent adopts the slot's
  views as the step's ``FrameBatch``); two slots so the output a step returned stays valid until the next step
  has completed (the actor consumes it before calling ``step`` again).

The pipe then carries only ``("step", ("shm", slot))`` and ``("state", None)`` tokens.  The leaves are the
same numbers as the pickled path's (``tests/rl/test_shm.py`` proves the outputs bit-identical on the fake
consoles); nothing about what the environments do changes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from melee_rl.env.libmelee_frames import FrameBatch, leaf_dtype, leaf_neutral, leaf_shape
from melee_rl.env.libmelee_frames import leaf_paths as frame_leaf_paths
from tensor_batch import BUTTON_ORDER

FRAME_INDEX: Final[str] = "frame_index"
ANALOG_FIELDS: Final[tuple[str, ...]] = ("main_x", "main_y", "c_x", "c_y", "shoulder")
CONTROLLER_FIELDS: Final[tuple[str, ...]] = ANALOG_FIELDS + tuple(BUTTON_ORDER)
_ALIGN: Final[int] = 64


@dataclass(frozen=True)
class Entry:
    path: str
    dtype: str
    shape: tuple[int, ...]
    offset: int


@dataclass(frozen=True)
class Layout:
    """Where each named array sits inside one segment."""

    entries: tuple[Entry, ...]
    nbytes: int

    def entry(self, path: str) -> Entry:
        for entry in self.entries:
            if entry.path == path:
                return entry
        raise KeyError(path)


def _layout(specs: Sequence[tuple[str, str, tuple[int, ...]]]) -> Layout:
    entries: list[Entry] = []
    offset = 0
    for path, dtype, shape in specs:
        entries.append(Entry(path, dtype, tuple(shape), offset))
        size = int(np.dtype(dtype).itemsize * int(np.prod(shape)))
        offset += (size + _ALIGN - 1) // _ALIGN * _ALIGN
    return Layout(tuple(entries), max(offset, _ALIGN))


def frame_layout(num_envs: int) -> Layout:
    """The 78 frame leaves in ``FrameBatch`` order plus ``frame_index`` (int64)."""

    specs = [(path, leaf_dtype(path).str, leaf_shape(path, num_envs)) for path in frame_leaf_paths()]
    specs.append((FRAME_INDEX, np.dtype(np.int64).str, (num_envs,)))
    return _layout(specs)


def controller_layout(num_envs: int, ports: Sequence[int]) -> Layout:
    """Per controlled port the 13 controller fields (analog float32, buttons bool), ``[num_envs]`` each."""

    specs = []
    for port in ports:
        for field in CONTROLLER_FIELDS:
            dtype = np.dtype(np.float32) if field in ANALOG_FIELDS else np.dtype(np.bool_)
            specs.append((f"{port}.{field}", dtype.str, (num_envs,)))
    return _layout(specs)


class Segment:
    """One shared-memory segment with a :class:`Layout`; the creator unlinks it."""

    def __init__(self, layout: Layout, *, name: str | None = None, create: bool) -> None:
        self.layout = layout
        if create:
            self._memory = shared_memory.SharedMemory(create=True, size=layout.nbytes)
        else:
            if name is None:
                raise ValueError("attaching needs the segment's name")
            self._memory = shared_memory.SharedMemory(name=name)
        self._owner = create
        self.name = self._memory.name

    def view(self, path: str, start: int | None = None, stop: int | None = None) -> npt.NDArray[Any]:
        entry = self.layout.entry(path)
        array: npt.NDArray[Any] = np.ndarray(
            entry.shape, dtype=np.dtype(entry.dtype), buffer=self._memory.buf, offset=entry.offset
        )
        if start is None and stop is None:
            return array
        return array[start:stop]

    def close(self) -> None:
        # Views may still be alive (numpy arrays over the buffer); the OS reclaims the mapping at exit.
        with contextlib.suppress(BufferError, OSError):
            self._memory.close()

    def unlink(self) -> None:
        if self._owner:
            with contextlib.suppress(FileNotFoundError):
                self._memory.unlink()


@dataclass(frozen=True)
class ExchangeSpec:
    """What a worker needs to attach: the segment names and the batch geometry (picklable)."""

    num_envs: int
    ports: tuple[int, ...]
    frame_names: tuple[str, str]
    controller_name: str
    slots: int = 2


class FrameExchange:
    """The parent's (``create=True``) or a worker's (``spec=``) side of the shared blocks."""

    def __init__(
        self,
        num_envs: int,
        ports: Sequence[int],
        *,
        spec: ExchangeSpec | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.ports = tuple(int(port) for port in ports)
        frames = frame_layout(self.num_envs)
        controllers = controller_layout(self.num_envs, self.ports)
        if spec is None:
            self._frames = [Segment(frames, create=True) for _ in range(2)]
            self._controllers = Segment(controllers, create=True)
        else:
            if spec.num_envs != self.num_envs or spec.ports != self.ports:
                raise ValueError(
                    f"exchange spec {spec} does not match {self.num_envs} envs on ports {self.ports}"
                )
            self._frames = [Segment(frames, name=name, create=False) for name in spec.frame_names]
            self._controllers = Segment(controllers, name=spec.controller_name, create=False)
        self.spec = ExchangeSpec(
            num_envs=self.num_envs,
            ports=self.ports,
            frame_names=(self._frames[0].name, self._frames[1].name),
            controller_name=self._controllers.name,
        )
        self._closed = False

    @property
    def slots(self) -> int:
        return len(self._frames)

    # -- frames ---------------------------------------------------------------------------------

    def frame_leaves(self, slot: int, start: int = 0, stop: int | None = None) -> dict[str, npt.NDArray[Any]]:
        """Views of the 78 leaves of ``slot`` over rows ``[start, stop)`` (the whole batch by default)."""

        segment = self._frames[slot]
        end = self.num_envs if stop is None else stop
        return {path: segment.view(path, start, end) for path in frame_leaf_paths()}

    def frame_index(self, slot: int, start: int = 0, stop: int | None = None) -> npt.NDArray[Any]:
        end = self.num_envs if stop is None else stop
        return self._frames[slot].view(FRAME_INDEX, start, end)

    def frame_batch(self, slot: int, start: int = 0, stop: int | None = None) -> FrameBatch:
        """A :class:`FrameBatch` adopting the slot's views (a worker's slice, or the parent's whole batch)."""

        return FrameBatch.from_leaves(self.frame_leaves(slot, start, stop))

    # -- controllers ----------------------------------------------------------------------------

    def write_controllers(self, port: int, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
        """The parent copies one port's 13 ``[num_envs]`` arrays into the block."""

        for field in CONTROLLER_FIELDS:
            target = self._controllers.view(f"{port}.{field}")
            np.copyto(target, arrays[field], casting="unsafe")

    def controller_arrays(
        self, port: int, start: int = 0, stop: int | None = None
    ) -> dict[str, npt.NDArray[Any]]:
        """Views of one port's controller fields over rows ``[start, stop)`` (a worker reads its slice)."""

        end = self.num_envs if stop is None else stop
        return {field: self._controllers.view(f"{port}.{field}", start, end) for field in CONTROLLER_FIELDS}

    # -- lifecycle ------------------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for segment in self._frames:
            segment.close()
        self._controllers.close()

    def unlink(self) -> None:
        for segment in self._frames:
            segment.unlink()
        self._controllers.unlink()


def neutral_fill(leaves: Mapping[str, npt.NDArray[Any]]) -> None:
    """Reset adopted leaves to the neutral frame in place (what a fresh ``FrameBatch`` starts as): the
    converter writes only what a frame carries (Randall on Yoshi's, present items, an existing Nana)."""

    for path, array in leaves.items():
        array.fill(leaf_neutral(path))


__all__ = [
    "ANALOG_FIELDS",
    "CONTROLLER_FIELDS",
    "FRAME_INDEX",
    "Entry",
    "ExchangeSpec",
    "FrameExchange",
    "Layout",
    "Segment",
    "controller_layout",
    "frame_layout",
    "neutral_fill",
]
