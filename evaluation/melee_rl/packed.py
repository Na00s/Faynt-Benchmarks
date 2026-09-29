"""Packed frame transfers and the packed history ring (5 Sep 2026, ``/rl-speedup`` lever 3).

The canonical frame tree has 78 leaves (``melee_rl.frames.FRAME_SPEC``): 73 scalars per row and the five
``items`` leaves of 15 slots each, in three dtypes (float32, int64, bool).  The actor used to move a frame to
the policy device with one pageable copy per leaf (each a host sync: 79 per frame on the bench of 5 Sep 2026)
and to write it into its history ring with one index-put per leaf.  Here the tree is laid out
**feature-major** per dtype -- a ``[rows, *batch_shape]`` tensor per dtype in which every leaf is a
contiguous slab of rows -- so that:

* :class:`FramePacker` moves a frame in one (pinned, non-blocking on CUDA) copy per dtype into static device
  buffers and hands out the *same* view tree every frame (a CUDA graph can be captured against it);
* :class:`PackedRing` keeps the worker's ``[B, W]`` history in the same layout, writes a frame with one
  index-put per dtype, and exposes the view tree the rollout tail's gather indexes as before.

Every leaf is a view of the packed numbers, so trajectories built through this path are bit-identical to the
leaf-wise ones (``tests/rl/test_packed.py``).  ``FrameLayout`` derives the offsets from ``FRAME_SPEC`` in tree
order; ``items`` leaves are the only ones with a trailing dimension (``MAX_ITEMS``), which becomes ``width``
rows of the slab and comes back as the last axis of the view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch

from melee_rl.frames import FRAME_SPEC, FrameTree, neutral_frame, tree_leaves
from tensor_batch import MAX_ITEMS

KIND_DTYPES: Final[dict[str, torch.dtype]] = {"float": torch.float32, "int": torch.int64, "bool": torch.bool}


@dataclass(frozen=True)
class PackedLeaf:
    """One leaf's slab: ``rows [offset, offset + width)`` of the ``kind`` buffer."""

    path: tuple[str, ...]
    kind: str
    offset: int
    width: int


class FrameLayout:
    """Feature-major layout of a frame tree spec (default: the canonical :data:`FRAME_SPEC`)."""

    def __init__(self, spec: Mapping[str, Any] = FRAME_SPEC) -> None:
        self.spec = spec
        leaves: list[PackedLeaf] = []
        rows = dict.fromkeys(KIND_DTYPES, 0)
        for path, kind in tree_leaves(spec):
            if kind not in KIND_DTYPES:
                raise ValueError(f"leaf {path} has kind {kind!r}, expected one of {tuple(KIND_DTYPES)}")
            parts = tuple(path.split("."))
            width = MAX_ITEMS if parts[0] == "items" else 1
            leaves.append(PackedLeaf(parts, kind, rows[kind], width))
            rows[kind] += width
        self.leaves: tuple[PackedLeaf, ...] = tuple(leaves)
        self.rows: dict[str, int] = rows

    def empty(
        self, batch_shape: Sequence[int], device: torch.device | str, *, pin_memory: bool = False
    ) -> dict[str, torch.Tensor]:
        """One ``[rows, *batch_shape]`` buffer per dtype kind."""

        return {
            kind: torch.empty(
                (count, *batch_shape), dtype=KIND_DTYPES[kind], device=device, pin_memory=pin_memory
            )
            for kind, count in self.rows.items()
        }

    def pack(self, trees: Sequence[Mapping[str, Any]], out: Mapping[str, torch.Tensor]) -> None:
        """Copy the leaves of ``trees`` (one per port, leaves ``[E, ...]``) into ``out``
        (``[rows, P * E, ...]``) port-major: port ``p`` fills columns ``[p * E, (p + 1) * E)``."""

        if not trees:
            raise ValueError("pack needs at least one tree")
        columns = next(iter(out.values())).shape[1]
        if columns % len(trees):
            raise ValueError(f"{columns} batch rows do not split over {len(trees)} ports")
        per_port = columns // len(trees)
        for index, tree in enumerate(trees):
            span = slice(index * per_port, (index + 1) * per_port)
            for leaf in self.leaves:
                value = _leaf(tree, leaf.path)
                target = out[leaf.kind][leaf.offset : leaf.offset + leaf.width, span]
                if leaf.width == 1:
                    target[0].copy_(value)
                else:
                    target.copy_(value.movedim(-1, 0))

    def unpack(self, buffers: Mapping[str, torch.Tensor]) -> FrameTree:
        """The view tree over ``buffers`` (``[rows, *batch_shape]`` per kind): scalar leaves
        ``[*batch_shape]``, item leaves ``[*batch_shape, MAX_ITEMS]``."""

        tree: FrameTree = {}
        for leaf in self.leaves:
            slab = buffers[leaf.kind][leaf.offset : leaf.offset + leaf.width]
            value = slab[0] if leaf.width == 1 else slab.movedim(0, -1)
            node = tree
            for part in leaf.path[:-1]:
                node = node.setdefault(part, {})
            node[leaf.path[-1]] = value
        return tree


def _leaf(tree: Mapping[str, Any], path: Sequence[str]) -> torch.Tensor:
    node: Any = tree
    for part in path:
        node = node[part]
    return torch.as_tensor(node)


class StaticFrameTree(dict[str, Any]):
    """A frame tree whose leaves are static device buffers (a :class:`FramePacker`'s): the same tensors every
    frame, so a CUDA graph may be captured against them (``melee_rl.graph_step``)."""


class FramePacker:
    """Static device buffers for one frame of ``batch`` rows and the pinned staging that feeds them."""

    def __init__(
        self,
        batch: int,
        device: torch.device | str,
        layout: FrameLayout | None = None,
        *,
        ports: int = 1,
        pin_memory: bool | None = None,
    ) -> None:
        if ports < 1 or batch % ports:
            raise ValueError(f"{batch} rows do not split over {ports} ports")
        self.layout = FrameLayout() if layout is None else layout
        self.batch = batch
        self.ports = ports
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        pin = cuda if pin_memory is None else pin_memory
        self.buffers: dict[str, torch.Tensor] = self.layout.empty((batch,), self.device)
        self.tree: StaticFrameTree = StaticFrameTree(self.layout.unpack(self.buffers))
        # Two host staging sets: the copy from one may still be in flight while the next frame packs the
        # other.
        self._staging = [self.layout.empty((batch,), "cpu", pin_memory=pin) for _ in range(2)]
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._turn = 0
        self._cuda = cuda

    def upload(self, trees: Sequence[Mapping[str, Any]]) -> StaticFrameTree:
        """Pack the port trees (leaves ``[E]``, port-major into the ``batch`` rows) and copy them over.

        Returns the static view tree (the same object every call; its leaves are overwritten in place).
        """

        if len(trees) != self.ports:
            raise ValueError(f"expected one tree per port ({self.ports}), got {len(trees)}")
        turn = self._turn
        event = self._events[turn]
        if event is not None:
            event.synchronize()
        staging = self._staging[turn]
        self.layout.pack(trees, staging)
        for kind, buffer in self.buffers.items():
            buffer.copy_(staging[kind], non_blocking=self._cuda)
        if self._cuda:
            done = torch.cuda.Event()
            done.record()
            self._events[turn] = done
        self._turn = 1 - turn
        return self.tree


class PackedRing:
    """The actor's ``[B, W]`` frame history in packed form, neutral-initialised."""

    def __init__(
        self, batch: int, capacity: int, device: torch.device | str, layout: FrameLayout | None = None
    ) -> None:
        self.layout = FrameLayout() if layout is None else layout
        self.batch = batch
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffers: dict[str, torch.Tensor] = self.layout.empty((batch, capacity), self.device)
        self.layout.pack([neutral_frame((batch, capacity), device=self.device)], self.buffers)
        self.tree: FrameTree = self.layout.unpack(self.buffers)

    def write(self, slot: int, packed: Mapping[str, torch.Tensor]) -> None:
        """Store one packed frame (``[rows, B]`` per kind, e.g. a :class:`FramePacker`'s buffers) at
        ``slot``."""

        for kind, buffer in self.buffers.items():
            buffer[:, :, slot] = packed[kind]


__all__ = ["KIND_DTYPES", "FrameLayout", "FramePacker", "PackedLeaf", "PackedRing", "StaticFrameTree"]
