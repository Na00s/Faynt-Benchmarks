"""Frame trees and label helpers -- the observation contract of the RL package.

A *frame tree* is a nested ``dict`` whose leaves are tensors sharing a leading (batch)
shape ``S`` and whose keys are the ``tensor_batch.GameStateBatch`` field names
(INTERFACE.md §2): ``p0`` / ``p1`` (``PlayerBatch`` scalars plus ``controller`` and
``nana``), ``stage``, ``randall``, ``fod_platforms`` and ``items`` (leaves
``[*S, 15]``).  ``p0`` is always the controlled player.  ``SlippiEncoder.forward``
consumes such trees directly (``model.py:466-479`` reads mappings and attributes
alike); :func:`frame_tree_to_game_state` gives the typed ``GameStateBatch`` view.

Everything is batch-major and device-agnostic.  The valid ranges enforced by the
encoder (raise, not clamp: ``model.py:755-756`` character, ``:765-766`` jumps,
``:903-904`` stage) are taken from ``SlippiEncoder`` so that :func:`neutral_frame` and
:func:`random_valid_frames` always produce encodable rows -- padded rows must hold a
valid frame because the encoder validates every row (PLAN.md §3.2).

No slippi-ai or MIMIC code is ported in this module.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Literal

import torch

from controller_codec import ControllerLabels
from model import SlippiEncoder
from tensor_batch import (
    BUTTON_ORDER,
    MAX_ITEMS,
    ButtonsBatch,
    ControllerBatch,
    FoDPlatformsBatch,
    GameStateBatch,
    ItemsBatch,
    NanaBatch,
    PlayerBatch,
    RandallBatch,
    StickBatch,
)

FrameTree = dict[str, Any]
"""Nested ``dict`` of tensors (values are tensors or nested ``FrameTree``)."""

LeafKind = Literal["bool", "int", "float"]

PLAYER_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (
    ("percent", "int"),
    ("facing", "bool"),
    ("x", "float"),
    ("y", "float"),
    ("action", "int"),
    ("invulnerable", "bool"),
    ("character", "int"),
    ("jumps_left", "int"),
    ("shield_strength", "float"),
    ("on_ground", "bool"),
)
"""Scalar ``PlayerBatch`` leaves in ``tensor_batch.py:111-120`` order."""

NANA_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (("exists", "bool"), *PLAYER_FIELDS)
ITEM_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (
    ("exists", "bool"),
    ("type", "int"),
    ("state", "int"),
    ("x", "float"),
    ("y", "float"),
)

_STICK_SPEC: Final[dict[str, Any]] = {"x": "float", "y": "float"}
_CONTROLLER_SPEC: Final[dict[str, Any]] = {
    "main_stick": _STICK_SPEC,
    "c_stick": _STICK_SPEC,
    "shoulder": "float",
    "buttons": dict.fromkeys(BUTTON_ORDER, "bool"),
}
_PLAYER_SPEC: Final[dict[str, Any]] = {
    **dict(PLAYER_FIELDS),
    "controller": _CONTROLLER_SPEC,
    "nana": dict(NANA_FIELDS),
}
FRAME_SPEC: Final[dict[str, Any]] = {
    "p0": _PLAYER_SPEC,
    "p1": _PLAYER_SPEC,
    "stage": "int",
    "randall": _STICK_SPEC,
    "fod_platforms": {"left": "float", "right": "float"},
    "items": dict(ITEM_FIELDS),
}
"""The full tree: nested dicts ending in a :data:`LeafKind`; ``items`` leaves carry ``[*S, 15]``."""

_KIND_DTYPES: Final[dict[str, torch.dtype]] = {
    "bool": torch.bool,
    "int": torch.int64,
    "float": torch.float32,
}

# Half-open ranges for random frames; every upper bound is what the encoder accepts.
_INT_RANGES: Final[dict[str, tuple[int, int]]] = {
    "percent": (0, 300),
    "action": (0, SlippiEncoder.ACTION_SIZE),
    "character": (0, SlippiEncoder.CHARACTER_SIZE),
    "jumps_left": (0, SlippiEncoder.JUMPS_SIZE),
    "stage": (0, SlippiEncoder.STAGE_SIZE),
    "type": (0, SlippiEncoder.ITEM_TYPE_INPUT_SIZE),
    "state": (0, SlippiEncoder.ITEM_STATE_INPUT_SIZE),
}
_FLOAT_RANGES: Final[dict[str, tuple[float, float]]] = {
    "x": (-100.0, 100.0),
    "y": (-50.0, 150.0),
    "shield_strength": (0.0, 60.0),
    "shoulder": (0.0, 1.0),
    "left": (-50.0, 50.0),
    "right": (-50.0, 50.0),
}


# ---------------------------------------------------------------------------
# Generic tree operations
# ---------------------------------------------------------------------------


def tree_map(fn: Callable[..., Any], tree: Mapping[str, Any], *rest: Mapping[str, Any]) -> FrameTree:
    """Apply ``fn`` leaf-wise over ``tree`` (and the same leaves of ``rest``)."""

    result: FrameTree = {}
    for key, value in tree.items():
        others = [other[key] for other in rest]
        if isinstance(value, Mapping):
            result[key] = tree_map(fn, value, *others)
        else:
            result[key] = fn(value, *others)
    return result


def tree_leaves(tree: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """``(dotted_path, leaf)`` pairs in insertion order."""

    leaves: list[tuple[str, Any]] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            leaves.extend(tree_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def tree_stack(trees: Sequence[Mapping[str, Any]], dim: int = 0) -> FrameTree:
    """``torch.stack`` every leaf across ``trees`` along ``dim``."""

    if not trees:
        raise ValueError("tree_stack needs at least one tree")
    return tree_map(lambda *leaves: torch.stack(leaves, dim=dim), *trees)


def tree_cat(trees: Sequence[Mapping[str, Any]], dim: int = 0) -> FrameTree:
    """``torch.cat`` every leaf across ``trees`` along ``dim``."""

    if not trees:
        raise ValueError("tree_cat needs at least one tree")
    return tree_map(lambda *leaves: torch.cat(leaves, dim=dim), *trees)


def tree_index(tree: Mapping[str, Any], index: Any) -> FrameTree:
    """``leaf[index]`` for every leaf (``index`` may be an int, slice, tuple or tensor)."""

    return tree_map(lambda leaf: leaf[index], tree)


def tree_narrow(tree: Mapping[str, Any], dim: int, start: int, length: int) -> FrameTree:
    """``leaf.narrow(dim, start, length)`` for every leaf (a view, no copy)."""

    return tree_map(lambda leaf: leaf.narrow(dim, start, length), tree)


def tree_to(tree: Mapping[str, Any], device: torch.device | str) -> FrameTree:
    """Move every leaf to ``device`` (non-tensor leaves are converted with ``as_tensor``)."""

    return tree_map(lambda leaf: torch.as_tensor(leaf).to(device), tree)


def tree_to_packed(tree: Mapping[str, Any], device: torch.device | str) -> FrameTree:
    """:func:`tree_to` with one copy per leaf dtype (P8-PLAN.md 3.1).

    The leaves are flattened in tree order, concatenated per dtype, moved in at most one
    ``.to(device)`` per dtype group (a frame tree has three) and split back into views --
    bit-identical values to :func:`tree_to`.  The pipelined actor packs its ``b`` buffered
    frames this way, so ``b x 78`` leaves cross to the policy device in <= 3 copies per
    call instead of 78 per frame.
    """

    leaves = [torch.as_tensor(leaf) for _, leaf in tree_leaves(tree)]
    groups: dict[torch.dtype, list[int]] = {}
    for index, leaf in enumerate(leaves):
        groups.setdefault(leaf.dtype, []).append(index)
    moved: dict[int, torch.Tensor] = {}
    for indices in groups.values():
        flat = torch.cat([leaves[index].reshape(-1) for index in indices]).to(device)
        pieces = torch.split(flat, [leaves[index].numel() for index in indices])
        for index, piece in zip(indices, pieces, strict=True):
            moved[index] = piece.reshape(leaves[index].shape)
    ordered = iter(moved[index] for index in range(len(leaves)))

    def rebuild(node: Mapping[str, Any]) -> FrameTree:
        result: FrameTree = {}
        for key, value in node.items():
            result[key] = rebuild(value) if isinstance(value, Mapping) else next(ordered)
        return result

    return rebuild(tree)


def tree_clone(tree: Mapping[str, Any]) -> FrameTree:
    """Deep copy of the leaves."""

    return tree_map(lambda leaf: torch.as_tensor(leaf).clone(), tree)


def leading_shape(tree: Mapping[str, Any]) -> torch.Size:
    """The batch shape ``S`` of a frame tree, read from ``p0.percent``."""

    try:
        percent = tree["p0"]["percent"]
    except (KeyError, TypeError) as error:
        raise ValueError("frame tree has no p0.percent leaf") from error
    return torch.Size(torch.as_tensor(percent).shape)


# ---------------------------------------------------------------------------
# Building and validating frames
# ---------------------------------------------------------------------------


def _build(spec: Mapping[str, Any], path: str, leaf: Callable[[str, str], torch.Tensor]) -> FrameTree:
    result: FrameTree = {}
    for key, kind in spec.items():
        child = f"{path}.{key}" if path else key
        result[key] = _build(kind, child, leaf) if isinstance(kind, Mapping) else leaf(child, kind)
    return result


def _leaf_shape(path: str, batch_shape: torch.Size) -> torch.Size:
    return torch.Size((*batch_shape, MAX_ITEMS)) if path.startswith("items.") else batch_shape


def _is_stick_axis(path: str) -> bool:
    return ".main_stick." in path or ".c_stick." in path


def neutral_frame(
    batch_shape: Sequence[int] = (),
    *,
    device: torch.device | str | None = None,
) -> FrameTree:
    """The valid all-default frame: zeros / ``False`` everywhere, sticks centred at 0.5.

    Used to left-pad windows (``padding_mask`` false) and as a template; it is never
    meant to be attended.  ``stage = 0``, ``character = 0``, ``action = 0`` are all inside
    the encoder's vocabularies.
    """

    shape = torch.Size(batch_shape)

    def leaf(path: str, kind: str) -> torch.Tensor:
        full = _leaf_shape(path, shape)
        if kind == "float":
            value = 0.5 if _is_stick_axis(path) else 0.0
            return torch.full(full, value, dtype=torch.float32, device=device)
        return torch.zeros(full, dtype=_KIND_DTYPES[kind], device=device)

    return _build(FRAME_SPEC, "", leaf)


def random_valid_frames(
    batch_shape: Sequence[int],
    generator: torch.Generator,
    *,
    device: torch.device | str | None = None,
) -> FrameTree:
    """Independent random frames inside every range the encoder accepts (tests only).

    Drawn on the CPU from ``generator`` (deterministic for a given seed), then moved to
    ``device``.  Sticks lie in ``[0, 1]`` (slippi-ai convention), percents in ``[0, 300)``,
    categorical ids strictly below the encoder vocabularies.
    """

    shape = torch.Size(batch_shape)

    def leaf(path: str, kind: str) -> torch.Tensor:
        full = _leaf_shape(path, shape)
        name = path.rsplit(".", 1)[-1]
        if kind == "bool":
            return torch.rand(full, generator=generator) < 0.5
        if kind == "int":
            low, high = _INT_RANGES.get(name, (0, 2))
            return torch.randint(low, high, full, generator=generator)
        low_f, high_f = (0.0, 1.0) if _is_stick_axis(path) else _FLOAT_RANGES.get(name, (-50.0, 50.0))
        return torch.rand(full, generator=generator) * (high_f - low_f) + low_f

    tree = _build(FRAME_SPEC, "", leaf)
    return tree if device is None else tree_to(tree, device)


def validate_frame_tree(tree: Mapping[str, Any], batch_shape: Sequence[int] | None = None) -> torch.Size:
    """Check that every leaf of :data:`FRAME_SPEC` exists with the right shape and dtype kind.

    Returns the batch shape.  Extra keys are allowed and ignored (an environment may add
    diagnostics such as a frame counter).  Raises ``ValueError`` for missing leaves or wrong
    shapes and ``TypeError`` for non-tensor leaves or wrong dtype kinds.
    """

    if not isinstance(tree, Mapping):
        raise TypeError("a frame tree must be a mapping")
    shape = leading_shape(tree) if batch_shape is None else torch.Size(batch_shape)

    def check(spec: Mapping[str, Any], node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            raise TypeError(f"{path or 'frame tree'} must be a mapping")
        for key, kind in spec.items():
            child = f"{path}.{key}" if path else key
            if key not in node:
                raise ValueError(f"frame tree is missing {child}")
            value = node[key]
            if isinstance(kind, Mapping):
                check(kind, value, child)
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{child} must be a tensor, got {type(value).__name__}")
            expected = _leaf_shape(child, shape)
            if value.shape != expected:
                raise ValueError(f"{child} must have shape {tuple(expected)}, got {tuple(value.shape)}")
            if kind == "bool" and value.dtype != torch.bool:
                raise TypeError(f"{child} must be torch.bool, got {value.dtype}")
            not_integer = value.dtype == torch.bool or value.is_floating_point() or value.is_complex()
            if kind == "int" and not_integer:
                raise TypeError(f"{child} must have an integer dtype, got {value.dtype}")
            if kind == "float" and not value.is_floating_point():
                raise TypeError(f"{child} must have a floating dtype, got {value.dtype}")

    check(FRAME_SPEC, tree, "")
    return shape


# ---------------------------------------------------------------------------
# GameStateBatch <-> tree
# ---------------------------------------------------------------------------


def _to_tree(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _to_tree(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return value


def game_state_to_tree(state: GameStateBatch) -> FrameTree:
    """Nested dict with the same leaves (no copies)."""

    tree: FrameTree = _to_tree(state)
    return tree


def _stick(tree: Mapping[str, Any]) -> StickBatch:
    return StickBatch(x=tree["x"], y=tree["y"])


def _controller(tree: Mapping[str, Any]) -> ControllerBatch:
    buttons = tree["buttons"]
    return ControllerBatch(
        main_stick=_stick(tree["main_stick"]),
        c_stick=_stick(tree["c_stick"]),
        shoulder=tree["shoulder"],
        buttons=ButtonsBatch(**{name: buttons[name] for name in BUTTON_ORDER}),
    )


def _player(tree: Mapping[str, Any]) -> PlayerBatch:
    nana = tree["nana"]
    return PlayerBatch(
        **{name: tree[name] for name, _ in PLAYER_FIELDS},
        controller=_controller(tree["controller"]),
        nana=NanaBatch(**{name: nana[name] for name, _ in NANA_FIELDS}),
    )


def frame_tree_to_game_state(tree: Mapping[str, Any]) -> GameStateBatch:
    """Typed view of a validated frame tree (shares the leaf tensors)."""

    validate_frame_tree(tree)
    items = tree["items"]
    return GameStateBatch(
        p0=_player(tree["p0"]),
        p1=_player(tree["p1"]),
        stage=tree["stage"],
        randall=RandallBatch(x=tree["randall"]["x"], y=tree["randall"]["y"]),
        fod_platforms=FoDPlatformsBatch(
            left=tree["fod_platforms"]["left"],
            right=tree["fod_platforms"]["right"],
        ),
        items=ItemsBatch(**{name: items[name] for name, _ in ITEM_FIELDS}),
    )


# ---------------------------------------------------------------------------
# Controller labels
# ---------------------------------------------------------------------------


def neutral_labels(
    batch_shape: Sequence[int] = (),
    *,
    device: torch.device | str | None = None,
) -> ControllerLabels:
    """Labels ``(0, 0)``: main stick centred, no buttons (``tests/fixtures/custom_v1_pinned.json``).

    Pre-fills the delay queue and is the ``controller_prev`` input at reset and padded rows.
    """

    shape = torch.Size(batch_shape)
    return ControllerLabels(
        buttons=torch.zeros(shape, dtype=torch.long, device=device),
        main_stick=torch.zeros(shape, dtype=torch.long, device=device),
    )


def labels_map(
    fn: Callable[..., torch.Tensor],
    labels: ControllerLabels,
    *rest: ControllerLabels,
) -> ControllerLabels:
    """Apply ``fn`` component-wise over one or more ``ControllerLabels``."""

    return ControllerLabels(
        buttons=fn(labels.buttons, *(other.buttons for other in rest)),
        main_stick=fn(labels.main_stick, *(other.main_stick for other in rest)),
    )


def labels_stack(items: Sequence[ControllerLabels], dim: int = 0) -> ControllerLabels:
    if not items:
        raise ValueError("labels_stack needs at least one element")
    return labels_map(lambda *components: torch.stack(components, dim=dim), *items)


def labels_cat(items: Sequence[ControllerLabels], dim: int = 0) -> ControllerLabels:
    if not items:
        raise ValueError("labels_cat needs at least one element")
    return labels_map(lambda *components: torch.cat(components, dim=dim), *items)


def labels_index(labels: ControllerLabels, index: Any) -> ControllerLabels:
    return labels_map(lambda component: component[index], labels)


def labels_to(labels: ControllerLabels, device: torch.device | str) -> ControllerLabels:
    return labels_map(lambda component: component.to(device), labels)


def labels_clone(labels: ControllerLabels) -> ControllerLabels:
    return labels_map(lambda component: component.clone(), labels)


__all__ = [
    "FRAME_SPEC",
    "ITEM_FIELDS",
    "NANA_FIELDS",
    "PLAYER_FIELDS",
    "FrameTree",
    "LeafKind",
    "frame_tree_to_game_state",
    "game_state_to_tree",
    "labels_cat",
    "labels_clone",
    "labels_index",
    "labels_map",
    "labels_stack",
    "labels_to",
    "leading_shape",
    "neutral_frame",
    "neutral_labels",
    "random_valid_frames",
    "tree_cat",
    "tree_clone",
    "tree_index",
    "tree_leaves",
    "tree_map",
    "tree_narrow",
    "tree_stack",
    "tree_to",
    "tree_to_packed",
    "validate_frame_tree",
]
