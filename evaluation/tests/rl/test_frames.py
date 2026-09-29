from __future__ import annotations

import pytest
import torch

from controller_codec import ControllerLabels
from melee_rl import frames
from melee_rl.adapter import MeleePolicyAdapter
from model import SlippiEncoder
from tensor_batch import MAX_ITEMS, GameStateBatch


def test_neutral_frame_is_valid_encodable_and_round_trips(tiny_adapter: MeleePolicyAdapter) -> None:
    tree = frames.neutral_frame((2, 3))
    assert frames.validate_frame_tree(tree) == torch.Size((2, 3))
    assert frames.leading_shape(tree) == torch.Size((2, 3))
    assert tree["items"]["exists"].shape == (2, 3, MAX_ITEMS)
    assert tree["p0"]["controller"]["main_stick"]["x"].tolist() == [[0.5] * 3] * 2
    assert tree["stage"].dtype == torch.int64 and tree["p1"]["facing"].dtype == torch.bool

    state = frames.frame_tree_to_game_state(tree)
    assert isinstance(state, GameStateBatch)
    assert state.p0.controller.main_stick.x is tree["p0"]["controller"]["main_stick"]["x"]
    back = frames.game_state_to_tree(state)
    pairs = zip(frames.tree_leaves(tree), frames.tree_leaves(back), strict=True)
    for (path, leaf), (other_path, other) in pairs:
        assert path == other_path
        assert leaf is other

    encoded = tiny_adapter.encoder(tree, frames.neutral_labels((2, 3)))
    assert encoded.shape == (2, 3, tiny_adapter.encoder.output_dim)
    assert torch.isfinite(encoded).all()
    typed = tiny_adapter.encoder(state, frames.neutral_labels((2, 3)))
    torch.testing.assert_close(typed, encoded, rtol=0.0, atol=0.0)


def test_random_valid_frames_deterministic_in_range_encodable(tiny_adapter: MeleePolicyAdapter) -> None:
    first = frames.random_valid_frames((3, 5), torch.Generator().manual_seed(11))
    second = frames.random_valid_frames((3, 5), torch.Generator().manual_seed(11))
    for (path, leaf), (_, other) in zip(frames.tree_leaves(first), frames.tree_leaves(second), strict=True):
        assert torch.equal(leaf, other), path
    frames.validate_frame_tree(first, (3, 5))

    for player in ("p0", "p1"):
        assert int(first[player]["character"].max()) < SlippiEncoder.CHARACTER_SIZE
        assert int(first[player]["jumps_left"].max()) < SlippiEncoder.JUMPS_SIZE
        assert int(first[player]["action"].max()) < SlippiEncoder.ACTION_SIZE
        assert int(first[player]["nana"]["character"].max()) < SlippiEncoder.CHARACTER_SIZE
        stick = first[player]["controller"]["main_stick"]
        assert bool(((stick["x"] >= 0.0) & (stick["x"] <= 1.0)).all())
    assert int(first["stage"].max()) < SlippiEncoder.STAGE_SIZE
    assert int(first["items"]["type"].max()) < SlippiEncoder.ITEM_TYPE_INPUT_SIZE
    assert int(first["items"]["state"].max()) < SlippiEncoder.ITEM_STATE_INPUT_SIZE

    encoded = tiny_adapter.encoder(first, frames.neutral_labels((3, 5)))
    assert encoded.shape == (3, 5, tiny_adapter.encoder.output_dim)
    assert torch.isfinite(encoded).all()
    assert tiny_adapter.encode_observation(frames.tree_index(first, (slice(None), 0))).stage.shape == (3,)


def test_tree_operations_round_trip() -> None:
    generator = torch.Generator().manual_seed(5)
    steps = [frames.random_valid_frames((4,), generator) for _ in range(6)]
    stacked = frames.tree_stack(steps, dim=1)
    assert frames.validate_frame_tree(stacked) == torch.Size((4, 6))
    for index, step in enumerate(steps):
        picked = frames.tree_index(stacked, (slice(None), index))
        pairs = zip(frames.tree_leaves(picked), frames.tree_leaves(step), strict=True)
        for (path, leaf), (_, other) in pairs:
            assert torch.equal(leaf, other), path

    head = frames.tree_narrow(stacked, 1, 0, 2)
    tail = frames.tree_narrow(stacked, 1, 2, 4)
    assert frames.leading_shape(head) == torch.Size((4, 2))
    joined = frames.tree_cat([head, tail], dim=1)
    for (path, leaf), (_, other) in zip(frames.tree_leaves(joined), frames.tree_leaves(stacked), strict=True):
        assert torch.equal(leaf, other), path

    moved = frames.tree_to(stacked, "cpu")
    cloned = frames.tree_clone(stacked)
    assert moved["stage"].device.type == "cpu"
    assert cloned["stage"] is not stacked["stage"] and torch.equal(cloned["stage"], stacked["stage"])
    assert frames.tree_map(lambda leaf: leaf.shape[0], stacked)["randall"]["x"] == 4
    with pytest.raises(ValueError, match="at least one"):
        frames.tree_stack([])


def test_tree_to_packed_matches_tree_to() -> None:
    """P8-PLAN.md 3.1: dtype-grouped packing is bit-identical to the leaf-wise move."""

    generator = torch.Generator().manual_seed(31)
    tree = frames.random_valid_frames((3, 2), generator)
    packed = frames.tree_to_packed(tree, "cpu")
    plain = frames.tree_to(tree, "cpu")
    pairs = zip(frames.tree_leaves(packed), frames.tree_leaves(plain), strict=True)
    for (path, leaf), (other_path, other) in pairs:
        assert path == other_path
        assert leaf.dtype == other.dtype and leaf.shape == other.shape, path
        assert leaf.device == other.device, path
        assert torch.equal(leaf, other), path
    assert frames.validate_frame_tree(packed) == torch.Size((3, 2))
    assert packed["items"]["x"].shape == (3, 2, MAX_ITEMS)

    # The point of the function: one copy per dtype group, so leaves of a dtype share one storage.
    for dtype in (torch.float32, torch.int64, torch.bool):
        pointers = {
            leaf.untyped_storage().data_ptr() for _, leaf in frames.tree_leaves(packed) if leaf.dtype == dtype
        }
        assert len(pointers) == 1, dtype

    # Non-tensor leaves go through as_tensor, exactly like tree_to; an empty tree stays empty.
    mixed = {"a": {"flag": [1.0, 2.0]}, "count": torch.tensor([3, 4])}
    packed_mixed = frames.tree_to_packed(mixed, "cpu")
    assert torch.equal(packed_mixed["a"]["flag"], torch.tensor([1.0, 2.0]))
    assert torch.equal(packed_mixed["count"], torch.tensor([3, 4]))
    assert frames.tree_to_packed({}, "cpu") == {}


def test_validate_frame_tree_errors() -> None:
    tree = frames.neutral_frame((2,))
    del tree["p1"]["nana"]["exists"]
    with pytest.raises(ValueError, match=r"missing p1\.nana\.exists"):
        frames.validate_frame_tree(tree)

    tree = frames.neutral_frame((2,))
    tree["stage"] = torch.zeros((3,), dtype=torch.int64)
    with pytest.raises(ValueError, match="stage must have shape"):
        frames.validate_frame_tree(tree)

    tree = frames.neutral_frame((2,))
    tree["p0"]["facing"] = torch.zeros((2,), dtype=torch.float32)
    with pytest.raises(TypeError, match=r"torch\.bool"):
        frames.validate_frame_tree(tree)

    tree = frames.neutral_frame((2,))
    tree["p0"]["percent"] = torch.zeros((2,), dtype=torch.float32)
    with pytest.raises(TypeError, match="integer dtype"):
        frames.validate_frame_tree(tree)

    tree = frames.neutral_frame((2,))
    tree["items"]["x"] = [0.0] * 15
    with pytest.raises(TypeError, match="must be a tensor"):
        frames.validate_frame_tree(tree)
    with pytest.raises(ValueError, match=r"p0\.percent"):
        frames.validate_frame_tree({"p1": {}})


def test_label_helpers() -> None:
    neutral = frames.neutral_labels((2, 3))
    assert neutral.buttons.dtype == torch.long and neutral.main_stick.shape == (2, 3)
    assert int(neutral.buttons.sum()) == 0 and int(neutral.main_stick.sum()) == 0

    generator = torch.Generator().manual_seed(2)
    steps = [
        ControllerLabels(
            buttons=torch.randint(0, 728, (4,), generator=generator),
            main_stick=torch.randint(0, 85, (4,), generator=generator),
        )
        for _ in range(3)
    ]
    stacked = frames.labels_stack(steps, dim=1)
    assert stacked.buttons.shape == (4, 3)
    for index, step in enumerate(steps):
        picked = frames.labels_index(stacked, (slice(None), index))
        assert torch.equal(picked.buttons, step.buttons) and torch.equal(picked.main_stick, step.main_stick)
    joined = frames.labels_cat([frames.labels_index(stacked, (slice(None), slice(0, 1))), stacked], dim=1)
    assert joined.main_stick.shape == (4, 4)
    cloned = frames.labels_clone(stacked)
    assert cloned.buttons is not stacked.buttons and torch.equal(cloned.buttons, stacked.buttons)
    assert frames.labels_to(stacked, "cpu").buttons.device.type == "cpu"
    with pytest.raises(ValueError, match="at least one"):
        frames.labels_stack([])
