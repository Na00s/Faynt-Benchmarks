"""The shared-memory frame exchange (5 Sep 2026, lever 0 / IPC): layouts, views, and the worker environment's
outputs bit-identical to the pickled path's on the fake consoles."""

from __future__ import annotations

import numpy as np
import pytest
import torch

melee = pytest.importorskip("melee")  # the ``dolphin`` extra (melee==0.47.3)

from controller_codec import ControllerState, CustomV1Codec  # noqa: E402
from melee_rl.env.dolphin import ControllerRow, DolphinEnv, DolphinEnvConfig, neutral_row  # noqa: E402
from melee_rl.env.dolphin_mp import WorkerDolphinEnv, controller_arrays  # noqa: E402
from melee_rl.env.libmelee_frames import FrameBatch, leaf_neutral, leaf_paths  # noqa: E402
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvOutput  # noqa: E402
from melee_rl.env.shm import (  # noqa: E402
    CONTROLLER_FIELDS,
    ExchangeSpec,
    FrameExchange,
    controller_layout,
    frame_layout,
    neutral_fill,
)
from melee_rl.frames import neutral_labels, tree_leaves  # noqa: E402
from tests.rl import _mp_backend  # noqa: E402
from tests.rl._mp_backend import Script, _boot  # noqa: E402

START = INITIAL_FRAME_INDEX


def _neutral(batch: int) -> ControllerState:
    return CustomV1Codec().decode(neutral_labels((batch,)))


def _pressed(batch: int, name: str) -> ControllerState:
    state = _neutral(batch)
    buttons = state.buttons
    values = {field: getattr(buttons, field) for field in buttons.as_dict()}
    values[name] = torch.ones(batch, dtype=torch.bool)
    return ControllerState(
        main_stick=state.main_stick,
        c_stick=state.c_stick,
        shoulder=state.shoulder,
        buttons=type(buttons)(**values),
    )


def _assert_outputs_equal(a: EnvOutput, b: EnvOutput) -> None:
    assert torch.equal(a.frame_index, b.frame_index) and torch.equal(a.needs_reset, b.needs_reset)
    assert set(a.frames) == set(b.frames)
    for port in a.frames:
        left = dict(tree_leaves(a.frames[port]))
        right = dict(tree_leaves(b.frames[port]))
        assert set(left) == set(right)
        for path in left:
            assert torch.equal(left[path], right[path]), (port, path)
        assert torch.equal(a.rewards[port], b.rewards[port])


def test_layouts_and_views_round_trip() -> None:
    frames = frame_layout(5)
    paths = leaf_paths()
    assert [entry.path for entry in frames.entries][: len(paths)] == list(paths)
    assert frames.entries[-1].path == "frame_index" and frames.entries[-1].shape == (5,)
    assert all(entry.offset % 64 == 0 for entry in frames.entries) and frames.nbytes % 64 == 0
    assert (
        frames.entry("items.x").shape == (5, 15) and frames.entry("items.x").dtype == np.dtype(np.float32).str
    )
    controllers = controller_layout(5, (0, 1))
    assert len(controllers.entries) == 2 * len(CONTROLLER_FIELDS)
    assert controllers.entry("1.A").dtype == np.dtype(np.bool_).str

    parent = FrameExchange(5, (0, 1))
    try:
        worker = FrameExchange(5, (0, 1), spec=parent.spec)
        try:
            assert parent.slots == 2 and worker.spec == parent.spec
            # A worker's slice written in place is what the parent's whole-batch view shows.
            slice_batch = worker.frame_batch(1, 2, 5)
            assert slice_batch.batch == 3
            slice_batch.leaves["p0.x"][:] = [1.0, 2.0, 3.0]
            slice_batch.leaves["items.exists"][0, 4] = True
            worker.frame_index(1, 2, 5)[:] = [7, 8, 9]
            whole = parent.frame_batch(1)
            assert whole.leaves["p0.x"].tolist() == [0.0, 0.0, 1.0, 2.0, 3.0]
            assert whole.leaves["items.exists"][2, 4] and not whole.leaves["items.exists"][1, 4]
            assert parent.frame_index(1).tolist() == [0, 0, 7, 8, 9]
            # The other slot is untouched.
            assert parent.frame_batch(0).leaves["p0.x"].tolist() == [0.0] * 5
            # Controllers: the parent writes, the worker reads its slice.
            parent.write_controllers(1, controller_arrays(_pressed(5, "B"), 5))
            arrays = worker.controller_arrays(1, 3, 5)
            assert arrays["B"].tolist() == [True, True] and arrays["A"].tolist() == [False, False]
            assert arrays["main_x"].dtype == np.float32 and arrays["main_x"].tolist() == [0.5, 0.5]
            # neutral_fill restores the fresh-batch values (sticks at 0.5) on the slice it is given; a
            # new segment starts as zeros, so the untouched rows 0-1 keep 0.0 until a worker clears them.
            neutral_fill(slice_batch.leaves)
            assert whole.leaves["p0.x"].tolist() == [0.0] * 5
            assert whole.leaves["p0.controller.main_stick.x"].tolist() == [0.0, 0.0, 0.5, 0.5, 0.5]
            assert not whole.leaves["items.exists"].any()
            with pytest.raises(ValueError, match="does not match"):
                FrameExchange(4, (0, 1), spec=parent.spec)
        finally:
            worker.close()
    finally:
        parent.close()
        parent.unlink()
    assert ExchangeSpec(5, (0, 1), ("a", "b"), "c").slots == 2


def test_frame_batch_clear_and_adopted_step_rows() -> None:
    batch = FrameBatch(2)
    batch.leaves["p1.y"][:] = 3.0
    batch.leaves["items.type"][1, 2] = 9
    batch.clear()
    assert all(np.all(batch.leaves[path] == leaf_neutral(path)) for path in leaf_paths()) and batch.leaves[
        "p0.controller.c_stick.y"
    ].tolist() == [0.5, 0.5]
    backend = _mp_backend.FakeBackendMP({i: [Script(_boot())] for i in range(2)})
    env = DolphinEnv(
        DolphinEnvConfig(num_envs=2, check_iso=False, slippi_port=51900), cpu_level=9, backend=backend
    )
    try:
        rows = {0: [_row() for _ in range(2)]}
        fresh, index = env.step_rows(rows)
        adopted = FrameBatch(2)
        adopted.leaves["randall.x"][:] = 5.0  # stale content a reused slot might carry
        again, index_again = env.step_rows(rows, into=adopted, output=False)
        assert again is adopted and index_again == [value + 1 for value in index]
        assert adopted.leaves["randall.x"].tolist() == [0.0, 0.0]  # cleared before the write
        assert adopted.leaves["p0.x"].tolist() == [value + 1 for value in fresh.leaves["p0.x"].tolist()]
        assert env.current().frame_index.tolist() == index  # output=False left the pending state alone
        with pytest.raises(ValueError, match="rows"):
            env.step_rows(rows, into=FrameBatch(3))
    finally:
        env.close()


def _row() -> ControllerRow:
    return neutral_row()


def test_worker_env_shared_memory_matches_the_pickled_path() -> None:
    def build(shared: bool, port: int) -> WorkerDolphinEnv:
        config = DolphinEnvConfig(
            num_envs=4,
            worker_processes=2,
            mp_start_method="forkserver",
            console_timeout_s=0.05,
            launch_retries=1,
            slippi_port=port,
            check_iso=False,
            shared_memory=shared,
        )
        return WorkerDolphinEnv(config, cpu_level=9, backend_factory=_mp_backend.flaky_backend)

    with pytest.raises(ValueError, match="shared_memory"):
        DolphinEnvConfig(num_envs=2, worker_processes=0, shared_memory=True)
    pickled = build(False, 51910)
    shared = build(True, 51920)
    try:
        assert shared.shared_memory and not pickled.shared_memory
        _assert_outputs_equal(shared.current(), pickled.current())
        controls = [_neutral(4), _pressed(4, "A"), _neutral(4), _pressed(4, "X"), _neutral(4), _neutral(4)]
        outputs = []
        for control in controls:  # step 3 relaunches worker 1's first env in both modes (the flaky backend)
            a = shared.step({0: control})
            b = pickled.step({0: control})
            a.validate(4)
            _assert_outputs_equal(a, b)
            outputs.append(a)
        assert outputs[2].needs_reset.tolist() == [False, False, True, False]
        # The step before last is still readable after the last one (two slots).
        assert outputs[-2].frame_index.tolist() != outputs[-1].frame_index.tolist()
        assert torch.equal(outputs[-2].frame_index + 1, outputs[-1].frame_index)
        # Every leaf of every step (the echoed controller included) was compared above; the frames also
        # advanced, so the shared slots carried fresh conversions, not the boot state.
        assert outputs[0].frame_index.tolist() == [START + 1] * 4
        # stats / timing / reset keep working; the timing counters count the shared steps.
        assert [row["relaunches"] for row in shared.stats()] == [0, 0, 1, 0]
        timing = shared.timing()
        assert timing["parent"]["steps"] == 6 and [
            row["worker"]["messages"] for row in timing["workers"]
        ] == [6, 6]
        # (reset() relaunches every Dolphin; the flaky backend scripts one launch per env, so the reset
        # path -- unchanged, pickled -- is covered by test_dolphin_mp, not here.)
    finally:
        shared.close()
        pickled.close()
    assert shared.closed and [process.exitcode for process in shared._procs] == [0, 0]
