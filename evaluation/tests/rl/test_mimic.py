"""MIMIC as a live opponent (P12): the released decoder, the per-environment context window, the one
batched forward per frame, and the ``ControllerState`` the rollout worker sends.

The real MIMIC package is not installed here, so every test drives :class:`melee_rl.mimic.MimicApi`
with fakes.  What is pinned by these tests is *our* half of the boundary: the five upstream callables
we use (``tools.inference_utils``), the order we call them in, the state we keep per environment, and
the fact that MIMIC's sampling never touches the caller's global torch RNG.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest
import torch

from melee_rl.env.dolphin import ControllerRow, neutral_row
from melee_rl.mimic import (
    ControllerRecorder,
    MimicApi,
    MimicConfig,
    MimicPlayer,
    MimicRuntime,
    rows_to_state,
)
from melee_rl.opponents import MimicOpponent

SEQ = 3
"""The fake context window; the real bundle uses ``max_seq_len`` 60."""


# ---------------------------------------------------------------------------
# fakes: the five upstream callables plus PlayerState
# ---------------------------------------------------------------------------


class FakeButton:
    """A libmelee ``enums.Button`` stand-in: what the recorder is allowed to read is ``value``."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakeButton({self.value!r})"


MAIN = FakeButton("MAIN")
C = FakeButton("C")
A = FakeButton("A")
B = FakeButton("B")
X = FakeButton("X")
L = FakeButton("L")


class FakeGamestate:
    """Only ``frame`` and ``tag`` matter: the frame builder is faked."""

    def __init__(self, tag: int, frame: int = 0) -> None:
        self.tag = tag
        self.frame = frame


class FakePlayerState:
    """``tools.inference_utils.PlayerState`` semantics: a bounded deque with a mock prefill."""

    def __init__(self, model: Any, seq_len: int, device: str, ctx: Any = None) -> None:
        self.model = model
        self.seq_len = seq_len
        self.device = device
        self._ctx = ctx
        self._frame_cache: deque[dict[str, torch.Tensor]] = deque(maxlen=seq_len)
        self.prev_sent: dict[str, Any] | None = None

    def push_frame(self, frame: dict[str, torch.Tensor]) -> None:
        if not self._frame_cache:
            mock = {key: torch.full_like(value, -1.0) for key, value in frame.items()}
            for _ in range(self.seq_len - 1):
                self._frame_cache.append({key: value.clone() for key, value in mock.items()})
        self._frame_cache.append(frame)


class FakeModel:
    """Records every batch it is given and answers with per-row logits keyed off the row's value."""

    def __init__(self, *, n_btn: int = 7, n_cdir: int = 9) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []
        self.n_btn = n_btn
        self.n_cdir = n_cdir

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls.append({key: value.clone() for key, value in batch.items()})
        rows = int(batch["state"].shape[0])
        window = int(batch["state"].shape[1])
        # The last frame of each row carries the tag the frame builder wrote.
        tags = batch["state"][:, window - 1, 0]
        return {
            "main_xy": self._logits(tags, 37, offset=0),
            "shoulder_val": self._logits(tags, 3, offset=1),
            "c_dir_logits": self._logits(tags, self.n_cdir, offset=2),
            "btn_logits": self._logits(tags, self.n_btn, offset=3),
            "rows": torch.arange(rows, dtype=torch.float32).reshape(rows, 1, 1),
        }

    @staticmethod
    def _logits(tags: torch.Tensor, classes: int, *, offset: int) -> torch.Tensor:
        """``[R, 1, classes]`` one-hot at ``(tag + offset) % classes``: argmax is a function of the tag."""

        rows = int(tags.shape[0])
        out = torch.zeros(rows, 1, classes)
        for row in range(rows):
            out[row, 0, (int(tags[row].item()) + offset) % classes] = 10.0
        return out


def fake_build_frame(gs: Any, prev_sent: dict[str, Any] | None, ctx: dict[str, Any]) -> Any:
    """``[1, 2]``: the gamestate tag and whether a previous command was threaded in."""

    if gs is None or getattr(gs, "tag", None) is None:
        return None
    threaded = 0.0 if prev_sent is None else float(prev_sent.get("main_x", 0.0))
    return {"state": torch.tensor([[float(gs.tag), threaded]])}


def fake_build_frame_p2(gs: Any, prev_sent: dict[str, Any] | None, ctx: dict[str, Any]) -> Any:
    frame = fake_build_frame(gs, prev_sent, ctx)
    if frame is None:
        return None
    frame["state"] = frame["state"] + torch.tensor([[100.0, 0.0]])
    return frame


def fake_decode_and_press(
    ctrl: Any,
    preds: dict[str, torch.Tensor],
    prev_sent: dict[str, Any] | None,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """A faithful miniature of the released decoder: sample each head, press, return ``prev_sent``."""

    main = int(torch.argmax(preds["main_xy"][0, -1]).item())
    shoulder = [0.0, 0.4, 1.0][int(torch.argmax(preds["shoulder_val"][0, -1]).item())]
    cdir = int(torch.argmax(preds["c_dir_logits"][0, -1]).item())
    button = int(torch.argmax(preds["btn_logits"][0, -1]).item())
    # A live draw on the global RNG: the isolation test proves it never leaks out.
    jitter = float(torch.rand(1).item()) * 1e-6
    main_x = (main % 7) / 6.0 + jitter
    main_y = (main % 5) / 4.0
    ctrl.tilt_analog(MAIN, main_x, main_y)
    ctrl.tilt_analog(C, cdir / 8.0, 0.5)
    for released in (A, B, X, L):
        ctrl.release_button(released)
    ctrl.press_shoulder(L, shoulder)
    pressed: list[str] = []
    if button == 0:
        ctrl.press_button(A)
        pressed.append("A")
    elif button == 4:
        ctrl.press_button(L)
        pressed.append("TRIG")
    elif button == 5:
        ctrl.press_button(A)
        ctrl.press_button(L)
        pressed.append("A+TRIG")
    # The released decoder ends every frame with a flush; the recorder must accept one.
    ctrl.flush()
    sent = {
        "main_x": main_x,
        "main_y": main_y,
        "c_x": cdir / 8.0,
        "c_y": 0.5,
        "l_shldr": shoulder,
        "r_shldr": 0.0,
        "btn_A": 1 if "A" in pressed else 0,
    }
    return sent, pressed, ["A", "B", "Z", "JUMP", "TRIG", "A+TRIG", "NONE"]


def fake_api(model: FakeModel) -> MimicApi:
    return MimicApi(
        load_mimic_model=lambda path, device: (model, _FakeConfig()),
        load_inference_context=lambda directory: {"marker": str(directory)},
        build_frame=fake_build_frame,
        build_frame_p2=fake_build_frame_p2,
        decode_and_press=fake_decode_and_press,
        player_state=FakePlayerState,
    )


class _FakeConfig:
    max_seq_len = SEQ


def make_runtime(model: FakeModel | None = None) -> MimicRuntime:
    used = FakeModel() if model is None else model
    return MimicRuntime(
        api=fake_api(used),
        model=used,
        max_seq_len=SEQ,
        context={"marker": "fake"},
        device="cpu",
        checkpoint="/opt/mimic/fox-master/model.pt",
    )


def make_player(runtime: MimicRuntime, envs: int, *, player: int = 1, seed: int = 0) -> MimicPlayer:
    return MimicPlayer(runtime, envs, player=player, seed=seed)


def resets(envs: int, *, value: bool = False) -> torch.Tensor:
    return torch.full((envs,), value, dtype=torch.bool)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_config_defaults_to_the_bundle_checkpoint() -> None:
    config = MimicConfig(source_dir="/opt/mimic/source", asset_dir="/opt/mimic/fox-master")
    assert config.resolved_checkpoint == "/opt/mimic/fox-master/model.pt"


def test_config_takes_an_explicit_checkpoint() -> None:
    config = MimicConfig(
        source_dir="/opt/mimic/source", asset_dir="/opt/mimic/fox-master", checkpoint="/tmp/other.pt"
    )
    assert config.resolved_checkpoint == "/tmp/other.pt"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_dir": ""},
        {"asset_dir": ""},
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"top_k": -1},
        {"top_p": 1.5},
        {"player": 2},
        {"device": ""},
    ],
)
def test_config_rejects_nonsense(kwargs: dict[str, Any]) -> None:
    base = {"source_dir": "/opt/mimic/source", "asset_dir": "/opt/mimic/fox-master"}
    with pytest.raises(ValueError):
        MimicConfig(**{**base, **kwargs})


# ---------------------------------------------------------------------------
# the controller recorder
# ---------------------------------------------------------------------------


def test_recorder_collects_one_frame_of_the_released_decoder() -> None:
    recorder = ControllerRecorder()
    recorder.tilt_analog(MAIN, 0.25, 0.75)
    recorder.tilt_analog(C, 1.0, 0.5)
    for button in (A, B, X, L):
        recorder.release_button(button)
    recorder.press_shoulder(L, 0.4)
    recorder.press_button(A)
    recorder.press_button(L)

    row = recorder.row()
    assert row.main_x == pytest.approx(0.25)
    assert row.main_y == pytest.approx(0.75)
    assert row.c_x == pytest.approx(1.0)
    assert row.c_y == pytest.approx(0.5)
    assert row.shoulder == pytest.approx(0.4)
    # BUTTON_ORDER is ("A", "B", "X", "Y", "Z", "L", "R", "D_UP").
    assert row.buttons == (True, False, False, False, False, True, False, False)


def test_recorder_release_after_press_wins() -> None:
    """The decoder releases everything and then presses; a stale press must not survive a release."""

    recorder = ControllerRecorder()
    recorder.press_button(A)
    recorder.release_button(A)
    assert recorder.row().buttons[0] is False


def test_recorder_starts_neutral() -> None:
    assert ControllerRecorder().row() == neutral_row()


def test_recorder_counts_flushes_without_performing_them() -> None:
    """``decode_and_press`` calls ``ctrl.flush()``; ``Console.step`` is what actually sends the frame."""

    recorder = ControllerRecorder()
    assert recorder.flushes == 0
    recorder.tilt_analog(MAIN, 0.25, 0.75)
    recorder.flush()
    assert recorder.flushes == 1
    assert recorder.row().main_x == pytest.approx(0.25)  # the values survive the flush


def test_recorder_ignores_the_right_shoulder_axis_but_keeps_the_digital_press() -> None:
    """``ControllerRow`` carries one analog shoulder (L); R stays digital, as in ``send_controller``."""

    recorder = ControllerRecorder()
    recorder.press_shoulder(FakeButton("R"), 1.0)
    recorder.press_button(FakeButton("R"))
    row = recorder.row()
    assert row.shoulder == pytest.approx(0.0)
    assert row.buttons[6] is True


def test_recorder_rejects_an_unknown_button() -> None:
    with pytest.raises(ValueError, match="unknown"):
        ControllerRecorder().press_button(FakeButton("START"))


# ---------------------------------------------------------------------------
# the player: one batched forward, per-environment state
# ---------------------------------------------------------------------------


def test_one_batched_forward_per_frame_over_every_environment() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 4)
    player.rows([FakeGamestate(i) for i in range(4)], resets(4))

    assert len(model.calls) == 1
    batch = model.calls[0]["state"]
    assert tuple(batch.shape) == (4, SEQ, 2)


def test_rows_differ_per_environment_and_track_the_gamestate() -> None:
    player = make_player(make_runtime(), 3)
    rows = player.rows([FakeGamestate(i) for i in (0, 1, 2)], resets(3))

    assert len(rows) == 3
    assert len({row.main_x for row in rows}) == 3


def test_the_previous_command_is_threaded_back_into_the_next_frame() -> None:
    """MIMIC sees the controller it sent last frame, per environment (``prev_sent``)."""

    model = FakeModel()
    player = make_player(make_runtime(model), 2)
    first = player.rows([FakeGamestate(1), FakeGamestate(2)], resets(2))
    player.rows([FakeGamestate(1), FakeGamestate(2)], resets(2))

    threaded = model.calls[1]["state"][:, SEQ - 1, 1]
    assert threaded.tolist() == pytest.approx([first[0].main_x, first[1].main_x])
    # The first frame has nothing to thread.
    assert model.calls[0]["state"][:, SEQ - 1, 1].tolist() == [0.0, 0.0]


def test_the_context_window_fills_and_then_slides() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 1, player=0)
    for tag in range(SEQ + 2):
        player.rows([FakeGamestate(tag)], resets(1))

    # The prefill is the mock frame (-1.0); after SEQ pushes only real frames remain.
    assert model.calls[0]["state"][0, 0, 0].item() == pytest.approx(-1.0)
    tail = model.calls[-1]["state"][0, :, 0].tolist()
    assert tail == pytest.approx([float(SEQ - 1), float(SEQ), float(SEQ + 1)])


def test_a_reset_forgets_that_environment_only() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 2, player=0)
    for tag in range(SEQ):
        player.rows([FakeGamestate(tag), FakeGamestate(tag + 50)], resets(2))

    mask = torch.tensor([True, False])
    player.rows([FakeGamestate(7), FakeGamestate(57)], mask)

    window = model.calls[-1]["state"][:, :, 0]
    # Environment 0 was reset: its window is mock-prefilled again around the single new frame.
    assert window[0, 0].item() == pytest.approx(-1.0)
    assert window[0, SEQ - 1].item() == pytest.approx(7.0)
    # Environment 1 kept its history.
    assert window[1, 0].item() == pytest.approx(51.0)
    assert window[1, SEQ - 1].item() == pytest.approx(57.0)


def test_a_reset_also_drops_the_previous_command() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 1)
    player.rows([FakeGamestate(1)], resets(1))
    player.rows([FakeGamestate(1)], torch.tensor([True]))

    assert model.calls[-1]["state"][0, SEQ - 1, 1].item() == 0.0


def test_a_missing_gamestate_yields_a_neutral_row_and_no_push() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 2)
    rows = player.rows([None, FakeGamestate(3)], resets(2))

    assert rows[0] == neutral_row()
    assert len(model.calls) == 1
    # Only the live environment reaches the forward; the parked one has no context to send.
    assert tuple(model.calls[0]["state"].shape) == (1, SEQ, 2)


def test_a_builder_that_declines_the_frame_yields_a_neutral_row() -> None:
    """``build_frame`` returns ``None`` before both players exist; that frame is skipped."""

    runtime = make_runtime()
    player = make_player(runtime, 1)
    rows = player.rows([FakeGamestate(None)], resets(1))  # type: ignore[arg-type]
    assert rows[0] == neutral_row()


def test_every_environment_declining_skips_the_forward_entirely() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 2)
    rows = player.rows([None, None], resets(2))

    assert rows == [neutral_row(), neutral_row()]
    assert model.calls == []


def test_player_two_uses_the_second_perspective_builder() -> None:
    model = FakeModel()
    make_player(make_runtime(model), 1, player=1).rows([FakeGamestate(4)], resets(1))
    first = model.calls[-1]["state"][0, SEQ - 1, 0].item()

    other = FakeModel()
    make_player(make_runtime(other), 1, player=0).rows([FakeGamestate(4)], resets(1))
    second = other.calls[-1]["state"][0, SEQ - 1, 0].item()

    assert first == pytest.approx(104.0)  # build_frame_p2 adds 100
    assert second == pytest.approx(4.0)


def test_the_batch_size_must_match_the_environment_count() -> None:
    player = make_player(make_runtime(), 2)
    with pytest.raises(ValueError, match="2"):
        player.rows([FakeGamestate(1)], resets(2))
    with pytest.raises(ValueError, match="2"):
        player.rows([FakeGamestate(1), FakeGamestate(2)], resets(3))


# ---------------------------------------------------------------------------
# sampling RNG
# ---------------------------------------------------------------------------


def test_the_same_seed_replays_the_same_match() -> None:
    one = make_player(make_runtime(), 2, seed=11)
    two = make_player(make_runtime(), 2, seed=11)
    states = [FakeGamestate(5), FakeGamestate(6)]

    assert one.rows(states, resets(2)) == two.rows(states, resets(2))


def test_different_seeds_diverge() -> None:
    one = make_player(make_runtime(), 1, seed=1)
    two = make_player(make_runtime(), 1, seed=2)
    states = [FakeGamestate(5)]

    assert one.rows(states, resets(1)) != two.rows(states, resets(1))


def test_mimic_sampling_leaves_the_caller_rng_untouched() -> None:
    """Our policy samples from the global generator too; MIMIC must not consume its stream."""

    player = make_player(make_runtime(), 2, seed=3)
    torch.manual_seed(1234)
    expected = torch.rand(4)

    torch.manual_seed(1234)
    player.rows([FakeGamestate(1), FakeGamestate(2)], resets(2))
    observed = torch.rand(4)

    assert torch.equal(expected, observed)


# ---------------------------------------------------------------------------
# rows -> ControllerState
# ---------------------------------------------------------------------------


def test_rows_to_state_batches_the_rows_in_order() -> None:
    rows = [
        ControllerRow(0.1, 0.2, 0.3, 0.4, 0.5, (True, False, False, False, False, False, False, False)),
        ControllerRow(0.9, 0.8, 0.7, 0.6, 1.0, (False, True, False, False, False, True, False, False)),
    ]
    state = rows_to_state(rows)

    assert tuple(state.shoulder.shape) == (2,)
    assert state.main_stick.x.tolist() == pytest.approx([0.1, 0.9])
    assert state.main_stick.y.tolist() == pytest.approx([0.2, 0.8])
    assert state.c_stick.x.tolist() == pytest.approx([0.3, 0.7])
    assert state.c_stick.y.tolist() == pytest.approx([0.4, 0.6])
    assert state.shoulder.tolist() == pytest.approx([0.5, 1.0])
    assert state.buttons.A.tolist() == [True, False]
    assert state.buttons.B.tolist() == [False, True]
    assert state.buttons.L.tolist() == [False, True]


def test_rows_to_state_rejects_an_empty_batch() -> None:
    with pytest.raises(ValueError):
        rows_to_state([])


# ---------------------------------------------------------------------------
# the opponent
# ---------------------------------------------------------------------------


def test_opponent_reports_its_port_and_that_it_drives_it() -> None:
    player = make_player(make_runtime(), 2)
    opponent = MimicOpponent(player, lambda: [FakeGamestate(1), FakeGamestate(2)], port=1)

    assert opponent.port == 1
    assert opponent.controls_port is True


def test_opponent_act_state_matches_the_player_rows() -> None:
    states = [FakeGamestate(1), FakeGamestate(2)]
    expected = rows_to_state(make_player(make_runtime(), 2, seed=5).rows(states, resets(2)))

    opponent = MimicOpponent(make_player(make_runtime(), 2, seed=5), lambda: states, port=1)
    observed = opponent.act_state(None, resets(2))

    assert observed.main_stick.x.tolist() == pytest.approx(expected.main_stick.x.tolist())
    assert observed.buttons.A.tolist() == expected.buttons.A.tolist()


def test_opponent_refuses_the_label_path() -> None:
    """MIMIC's 37 stick clusters do not survive Ali's 85-value codec; the worker must use ``act_state``."""

    opponent = MimicOpponent(make_player(make_runtime(), 1), lambda: [FakeGamestate(1)], port=1)
    with pytest.raises(NotImplementedError, match="act_state"):
        opponent.act(None, resets(1))


def test_opponent_reset_state_clears_every_environment() -> None:
    model = FakeModel()
    player = make_player(make_runtime(model), 1)
    states = [FakeGamestate(9)]
    opponent = MimicOpponent(player, lambda: states, port=1)
    opponent.act_state(None, resets(1))
    opponent.reset_state()
    opponent.act_state(None, resets(1))

    assert model.calls[-1]["state"][0, 0, 0].item() == pytest.approx(-1.0)


def test_opponent_refresh_is_a_noop() -> None:
    opponent = MimicOpponent(make_player(make_runtime(), 1), lambda: [FakeGamestate(1)], port=1)
    opponent.refresh(None)  # type: ignore[arg-type]
