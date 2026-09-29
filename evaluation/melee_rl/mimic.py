"""MIMIC as a live opponent (P12, 29 Aug 2026).

MIMIC (`erickfm/MIMIC <https://github.com/erickfm/MIMIC>`_, MIT) is a released behaviour-cloning Melee
policy with its own observation, its own action space and its own decoder.  This module drives it
through **its own released inference path** -- the five callables of ``tools/inference_utils.py`` --
so that nothing of MIMIC's behaviour is re-derived here:

``load_mimic_model`` / ``load_inference_context``
    build the ``FramePredictor`` and the normalisation/enum bundle from ``fox-master``.
``build_frame`` / ``build_frame_p2``
    turn one raw libmelee ``GameState`` plus *the controller MIMIC last sent* into one frame of its
    context window, from player 1's or player 2's perspective.
``decode_and_press``
    sample its four heads and press a libmelee controller.

Two things had to be built around them.

* **A controller recorder.**  ``decode_and_press`` presses a controller instead of returning values,
  so :class:`ControllerRecorder` stands in for one (the same adapter trick as
  ``melee-policy``'s ``mimic_vs_hal._MimicCompleteControllerFrame``) and hands back a
  :class:`~melee_rl.env.dolphin.ControllerRow`.  Going through Ali's codec instead would quantise
  MIMIC's 37 stick clusters onto the 85-value ``main_stick`` vocabulary and lose the distinction
  between a digital ``L`` press and an analog shoulder, so the rows reach ``DolphinEnv`` unrounded via
  :func:`rows_to_state`.
* **A batch.**  MIMIC has no KV cache: every frame it re-runs its whole ``max_seq_len`` window
  (60 frames, ``d_model`` 1024 x 4 layers ~ 96 GFLOP at batch 16), which is seconds per frame on a CPU.
  :class:`MimicPlayer` keeps one upstream ``PlayerState`` per environment but concatenates their
  windows into **one forward per frame** for the whole batch, which is what makes a GPU worth having.

The sampling RNG is MIMIC's own: ``_safe_sample`` draws from the *global* torch generator, so every
decode of a frame runs inside :class:`_RngStream`, which swaps a private generator state in and the
caller's state back out.  Our policy samples from its own ``torch.Generator`` and is never disturbed.

Pinned upstream (``melee-policy/configs/integration_mimic_vs_hal.toml``, Ali's proven pair): source
:data:`MIMIC_SOURCE_REVISION`, weights :data:`MIMIC_ASSET_REVISION` (``fox-master``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from controller_codec import ButtonState, ControllerState, StickState
from melee_rl.env.dolphin import ControllerRow, neutral_row
from tensor_batch import BUTTON_ORDER

MIMIC_SOURCE_REVISION: Final[str] = "70c925b7675202d853472c4e54332790ad76efe7"
"""The MIMIC checkout the weights were validated against (E001-M1 and Ali's benchmark launcher)."""
MIMIC_ASSET_REVISION: Final[str] = "e9a805cefc590f3f06eb1d520fc6a33ea9c66c07"
"""The Hugging Face revision of ``erickfm/MIMIC`` holding the ``fox-master`` bundle."""
MIMIC_ASSET_SUBDIR: Final[str] = "fox-master"
DEFAULT_SOURCE_DIR: Final[str] = "/opt/mimic/source"
"""Default location for the separately supplied :data:`MIMIC_SOURCE_REVISION` source tree."""
DEFAULT_ASSET_DIR: Final[str] = "/opt/mimic/fox-master"
CHECKPOINT_NAME: Final[str] = "model.pt"

_MAIN: Final[str] = "MAIN"
_C: Final[str] = "C"
_L: Final[str] = "L"
_R: Final[str] = "R"


# ---------------------------------------------------------------------------
# configuration and the upstream boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MimicConfig:
    """Where MIMIC lives and how it is sampled.

    ``player`` is the *environment* player index (0 = Dolphin port 1, 1 = port 2), which selects the
    perspective builder; ``melee-policy`` puts MIMIC on port 2 and we keep that.  The sampling knobs
    are the released defaults of ``configs/integration_mimic_vs_hal.toml`` ``[mimic]``.
    """

    source_dir: str = DEFAULT_SOURCE_DIR
    asset_dir: str = DEFAULT_ASSET_DIR
    checkpoint: str = ""
    device: str = "cpu"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0
    seed: int = 0
    player: int = 1

    def __post_init__(self) -> None:
        if not self.source_dir:
            raise ValueError("[mimic] source_dir must name the MIMIC checkout")
        if not self.asset_dir:
            raise ValueError("[mimic] asset_dir must name the released bundle directory")
        if not self.device:
            raise ValueError("[mimic] device must be a torch device string")
        if self.temperature <= 0.0:
            raise ValueError(f"[mimic] temperature must be > 0, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"[mimic] top_k must be >= 0, got {self.top_k}")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(f"[mimic] top_p must be in [0, 1], got {self.top_p}")
        if self.player not in (0, 1):
            raise ValueError(f"[mimic] player must be 0 or 1, got {self.player}")

    @property
    def resolved_checkpoint(self) -> str:
        """The explicit ``checkpoint``, or ``<asset_dir>/model.pt`` (the released bundle layout)."""

        if self.checkpoint:
            return self.checkpoint
        return str(Path(self.asset_dir) / CHECKPOINT_NAME)


@dataclass(frozen=True)
class MimicApi:
    """The five upstream callables plus ``PlayerState``; faked wholesale in ``tests/rl/test_mimic.py``."""

    load_mimic_model: Callable[[str, str], tuple[Any, Any]]
    load_inference_context: Callable[[str], dict[str, Any]]
    build_frame: Callable[[Any, dict[str, Any] | None, dict[str, Any]], dict[str, Any] | None]
    build_frame_p2: Callable[[Any, dict[str, Any] | None, dict[str, Any]], dict[str, Any] | None]
    decode_and_press: Callable[..., tuple[dict[str, Any], list[str], list[str]]]
    player_state: Callable[[Any, int, str, Any], Any]


def load_mimic_api(source_dir: str | Path) -> MimicApi:
    """Import ``tools.inference_utils`` from a MIMIC checkout, putting it on ``sys.path`` first."""

    root = Path(source_dir).expanduser().resolve()
    module = root / "tools" / "inference_utils.py"
    if not module.is_file():
        raise FileNotFoundError(f"MIMIC checkout has no tools/inference_utils.py: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.inference_utils import (  # type: ignore[import-not-found]
        PlayerState,
        build_frame,
        build_frame_p2,
        decode_and_press,
        load_inference_context,
        load_mimic_model,
    )

    return MimicApi(
        load_mimic_model=load_mimic_model,
        load_inference_context=load_inference_context,
        build_frame=build_frame,
        build_frame_p2=build_frame_p2,
        decode_and_press=decode_and_press,
        player_state=PlayerState,
    )


@dataclass(frozen=True)
class MimicRuntime:
    """One loaded MIMIC: the model, its context window length and its normalisation bundle."""

    api: MimicApi
    model: Any
    max_seq_len: int
    context: dict[str, Any]
    device: str
    checkpoint: str


def load_runtime(config: MimicConfig, api: MimicApi | None = None) -> MimicRuntime:
    """Load the released bundle (or drive an injected :class:`MimicApi`)."""

    used = load_mimic_api(config.source_dir) if api is None else api
    checkpoint = config.resolved_checkpoint
    model, model_config = used.load_mimic_model(checkpoint, config.device)
    context = used.load_inference_context(config.asset_dir)
    return MimicRuntime(
        api=used,
        model=model,
        max_seq_len=int(model_config.max_seq_len),
        context=context,
        device=config.device,
        checkpoint=checkpoint,
    )


# ---------------------------------------------------------------------------
# the controller recorder
# ---------------------------------------------------------------------------


class ControllerRecorder:
    """A libmelee ``Controller`` stand-in that collects one frame into a :class:`ControllerRow`.

    Only the four methods ``decode_and_press`` calls are implemented, and only ``button.value`` is
    read, so the recorder is independent of the installed libmelee version.  The analog ``R`` axis is
    tracked but not carried into the row: ``send_controller`` sends one analog shoulder (``L``), which
    is all MIMIC's decoder ever sets.
    """

    __slots__ = ("_buttons", "_c", "_flushes", "_l_shoulder", "_main", "_r_shoulder")

    def __init__(self) -> None:
        neutral = neutral_row()
        self._buttons = dict.fromkeys(BUTTON_ORDER, False)
        self._main = (neutral.main_x, neutral.main_y)
        self._c = (neutral.c_x, neutral.c_y)
        self._l_shoulder = neutral.shoulder
        self._r_shoulder = 0.0
        self._flushes = 0

    @property
    def r_shoulder(self) -> float:
        """The analog ``R`` value the decoder asked for; not carried into the row."""

        return self._r_shoulder

    @property
    def flushes(self) -> int:
        """How often the decoder asked to send the frame (``decode_and_press`` ends with one flush)."""

        return self._flushes

    def flush(self) -> None:
        """Accepted and counted, not performed: ``DolphinEnv.step`` owns the pipe and flushes it once."""

        self._flushes += 1

    def _name(self, button: Any) -> str:
        return str(getattr(button, "value", button))

    def tilt_analog(self, button: Any, x: float, y: float) -> None:
        name = self._name(button)
        if name == _MAIN:
            self._main = (float(x), float(y))
        elif name == _C:
            self._c = (float(x), float(y))
        else:
            raise ValueError(f"tilt_analog on a non-stick button: {name}")

    def press_shoulder(self, button: Any, amount: float) -> None:
        name = self._name(button)
        if name == _L:
            self._l_shoulder = float(amount)
        elif name == _R:
            self._r_shoulder = float(amount)
        else:
            raise ValueError(f"press_shoulder on a non-shoulder button: {name}")

    def press_button(self, button: Any) -> None:
        self._set(button, True)

    def release_button(self, button: Any) -> None:
        self._set(button, False)

    def _set(self, button: Any, pressed: bool) -> None:
        name = self._name(button)
        if name not in self._buttons:
            raise ValueError(f"unknown controller button: {name}")
        self._buttons[name] = pressed

    def row(self) -> ControllerRow:
        return ControllerRow(
            main_x=self._main[0],
            main_y=self._main[1],
            c_x=self._c[0],
            c_y=self._c[1],
            shoulder=self._l_shoulder,
            buttons=tuple(self._buttons[name] for name in BUTTON_ORDER),
        )


def rows_to_state(rows: Sequence[ControllerRow]) -> ControllerState:
    """Batch ``[E]`` rows into a ``ControllerState`` with the exact float values MIMIC chose."""

    if not rows:
        raise ValueError("rows_to_state needs at least one row")
    columns = {
        name: torch.tensor([row.buttons[index] for row in rows], dtype=torch.bool)
        for index, name in enumerate(BUTTON_ORDER)
    }
    return ControllerState(
        main_stick=StickState(
            x=torch.tensor([row.main_x for row in rows], dtype=torch.float32),
            y=torch.tensor([row.main_y for row in rows], dtype=torch.float32),
        ),
        c_stick=StickState(
            x=torch.tensor([row.c_x for row in rows], dtype=torch.float32),
            y=torch.tensor([row.c_y for row in rows], dtype=torch.float32),
        ),
        shoulder=torch.tensor([row.shoulder for row in rows], dtype=torch.float32),
        buttons=ButtonState(**columns),
    )


# ---------------------------------------------------------------------------
# the sampling stream
# ---------------------------------------------------------------------------


class _RngStream:
    """A private global-RNG stream: MIMIC's ``_safe_sample`` draws from ``torch``'s default generator."""

    __slots__ = ("_seed", "_state")

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._state: torch.Tensor | None = None

    def reset(self) -> None:
        self._state = None

    def invoke(self, call: Callable[[], Any]) -> Any:
        outer = torch.get_rng_state()
        if self._state is None:
            torch.manual_seed(self._seed)
        else:
            torch.set_rng_state(self._state)
        try:
            return call()
        finally:
            self._state = torch.get_rng_state()
            torch.set_rng_state(outer)


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------


class MimicPlayer:
    """MIMIC on one port across ``num_envs`` environments; one batched forward per frame."""

    def __init__(
        self,
        runtime: MimicRuntime,
        num_envs: int,
        *,
        player: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        seed: int = 0,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if player not in (0, 1):
            raise ValueError(f"player must be 0 or 1, got {player}")
        self.runtime = runtime
        self.num_envs = num_envs
        self.player = player
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self._builder = runtime.api.build_frame_p2 if player == 1 else runtime.api.build_frame
        self._rng = _RngStream(seed)
        self._states: list[Any] = [self._new_state() for _ in range(num_envs)]
        self._prev_sent: list[dict[str, Any] | None] = [None] * num_envs
        self.frames_seen = 0
        self.forwards = 0

    def _new_state(self) -> Any:
        return self.runtime.api.player_state(
            self.runtime.model, self.runtime.max_seq_len, self.runtime.device, self.runtime.context
        )

    def reset(self, index: int) -> None:
        """Forget one environment's context window and the controller it last sent."""

        self._states[index] = self._new_state()
        self._prev_sent[index] = None

    def reset_all(self) -> None:
        for index in range(self.num_envs):
            self.reset(index)
        self._rng.reset()

    def rows(self, gamestates: Sequence[Any], needs_reset: torch.Tensor) -> list[ControllerRow]:
        """One frame: build, push, one forward for the live rows, decode each into a row."""

        if len(gamestates) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} gamestates, got {len(gamestates)}")
        if int(needs_reset.shape[0]) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset flags, got {int(needs_reset.shape[0])}")

        flags = needs_reset.to(dtype=torch.bool, device="cpu").tolist()
        out = [neutral_row()] * self.num_envs
        live: list[int] = []
        for index, gamestate in enumerate(gamestates):
            if flags[index]:
                self.reset(index)
            if gamestate is None:
                continue
            frame = self._builder(gamestate, self._prev_sent[index], self.runtime.context)
            if frame is None:
                continue
            self._states[index].push_frame(frame)
            live.append(index)
        if not live:
            return out

        predictions = self._forward(live)
        self._rng.invoke(lambda: self._decode(live, predictions, out))
        self.frames_seen += len(live)
        return out

    def _forward(self, live: Sequence[int]) -> dict[str, torch.Tensor]:
        """Concatenate every live window into ``[len(live), T, ...]`` and run the model once."""

        windows = [list(self._states[index]._frame_cache) for index in live]
        keys = windows[0][0].keys()
        batch = {
            key: torch.cat(
                [torch.cat([frame[key] for frame in window], dim=0).unsqueeze(0) for window in windows],
                dim=0,
            ).to(self.runtime.device)
            for key in keys
        }
        with torch.inference_mode():
            predictions = self.runtime.model(batch)
        self.forwards += 1
        # The heads are tiny; decoding on the CPU keeps the sampling stream reproducible on any device.
        return {
            key: value.detach().to("cpu")
            for key, value in predictions.items()
            if isinstance(value, torch.Tensor)
        }

    def _decode(
        self,
        live: Sequence[int],
        predictions: dict[str, torch.Tensor],
        out: list[ControllerRow],
    ) -> None:
        decode = self.runtime.api.decode_and_press
        for row, index in enumerate(live):
            recorder = ControllerRecorder()
            sliced = {key: value[row : row + 1] for key, value in predictions.items()}
            sent, _pressed, _names = decode(
                recorder,
                sliced,
                self._prev_sent[index],
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            self._prev_sent[index] = dict(sent)
            self._states[index].prev_sent = self._prev_sent[index]
            out[index] = recorder.row()


__all__ = [
    "CHECKPOINT_NAME",
    "DEFAULT_ASSET_DIR",
    "DEFAULT_SOURCE_DIR",
    "MIMIC_ASSET_REVISION",
    "MIMIC_ASSET_SUBDIR",
    "MIMIC_SOURCE_REVISION",
    "ControllerRecorder",
    "MimicApi",
    "MimicConfig",
    "MimicPlayer",
    "MimicRuntime",
    "load_mimic_api",
    "load_runtime",
    "rows_to_state",
]
