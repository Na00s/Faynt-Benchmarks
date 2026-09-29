"""What the controller did, and what the game did about it (P26, 9 Sep 2026).

:mod:`melee_rl.slp` answers "who won"; this module answers "how well was it played".  P25 published
a SmashBot that beat Dolphin's CPU 9 sixteen games to nil and still took 1.9 stocks more damage per
game than the reference bundle the same bot reaches on libmelee 0.41.1 -- a deficit invisible to a
win rate and visible only because the four-stock rate was published beside it.  A finer instrument
was needed, and it has to read *input fidelity* without depending on who won.

**Waveshine chains** are that instrument.  A jump-cancelled shine is the most timing-sensitive thing
in Melee that a bot does on purpose: the shine has to be cancelled inside its 3-frame jump window,
frame after frame, so the length of an uninterrupted chain collapses the moment inputs land late.
The external 2026-08-20 bridge analysis reports exactly this scale -- chains of 17-23 at
``online_delay = 0`` against a maximum of 2 at ``online_delay = 2`` -- which makes it the one
statistic where a published reference value exists to compare ours against.

**Uncontested deaths** are the second: a stock lost with no damage taken in the preceding window is a
self-destruct or a failed recovery, not a punish.  Session 52 could only *infer* from the delay sweep
that SmashBot's cheap deaths were aggressive edgeguarding rather than broken recoveries; this
measures it.

Both come out of the replay, so they can be computed after the fact on every match already recorded,
and neither needs the agent to be instrumented.  The pre-frame parser follows libmelee's own offsets
(``console.py:1145-1212``): frame ``0x1``, port ``0x5``, follower ``0x6``, main stick ``0x19`` /
``0x1D``, C-stick ``0x21`` / ``0x25``, trigger ``0x29``, processed buttons ``0x2D``, physical buttons
``0x31``.  The physical bits are the ones an agent pressed; the processed bits are what the game made
of them.

No torch, no numpy: this parses bytes.
"""

from __future__ import annotations

import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from melee_rl import slp

# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

BUTTON_D_LEFT: Final[int] = 0x0001
BUTTON_D_RIGHT: Final[int] = 0x0002
BUTTON_D_DOWN: Final[int] = 0x0004
BUTTON_D_UP: Final[int] = 0x0008
BUTTON_Z: Final[int] = 0x0010
BUTTON_R: Final[int] = 0x0020
BUTTON_L: Final[int] = 0x0040
BUTTON_A: Final[int] = 0x0100
BUTTON_B: Final[int] = 0x0200
BUTTON_X: Final[int] = 0x0400
BUTTON_Y: Final[int] = 0x0800
BUTTON_START: Final[int] = 0x1000
"""The physical button bits, exactly as libmelee's ``parse_button_bits`` reads them."""

PRESSABLE: Final[int] = (
    BUTTON_D_LEFT
    | BUTTON_D_RIGHT
    | BUTTON_D_DOWN
    | BUTTON_D_UP
    | BUTTON_Z
    | BUTTON_R
    | BUTTON_L
    | BUTTON_A
    | BUTTON_B
    | BUTTON_X
    | BUTTON_Y
    | BUTTON_START
)
"""Every bit the parser knows; anything else in the word is ignored rather than counted as motion."""

# ---------------------------------------------------------------------------
# the action states a shine passes through
# ---------------------------------------------------------------------------

SHINE_GROUND_START: Final[int] = 0x168
"""``Action.DOWN_B_GROUND_START`` -- the frame a grounded shine comes out."""

SHINE_GROUND: Final[int] = 0x169
SHINE_TURN: Final[int] = 0x16A
SHINE_STUN: Final[int] = 0x16D
SHINE_AIR: Final[int] = 0x16E
"""``Action.DOWN_B_AIR`` -- the aerial shine a waveshine alternates into."""

SHINE_ONSETS: Final[frozenset[int]] = frozenset({SHINE_GROUND_START, SHINE_AIR})
"""Transitions *into* one of these are shines; the states they hold in are not counted again."""

SHINE_STATES: Final[frozenset[int]] = frozenset(
    {SHINE_GROUND_START, SHINE_GROUND, SHINE_TURN, SHINE_STUN, SHINE_AIR}
)

SHINE_GROUNDED: Final[frozenset[int]] = frozenset({SHINE_GROUND_START, SHINE_GROUND})
"""The two states SmashBot's jump-cancel rule names -- ``DOWN_B_STUN`` and the aerial shine are *not*
in it (``Chains/waveshine.py:94``), and counting them turns a 1.00-frame round trip into 2.37."""

DEFAULT_GAP_FRAMES: Final[int] = 25
"""A jump-cancelled waveshine cycle is ~12 frames; twice that separates a chain from a fresh one."""

DEFAULT_DEATH_WINDOW_FRAMES: Final[int] = 120
"""Two seconds: longer than any combo string that could still be called the cause of a death."""

SHINE_JUMP_CANCEL_FRAME: Final[int] = 3
"""SmashBot jumps out of a shine on the first gamestate with ``action_frame >= 3``
(``Chains/waveshine.py:107-110`` at revision ``67c4120``).  The frame that rule first fires on is the
frame it *observed*; the frame Y reaches the game is the frame its answer *landed* on."""

STICK_DEADZONE: Final[float] = 0.2875
"""Melee's main-stick deadzone, 23/80 -- below this a component does not move the character."""

STICK_EPSILON: Final[float] = 1e-4
"""SmashBot's working wavedash records at *exactly* the deadzone edge, and the replay stores it as a
float32 (-0.28749999403...), so the edge has to count as usable or every good wavedash reads as flat."""

AIRDODGE: Final[int] = 0xEC
"""``Action.AIRDODGE``.  L is *also* the shield button, so a shoulder press only counts as an
airdodge when the game actually entered this state -- a third of SmashBot's L presses are shields."""

AIRDODGE_WINDOW_FRAMES: Final[int] = 3
"""Frames after the press within which the airdodge has to appear (jumpsquat is 3 for Fox)."""

JUMPSQUAT: Final[int] = 0x18
"""``Action.KNEE_BEND``.  A wavedash is jumpsquat -> shoulder -> the ground, so the three frames before
the press are what separates a wavedash from a shine landing that also ends in ``LANDING_SPECIAL``."""

LANDING_SPECIAL: Final[int] = 0x2B
"""``Action.LANDING_SPECIAL`` -- an airdodge that reached the ground.  A wavedash with its angle intact
goes straight here and never shows ``AIRDODGE`` at all, which is why the outcome has to be read from
both states rather than from ``AIRDODGE`` alone."""


# ---------------------------------------------------------------------------
# one port's frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortFrames:
    """One port's frame stream: what it pressed (pre-frame) and what it became (post-frame).

    Every tuple is the same length and indexed by the same frame, so a metric can walk them
    together.  A port with no pre-frame rows (a CPU, or a replay written without them) reads as
    all-zero buttons rather than as a missing port.
    """

    port: int
    frames: tuple[int, ...]
    actions: tuple[int, ...]
    action_frame: tuple[int, ...]
    percent: tuple[float, ...]
    stocks: tuple[int, ...]
    position: tuple[tuple[float, float], ...]
    buttons: tuple[int, ...]
    main_stick: tuple[tuple[float, float], ...]
    trigger: tuple[float, ...]
    has_inputs: bool

    def __len__(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class ReplayFrames:
    """Every occupied port of one replay."""

    path: Path
    ports: dict[int, PortFrames]


def _pre_frame_row(event: bytes) -> tuple[int, int, int, tuple[float, float], float]:
    frame = struct.unpack_from(">i", event, 0x1)[0]
    port = event[0x5] + 1
    main_x = struct.unpack_from(">f", event, 0x19)[0]
    main_y = struct.unpack_from(">f", event, 0x1D)[0]
    trigger = struct.unpack_from(">f", event, 0x29)[0]
    buttons = struct.unpack_from(">H", event, 0x31)[0]
    return frame, port, buttons, (main_x, main_y), trigger


def _post_frame_row(event: bytes) -> tuple[int, int, int, int, float, int, tuple[float, float]]:
    frame = struct.unpack_from(">i", event, 0x1)[0]
    port = event[0x5] + 1
    action = struct.unpack_from(">H", event, 0x8)[0]
    x = struct.unpack_from(">f", event, 0xA)[0]
    y = struct.unpack_from(">f", event, 0xE)[0]
    percent = struct.unpack_from(">f", event, 0x16)[0]
    action_frame = int(struct.unpack_from(">f", event, 0x22)[0])
    return frame, port, action, action_frame, percent, event[0x21], (x, y)


def read_frames(path: str | Path) -> ReplayFrames:
    """Parse the whole frame stream of a replay into one :class:`PortFrames` per occupied port."""

    file = Path(path)
    raw = slp._raw_section(file.read_bytes())
    order: dict[int, list[int]] = {}
    actions: dict[int, dict[int, int]] = {}
    action_frames: dict[int, dict[int, int]] = {}
    percent: dict[int, dict[int, float]] = {}
    stocks: dict[int, dict[int, int]] = {}
    position: dict[int, dict[int, tuple[float, float]]] = {}
    buttons: dict[int, dict[int, int]] = {}
    sticks: dict[int, dict[int, tuple[float, float]]] = {}
    triggers: dict[int, dict[int, float]] = {}

    for command, event in slp.iter_events(raw):
        if command not in (slp.PRE_FRAME, slp.POST_FRAME) or event[0x6] == 1:
            continue  # follower rows (Nana) share their leader's port and are not their own player
        if command == slp.PRE_FRAME:
            frame, port, pressed, stick, trigger = _pre_frame_row(event)
            buttons.setdefault(port, {})[frame] = pressed
            sticks.setdefault(port, {})[frame] = stick
            triggers.setdefault(port, {})[frame] = trigger
        else:
            frame, port, action, action_frame, damage, stock, where = _post_frame_row(event)
            actions.setdefault(port, {})[frame] = action
            action_frames.setdefault(port, {})[frame] = action_frame
            percent.setdefault(port, {})[frame] = damage
            stocks.setdefault(port, {})[frame] = stock
            position.setdefault(port, {})[frame] = where
        seen = order.setdefault(port, [])
        if not seen or seen[-1] != frame:
            seen.append(frame)

    ports: dict[int, PortFrames] = {}
    for port, seen in order.items():
        frames = tuple(sorted(set(seen)))
        port_buttons = buttons.get(port, {})
        ports[port] = PortFrames(
            port=port,
            frames=frames,
            actions=tuple(actions.get(port, {}).get(frame, 0) for frame in frames),
            action_frame=tuple(action_frames.get(port, {}).get(frame, 0) for frame in frames),
            percent=tuple(percent.get(port, {}).get(frame, 0.0) for frame in frames),
            stocks=tuple(stocks.get(port, {}).get(frame, 0) for frame in frames),
            position=tuple(position.get(port, {}).get(frame, (0.0, 0.0)) for frame in frames),
            buttons=tuple(port_buttons.get(frame, 0) for frame in frames),
            main_stick=tuple(sticks.get(port, {}).get(frame, (0.0, 0.0)) for frame in frames),
            trigger=tuple(triggers.get(port, {}).get(frame, 0.0) for frame in frames),
            has_inputs=bool(port_buttons),
        )
    return ReplayFrames(path=file, ports=ports)


# ---------------------------------------------------------------------------
# waveshine chains
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShineChain:
    """A run of shines each following the last inside ``gap_frames``."""

    start_frame: int
    length: int
    span: int
    """Frames from the first shine to the last -- a chain of one spans zero."""


def shine_onsets(port: PortFrames) -> tuple[int, ...]:
    """The frames on which a shine came out: transitions *into* a shine state, not held frames."""

    onsets: list[int] = []
    previous = -1
    for frame, action in zip(port.frames, port.actions, strict=True):
        if action in SHINE_ONSETS and previous not in SHINE_STATES:
            onsets.append(frame)
        previous = action
    return tuple(onsets)


def shine_chains(port: PortFrames, *, gap_frames: int = DEFAULT_GAP_FRAMES) -> tuple[ShineChain, ...]:
    """Group :func:`shine_onsets` into chains; a gap longer than ``gap_frames`` starts a new one."""

    if gap_frames <= 0:
        raise ValueError(f"gap_frames must be positive, got {gap_frames}")
    chains: list[ShineChain] = []
    run: list[int] = []
    for frame in shine_onsets(port):
        if run and frame - run[-1] > gap_frames:
            chains.append(ShineChain(start_frame=run[0], length=len(run), span=run[-1] - run[0]))
            run = []
        run.append(frame)
    if run:
        chains.append(ShineChain(start_frame=run[0], length=len(run), span=run[-1] - run[0]))
    return tuple(chains)


# ---------------------------------------------------------------------------
# deaths
# ---------------------------------------------------------------------------


def death_frames(port: PortFrames) -> tuple[int, ...]:
    """Rising edges into a dying action -- the same rule the reward uses (``reward.py:45``)."""

    dying: list[int] = []
    previous = -1
    for frame, action in zip(port.frames, port.actions, strict=True):
        if action <= slp.DYING_ACTION_MAX < previous:
            dying.append(frame)
        previous = action
    return tuple(dying)


def deaths(port: PortFrames) -> int:
    return len(death_frames(port))


def uncontested_deaths(port: PortFrames, *, window_frames: int = DEFAULT_DEATH_WINDOW_FRAMES) -> int:
    """Deaths with no damage taken in the preceding ``window_frames``: SDs and failed recoveries."""

    if window_frames < 0:
        raise ValueError(f"window_frames must be >= 0, got {window_frames}")
    index = {frame: position for position, frame in enumerate(port.frames)}
    count = 0
    for frame in death_frames(port):
        end = index[frame]
        start = max(0, end - window_frames)
        window = port.percent[start : end + 1]
        if len(window) < 2 or max(window) <= window[0]:
            count += 1
    return count


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------


def jump_cancel_latencies(port: PortFrames) -> tuple[int, ...]:
    """Frames from the shine gamestate SmashBot decides on to the Y press reaching the game.

    In a lockstep emulator the floor is 1: the agent sees frame N and its controller is what the
    emulator reads for frame N + 1.  Measured on the recorded delay sweep this reads 1.00 at
    ``online_delay = 0`` (369 of 369), 1.95 at 1 and 2.98 at 2 -- a slope of exactly one frame per
    configured frame, which is what makes it a calibrated instrument rather than a statistic.
    """

    out: list[int] = []
    trigger: int | None = None
    for index, frame in enumerate(port.frames):
        if trigger is not None and port.buttons[index] & BUTTON_Y:
            out.append(frame - trigger)
            trigger = None
            continue
        action = port.actions[index]
        if action in SHINE_GROUNDED and port.action_frame[index] >= SHINE_JUMP_CANCEL_FRAME:
            if trigger is None:
                trigger = frame
        elif action not in SHINE_GROUNDED:
            trigger = None
    return tuple(out)


# ---------------------------------------------------------------------------
# airdodges: a wavedash, or a glide off the stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AirdodgeInput:
    """One airdodge: the frame the shoulder went down, the stick that aimed it, and what came of it."""

    frame: int
    x: float
    y: float
    landed: bool
    """The airdodge reached the ground on the spot (``LANDING_SPECIAL``) -- a wavedash.  ``False`` is a
    glide: the character stayed in ``AIRDODGE``, on our recordings for a mean of 35 frames, and off a
    ledge that ends in ``DEAD_FALL``."""

    @property
    def flat(self) -> bool:
        """No usable downward component in the *input* -- a horizontal airdodge, not a wavedash."""

        return self.y > -STICK_DEADZONE + STICK_EPSILON


def airdodges(port: PortFrames) -> tuple[AirdodgeInput, ...]:
    """Every shoulder press that actually became an airdodge, with the stick that aimed it.

    SmashBot's wavedash always asks for ``tilt_analog(MAIN, x, 0.35)`` -- ``.35`` is commented "near
    perfect wavedash angle" (``Chains/wavedash.py:67``, ``Chains/waveshine.py:148``) -- so every one
    of these *should* carry a downward component of -0.30.  The action check matters: L is the shield
    button too, and a shield press carries whatever stick happened to be held.
    """

    shoulder = BUTTON_L | BUTTON_R
    out: list[AirdodgeInput] = []
    previous = 0
    for index, frame in enumerate(port.frames):
        pressed = port.buttons[index] & shoulder
        if pressed and not previous:
            window = port.actions[index : index + 1 + AIRDODGE_WINDOW_FRAMES]
            before = port.actions[max(0, index - AIRDODGE_WINDOW_FRAMES) : index]
            landed = LANDING_SPECIAL in window and JUMPSQUAT in before
            if AIRDODGE in window or landed:
                x, y = port.main_stick[index]
                out.append(AirdodgeInput(frame=frame, x=x, y=y, landed=landed))
        previous = pressed
    return tuple(out)


def airdodges_landed(port: PortFrames) -> int:
    return sum(1 for dodge in airdodges(port) if dodge.landed)


def airdodge_landing_rate(port: PortFrames) -> float:
    """Share of airdodges that reached the ground rather than gliding.

    The headline fidelity number, and the one that does not depend on reading the stick: with our
    libmelee pin's analog remap on, SmashBot landed 24 % of its airdodges; with it off, 96 %.
    """

    dodges = airdodges(port)
    return (airdodges_landed(port) / len(dodges)) if dodges else 0.0


def flat_airdodges(port: PortFrames) -> int:
    return sum(1 for dodge in airdodges(port) if dodge.flat)


def flat_airdodge_rate(port: PortFrames) -> float:
    dodges = airdodges(port)
    return (flat_airdodges(port) / len(dodges)) if dodges else 0.0


# ---------------------------------------------------------------------------
# how busy the controller is
# ---------------------------------------------------------------------------


def button_change_rate(port: PortFrames) -> float:
    """Share of frames on which the physical buttons differ from the frame before.

    Session 48 measured a human Fox changing its controller on 26 % of frames; a bot whose inputs are
    being dropped or coalesced shows here before it shows in a scoreline.
    """

    if not port.has_inputs or len(port) < 2:
        return 0.0
    changes = sum(
        1
        for previous, current in zip(port.buttons[:-1], port.buttons[1:], strict=True)
        if (previous ^ current) & PRESSABLE
    )
    return changes / (len(port) - 1)


# ---------------------------------------------------------------------------
# one replay, every port
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortMetrics:
    """Everything this module measures about one port of one replay."""

    port: int
    frames: int
    shines: int
    longest_shine_chain: int
    mean_shine_chain: float
    chains: tuple[ShineChain, ...]
    deaths: int
    uncontested_deaths: int
    button_change_rate: float
    jump_cancel_latencies: tuple[int, ...]
    airdodges: int
    flat_airdodges: int
    airdodges_landed: int


@dataclass(frozen=True)
class ReplayMetrics:
    path: Path
    ports: dict[int, PortMetrics]


def port_metrics(
    port: PortFrames,
    *,
    gap_frames: int = DEFAULT_GAP_FRAMES,
    window_frames: int = DEFAULT_DEATH_WINDOW_FRAMES,
) -> PortMetrics:
    chains = shine_chains(port, gap_frames=gap_frames)
    lengths = [chain.length for chain in chains]
    return PortMetrics(
        port=port.port,
        frames=len(port),
        shines=sum(lengths),
        longest_shine_chain=max(lengths, default=0),
        mean_shine_chain=(sum(lengths) / len(lengths)) if lengths else 0.0,
        chains=chains,
        deaths=deaths(port),
        uncontested_deaths=uncontested_deaths(port, window_frames=window_frames),
        button_change_rate=button_change_rate(port),
        jump_cancel_latencies=jump_cancel_latencies(port),
        airdodges=len(airdodges(port)),
        flat_airdodges=flat_airdodges(port),
        airdodges_landed=airdodges_landed(port),
    )


def replay_metrics(
    path: str | Path,
    *,
    gap_frames: int = DEFAULT_GAP_FRAMES,
    window_frames: int = DEFAULT_DEATH_WINDOW_FRAMES,
) -> ReplayMetrics:
    frames = read_frames(path)
    return ReplayMetrics(
        path=frames.path,
        ports={
            port: port_metrics(rows, gap_frames=gap_frames, window_frames=window_frames)
            for port, rows in frames.ports.items()
        },
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_metrics(metrics: ReplayMetrics) -> str:
    """One line per port: shines, the longest chain, deaths and how busy the controller was."""

    lines = []
    for port in sorted(metrics.ports):
        side = metrics.ports[port]
        lines.append(
            f"{metrics.path.name}  p{port}  frames {side.frames:5d}  shines {side.shines:4d}  "
            f"longest chain {side.longest_shine_chain:3d}  mean {side.mean_shine_chain:5.2f}  "
            f"deaths {side.deaths} ({side.uncontested_deaths} uncontested)  "
            f"airdodges {side.airdodges:3d} ({side.airdodges_landed} landed, {side.flat_airdodges} flat)  "
            f"buttons {side.button_change_rate:.3f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m melee_rl.slp_inputs <replay.slp | directory> ...``"""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(main.__doc__)
        return 2
    paths: list[Path] = []
    for argument in arguments:
        target = Path(argument)
        paths.extend(sorted(target.rglob("*.slp")) if target.is_dir() else [target])
    for path in paths:
        print(format_metrics(replay_metrics(path)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
