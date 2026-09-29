"""Read a Slippi ``.slp`` replay: the game-start block, and the frame stream on demand.

Why this module exists (28 Aug 2026): the replay is the only durable record of what the emulator
actually did.  It carries

* the **Melee RNG seed** the match started from (game-start payload ``0x13D``) -- the thing a seeded
  recording has to be checked against, because nothing in libmelee exposes it;
* the **rule set** actually in force (stock start count per port, CPU level, stage);
* how the match **ended** (``GAME_END``), which is how a recorder knows a real match is over.

It is what showed that our matches ran with ``infinite_time`` -- endless, nobody eliminated: every
recorded replay has 2-4 deaths per port and a stock counter frozen at 4, and no ``GAME_END`` at all.

Offsets follow libmelee's own parser so the two agree by construction: the game start is
``melee/console.py:1093-1145`` (version ``0x1``, stage ``0x13``, per-port block ``0x65 + 0x24 * i``
with character ``+0x00``, type ``+0x01``, stock start ``+0x02``, costume ``+0x03``, CPU level
``+0x0F``) and the post frame is ``:1240-1250`` (frame ``0x1``, port ``0x5``, follower ``0x6``,
action ``0x8``, percent ``0x16``, stocks ``0x21``).  The RNG seed is not read by libmelee at all.

A replay Dolphin is **still writing** declares ``raw`` length 0 -- Slippi backfills it when it closes
the file -- and a replay from a *killed* Dolphin ends mid-event, so both are read to whatever is
there rather than to the declared length.

No torch: this parses bytes, and the probe tooling that uses it should start in milliseconds.
"""

from __future__ import annotations

import struct
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# the format
# ---------------------------------------------------------------------------

RAW_MARKER: Final[bytes] = b"raw[$U#l"
"""The UBJSON key that opens the event stream; the next four bytes are its declared length."""

METADATA_MARKER: Final[bytes] = b"U\x08metadata"
"""What follows the event stream in a closed file: the bound of ``raw`` when no length was written."""

PAYLOADS: Final[int] = 0x35
GAME_START: Final[int] = 0x36
PRE_FRAME: Final[int] = 0x37
POST_FRAME: Final[int] = 0x38
GAME_END: Final[int] = 0x39

GAME_START_SIZE: Final[int] = 0x2F8
"""Bytes after the command byte of a game start (760, as Slippi 3.19 writes it)."""

PRE_FRAME_SIZE: Final[int] = 0x42
"""Bytes after the command byte of a pre-frame update, as Slippi 3.19 writes it."""

POST_FRAME_SIZE: Final[int] = 0x54
GAME_END_SIZE: Final[int] = 0x6

END_TIME: Final[int] = 1
END_GAME: Final[int] = 2
END_NO_CONTEST: Final[int] = 7
END_METHODS: Final[dict[int, str]] = {END_TIME: "TIME!", END_GAME: "GAME!", END_NO_CONTEST: "no contest"}

EMPTY_PORT: Final[int] = 3
"""``player_type`` of a port nobody is on (0 human/bot, 1 CPU, 2 demo, 3 empty)."""

INITIAL_FRAME: Final[int] = -123
"""The first frame index of a Melee game (``melee_rl.env.protocol.INITIAL_FRAME_INDEX``)."""

DYING_ACTION_MAX: Final[int] = 0xA
"""Action ids ``<= 0xA`` are the dying states, exactly as the reward reads them (``reward.py:45``)."""

FRAMES_PER_SECOND: Final[float] = 60.0

_HEAD_BYTES: Final[int] = 1 << 16
"""Enough of a file to hold the payload table and the game start (both land in the first ~1.5 KiB)."""


# ---------------------------------------------------------------------------
# what a replay says
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerStart:
    """One occupied port at the game start."""

    port: int
    character: int
    player_type: int
    stocks: int
    costume: int
    cpu_level: int


@dataclass(frozen=True)
class GameStart:
    """The game-start block: what the match was set up as, and the RNG seed it started from."""

    slp_version: tuple[int, int, int]
    stage: int
    seed: int
    players: tuple[PlayerStart, ...]

    @property
    def stocks(self) -> int:
        """The stock start count (Melee gives every port the same one; the max if they ever differ)."""

        return max((player.stocks for player in self.players), default=0)

    @property
    def cpu_levels(self) -> dict[int, int]:
        return {player.port: player.cpu_level for player in self.players}


@dataclass(frozen=True)
class PlayerResult:
    """What one port did over the replay: stocks, damage and deaths."""

    port: int
    stocks_start: int
    stocks_end: int
    percent_end: float
    deaths: int
    damage_taken: float
    damage_dealt: float


@dataclass(frozen=True)
class ReplaySummary:
    """A whole replay: its game start, its frame span, its result."""

    path: Path
    start: GameStart
    frames: int
    first_frame: int
    last_frame: int
    end_method: int | None
    lras_initiator: int | None
    players: tuple[PlayerResult, ...]

    @property
    def seed(self) -> int:
        return self.start.seed

    @property
    def finished(self) -> bool:
        """True when the replay holds a ``GAME_END``: the match actually ended inside it."""

        return self.end_method is not None

    @property
    def end_method_name(self) -> str:
        if self.end_method is None:
            return "unfinished"
        return END_METHODS.get(self.end_method, f"method {self.end_method}")

    @property
    def seconds(self) -> float:
        return self.frames / FRAMES_PER_SECOND

    @property
    def winner(self) -> int | None:
        """The port that won: most stocks left, ties broken by lower percent (Melee's own rule).

        ``None`` for an unfinished replay, a no contest, or an exact tie.  In a ``GAME!`` ending the
        loser is at zero stocks, so this is only interesting for a ``TIME!`` ending or a cut replay.
        """

        if self.end_method is None or self.end_method == END_NO_CONTEST or len(self.players) < 2:
            return None
        ranked = sorted(self.players, key=lambda player: (player.stocks_end, -player.percent_end))
        best, runner_up = ranked[-1], ranked[-2]
        if (best.stocks_end, -best.percent_end) == (runner_up.stocks_end, -runner_up.percent_end):
            return None
        return best.port


# ---------------------------------------------------------------------------
# reading the byte stream
# ---------------------------------------------------------------------------


def _raw_section(data: bytes) -> bytes:
    """The event stream of ``data``: the declared length when it is usable, else what is there.

    A live replay declares 0 (Slippi writes the length on close) and a killed Dolphin leaves the
    stream short of its declaration, so both fall back to the metadata marker or the end of file.
    """

    marker = data.find(RAW_MARKER)
    if marker < 0:
        raise ValueError("not a .slp file: no raw event block")
    begin = marker + len(RAW_MARKER) + 4
    if begin > len(data):
        raise ValueError("not a .slp file: the raw block has no length")
    declared = struct.unpack_from(">I", data, marker + len(RAW_MARKER))[0]
    end = len(data)
    metadata = data.find(METADATA_MARKER, begin)
    if metadata >= 0:
        end = metadata
    if declared and begin + declared <= end:
        end = begin + declared
    return data[begin:end]


def payload_sizes(raw: bytes) -> tuple[dict[int, int], int]:
    """The event-payload table that opens every stream, and the offset of the first event after it."""

    if not raw or raw[0] != PAYLOADS:
        raise ValueError("the event stream does not open with an event-payloads command")
    table_size = raw[1]
    sizes: dict[int, int] = {}
    offset = 2
    while offset + 3 <= 1 + table_size:
        command, size = struct.unpack_from(">BH", raw, offset)
        sizes[command] = size
        offset += 3
    return sizes, 1 + table_size


def iter_events(raw: bytes) -> Iterator[tuple[int, bytes]]:
    """``(command, event bytes)`` for every complete event; a truncated tail simply ends the stream."""

    sizes, offset = payload_sizes(raw)
    while offset < len(raw):
        command = raw[offset]
        size = sizes.get(command)
        if size is None or offset + 1 + size > len(raw):
            return
        yield command, raw[offset : offset + 1 + size]
        offset += 1 + size


def parse_game_start(event: bytes) -> GameStart:
    """The game-start event -> :class:`GameStart` (libmelee's offsets, plus the RNG seed at 0x13D)."""

    version = (event[0x1], event[0x2], event[0x3])
    stage = struct.unpack_from(">H", event, 0x13)[0]
    seed = struct.unpack_from(">I", event, 0x13D)[0] if len(event) >= 0x141 else 0
    players: list[PlayerStart] = []
    for index in range(4):
        base = 0x65 + 0x24 * index
        if base + 0x10 > len(event):
            break
        player_type = event[base + 0x01]
        if player_type == EMPTY_PORT:
            continue
        players.append(
            PlayerStart(
                port=index + 1,
                character=event[base],
                player_type=player_type,
                stocks=event[base + 0x02],
                costume=event[base + 0x03],
                # libmelee zeroes the slider byte for a port that is not a CPU (console.py:1129-1131)
                cpu_level=event[base + 0x0F] if player_type == 1 else 0,
            )
        )
    return GameStart(slp_version=version, stage=stage, seed=seed, players=tuple(players))


def read_game_start(path: str | Path) -> GameStart:
    """The game start of ``path``, reading only the head of the file (works on a live replay)."""

    file = Path(path)
    with file.open("rb") as handle:
        head = handle.read(_HEAD_BYTES)
    for command, event in iter_events(_raw_section(head)):
        if command == GAME_START:
            return parse_game_start(event)
    raise ValueError(f"{file}: no game-start event in the first {_HEAD_BYTES} bytes")


def summarize_replay(path: str | Path) -> ReplaySummary:
    """Parse the whole replay: its game start, its frame span and what every port ended on."""

    file = Path(path)
    raw = _raw_section(file.read_bytes())
    start: GameStart | None = None
    end_method: int | None = None
    lras: int | None = None
    frames: set[int] = set()
    stocks: dict[int, int] = {}
    percent: dict[int, float] = {}
    previous_action: dict[int, int] = {}
    deaths: dict[int, int] = {}
    taken: dict[int, float] = {}

    for command, event in iter_events(raw):
        if command == GAME_START:
            if start is None:
                start = parse_game_start(event)
            continue
        if command == GAME_END:
            end_method = event[0x1]
            lras = struct.unpack_from(">b", event, 0x2)[0] if len(event) > 0x2 else None
            continue
        if command != POST_FRAME or event[0x6] == 1:  # follower rows (Nana) are not their own player
            continue
        frame = struct.unpack_from(">i", event, 0x1)[0]
        port = event[0x5] + 1
        action = struct.unpack_from(">H", event, 0x8)[0]
        damage = struct.unpack_from(">f", event, 0x16)[0]
        frames.add(frame)
        if port in percent and damage > percent[port]:
            taken[port] = taken.get(port, 0.0) + (damage - percent[port])
        if port in previous_action and action <= DYING_ACTION_MAX < previous_action[port]:
            deaths[port] = deaths.get(port, 0) + 1
        percent[port] = damage
        previous_action[port] = action
        stocks[port] = event[0x21]

    if start is None:
        raise ValueError(f"{file}: no game-start event")
    results = tuple(
        PlayerResult(
            port=player.port,
            stocks_start=player.stocks,
            stocks_end=stocks.get(player.port, player.stocks),
            percent_end=percent.get(player.port, 0.0),
            deaths=deaths.get(player.port, 0),
            damage_taken=taken.get(player.port, 0.0),
            damage_dealt=sum(value for other, value in taken.items() if other != player.port),
        )
        for player in start.players
    )
    return ReplaySummary(
        path=file,
        start=start,
        frames=len(frames),
        first_frame=min(frames, default=0),
        last_frame=max(frames, default=0),
        end_method=end_method,
        lras_initiator=lras,
        players=results,
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_summary(summary: ReplaySummary) -> str:
    """One line per replay: seed, span, how it ended, and each port's stocks / damage / deaths."""

    winner = summary.winner
    ending = summary.end_method_name + (f" -> port {winner}" if winner is not None else "")
    ports = "; ".join(
        f"p{player.port} stocks {player.stocks_start}->{player.stocks_end} "
        f"{player.damage_taken:.0f}% taken, {player.deaths} deaths"
        for player in summary.players
    )
    return (
        f"{summary.path.name}: seed 0x{summary.seed:08x} stage {summary.start.stage}, "
        f"{summary.frames} frames ({summary.seconds:.1f} s), {ending}; {ports}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m melee_rl.slp <replay>...``: one summary line per replay."""

    paths = list(sys.argv[1:] if argv is None else argv)
    if not paths:
        print("usage: python -m melee_rl.slp <replay.slp>...", file=sys.stderr)
        return 2
    for path in paths:
        print(format_summary(summarize_replay(path)))
    return 0


__all__ = [
    "END_GAME",
    "END_METHODS",
    "END_NO_CONTEST",
    "END_TIME",
    "GAME_END",
    "GAME_START",
    "INITIAL_FRAME",
    "POST_FRAME",
    "GameStart",
    "PlayerResult",
    "PlayerStart",
    "ReplaySummary",
    "format_summary",
    "iter_events",
    "main",
    "parse_game_start",
    "payload_sizes",
    "read_game_start",
    "summarize_replay",
]


if __name__ == "__main__":  # pragma: no cover - the CLI is a thin wrapper over format_summary
    raise SystemExit(main())
