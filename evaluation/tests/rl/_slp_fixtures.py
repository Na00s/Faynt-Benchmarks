"""Synthetic Slippi replays for the tests (P11).

One home for the byte layout so ``test_slp`` (which pins every offset) and the tests that merely need
*a* replay of a known length cannot drift apart.  Offsets are libmelee's, listed in ``melee_rl.slp``.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from pathlib import Path

from melee_rl import slp

TWO_FOX: Sequence[tuple[int, int, int, int, int]] = (
    (0x02, 0, 4, 1, 1),  # port 1: Fox, bot, 4 stocks, costume 1
    (0x02, 1, 4, 0, 9),  # port 2: Fox, CPU level 9, 4 stocks
    (0x1A, 3, 4, 0, 1),  # ports 3 and 4 empty
    (0x1A, 3, 4, 0, 1),
)


def game_start(
    *,
    seed: int,
    stage: int = 32,
    version: tuple[int, int, int, int] = (3, 19, 0, 0),
    players: Sequence[tuple[int, int, int, int, int]] = TWO_FOX,
) -> bytes:
    """``(character, player_type, stocks, costume, cpu_level)`` per port -> a game-start event."""

    event = bytearray(1 + slp.GAME_START_SIZE)
    event[0] = slp.GAME_START
    event[0x1:0x5] = bytes(version)
    struct.pack_into(">H", event, 0x13, stage)
    struct.pack_into(">I", event, 0x13D, seed)
    for index, (character, kind, stocks, costume, cpu_level) in enumerate(players):
        base = 0x65 + 0x24 * index
        event[base] = character
        event[base + 0x01] = kind
        event[base + 0x02] = stocks
        event[base + 0x03] = costume
        event[base + 0x0F] = cpu_level
    return bytes(event)


def pre_frame(
    *,
    frame: int,
    port: int,
    buttons: int = 0,
    main_stick: tuple[float, float] = (0.0, 0.0),
    c_stick: tuple[float, float] = (0.0, 0.0),
    trigger: float = 0.0,
    action: int = 0x0E,
    follower: bool = False,
) -> bytes:
    """The controller state the game read on ``frame`` (libmelee ``console.py:1145-1212``).

    ``main_stick`` / ``c_stick`` are in the replay's own [-1, 1] units, not libmelee's rescaled
    [0, 1]; ``buttons`` are the *physical* bits at 0x31, which is what an agent pressed.
    """

    event = bytearray(1 + slp.PRE_FRAME_SIZE)
    event[0] = slp.PRE_FRAME
    struct.pack_into(">i", event, 0x1, frame)
    event[0x5] = port - 1
    event[0x6] = 1 if follower else 0
    struct.pack_into(">H", event, 0xB, action)
    struct.pack_into(">f", event, 0x19, main_stick[0])
    struct.pack_into(">f", event, 0x1D, main_stick[1])
    struct.pack_into(">f", event, 0x21, c_stick[0])
    struct.pack_into(">f", event, 0x25, c_stick[1])
    struct.pack_into(">f", event, 0x29, trigger)
    struct.pack_into(">I", event, 0x2D, buttons)
    struct.pack_into(">H", event, 0x31, buttons & 0xFFFF)
    return bytes(event)


def post_frame(
    *,
    frame: int,
    port: int,
    action: int = 0x0E,
    percent: float = 0.0,
    stocks: int = 4,
    follower: bool = False,
    position: tuple[float, float] = (0.0, 0.0),
    action_frame: int = 0,
) -> bytes:
    event = bytearray(1 + slp.POST_FRAME_SIZE)
    event[0] = slp.POST_FRAME
    struct.pack_into(">i", event, 0x1, frame)
    event[0x5] = port - 1
    event[0x6] = 1 if follower else 0
    struct.pack_into(">H", event, 0x8, action)
    struct.pack_into(">f", event, 0xA, position[0])
    struct.pack_into(">f", event, 0xE, position[1])
    struct.pack_into(">f", event, 0x16, percent)
    event[0x21] = stocks
    struct.pack_into(">f", event, 0x22, float(action_frame))
    return bytes(event)


def game_end(*, method: int = slp.END_GAME, lras: int = -1) -> bytes:
    event = bytearray(1 + slp.GAME_END_SIZE)
    event[0] = slp.GAME_END
    event[0x1] = method
    struct.pack_into(">b", event, 0x2, lras)
    return bytes(event)


def payload_sizes() -> bytes:
    sizes = {
        slp.GAME_START: slp.GAME_START_SIZE,
        slp.PRE_FRAME: slp.PRE_FRAME_SIZE,
        slp.POST_FRAME: slp.POST_FRAME_SIZE,
        slp.GAME_END: slp.GAME_END_SIZE,
    }
    body = b"".join(struct.pack(">BH", command, size) for command, size in sorted(sizes.items()))
    return bytes([0x35, len(body) + 1]) + body


def write_replay(
    path: Path, events: Sequence[bytes], *, declare_length: bool = True, metadata: bool = True
) -> Path:
    """A ``.slp`` file: the UBJSON header, the raw event stream, the metadata block.

    ``declare_length=False`` + ``metadata=False`` is what a replay looks like while Dolphin is still
    writing it: Slippi backfills the ``raw`` length only when it closes the file.
    """

    raw = payload_sizes() + b"".join(events)
    header = b"{U\x03raw[$U#l" + struct.pack(">I", len(raw) if declare_length else 0)
    tail = b"U\x08metadata{}}" if metadata else b""
    path.write_bytes(header + raw + tail)
    return path


def write_match(path: Path, *, frames: int, seed: int = 0xD831FED0, finished: bool = True) -> Path:
    """A replay of exactly ``frames`` frames, starting at ``-123`` as a Melee game does."""

    events = [game_start(seed=seed)]
    for offset in range(frames):
        frame = slp.INITIAL_FRAME + offset
        events.append(post_frame(frame=frame, port=1))
        events.append(post_frame(frame=frame, port=2))
    if finished:
        events.append(game_end())
    return write_replay(path, events)


__all__ = [
    "TWO_FOX",
    "game_end",
    "game_start",
    "payload_sizes",
    "post_frame",
    "pre_frame",
    "write_match",
    "write_replay",
]
