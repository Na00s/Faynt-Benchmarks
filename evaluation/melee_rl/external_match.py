"""Matches where neither side is one of our policies (P25, 9 Sep 2026).

The recorder's matchup grammar is ``<policy|bc>:<opponent>`` (:mod:`melee_rl.video`): one side is
always a checkpoint of ours, driven through ``RolloutWorker`` with a trajectory, a reward and a value
estimate.  The SmashBot calibration arms need the other shape -- **SmashBot vs Dolphin's CPU 9** and
**SmashBot vs MIMIC** -- where none of our weights are in the game at all.

Those two arms are the gate.  Before any number is published against SmashBot it has to be shown that
SmashBot is *healthy inside our harness*: the libmelee drift it suffers from
(:mod:`melee_rl.smashbot_compat`) fails silently, and upstream swallows the exceptions it does raise,
so a broken SmashBot plays like a passive Fox rather than crashing.  CPU 9 alone cannot settle it --
it is known to mis-rank agents rather than merely flatter them (P12, GOTCHAS S30) -- so MIMIC, which
every model in ``RESULTS.md`` is measured against, is the second anchor.

The driver is therefore deliberately thin: no policy, no trajectory, no reward, no torch model.  Read
``current()``, ask each driven side's opponent for a ``ControllerState``, ``step``, and stop on exactly
the recorder's match-mode rule -- every Dolphin has begun a second game, so its first one is complete
on disc.  Results are read back out of the ``.slp`` by :mod:`melee_rl.slp` through the recorder's own
:func:`melee_rl.video.match_record`, so they are directly comparable with the P14 / P15 / P21 tables.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from controller_codec import ControllerState
from melee_rl import slp
from melee_rl.env.protocol import EnvProtocol
from melee_rl.phillip import PHILLIP_REVISION
from melee_rl.video import collect_replays, games_started, match_record

if TYPE_CHECKING:
    from melee_rl.config import RLConfig

LOG: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_POLL_FRAMES: Final[int] = 128
"""How often the stop rule reads the replay directories -- the recorder polls once per rollout."""
STOP_REASONS: Final[tuple[str, ...]] = ("complete", "cap")
"""``complete``: every Dolphin began a second game.  ``cap``: ``max_frames`` ran out first."""
EXTERNAL_SIDE_KINDS: Final[tuple[str, ...]] = ("smashbot", "mimic", "slippi_ai", "phillip", "cpu")
"""What may play a side: altf4/SmashBot, MIMIC, a slippi-ai release (P27), a vladfi1/phillip agent (P32),
or Dolphin's own CPU."""
MANIFEST_NAME: Final[str] = "manifest.json"
FRAMES_PER_SECOND: Final[int] = 60
DEFAULT_SECONDS: Final[float] = 520.0
"""The recorder's per-match cap, deliberately past Melee's eight-minute timer (P12, GOTCHAS S30)."""
_CPU_SIDE: Final[re.Pattern[str]] = re.compile(r"^cpu([1-9])$")


@dataclass(frozen=True)
class ExternalSide:
    """One side of the match.

    ``player`` is the *environment* player index (0 or 1) -- the key ``env.step`` and
    ``Opponent.port`` use, not the libmelee port.  A side the environment drives itself (Dolphin's
    in-game CPU) has ``player = None`` and no opponent: it is named so it reaches the manifest, but it
    never receives a controller.
    """

    name: str
    player: int | None
    opponent: Any | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a side must be named")
        if self.player is None and self.opponent is not None:
            raise ValueError(f"side {self.name!r} has an opponent but no player index to send it to")
        if self.player is not None and self.opponent is None:
            raise ValueError(f"side {self.name!r} has a player index but no opponent to drive it")
        if self.player is not None and self.player not in (0, 1):
            raise ValueError(f"side {self.name!r}: player must be 0 or 1, got {self.player}")

    @property
    def driven_by_env(self) -> bool:
        """True for Dolphin's own CPU: a side of the match that we never send inputs for."""

        return self.player is None


def side_health(opponent: Any | None) -> dict[str, Any]:
    """An opponent's own health counters, or ``{}`` when it keeps none.

    :class:`melee_rl.smashbot.SmashBotPlayer` exposes ``stats()`` -- exceptions, the neutral-input
    share and the Tactic/Chain histogram -- which is what the calibration gate reads.
    """

    stats = getattr(getattr(opponent, "player", None), "stats", None)
    if not callable(stats):
        return {}
    result = stats()
    return dict(result) if isinstance(result, Mapping) else {}


def ensure_empty_replay_root(replay_root: str | Path) -> None:
    """Refuse a replay directory that already holds ``.slp`` files.

    Both the stop rule and the results come from reading this directory, so a re-used one makes a job
    stop on its first poll and report the previous run's games (measured: a smoke re-run into a used
    ``--output-dir`` said "128 frames in 2 s (complete), 4 matches read", all of them unfinished
    leftovers).  This must be called **before the environment is built**: booting the Dolphins starts
    their first game, which writes one ``.slp`` each -- and that file is precisely what the ``>= 2``
    stop rule counts from, so by the time the driver runs the directory is legitimately non-empty.
    """

    root = Path(replay_root)
    stale = sorted(root.rglob("*.slp")) if root.is_dir() else []
    if stale:
        raise ValueError(
            f"{root} already holds {len(stale)} .slp file(s) (e.g. {stale[0].name}): the stop rule "
            "counts games by reading this directory and the results are read back out of it, so a "
            "re-used directory stops the job on its first poll and reports the previous run's games"
        )


def _record_for(path: Path) -> dict[str, Any]:
    """One replay's result row, in the recorder's own shape (patched in tests)."""

    return match_record(slp.summarize_replay(path))


@dataclass
class ExternalMatchRun:
    """What one job did: the sides, the frames it took, the results and each side's health."""

    sides: tuple[str, ...]
    num_envs: int
    frames: int
    seconds: float
    stopped: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    health: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "melee_rl.external_match.v1",
            "sides": list(self.sides),
            "num_envs": self.num_envs,
            "frames": self.frames,
            "seconds": self.seconds,
            "stopped": self.stopped,
            "matches": self.matches,
            "health": self.health,
        }


def run_external_matches(
    env: EnvProtocol,
    sides: Sequence[ExternalSide],
    *,
    replay_root: str | Path,
    max_frames: int,
    poll_frames: int = DEFAULT_POLL_FRAMES,
    log: Callable[[str], None] | None = None,
) -> ExternalMatchRun:
    """Play one match per Dolphin between ``sides``; return the manifest of the job.

    ``max_frames`` is a *ceiling*, not a target: like the recorder's ``[video] seconds`` it should sit
    past Melee's own eight-minute timer (520 s = 31 200 frames), because the run starts at frame -123
    and a cap that lands before ``GAME_END`` reads as an unfinished match (P12, GOTCHAS S30).  The
    environment's lifetime belongs to the caller, so this never closes it.
    """

    say = log or (lambda message: LOG.info("%s", message))
    root = Path(replay_root)
    drivers: dict[int, Any] = {}
    for side in sides:
        if side.player is not None and side.opponent is not None:
            drivers[side.player] = side.opponent
    started = time.perf_counter()
    frames = 0
    stopped = "cap"

    while frames < max_frames:
        pending = env.current()
        actions: dict[int, ControllerState] = {
            player: driver.act_state(None, pending.needs_reset) for player, driver in drivers.items()
        }
        env.step(actions)
        frames += 1
        # Reading the directories is the direct evidence a match finished; doing it once per
        # ``poll_frames`` rather than per frame keeps the filesystem walk off the frame path.
        if frames % poll_frames == 0 and all(count >= 2 for count in games_started(root, env.num_envs)):
            stopped = "complete"
            break

    seconds = time.perf_counter() - started
    run = ExternalMatchRun(
        sides=tuple(side.name for side in sides),
        num_envs=env.num_envs,
        frames=frames,
        seconds=seconds,
        stopped=stopped,
        health={side.name: health for side in sides if (health := side_health(side.opponent))},
    )
    for path in collect_replays(root, env.num_envs, first=True):
        if path is None:
            continue
        try:
            run.matches.append(_record_for(path))
        except Exception as error:  # one truncated .slp must not lose the job's other matches
            LOG.warning("could not read %s: %s", path, error)
            run.matches.append({"path": str(path), "error": f"{type(error).__name__}: {error}"})
    say(
        f"{'/'.join(run.sides)}: {run.frames} frames in {run.seconds:.0f} s ({run.stopped}), "
        f"{len(run.matches)} matches read"
    )
    return run


# ---------------------------------------------------------------------------
# the side grammar and the environment layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideSpec:
    """A parsed side: ``smashbot``, ``mimic``, ``slippi_ai[:<release>]``, ``phillip[:<agent>]`` or
    ``cpu<1-9>``."""

    spec: str
    kind: str
    cpu_level: int = 0
    model: str = ""
    """For ``slippi_ai:<release>``, the release that overrides ``[slippi_ai] model``; for ``phillip:<agent>``,
    the agent directory that overrides ``[phillip] agent``."""

    @property
    def driven_by_env(self) -> bool:
        return self.kind == "cpu"


def parse_side(spec: str) -> SideSpec:
    """``"smashbot"`` / ``"mimic"`` / ``"slippi_ai[:gm]"`` / ``"phillip[:delay0/FoxFD]"`` / ``"cpu9"`` -> a
    :class:`SideSpec`."""

    usage = (
        f"side must be 'smashbot', 'mimic', 'slippi_ai' (optionally 'slippi_ai:<release>'), "
        f"'phillip' (optionally 'phillip:<agent>') or 'cpu1'-'cpu9', got {spec!r}"
    )
    if not isinstance(spec, str) or not spec:
        raise ValueError(usage)
    if spec in ("smashbot", "mimic", "slippi_ai", "phillip"):
        return SideSpec(spec=spec, kind=spec)
    if spec.startswith("phillip:"):
        agent = spec.split(":", 1)[1]
        if not agent:
            raise ValueError(usage)
        return SideSpec(spec=spec, kind="phillip", model=agent)
    if spec.startswith("slippi_ai:"):
        from melee_rl.slippi_ai_agent import RELEASED_MODELS

        model = spec.split(":", 1)[1]
        if model not in RELEASED_MODELS:
            raise ValueError(f"side {spec!r} names no slippi-ai release; have {sorted(RELEASED_MODELS)}")
        return SideSpec(spec=spec, kind="slippi_ai", model=model)
    match = _CPU_SIDE.match(spec)
    if match is None:
        raise ValueError(usage)
    return SideSpec(spec=spec, kind="cpu", cpu_level=int(match.group(1)))


def env_layout(sides: Sequence[str]) -> tuple[tuple[str, str], int]:
    """``([env.dolphin] players, cpu_level)`` for two sides.

    Dolphin's in-game CPU can only be the *second* port here: ``DolphinEnvConfig.players[1]`` is the
    only slot the menu helper sets a CPU level on (``env/dolphin.py:969-972``), and ``_check_characters``
    verifies exactly that port's level at every game start.
    """

    if len(sides) != 2:
        raise ValueError(f"an external match needs exactly two sides, got {list(sides)}")
    parsed = [parse_side(spec) for spec in sides]
    if all(side.driven_by_env for side in parsed):
        raise ValueError("at least one side must be an agent we drive; two CPUs would play themselves")
    if parsed[0].driven_by_env:
        raise ValueError(
            f"a cpu side must be the second one (Dolphin only takes a CPU on port 2), got {list(sides)}"
        )
    if parsed[1].driven_by_env:
        return ("policy", "cpu"), parsed[1].cpu_level
    return ("policy", "policy"), 9


def write_external_manifest(
    run: ExternalMatchRun, directory: str | Path, *, extra: Mapping[str, Any] | None = None
) -> Path:
    """``<directory>/manifest.json``: the run plus whatever the job wants recorded beside it."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    target = path / MANIFEST_NAME
    target.write_text(json.dumps({**run.to_dict(), **(dict(extra) if extra else {})}, indent=2))
    return target


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


def _build_opponent(
    side: SideSpec, player: int, config: RLConfig, gamestates: Callable[[], Sequence[Any]], matches: int
) -> Any:
    from dataclasses import replace as dataclass_replace

    if side.kind == "slippi_ai":
        from melee_rl.opponents import SlippiAIOpponent
        from melee_rl.slippi_ai_agent import SlippiAIPlayer

        slippi = dataclass_replace(config.slippi_ai, player=player)
        if side.model:
            # ``slippi_ai:<release>`` on the side overrides the run file, so one file covers every arm.
            slippi = dataclass_replace(slippi, model=side.model)
        wanted = slippi.resolved_character
        playing = config.env.dolphin.characters[player]
        if playing != wanted:
            raise ValueError(
                f"[slippi_ai] {slippi.model} plays libmelee character {wanted}, but "
                f"[env.dolphin] characters starts player {player} as {playing}"
            )
        return SlippiAIOpponent(SlippiAIPlayer(slippi, matches), gamestates, port=player)
    if side.kind == "phillip":
        from melee_rl.opponents import PhillipOpponent
        from melee_rl.phillip import PhillipPlayer, check_phillip_character

        phillip = dataclass_replace(config.phillip, player=player)
        if side.model:
            # ``phillip:<agent>`` on the side overrides the run file, so one file covers every arm.
            phillip = dataclass_replace(phillip, agent=side.model)
        check_phillip_character(phillip, config.env.dolphin.characters)
        return PhillipOpponent(PhillipPlayer(phillip, matches), gamestates, port=player)
    if side.kind == "smashbot":
        from melee_rl.opponents import SmashBotOpponent
        from melee_rl.smashbot import SmashBotPlayer

        smashbot = dataclass_replace(config.smashbot, player=player)
        return SmashBotOpponent(
            SmashBotPlayer(smashbot, matches),
            gamestates,
            port=player,
            delay_frames=smashbot.delay_frames,
        )
    from melee_rl.mimic import MimicPlayer, load_runtime
    from melee_rl.opponents import MimicOpponent

    mimic = dataclass_replace(config.mimic, player=player)
    return MimicOpponent(
        MimicPlayer(
            load_runtime(mimic),
            matches,
            player=player,
            temperature=mimic.temperature,
            top_k=mimic.top_k,
            top_p=mimic.top_p,
            seed=mimic.seed,
        ),
        gamestates,
        port=player,
    )


def record_external_matches(
    config: RLConfig,
    *,
    sides: Sequence[str],
    output_dir: str | Path,
    label: str | None = None,
    matches: int | None = None,
    seconds: float | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Play ``matches`` real four-stock games between two outside agents and read the results back.

    One Dolphin per match, in this process (the raw gamestates an outside agent reads do not survive a
    worker process, P12).  ``seconds`` caps each match; the run stops as soon as every Dolphin has
    begun a second game.
    """

    from melee_rl.evaluate import build_eval_env

    say = log or (lambda message: LOG.info("%s", message))
    count = matches if matches is not None else config.video.clips
    cap = seconds if seconds is not None else DEFAULT_SECONDS
    players, cpu_level = env_layout(sides)
    parsed = [parse_side(spec) for spec in sides]
    # A fresh directory per job.  ``run_record_video`` does the same (``<output_dir>/<stem>-<UTC>``)
    # and for the same reason: the stop rule and the results both come from reading the replay
    # directory, so two jobs sharing one would each report the other's games.
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root = Path(output_dir) / f"{label or '-'.join(sides)}-{stamp}"
    replay_root = root / "replays"
    ensure_empty_replay_root(replay_root)  # before the Dolphins boot and write their first game
    replay_root.mkdir(parents=True, exist_ok=True)

    if any(side.kind == "smashbot" for side in parsed):
        # Fail before any Dolphin boots: a missing checkout is a five-second error here and a
        # forty-minute one after a pool of emulators has come up (the same preflight video.py does).
        from melee_rl.smashbot import SMASHBOT_REVISION, load_smashbot_api

        say(f"loading SmashBot from {config.smashbot.source_dir} (revision {SMASHBOT_REVISION[:7]})")
        load_smashbot_api(config.smashbot.source_dir, compat=config.smashbot.compat)
    phillip_sides = [side for side in parsed if side.kind == "phillip"]
    if phillip_sides:
        # The same, one step further: start the sidecar interpreter, restore the agent and play a few
        # synthetic frames before the Dolphins boot (P32).
        from dataclasses import replace as dataclass_replace

        from melee_rl.phillip import check_phillip_character, format_preflight
        from melee_rl.phillip import preflight as phillip_preflight

        for index, side in enumerate(parsed):
            if side.kind != "phillip":
                continue
            phillip = dataclass_replace(config.phillip, player=index)
            if side.model:
                phillip = dataclass_replace(phillip, agent=side.model)
            check_phillip_character(phillip, config.env.dolphin.characters)
            say(f"preflighting phillip {phillip.agent} (revision {PHILLIP_REVISION[:7]})")
            say(format_preflight(phillip_preflight(phillip, log=say)))

    started = time.perf_counter()
    say(f"{' vs '.join(sides)}: {count} matches, cap {cap:g} s, players {players}")
    env = build_eval_env(
        config,
        num_envs=count,
        players=players,
        cpu_level=cpu_level,
        port_offset=config.video.port_offset,
        save_replays=True,
        replay_dir=replay_root,
        in_process=True,
    )
    try:
        gamestates = getattr(env, "gamestates", None)
        if gamestates is None:
            raise ValueError(
                "an external match needs the raw gamestates: set [env.dolphin] worker_processes = 0"
            )
        built = [
            ExternalSide(
                name=side.spec,
                player=None if side.driven_by_env else index,
                opponent=(
                    None if side.driven_by_env else _build_opponent(side, index, config, gamestates, count)
                ),
            )
            for index, side in enumerate(parsed)
        ]
        run = run_external_matches(
            env,
            built,
            replay_root=replay_root,
            max_frames=round(cap * FRAMES_PER_SECOND),
            log=say,
        )
    finally:
        env.close()
        for driven in built:
            closer = getattr(driven.opponent, "close", None)
            if callable(closer):
                closer()  # a Phillip side's sidecar interpreter

    extra: dict[str, Any] = {
        "label": label or "-".join(sides),
        "output_dir": str(root),
        "seconds_cap": cap,
        "duration_s": time.perf_counter() - started,
    }
    if any(side.kind == "smashbot" for side in parsed):
        extra["smashbot_revision"] = config.smashbot.source_dir
        extra["delay_frames"] = config.smashbot.delay_frames
    for index, side in enumerate(parsed):
        if side.kind == "phillip":
            extra[f"phillip_{index}"] = {
                "agent": side.model or config.phillip.agent,
                "revision": PHILLIP_REVISION,
                "epsilon": config.phillip.epsilon,
            }
    manifest = write_external_manifest(run, root, extra=extra)
    say(f"manifest: {manifest}")
    return {**run.to_dict(), **extra, "manifest": str(manifest)}


__all__ = [
    "DEFAULT_POLL_FRAMES",
    "DEFAULT_SECONDS",
    "EXTERNAL_SIDE_KINDS",
    "FRAMES_PER_SECOND",
    "MANIFEST_NAME",
    "STOP_REASONS",
    "ExternalMatchRun",
    "ExternalSide",
    "SideSpec",
    "ensure_empty_replay_root",
    "env_layout",
    "parse_side",
    "record_external_matches",
    "run_external_matches",
    "side_health",
    "write_external_manifest",
]
