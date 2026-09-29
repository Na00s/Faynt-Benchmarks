"""Stage 1 of P9: record clips of a checkpoint playing, keep the ``.slp``, hand them to the renderer.

Every judgement about the runs so far is a number (``eval/other/reward_per_frame``, entropy, drift).
This module is the qualitative counterpart, on demand: for a selected checkpoint it plays

    the checkpoint vs ``cpu1`` / ``cpu9`` / the BC checkpoint,   the BC checkpoint vs ``cpu1`` / ``cpu9``

for ``clips`` clips of ``seconds`` seconds each and produces, per clip, a ``.slp`` replay, a captioned
mp4 (:mod:`melee_rl.render`) and a JSON row of the same slippi-ai statistics the in-loop evaluator
logs.  The four BC clips are the *baseline*: they are recorded once per BC checkpoint into a directory
keyed by its sha256 (:func:`baseline_dir`) and reused until that file changes (user decision,
26 Aug 2026).

Playing reuses the evaluation path unchanged -- ``evaluate.build_eval_env`` (with the P9 replay
overrides), ``RolloutWorker`` in ``ring`` context mode at temperature 1.0, Fox vs Fox on Final
Destination in the infinite-time match, so a clip is two continuous minutes of play with respawns and
never a menu.  One environment of ``clips`` Dolphins per matchup: each Dolphin writes its own clip.

The student is loaded with ``checkpoint.load_policy``, which builds the adapter from the checkpoint's
own ``ModelConfig``, so any profile works without touching the run file; the delay and the ``T``
window are then re-checked against the *loaded* policy (``delay_offset`` / ``context_length``).
"""

from __future__ import annotations

import json
import math
import re
import struct
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import torch

from melee_rl import checkpoint as checkpoint_lib
from melee_rl import slp
from melee_rl.actor import ActorConfig, RolloutWorker
from melee_rl.evaluate import EVAL_STATS, build_eval_env, stats_from_metrics, trajectory_metrics
from melee_rl.mimic import MimicPlayer, MimicRuntime, load_runtime
from melee_rl.opponents import (
    CpuOpponent,
    MimicOpponent,
    Opponent,
    OtherOpponent,
    PhillipOpponent,
    SlippiAIOpponent,
    SmashBotOpponent,
)
from melee_rl.phillip import PHILLIP_REVISION, PhillipPlayer, check_phillip_character, format_preflight
from melee_rl.phillip import preflight as phillip_preflight
from melee_rl.protocol import PolicyProtocol
from melee_rl.render import RenderConfig, RenderJob, RenderResult, render_many
from melee_rl.slippi_ai_agent import SlippiAIPlayer, load_slippi_ai_api
from melee_rl.smashbot import SMASHBOT_REVISION, SmashBotPlayer, load_smashbot_api
from melee_rl.trajectory import Trajectory

if TYPE_CHECKING:
    from melee_rl.config import RLConfig

FRAMES_PER_SECOND: Final[int] = 60
"""Melee runs at 60 frames per second; a clip of ``s`` seconds is ``60 s`` environment frames."""
MATCHUP_KINDS: Final[tuple[str, ...]] = ("policy", "bc")
"""Who may be the player under test in a ``<student>:<opponent>`` matchup."""
MIMIC_OPPONENT: Final[str] = "mimic"
"""``<student>:mimic`` plays the released MIMIC policy on port 2 (P12, :mod:`melee_rl.mimic`)."""
SMASHBOT_OPPONENT: Final[str] = "smashbot"
"""``<student>:smashbot`` plays altf4/SmashBot on port 2 (P25, :mod:`melee_rl.smashbot`)."""
SLIPPI_AI_OPPONENT: Final[str] = "slippi_ai"
"""``<student>:slippi_ai`` plays the ``[slippi_ai] model`` release on port 2 (P27).

Which release is a run-file setting rather than part of the matchup string: one job plays one
release, and its name, sha256, delay and nametag go into the manifest beside the results.
"""
PHILLIP_OPPONENT: Final[str] = "phillip"
"""``<student>:phillip`` plays the ``[phillip] agent`` release of vladfi1/phillip on port 2 (P32).

Which agent is a run-file setting, as a slippi-ai release is (per arm:
``--overrides "phillip.agent=delay0/FalcoFD,env.dolphin.characters=[1,22]"``): one job plays one
agent, and its name, character, ``act_every`` and trained delay go into the manifest beside the results.
"""
CLIP_STATS: Final[tuple[str, ...]] = EVAL_STATS
"""Per-clip statistics: the evaluator's, measured over that one Dolphin's rollout rows."""
MANIFEST_NAME: Final[str] = "manifest.json"
MANIFEST_FORMAT: Final[str] = "melee_rl.video_manifest.v1"
VIDEO_MODES: Final[tuple[str, ...]] = ("clip", "match")
"""``[video] mode``: fixed-length clips (P9) or one real match per Dolphin (P11)."""

REPLAY_SUBDIR: Final[str] = "replays"
"""Where the Dolphins of a matchup write while they play (``<dir>/replays/<matchup>/env<i>``)."""
_CPU_SPEC: Final[re.Pattern[str]] = re.compile(r"^cpu([1-9])$")

RenderFn = Callable[[Sequence[RenderJob]], list[RenderResult]]
"""What :func:`record_matches` calls for stage 2 (``render.render_many`` bound to a config)."""


# ---------------------------------------------------------------------------
# matchups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchSpec:
    """One matchup: who plays port 1, who plays port 2, how many clips of how many seconds."""

    student: str
    opponent: str
    clips: int
    seconds: float

    @property
    def name(self) -> str:
        return f"{self.student}_vs_{self.opponent}"

    @property
    def baseline(self) -> bool:
        """True for the BC checkpoint's own clips (recorded once per BC file)."""

        return self.student == "bc"

    @property
    def cpu_level(self) -> int | None:
        match = _CPU_SPEC.match(self.opponent)
        return None if match is None else int(match.group(1))

    @property
    def mimic(self) -> bool:
        """True when port 2 is the released MIMIC policy rather than a CPU or one of our checkpoints."""

        return self.opponent == MIMIC_OPPONENT

    @property
    def smashbot(self) -> bool:
        """True when port 2 is altf4/SmashBot, the expert system (P25)."""

        return self.opponent == SMASHBOT_OPPONENT

    @property
    def slippi_ai(self) -> bool:
        """True when port 2 is a released slippi-ai agent (P27, :mod:`melee_rl.slippi_ai_agent`)."""

        return self.opponent == SLIPPI_AI_OPPONENT

    @property
    def phillip(self) -> bool:
        """True when port 2 is a released vladfi1/phillip agent (P32, :mod:`melee_rl.phillip`)."""

        return self.opponent == PHILLIP_OPPONENT

    @property
    def external(self) -> bool:
        """True when port 2 is an outside agent with its own action space.

        MIMIC, SmashBot, slippi-ai or Phillip: all four build their observation from the raw libmelee
        gamestate, so all four need ``[env.dolphin] worker_processes = 0``.
        """

        return self.mimic or self.smashbot or self.slippi_ai or self.phillip

    @property
    def players(self) -> tuple[str, str]:
        return ("policy", "cpu") if self.cpu_level is not None else ("policy", "policy")

    @property
    def frames(self) -> int:
        """Environment frames per clip."""

        return round(self.seconds * FRAMES_PER_SECOND)


def parse_matchup(spec: str, *, clips: int, seconds: float) -> MatchSpec:
    """``"policy:cpu9"`` / ``"policy:bc"`` / ``"bc:cpu1"`` -> a :class:`MatchSpec`."""

    usage = (
        f"matchup must be '<policy|bc>:<cpu1-cpu9|bc|mimic|smashbot|slippi_ai|phillip>' with distinct "
        f"sides, got {spec!r} (P9-PLAN.md §1, P12, P25, P27, P32)"
    )
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise ValueError(usage)
    student, opponent = spec.split(":")
    if student not in MATCHUP_KINDS:
        raise ValueError(usage)
    known = ("bc", MIMIC_OPPONENT, SMASHBOT_OPPONENT, SLIPPI_AI_OPPONENT, PHILLIP_OPPONENT)
    if opponent not in known and _CPU_SPEC.match(opponent) is None:
        raise ValueError(usage)
    if student == opponent:
        raise ValueError(usage)
    return MatchSpec(student=student, opponent=opponent, clips=clips, seconds=seconds)


@dataclass(frozen=True)
class VideoConfig:
    """``[video]``: what one recording job produces (on demand; no in-loop hook).

    ``matchups`` are ``<student>:<opponent>`` strings, ``clips`` clips of ``seconds`` seconds each;
    ``bc`` is the BC checkpoint used both as an opponent and as the baseline player (the ``--bc``
    flag overrides it).  Outputs land under ``output_dir``: a fresh ``<stem>-step<N>-<UTC>``
    directory per job and the sha256-keyed ``<output_dir>/<baseline_subdir>/<stem>-<sha8>`` for the
    baseline clips.  ``rollout_length`` / ``temperature`` / ``seed`` drive the rollout worker,
    ``port_offset`` shifts the Slippi ports (a recording job normally has the container to itself),
    and ``wandb`` (off by default) uploads a ``wandb_seconds`` trim of every clip.  ``[video.render]``
    is stage 2.

    ``bc_delay_frames`` (11 Sep 2026) is the delay the ``bc`` file plays at as a rival or a baseline
    player; unset, it plays at the delay it was trained at (``checkpoint.trained_delay``), as the recorded
    checkpoint itself does unless ``[actor] delay_frames`` + ``[delay] allow_mismatch`` say otherwise.

    ``mode`` (P11) is ``"clip"`` -- play ``seconds`` of game and keep the newest replay, the P9
    behaviour -- or ``"match"``: every Dolphin plays **one real match**, the recording stops as soon
    as all of them have started a second game, and the **first** replay of each is the one kept.
    ``seconds`` is then the per-match cap rather than the clip length.  Match mode only means
    anything with ``[env.dolphin] infinite_time = false``, since the endless mode has no game to end
    (28 Aug 2026: every replay before P11 was that mode, four stocks that never fell).  One match per
    boot is not a preference: the stage drifts on a restart (Final Destination -> Battlefield ->
    Dream Land, measured in the P11 probe) and so does the RNG seed.
    """

    matchups: tuple[str, ...] = ("policy:cpu1", "policy:cpu9", "policy:bc", "bc:cpu1", "bc:cpu9")
    mode: str = "clip"
    clips: int = 2
    seconds: float = 120.0
    bc: str | None = None
    output_dir: str = "/runs/videos"
    baseline_subdir: str = "baseline"
    rollout_length: int = 128
    temperature: float = 1.0
    seed: int = 0
    port_offset: int = 0
    wandb: bool = False
    wandb_seconds: float = 30.0
    wandb_bitrate_kbps: int = 1200
    bc_delay_frames: int | None = None
    render: RenderConfig = field(default_factory=RenderConfig)

    def __post_init__(self) -> None:
        if self.mode not in VIDEO_MODES:
            raise ValueError(f"[video] mode must be one of {VIDEO_MODES}, got {self.mode!r}")
        if self.bc_delay_frames is not None and self.bc_delay_frames < 0:
            raise ValueError("[video] bc_delay_frames must be >= 0 or omitted (the file's trained delay)")
        if self.clips < 1:
            raise ValueError("[video] clips must be >= 1")
        if not self.seconds > 0.0:
            raise ValueError("[video] seconds must be positive")
        if not self.matchups:
            raise ValueError("[video] matchups must name at least one matchup")
        specs = expand_matches(self)
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError(f"[video] matchups must be distinct, got {self.matchups!r}")
        if self.rollout_length < 1:
            raise ValueError("[video] rollout_length must be >= 1")
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("[video] temperature must be finite and >= 0")
        if not self.wandb_seconds > 0.0:
            raise ValueError("[video] wandb_seconds must be positive")
        if self.wandb_bitrate_kbps < 1:
            raise ValueError("[video] wandb_bitrate_kbps must be >= 1")
        if self.port_offset < 0:
            raise ValueError("[video] port_offset must be >= 0")
        if not self.output_dir:
            raise ValueError("[video] output_dir must name a directory")

    @property
    def needs_mimic(self) -> bool:
        """True when any matchup plays the released MIMIC policy."""

        return any(spec.mimic for spec in expand_matches(self))

    @property
    def needs_smashbot(self) -> bool:
        """True when any matchup plays altf4/SmashBot (P25)."""

        return any(spec.smashbot for spec in expand_matches(self))

    @property
    def needs_slippi_ai(self) -> bool:
        """True when any matchup plays a released slippi-ai agent (P27)."""

        return any(spec.slippi_ai for spec in expand_matches(self))

    @property
    def needs_phillip(self) -> bool:
        """True when any matchup plays a released vladfi1/phillip agent (P32)."""

        return any(spec.phillip for spec in expand_matches(self))

    @property
    def needs_bc(self) -> bool:
        """True when any matchup mentions the BC checkpoint."""

        return any(spec.student == "bc" or spec.opponent == "bc" for spec in expand_matches(self))


def expand_matches(config: VideoConfig) -> tuple[MatchSpec, ...]:
    """The configured matchups as :class:`MatchSpec` objects, in order."""

    return tuple(parse_matchup(spec, clips=config.clips, seconds=config.seconds) for spec in config.matchups)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


@dataclass
class ClipRecord:
    """One clip: where its files are, what happened in it."""

    matchup: str
    index: int
    student: str
    opponent: str
    seconds: float
    frames: int
    caption: str
    slp: str | None = None
    video: str | None = None
    stats: dict[str, float] = field(default_factory=dict)
    """Rates over everything the Dolphin played, which in match mode runs past the kept game."""
    rendered: bool = False
    reused: bool = False
    duration_s: float | None = None
    render: dict[str, Any] | None = None
    match: dict[str, Any] | None = None
    error: str | None = None

    def record(self) -> dict[str, Any]:
        """A JSON-able row of the manifest."""

        return {
            "matchup": self.matchup,
            "index": self.index,
            "student": self.student,
            "opponent": self.opponent,
            "seconds": self.seconds,
            "frames": self.frames,
            "caption": self.caption,
            "slp": self.slp,
            "video": self.video,
            "stats": {name: float(value) for name, value in self.stats.items()},
            "rendered": self.rendered,
            "reused": self.reused,
            "duration_s": self.duration_s,
            "render": self.render,
            "match": self.match,
            "error": self.error,
        }


@dataclass
class VideoRun:
    """One recording job: its directory, the checkpoint it recorded and every clip."""

    directory: str
    checkpoint: str
    checkpoint_sha256: str
    step: int | None
    baseline_directory: str
    baseline_reused: bool
    clips: list[ClipRecord]
    seconds: float = 0.0
    manifest: str | None = None
    delay_frames: int = 0
    """The delay the recorded checkpoint played at (11 Sep 2026): its own trained delay by default."""

    def record(self) -> dict[str, Any]:
        return {
            "format": MANIFEST_FORMAT,
            "directory": self.directory,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "step": self.step,
            "delay_frames": self.delay_frames,
            "baseline_directory": self.baseline_directory,
            "baseline_reused": self.baseline_reused,
            "seconds": self.seconds,
            "clips": [clip.record() for clip in self.clips],
        }


def write_clip_records(clips: Sequence[ClipRecord], directory: str | Path) -> list[Path]:
    """One ``<matchup>_<index>.json`` per clip, beside its ``.slp`` and ``.mp4`` (user decision).

    Written as soon as a matchup has played, so the statistics survive a failure in a later matchup or
    in the render pass; :func:`record_matches` refreshes them once the videos are in.
    """

    written = []
    for clip in clips:
        path = Path(directory) / f"{clip.matchup}_{clip.index}.json"
        path.write_text(json.dumps(clip.record(), indent=1))
        written.append(path)
    return written


def write_manifest(run: VideoRun, path: str | Path) -> Path:
    """Write ``run`` as JSON to ``path`` (creating its directory) and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run.record(), indent=1))
    return target


def format_summary(run: VideoRun) -> str:
    """One console block: the directory, then a line per clip."""

    lines = [
        f"{len(run.clips)} clips in {run.directory} "
        f"(checkpoint {run.checkpoint}, baseline {'reused' if run.baseline_reused else 'recorded'}, "
        f"{run.seconds:.0f} s)"
    ]
    for clip in run.clips:
        stats = clip.stats
        detail = ""
        if stats:
            detail = (
                f" dealt {stats.get('damage_dealt_per_minute', 0.0):.1f}/min "
                f"taken {stats.get('damage_taken_per_minute', 0.0):.1f}/min "
                f"ko_diff {stats.get('ko_diff_per_minute', 0.0):+.2f}/min"
            )
        state = "video" if clip.rendered else (clip.error or "no video")
        lines.append(f"  {clip.matchup} {clip.index}/{clip.frames} frames: {state}{detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# files
# ---------------------------------------------------------------------------


def baseline_dir(output_dir: str | Path, bc: str | Path, *, subdir: str = "baseline") -> Path:
    """``<output_dir>/<subdir>/<bc stem>-<sha8>``: the BC clips are keyed by the checkpoint's sha256."""

    digest = checkpoint_lib.sha256_file(bc)
    return Path(output_dir) / subdir / f"{Path(bc).stem}-{digest[:8]}"


def games_started(replay_root: str | Path, count: int) -> list[int]:
    """How many games each of ``count`` Dolphins has written so far: one ``.slp`` per game started.

    Match mode's stop condition -- a second file means that Dolphin's first match is over and the
    next one has begun.  Reading the directory is the direct evidence; the trajectory's reset rows
    would say the same thing one abstraction further away.
    """

    root = Path(replay_root)
    return [len(list((root / f"env{index}").rglob("*.slp"))) for index in range(count)]


def collect_replays(replay_root: str | Path, count: int, *, first: bool = False) -> list[Path | None]:
    """The ``.slp`` each of ``count`` Dolphins wrote under ``<replay_root>/env<i>`` (newest; ``None``).

    ``first=True`` takes the **oldest** instead: in match mode that is the one complete match, while
    the newest is whatever game the recorder was cut off in the middle of.

    Searched recursively: Dolphin's own default is Slippi's ``<dir>/<YYYY-MM>/`` monthly layout, which
    ``DolphinEnvConfig.replay_monthly_folders`` turns off but an externally configured Dolphin may not.
    """

    found: list[Path | None] = []
    for index in range(count):
        directory = Path(replay_root) / f"env{index}"
        files = sorted(directory.rglob("*.slp"), key=lambda item: item.stat().st_mtime)
        if first and files:
            found.append(files[0])
            continue
        found.append(files[-1] if directory.is_dir() and files else None)
    return found


def match_record(summary: slp.ReplaySummary) -> dict[str, Any]:
    """The JSON-able result of one match, read out of its replay.

    Unlike :attr:`ClipRecord.stats` -- rates over everything the Dolphin played -- this describes the
    kept game exactly: who won, on how many stocks, and the per-minute rates of that game alone.
    """

    minutes = summary.seconds / 60.0
    return {
        "seed": f"0x{summary.seed:08x}",
        "stage": summary.start.stage,
        "frames": summary.frames,
        "seconds": round(summary.seconds, 2),
        "finished": summary.finished,
        "end_method": summary.end_method,
        "ending": summary.end_method_name,
        "winner": summary.winner,
        "players": [
            {
                "port": player.port,
                "stocks_start": player.stocks_start,
                "stocks_end": player.stocks_end,
                "deaths": player.deaths,
                "damage_taken": round(player.damage_taken, 1),
                "damage_dealt": round(player.damage_dealt, 1),
                "damage_dealt_per_minute": round(player.damage_dealt / minutes, 1) if minutes else 0.0,
                "damage_taken_per_minute": round(player.damage_taken / minutes, 1) if minutes else 0.0,
            }
            for player in summary.players
        ],
    }


def render_jobs(clips: Sequence[ClipRecord]) -> list[RenderJob]:
    """A :class:`RenderJob` per clip that has both a replay and a target path."""

    return [
        RenderJob(slp=Path(clip.slp), target=Path(clip.video), caption=clip.caption, frames=clip.frames)
        for clip in clips
        if clip.slp is not None and clip.video is not None
    ]


def caption_for(spec: MatchSpec, index: int, *, student_label: str, opponent_label: str) -> str:
    """``"<student> (P1) vs <opponent> (P2) - clip i/n"``, burned into the video."""

    return f"{student_label} (P1) vs {opponent_label} (P2) - clip {index}/{spec.clips}"


def _opponent_label(spec: MatchSpec, bc: Path | None) -> str:
    level = spec.cpu_level
    if level is not None:
        return f"CPU {level}"
    if spec.mimic:
        return "MIMIC"
    if spec.smashbot:
        return "SmashBot"
    return "bc" if bc is None else bc.stem


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------


def _load_policy(path: str | Path, device: torch.device | str) -> PolicyProtocol:
    policy = checkpoint_lib.load_policy(path, device=device)
    policy.eval()
    policy.requires_grad_(False)
    return policy


def _played_delay(config: RLConfig, trained_delay: int) -> int:
    """The delay a recorded file plays at (11 Sep 2026): the delay it was trained at, unless the run file
    names another one under ``[delay] allow_mismatch``.

    Without the flag a run file's ``actor.delay_frames`` may only be 0 (unset) or the file's own.  The old
    check compared it with the model's ``action_offset_frames - 1`` -- always 0 -- and the stock run files
    would have recorded a D = 18 checkpoint at 0.
    """

    requested = config.actor.delay_frames
    if config.delay.allow_mismatch:
        return requested
    if requested not in (0, trained_delay):
        raise ValueError(
            f"[actor] delay_frames {requested} disagrees with the checkpoint's trained delay "
            f"{trained_delay}; a file plays at its own delay by default -- set [delay] allow_mismatch = true "
            "to record it at another one"
        )
    return trained_delay


def _actor_config(config: RLConfig, policy: PolicyProtocol, *, trained_delay: int = 0) -> ActorConfig:
    """The evaluation-shaped actor config, re-checked against the *loaded* policy (P9-PLAN.md §7); the delay
    is the file's own unless the run file overrides it (:func:`_played_delay`)."""

    video = config.video
    actor = ActorConfig(
        rollout_length=video.rollout_length,
        context_frames=0,
        delay_frames=_played_delay(config, trained_delay),
        context_mode="ring",
        decision_stride=config.actor.decision_stride,  # lever 8 check (ii): the k-frame hold at play time
        temperature=video.temperature,
        seed=video.seed,
        reward=config.actor.reward,
        check_reward_bounds=config.actor.check_reward_bounds,
    )
    actor.resolve_context_frames(policy.context_length)  # T must fit the loaded model's context
    return actor


def _play(
    config: RLConfig,
    spec: MatchSpec,
    *,
    student: PolicyProtocol,
    bc: Path | None,
    replay_root: Path,
    actor: ActorConfig,
    device: torch.device | str,
    mimic: MimicRuntime | None = None,
    log: Callable[[str], None] | None = None,
    bc_delay: int = 0,
) -> tuple[list[dict[str, float]], int, float]:
    """Play one matchup; per-clip statistics, the frames each clip covers and the wall time.  ``bc_delay`` is
    the delay the ``bc`` rival plays at (its own trained delay unless ``[video] bc_delay_frames``)."""

    video = config.video
    rival: Opponent
    cpu_level = 9
    if spec.cpu_level is not None:
        rival = CpuOpponent(cpu_level=spec.cpu_level)
        cpu_level = spec.cpu_level
    elif not spec.external:
        assert bc is not None
        rival = OtherOpponent(
            bc,
            spec.clips,
            delay_frames=bc_delay,
            temperature=video.temperature,
            seed=video.seed + 1,
            device=device,
        )
    replay_root.mkdir(parents=True, exist_ok=True)
    env = build_eval_env(
        config,
        num_envs=spec.clips,
        players=spec.players,
        cpu_level=cpu_level,
        port_offset=video.port_offset,
        save_replays=True,
        replay_dir=replay_root,
    )
    gamestates: Callable[[], Sequence[Any]] | None = None
    if spec.external:
        # An outside agent builds its features from the raw libmelee gamestate, which only exists in
        # this process (P12, P25): worker processes keep it in the child, so such a matchup must run
        # with worker_processes = 0.
        found = getattr(env, "gamestates", None)
        if found is None:
            env.close()
            raise ValueError(
                f"a {spec.opponent!r} matchup needs the raw gamestates: set [env.dolphin] "
                "worker_processes = 0"
            )
        gamestates = cast(Callable[[], Sequence[Any]], found)
    if spec.smashbot:
        assert gamestates is not None
        rival = SmashBotOpponent(
            SmashBotPlayer(replace(config.smashbot, seed=config.smashbot.seed + video.seed), spec.clips),
            gamestates,
            port=config.smashbot.player,  # the ENV player index, not the libmelee port
            delay_frames=config.smashbot.delay_frames,
        )
    elif spec.slippi_ai:
        assert gamestates is not None
        wanted = config.slippi_ai.resolved_character
        playing = config.env.dolphin.characters[config.slippi_ai.player]
        if playing != wanted:
            raise ValueError(
                f"[slippi_ai] {config.slippi_ai.model} plays libmelee character {wanted}, but "
                f"[env.dolphin] characters starts player {config.slippi_ai.player} as {playing}"
            )
        rival = SlippiAIOpponent(
            SlippiAIPlayer(config.slippi_ai, spec.clips),
            gamestates,
            port=config.slippi_ai.player,  # the ENV player index, not the libmelee port
        )
    elif spec.phillip:
        assert gamestates is not None
        check_phillip_character(config.phillip, config.env.dolphin.characters)
        rival = PhillipOpponent(
            PhillipPlayer(config.phillip, spec.clips, log=log),
            gamestates,
            port=config.phillip.player,  # the ENV player index, not the libmelee port
        )
    elif spec.mimic:
        assert mimic is not None and gamestates is not None
        rival = MimicOpponent(
            MimicPlayer(
                mimic,
                spec.clips,
                player=1,
                temperature=config.mimic.temperature,
                top_k=config.mimic.top_k,
                top_p=config.mimic.top_p,
                seed=config.mimic.seed + video.seed,
            ),
            gamestates,
            port=1,
        )
    rollouts = math.ceil(spec.frames / actor.rollout_length)  # the clip length, or the match cap
    match_mode = video.mode == "match"
    try:
        worker = RolloutWorker(student, env, rival, actor)
        started = time.perf_counter()
        trajectories: list[Trajectory] = []
        for _ in range(rollouts):
            trajectories.append(worker.rollout())
            # Match mode stops when every Dolphin has written a second replay: its match is over and
            # the next game has begun.  Playing on would only record a game nobody keeps.
            if match_mode and all(count >= 2 for count in games_started(replay_root, spec.clips)):
                break
        seconds = time.perf_counter() - started
    finally:
        env.close()
        if spec.phillip:
            cast(PhillipOpponent, rival).close()  # the sidecar interpreter dies with the matchup
    if spec.phillip and log is not None:
        # P25's lesson once more: an outside agent that decides nothing holds a neutral stick and looks
        # like a weak opponent rather than a broken one.  Exceptions must be zero and the restore clean.
        health = cast(PhillipOpponent, rival).player.stats()
        log(
            f"{spec.name}: phillip {health['agent']} ({health['char']}, act_every {health['act_every']}, "
            f"delay {health['delay_frames']} frames) played {health['live_frames']} of {health['frames']} "
            f"frames, "
            f"{health['decisions']} decisions, {health['neutral_share']:.1%} neutral, "
            f"{health['exceptions']} exceptions, {health['sidecar_seconds']:.0f} s in the sidecar"
        )
        if health["exceptions"]:
            log(f"{spec.name}: phillip first error:\n{health['first_error']}")
    if spec.mimic and log is not None:
        # The one failure mode that looks like a working match: MIMIC's frame builder declining every
        # frame (it returns None until both players exist), which leaves it holding a neutral stick.
        player = cast(MimicOpponent, rival).player
        say = f"{player.frames_seen} frames in {player.forwards} batched forwards"
        log(f"{spec.name}: MIMIC saw {say}")
    if spec.slippi_ai and log is not None:
        # P25's lesson again: an outside agent that decides nothing holds a neutral stick and
        # looks like a weak opponent rather than a broken one.  Exceptions must be zero.
        health = cast(SlippiAIOpponent, rival).player.stats()
        log(
            f"{spec.name}: slippi-ai {health['model']} (delay {health['effective_delay']}, "
            f"name {health['name']!r}) played {health['frames_seen']} frames, "
            f"{health['neutral_share']:.1%} neutral, {health['held_frames']} held, "
            f"{health['exceptions']} exceptions"
        )
        if health["exceptions"]:
            log(f"{spec.name}: slippi-ai first error:\n{health['first_error']}")
    if spec.smashbot and log is not None:
        # The calibration gate (P25): exceptions must be zero and the tactic histogram must show real
        # play.  A SmashBot broken by the libmelee drift does not crash -- it holds a neutral stick.
        health = cast(SmashBotOpponent, rival).player.stats()
        top = ", ".join(f"{name} {count}" for name, count in list(health["tactics"].items())[:5])
        log(
            f"{spec.name}: SmashBot played {health['frames']} frames, "
            f"{health['neutral_share']:.1%} neutral, {health['exceptions']} exceptions; tactics: {top}"
        )
        if health["exceptions"]:
            log(f"{spec.name}: SmashBot first error:\n{health['first_error']}")
    frames = len(trajectories) * actor.rollout_length
    threshold = config.actor.reward.stalling_threshold
    stats = [
        stats_from_metrics(
            trajectory_metrics(
                [trajectory.index_envs([index]) for trajectory in trajectories],
                ports=1,
                stalling_threshold=threshold,
            ),
            frames=frames,
            seconds=seconds,
        )
        for index in range(spec.clips)
    ]
    return stats, frames, seconds


def _reusable_baseline(directory: Path, specs: Sequence[MatchSpec]) -> list[ClipRecord] | None:
    """The recorded baseline clips of ``directory`` when they cover ``specs``, else ``None``."""

    manifest = directory / MANIFEST_NAME
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text())
        rows = payload["clips"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    clips = [ClipRecord(**{**row, "reused": True}) for row in rows]
    for spec in specs:
        matching = [clip for clip in clips if clip.matchup == spec.name]
        complete = all(clip.video and Path(clip.video).is_file() for clip in matching)
        if len(matching) != spec.clips or not complete:
            return None
    return clips


def preload_slippi_ai(config: RLConfig, say: Callable[[str], None]) -> None:
    """Import the upstream tree the release will play on now, not after a pool of emulators has come up.

    ``sys.modules`` caches it for the per-matchup player -- which is why it has to be the SAME tree the player
    resolves (``SlippiAIConfig.resolved_source_dir``: the release's own revision, or an operator's
    ``source_dir``).  22 Sep 2026: preloading the run file's ``source_dir`` -- the pin -- ahead of the shared
    delay-0 Fox's own checkout tripped the one-tree guard and killed sixteen campaign jobs at construction.
    """

    settings = config.slippi_ai
    source_dir = settings.resolved_source_dir
    revision = settings.released.revision[:7]
    say(f"loading slippi-ai from {source_dir} (revision {revision}, model {settings.model})")
    load_slippi_ai_api(source_dir)


def record_matches(
    config: RLConfig,
    *,
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    bc: str | Path | None = None,
    step: int | None = None,
    label: str | None = None,
    device: torch.device | str = "cpu",
    render: bool = True,
    force_baseline: bool = False,
    renderer: RenderFn | None = None,
    stamp: str | None = None,
    log: Callable[[str], None] | None = None,
) -> VideoRun:
    """Record every configured matchup for ``checkpoint`` and return the manifest of the job.

    Stage 1 plays each matchup in a fresh environment of ``clips`` Dolphins that keep their ``.slp``;
    stage 2 (``render``) turns each replay into a captioned mp4 through ``renderer``.  Baseline
    matchups (the BC checkpoint as the player) are reused from their sha256-keyed directory unless
    ``force_baseline``.  Nothing here raises for a clip that fails: the failure is in its record.
    """

    video = config.video
    specs = expand_matches(video)
    bc_path = Path(bc) if bc is not None else (Path(video.bc) if video.bc else None)
    if video.needs_bc and bc_path is None:
        raise ValueError(
            "[video] matchups mention the BC checkpoint but none is configured: set [video] bc in the "
            "run file or pass --bc <path>"
        )
    if bc_path is not None and not bc_path.is_file():
        raise FileNotFoundError(f"[video] bc checkpoint {bc_path} does not exist")
    source = Path(checkpoint)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint {source} does not exist")
    say = log if log is not None else (lambda _message: None)

    stamped = stamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    root = Path(output_dir) if output_dir is not None else Path(video.output_dir)
    name = f"{source.stem}-step{step}-{stamped}" if step is not None else f"{source.stem}-{stamped}"
    run_dir = root / name
    baseline_root = (
        baseline_dir(root, bc_path, subdir=video.baseline_subdir) if bc_path is not None else run_dir
    )
    baseline_specs = [spec for spec in specs if spec.baseline]
    reused: list[ClipRecord] | None = None
    if baseline_specs and not force_baseline:
        reused = _reusable_baseline(baseline_root, baseline_specs)
    run_dir.mkdir(parents=True, exist_ok=True)
    if baseline_specs and reused is None:
        baseline_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    student_label = label or source.stem
    if step is not None:
        student_label = f"{student_label} step {step}"
    # 11 Sep 2026: a file plays at the delay it was trained at unless the run file says otherwise.
    trained = checkpoint_lib.trained_delay(source)
    bc_delay = 0
    if bc_path is not None:
        bc_delay = (
            video.bc_delay_frames
            if video.bc_delay_frames is not None
            else checkpoint_lib.trained_delay(bc_path)
        )
    student_delays = {"policy": trained, "bc": bc_delay}
    policies: dict[str, PolicyProtocol] = {}
    mimic_runtime: MimicRuntime | None = None
    if video.needs_mimic:
        say(f"loading MIMIC from {config.mimic.resolved_checkpoint}")
        mimic_runtime = load_runtime(config.mimic)
    if video.needs_smashbot:
        # Fail before any Dolphin boots: a missing checkout is a five-second error here and a
        # forty-minute one after a pool of emulators has come up.  ``sys.modules`` caches the import,
        # so the per-matchup ``SmashBotPlayer`` reuses exactly this one.
        say(f"loading SmashBot from {config.smashbot.source_dir} (revision {SMASHBOT_REVISION[:7]})")
        load_smashbot_api(config.smashbot.source_dir, compat=config.smashbot.compat)
    if video.needs_slippi_ai:
        preload_slippi_ai(config, say)
    if video.needs_phillip:
        # Same reason, one step further: start the sidecar interpreter, restore one agent and play a few
        # synthetic frames now.  A wrong interpreter, a missing checkout, a character mismatch or a
        # restore that initialised variables is a ten-second error here.
        check_phillip_character(config.phillip, config.env.dolphin.characters)
        say(f"preflighting phillip {config.phillip.agent} (revision {PHILLIP_REVISION[:7]})")
        say(format_preflight(phillip_preflight(config.phillip, log=say)))
    clips: list[ClipRecord] = []
    for spec in specs:
        if spec.baseline and reused is not None:
            clips.extend(clip for clip in reused if clip.matchup == spec.name)
            say(f"{spec.name}: reused from {baseline_root}")
            continue
        directory = baseline_root if spec.baseline else run_dir
        if spec.student not in policies:
            path = bc_path if spec.student == "bc" else source
            assert path is not None
            policies[spec.student] = _load_policy(path, device)
        policy = policies[spec.student]
        actor = _actor_config(config, policy, trained_delay=student_delays[spec.student])
        say(f"{spec.name}: {spec.clips} clips x {spec.seconds:g} s at delay {actor.delay_frames}")
        stats, frames, seconds = _play(
            config,
            spec,
            student=policy,
            bc=bc_path,
            replay_root=directory / REPLAY_SUBDIR / spec.name,
            actor=actor,
            device=device,
            mimic=mimic_runtime,
            log=say,
            bc_delay=bc_delay,
        )
        replays = collect_replays(
            directory / REPLAY_SUBDIR / spec.name, spec.clips, first=video.mode == "match"
        )
        this_label = bc_path.stem if spec.student == "bc" and bc_path is not None else student_label
        for index, (row, replay) in enumerate(zip(stats, replays, strict=True), start=1):
            clip = ClipRecord(
                matchup=spec.name,
                index=index,
                student=spec.student,
                opponent=spec.opponent,
                seconds=frames / FRAMES_PER_SECOND,
                frames=frames,
                caption=caption_for(
                    spec,
                    index,
                    student_label=this_label,
                    opponent_label=_opponent_label(spec, bc_path),
                ),
                stats=row,
            )
            if replay is None:
                clip.error = f"no replay written by Dolphin {index - 1} of {spec.name}"
            else:
                kept = directory / f"{spec.name}_{index}.slp"
                replay.replace(kept)
                clip.slp = str(kept)
                try:
                    clip.match = match_record(slp.summarize_replay(kept))
                except (OSError, ValueError, IndexError, struct.error) as error:
                    # A killed Dolphin can leave a replay that does not parse; the clip still stands.
                    clip.match = None
                    say(f"  {spec.name} {index}: could not read {kept.name} ({error})")
            clips.append(clip)
        write_clip_records(clips[-spec.clips :], directory)
        say(f"{spec.name}: {seconds:.0f} s of wall clock, {frames} frames per clip")

    fresh = [clip for clip in clips if not clip.reused]
    if render and fresh:
        for clip in fresh:
            if clip.slp is not None:
                clip.video = str(Path(clip.slp).with_suffix(".mp4"))
        jobs = render_jobs(fresh)
        if jobs:
            call = renderer if renderer is not None else _default_renderer(video.render)
            say(f"rendering {len(jobs)} clips")
            by_target = {str(result.target): result for result in call(jobs)}
            for clip in fresh:
                result = by_target.get(clip.video or "")
                if result is None:
                    continue
                clip.rendered = result.ok
                clip.video = result.video
                clip.duration_s = result.duration_s
                clip.render = result.record()
                clip.error = result.error

    run = VideoRun(
        directory=str(run_dir),
        checkpoint=str(source),
        checkpoint_sha256=checkpoint_lib.sha256_file(source),
        step=step,
        baseline_directory=str(baseline_root),
        baseline_reused=reused is not None,
        clips=clips,
        seconds=time.perf_counter() - started,
        delay_frames=_played_delay(config, trained),
    )
    for clip in fresh:  # refresh with what the render pass added (the mp4 path, its duration)
        if clip.slp is not None:
            write_clip_records([clip], Path(clip.slp).parent)
    run.manifest = str(write_manifest(run, run_dir / MANIFEST_NAME))
    if baseline_specs and reused is None:
        baseline_clips = [clip for clip in clips if clip.student == "bc"]
        record = replace(run, directory=str(baseline_root), clips=baseline_clips)
        write_manifest(record, baseline_root / MANIFEST_NAME)
    return run


def _default_renderer(config: RenderConfig) -> RenderFn:
    """``render.render_many`` with a scratch work directory outside the (slow) output volume."""

    work = Path(tempfile.gettempdir()) / "melee-rl-render"

    def call(jobs: Sequence[RenderJob]) -> list[RenderResult]:
        return render_many(config, jobs, work_dir=work)

    return call


def wandb_upload(
    config: RLConfig,
    run: VideoRun,
    *,
    mode: str = "offline",
    work_dir: str | Path | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Log a ``video.wandb_seconds`` trim of every rendered clip as ``wandb.Video`` (off by default).

    A dedicated ``job_type = "video"`` run in the same project keeps the clips out of the training
    run's history; the trim (about 4.5 MB at 30 s / 1200 kbps) respects W&B's "under 40 MB of video
    per minute" guidance where a full 2-minute clip at 3000 kbps would not (ENV.md §6.5).  The caption
    is already burned into the source, so the trim only cuts and re-encodes.
    """

    from melee_rl.render import ffmpeg_command, run_command

    video = config.video
    rendered = [clip for clip in run.clips if clip.rendered and clip.video]
    report: dict[str, Any] = {"uploaded": 0, "skipped": len(run.clips) - len(rendered), "error": None}
    if not rendered:
        return report
    root = Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir()) / "melee-rl-wandb"
    root.mkdir(parents=True, exist_ok=True)
    trims: list[tuple[ClipRecord, Path]] = []
    for clip in rendered:
        assert clip.video is not None
        target = root / f"{clip.matchup}_{clip.index}.mp4"
        command = ffmpeg_command(
            video.render,
            source=clip.video,
            target=target,
            trim_seconds=video.wandb_seconds,
            bitrate_kbps=video.wandb_bitrate_kbps,
        )
        result = run_command(command, timeout_s=video.render.timeout_s)
        if result.ok and target.is_file():
            trims.append((clip, target))
        else:
            report["skipped"] += 1
    if not trims:
        report["error"] = "no clip could be trimmed for the upload"
        return report
    import wandb

    logging_config = config.logging
    session = wandb.init(
        entity=logging_config.entity,
        project=logging_config.project,
        group=logging_config.group,
        job_type="video",
        name=name or f"video-{Path(run.directory).name}",
        mode=mode,  # type: ignore[arg-type]
        tags=[*logging_config.tags, "video", "p9"],
        config={"checkpoint": run.checkpoint, "sha256": run.checkpoint_sha256, "step": run.step},
    )
    try:
        for clip, trim in trims:
            key = f"video/{clip.matchup}_{clip.index}"
            payload: dict[str, Any] = {key: wandb.Video(str(trim), caption=clip.caption, format="mp4")}
            payload.update({f"{key}/{stat}": value for stat, value in clip.stats.items()})
            session.log(payload)
            report["uploaded"] += 1
    finally:
        report["run_id"] = getattr(session, "id", None)
        session.finish()
    return report


def clip_stats(
    trajectories: Sequence[Trajectory],
    index: int,
    *,
    frames: int,
    seconds: float,
    stalling_threshold: float,
) -> dict[str, float]:
    """:data:`CLIP_STATS` for one environment of ``trajectories`` (the reusable half of ``_play``)."""

    metrics: Mapping[str, float] = trajectory_metrics(
        [trajectory.index_envs([index]) for trajectory in trajectories],
        ports=1,
        stalling_threshold=stalling_threshold,
    )
    return stats_from_metrics(metrics, frames=frames, seconds=seconds)


__all__ = [
    "CLIP_STATS",
    "FRAMES_PER_SECOND",
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "MATCHUP_KINDS",
    "REPLAY_SUBDIR",
    "VIDEO_MODES",
    "ClipRecord",
    "MatchSpec",
    "RenderFn",
    "VideoConfig",
    "VideoRun",
    "baseline_dir",
    "caption_for",
    "clip_stats",
    "collect_replays",
    "expand_matches",
    "format_summary",
    "games_started",
    "match_record",
    "parse_matchup",
    "record_matches",
    "render_jobs",
    "wandb_upload",
    "write_clip_records",
    "write_manifest",
]
