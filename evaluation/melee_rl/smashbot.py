"""SmashBot as a live opponent (P25, 9 Sep 2026).

`altf4/SmashBot <https://github.com/altf4/SmashBot>`_ (GPL-3.0, pinned at
:data:`SMASHBOT_REVISION`) is the strongest public Melee agent that is not a neural network: a
hand-written expert system that picks a **Strategy**, which picks a **Tactic**, which picks a **Chain**
of button presses, and executes technical sequences frame by frame.  It plays Fox.  It is the opponent
the claim "best Melee bot" has to survive, where MIMIC and Dolphin's CPU 9 only test "best learned Fox".

Its interface is the one :mod:`melee_rl.mimic` already crosses: ``esagent.ESAgent.act(gamestate)``
takes a raw libmelee ``GameState`` and *presses a libmelee controller*.  So this module reuses the same
two pieces -- ``DolphinEnv.gamestates()`` for the observation and
:class:`~melee_rl.mimic.ControllerRecorder` for the controller -- and
:class:`~melee_rl.opponents.SmashBotOpponent` returns a ``ControllerState`` through ``act_state``
rather than labels through ``act``, so nothing is quantised onto Ali's 85-value ``main_stick``
vocabulary.  Upstream's own ``smashbot.py`` main loop is *not* used: our environment owns the console,
the menus and the flush.

Four things had to be built around it.

* **A persistent controller.**  libmelee's ``Controller.current`` is durable -- a button pressed on
  frame N stays pressed until it is released, and ``flush()`` only copies current to prev -- and
  SmashBot's chains rely on that, writing only the fields they mean to change.  MIMIC's decoder writes
  every field every frame, so :mod:`melee_rl.mimic` builds a *fresh* recorder per frame; doing that
  here would silently drop every held input.  :class:`PersistentControllerRecorder` lives for the whole
  game and adds the two methods upstream calls that MIMIC's decoder never does (``release_all`` /
  ``empty_input``, with libmelee's exact resting state) plus a no-op ``connect``.
* **A private ``random`` stream.**  Seven SmashBot modules draw from the *global* ``random`` module
  (``Tactics/recover.py`` picks its recovery angle and its Firefox height that way), so every frame runs
  inside :class:`_RandomStream`, which swaps a private state in and the caller's back out -- the same
  trick :class:`melee_rl.mimic._RngStream` plays for torch.
* **One shared ``FrameData``.**  ``melee.framedata.FrameData()`` measures **800 ms and 54 MB** per
  instance, and ``ESAgent.__init__`` builds one unconditionally.  One per Dolphin would be 19 s and
  1.3 GB at 24 environments and 76 s and 5.2 GB at 96, so a single instance is built here and injected
  by patching ``melee.framedata.FrameData`` for the duration of agent construction only
  (:func:`_shared_framedata`); the result is verified with an assertion rather than assumed.
* **An error policy that is not silence.**  Upstream wraps every ``act`` in
  ``except Exception: controller.empty_input()``, so a SmashBot broken by the libmelee drift
  (:mod:`melee_rl.smashbot_compat`) does not crash -- it plays like a passive, slightly-too-slow Fox,
  which is indistinguishable from the "modern Dolphin breaks its recoveries" report we are here to
  test.  ``strict`` (the default) lets the exception out; ``strict = false`` reproduces upstream's
  behaviour but *counts* it, and :meth:`SmashBotPlayer.stats` reports the count, the first traceback,
  the share of frames that held a neutral input and a histogram of which Tactic and Chain were live.
  That histogram is the calibration gate's main instrument: a SmashBot that never leaves ``Wait`` is
  broken however good its win rate against a CPU looks.

Pinned upstream: revision :data:`SMASHBOT_REVISION`, licence GPL-3.0. Users supply its source tree
separately. This adapter imports it into the evaluation process. Dependency combination and
distribution obligations require review as described in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import contextlib
import logging
import random
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

from melee_rl import smashbot_compat
from melee_rl.env.dolphin import PORTS, ControllerRow, neutral_row
from melee_rl.mimic import ControllerRecorder
from tensor_batch import BUTTON_ORDER

LOG: Final[logging.Logger] = logging.getLogger(__name__)

SMASHBOT_REPO_URL: Final[str] = "https://github.com/altf4/SmashBot.git"
SMASHBOT_REVISION: Final[str] = "67c412092e4ef1576e1357a79461df5faedad3c2"
"""``Fix stage API bugs with libmelee 0.41.0`` (15 May 2024), the repository's HEAD and its last commit."""
SMASHBOT_LICENSE: Final[str] = "GPL-3.0"
DEFAULT_SOURCE_DIR: Final[str] = "/opt/smashbot/source"
"""Default location for the separately supplied :data:`SMASHBOT_REVISION` source tree."""
ENTRY_MODULE: Final[str] = "esagent.py"
"""The one upstream file we import; its absence is what :func:`load_smashbot_api` reports."""
INITIAL_FRAME: Final[int] = -123
"""Melee's first frame; 0.41.1 clears any invulnerability there."""
DEFAULT_DIFFICULTY: Final[int] = 4
"""SmashBot's strongest real setting, and it must be passed explicitly.

``Strategies/bait.py`` sets ``self.difficulty = 4`` in ``__init__`` but *re-derives it every frame* in
``step`` (``:46-50``) from the value it was constructed with:

* ``-1`` -- upstream ``smashbot.py``'s argparse default -- means ``difficulty = smashbot_state.stock``,
  so SmashBot **self-handicaps as it loses stocks**;
* ``5`` is a debug/training dummy (``:98-100``): no attacks, no shielding, only DI and recovery;
* ``4`` is the strongest real setting -- the ``>= 4`` and ``== 4`` gates unlock its full behaviour.

Passing ``4`` is therefore load-bearing, not cosmetic: the constructor's initial value is overwritten on
the first frame."""

_IN_GAME_MENU_NAMES: Final[tuple[str, ...]] = ("IN_GAME", "SUDDEN_DEATH")
"""The menus SmashBot may act in -- upstream's ``menu_state == melee.Menu.IN_GAME`` plus sudden death,
which is what ``melee_rl.env.dolphin._DolphinInstance`` also treats as in-game."""

_MENUS_CACHE: frozenset[Any] | None = None


def _in_game_menus() -> frozenset[Any]:
    """``{melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH}``, resolved on first use.

    Resolved lazily so importing this module -- and therefore :mod:`melee_rl.opponents` -- does not
    require libmelee, matching the discipline of :mod:`melee_rl.env.dolphin`.
    """

    global _MENUS_CACHE
    if _MENUS_CACHE is None:
        from melee_rl.env.libmelee_frames import import_melee

        melee = import_melee()
        _MENUS_CACHE = frozenset(getattr(melee.Menu, name) for name in _IN_GAME_MENU_NAMES)
    return _MENUS_CACHE


def __getattr__(name: str) -> Any:
    """``IN_GAME_MENUS`` as a lazily resolved module attribute (PEP 562)."""

    if name == "IN_GAME_MENUS":
        return _in_game_menus()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# configuration and the upstream boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmashBotConfig:
    """Where SmashBot lives, which port it plays and how its inputs reach the game.

    ``player`` is the *environment* player index (0 = Dolphin port 1, 1 = port 2); we put SmashBot on
    port 2, as ``melee-policy`` does with MIMIC.

    ``compat`` installs :mod:`melee_rl.smashbot_compat`, the six-signature libmelee shim SmashBot needs
    on our pin.  It is on by default and there is no reason to turn it off in a real campaign -- the
    flag exists so calibration can *measure* what the drift costs in real games rather than assert it,
    which is the only honest way to answer "does a modern Dolphin break SmashBot".

    ``delay_frames`` holds its input back by that many
    frames *on its own port only*: our environment runs ``use_exi_inputs`` + ``blocking_input`` +
    ``online_delay = 0``, so an opponent decides on frame N and its input lands on frame N, which is
    tighter than the named-pipe path SmashBot was written against.  Chains that press a frame early to
    absorb that latency then land a frame early, which is exactly what a missed ledgedash or Firefox
    looks like -- so the calibration sweep measures 0, 1 and 2 before any campaign runs.
    """

    source_dir: str = DEFAULT_SOURCE_DIR
    player: int = 1
    seed: int = 0
    delay_frames: int = 0
    strict: bool = True
    compat: bool = True

    def __post_init__(self) -> None:
        if not self.source_dir:
            raise ValueError("[smashbot] source_dir must name the SmashBot checkout")
        if self.player not in (0, 1):
            raise ValueError(f"[smashbot] player must be 0 or 1, got {self.player}")
        if self.delay_frames < 0:
            raise ValueError(f"[smashbot] delay_frames must be >= 0, got {self.delay_frames}")

    @property
    def libmelee_port(self) -> int:
        """SmashBot's *libmelee* port, 1 or 2 (``melee_rl.env.dolphin.PORTS``).

        Distinct from :attr:`player`, the *environment* player index, which is what
        ``Opponent.port`` means everywhere else in ``melee_rl`` (``actor.py`` checks it against
        ``EnvProtocol.controlled_ports``).  ``ESAgent`` and ``gamestate.players`` are keyed by the
        libmelee port; the rollout worker and ``env.step`` are keyed by the player index.
        """

        return PORTS[self.player]

    @property
    def libmelee_opponent_port(self) -> int:
        """The libmelee port ``ESAgent`` should treat as its enemy."""

        return PORTS[1 - self.player]


@dataclass(frozen=True)
class SmashBotApi:
    """The single upstream symbol we use; faked wholesale in ``tests/rl/test_smashbot.py``."""

    agent_factory: Callable[..., Any]
    """``esagent.ESAgent(dolphin, smashbot_port, opponent_port, controller, difficulty)``."""


def load_smashbot_api(source_dir: str | Path, *, compat: bool = True) -> SmashBotApi:
    """Import ``esagent.ESAgent`` from a SmashBot checkout, putting it on ``sys.path`` first.

    With ``compat`` (the default) the compatibility shim is installed first.  SmashBot only ever
    reaches the drifted helpers through the ``melee`` module object (``import melee`` in 53 places; its
    ``from melee.enums import ...`` lines are enums, which did not change), so patch order is not
    actually load-bearing -- but a shim guaranteed to precede the first ``act`` is one less thing to
    reason about.

    The checkout root goes on ``sys.path`` because upstream imports its own top-level packages
    (``Strategies``, ``Tactics``, ``Chains``) absolutely.  Those names are generic; nothing in this
    repository or its dependencies uses them, and MIMIC's checkout is admitted the same way.
    """

    if compat:
        LOG.info("smashbot compat: %s", smashbot_compat.install() or "already installed")
    else:
        LOG.warning("smashbot compat is OFF: the libmelee drift is live (calibration only)")
    root = Path(source_dir).expanduser().resolve()
    module = root / ENTRY_MODULE
    if not module.is_file():
        raise FileNotFoundError(f"SmashBot checkout has no {ENTRY_MODULE}: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from esagent import ESAgent  # type: ignore[import-not-found]

    return SmashBotApi(agent_factory=ESAgent)


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------


_INVULN_ACTIONS: tuple[Any, Any, Any] | None = None

RESPAWN_INVULNERABILITY: Final[int] = 120
"""Frames granted on respawn (``ON_HALO_WAIT`` / ``ON_HALO_DESCENT``), libmelee 0.41.1."""
LEDGE_INVULNERABILITY: Final[int] = 36
"""Frames granted on the first frame of ``EDGE_CATCHING``, libmelee 0.41.1."""
FIRST_DESCENT_FRAME: Final[int] = 150
"""0.41.1 grants nothing for the opening drop-in, only for descents after this frame."""


def _invulnerability_actions() -> tuple[Any, Any, Any]:
    """``(ON_HALO_WAIT, ON_HALO_DESCENT, EDGE_CATCHING)``, resolved once."""

    global _INVULN_ACTIONS
    if _INVULN_ACTIONS is None:
        from melee_rl.env.libmelee_frames import import_melee

        action = import_melee().Action
        _INVULN_ACTIONS = (action.ON_HALO_WAIT, action.ON_HALO_DESCENT, action.EDGE_CATCHING)
    return _INVULN_ACTIONS


class _InvulnerabilityTracker:
    """libmelee 0.41.1's ``console.py:818-838``, which 0.47.3 stopped computing.

    0.41.1's *parser* maintained ``PlayerState.invulnerability_left`` -- 120 frames from a respawn, 36
    from a ledge grab, counted down per port against the frame the grant was made on.  0.47.3 keeps the
    field but never assigns it, so it is permanently ``0`` and SmashBot believes nobody is ever
    invulnerable.  It reads the field in eight places in ``Tactics/edgeguard.py`` alone, in
    ``Tactics/challenge.py``, and in its approach gate (``Strategies/bait.py:66``), so a dead countdown
    makes it attack into respawn and ledge invulnerability and mistime its edgeguards.

    One tracker per environment, applied to *both* ports (SmashBot reads its opponent's invulnerability
    as well as its own) before ``ESAgent.act`` sees the frame.

    **One deliberate deviation from 0.41.1:** it also did
    ``if invulnerability_left > 0: playerstate.invulnerable = True``.  That is not reproduced.  0.47.3
    sets the boolean from the game's own byte (``console.py:1273``), which is strictly better, and it is
    the field ``melee_rl`` feeds the *model* (``frames.py:55``) -- so leaving it alone keeps this
    restoration entirely on SmashBot's side of the boundary.
    """

    __slots__ = ("_seen", "_start")

    def __init__(self) -> None:
        self._start: dict[int, tuple[int, int]] = {}
        self._seen: set[int] = set()

    def reset(self) -> None:
        self._start.clear()
        self._seen.clear()

    def apply(self, gamestate: Any) -> None:
        halo_wait, halo_descent, edge_catching = _invulnerability_actions()
        frame = int(getattr(gamestate, "frame", 0))
        for port, player in getattr(gamestate, "players", {}).items():
            start = self._start.get(port)
            if port in self._seen and start is not None:
                player.invulnerability_left = max(0, start[1] - (frame - start[0]))
            action = getattr(player, "action", None)
            if action == halo_wait:
                player.invulnerability_left = RESPAWN_INVULNERABILITY
                self._start[port] = (frame, RESPAWN_INVULNERABILITY)
            # 0.41.1 skips the opening drop-in, which is not invulnerable.
            if action == halo_descent and frame > FIRST_DESCENT_FRAME:
                player.invulnerability_left = RESPAWN_INVULNERABILITY
                self._start[port] = (frame, RESPAWN_INVULNERABILITY)
            if action == edge_catching and int(getattr(player, "action_frame", 0)) == 1:
                player.invulnerability_left = LEDGE_INVULNERABILITY
                self._start[port] = (frame, LEDGE_INVULNERABILITY)
            if frame == INITIAL_FRAME:
                player.invulnerability_left = 0
                self._start[port] = (frame, 0)
            self._seen.add(port)


_CONTROLLER_BITS: tuple[Any, dict[str, Any]] | None = None


def _libmelee_controller_bits() -> tuple[Any, dict[str, Any]]:
    """``(ControllerState, {"A": Button.BUTTON_A, ...})``, resolved once per process.

    Cached deliberately.  ``import_melee`` re-runs ``importlib.metadata.version("melee")`` on every
    call, which is a site-packages scan of several milliseconds; :meth:`PersistentControllerRecorder.commit`
    runs once per Dolphin per frame and needs nine of these, so calling it live measured **73 ms per
    Dolphin-frame** against a 53-65 ms frame-step -- the opponent would have become the bottleneck.
    ``Button``'s enum values are exactly the names of ``BUTTON_ORDER``, which is what ``send_controller``
    already relies on (``env/dolphin.py:621``).
    """

    global _CONTROLLER_BITS
    if _CONTROLLER_BITS is None:
        from melee_rl.env.libmelee_frames import import_melee

        melee = import_melee()
        buttons = {name: melee.Button(name) for name in BUTTON_ORDER}
        _CONTROLLER_BITS = (melee.controller.ControllerState, buttons)
    return _CONTROLLER_BITS


class PersistentControllerRecorder(ControllerRecorder):
    """A libmelee ``Controller`` stand-in that lives for a whole game, as the real one does.

    ``ControllerRecorder`` already covers the four methods MIMIC's decoder calls.  SmashBot needs three
    more things from a controller, all of which MIMIC's decoder never touches:

    * ``release_all`` / ``empty_input`` (16 and 95 call sites: every tactic's idle branch), with
      libmelee's exact resting state -- every button up, both sticks ``0.5``, both shoulders ``0``.
    * ``connect``, on the construction path.
    * **``prev``** (28 call sites, e.g. ``Chains/dashdance.py:25`` asking whether the stick was already
      held right last frame).  In libmelee that is ``Controller.prev``, set by ``flush()`` as
      ``prev = copy(current)``.  ``DolphinEnv.step`` flushes the real controller once per frame, after
      the opponent has decided, so :meth:`commit` is called once per frame at the end of
      :meth:`SmashBotPlayer.rows` -- which makes ``prev`` inside frame N's ``act`` the state sent on
      frame N-1, exactly as upstream sees it.  It is a real libmelee ``ControllerState`` so its
      ``button`` dict is keyed by ``melee.Button``, the way SmashBot indexes it.

    The instance is *not* rebuilt per frame: :meth:`row` reports the accumulated state, exactly as
    ``Controller.flush`` sends ``Controller.current`` without clearing it.
    """

    __slots__ = ("prev",)

    def __init__(self) -> None:
        super().__init__()
        state_type, _ = _libmelee_controller_bits()
        self.prev: Any = state_type()

    def commit(self) -> None:
        """libmelee's ``flush()``: ``prev`` becomes what this frame sent.  Once per frame."""

        state_type, buttons = _libmelee_controller_bits()
        state = state_type()
        for name, pressed in self._buttons.items():
            state.button[buttons[name]] = pressed
        state.main_stick = self._main
        state.c_stick = self._c
        state.l_shoulder = self._l_shoulder
        state.r_shoulder = self._r_shoulder
        self.prev = state

    def connect(self) -> bool:
        """Accepted and ignored: ``DolphinEnv`` owns the real controller and its pipe."""

        return True

    def release_all(self) -> None:
        """libmelee's resting state: all buttons up, both sticks centred, both shoulders released."""

        neutral = neutral_row()
        for name in self._buttons:
            self._buttons[name] = False
        self._main = (neutral.main_x, neutral.main_y)
        self._c = (neutral.c_x, neutral.c_y)
        self._l_shoulder = neutral.shoulder
        self._r_shoulder = 0.0

    def empty_input(self) -> None:
        """libmelee aliases this to ``release_all``; upstream's error handler calls it."""

        self.release_all()


# ---------------------------------------------------------------------------
# the random stream
# ---------------------------------------------------------------------------


class _RandomStream:
    """A private global-``random`` stream: SmashBot draws from the module-level generator.

    One stream per player rather than per environment, mirroring :class:`melee_rl.mimic._RngStream`:
    ``random.getstate()`` builds a 625-element tuple, so a swap per environment per frame would be
    ~2 900 of them a second at 24 Dolphins for no behavioural gain.  The consequence is that the
    environments share one draw sequence, so a different number of live environments gives a different
    (still reproducible) sequence -- exactly MIMIC's property.
    """

    __slots__ = ("_seed", "_state")

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._state: tuple[Any, ...] | None = None

    def reset(self) -> None:
        self._state = None

    def invoke(self, call: Callable[[], Any]) -> Any:
        outer = random.getstate()
        if self._state is None:
            random.seed(self._seed)
        else:
            random.setstate(self._state)
        try:
            return call()
        finally:
            self._state = random.getstate()
            random.setstate(outer)


# ---------------------------------------------------------------------------
# the shared FrameData
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _shared_framedata(melee: Any, framedata: Any) -> Iterator[None]:
    """Make ``melee.framedata.FrameData()`` hand back ``framedata`` for the duration of the block.

    ``ESAgent.__init__`` builds its own unconditionally (800 ms, 54 MB), and its ``Bait`` strategy
    passes that instance down to every Tactic and Chain it later constructs, so replacing the
    attribute afterwards would be both wasteful and fragile.  Scoped to agent construction; nothing
    else in ``melee_rl`` constructs a ``FrameData`` at all.
    """

    original = melee.framedata.FrameData
    melee.framedata.FrameData = lambda *args, **kwargs: framedata
    try:
        yield
    finally:
        melee.framedata.FrameData = original


class _AgentHost:
    """The ``dolphin`` argument of ``ESAgent.__init__``, which only reads ``.logger``.

    ``.framedata`` is carried too so a faked agent factory can find the shared instance without the
    ``melee`` patch (``tests/rl/test_smashbot.py``).
    """

    __slots__ = ("framedata", "logger")

    def __init__(self, framedata: Any) -> None:
        self.logger = None
        self.framedata = framedata


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------


class SmashBotPlayer:
    """SmashBot on one port across ``num_envs`` environments; one ``ESAgent`` each, one ``FrameData``."""

    def __init__(
        self,
        config: SmashBotConfig,
        num_envs: int,
        *,
        api: SmashBotApi | None = None,
        framedata: Any | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.config = config
        self.num_envs = num_envs
        self.api = load_smashbot_api(config.source_dir, compat=config.compat) if api is None else api
        self._melee: Any | None = None
        if framedata is None:
            from melee_rl.env.libmelee_frames import import_melee

            self._melee = import_melee()
            framedata = self._melee.framedata.FrameData()
        self.framedata = framedata
        self._in_game = _in_game_menus()
        self._host = _AgentHost(self.framedata)
        self._rng = _RandomStream(config.seed)
        self._recorders: list[PersistentControllerRecorder] = []
        self._agents: list[Any] = []
        self._invuln: list[_InvulnerabilityTracker] = [_InvulnerabilityTracker() for _ in range(num_envs)]
        self.frames_seen = 0
        self.neutral_frames = 0
        self.exceptions = 0
        self.first_error: str = ""
        self.exception_envs: set[int] = set()
        self.tactics: Counter[str] = Counter()
        for index in range(num_envs):
            self._recorders.append(PersistentControllerRecorder())
            self._agents.append(self._build_agent(index))

    # -- construction ------------------------------------------------------

    def _build_agent(self, index: int) -> Any:
        recorder = self._recorders[index]
        build = lambda: self.api.agent_factory(  # noqa: E731 - one call, kept next to its guard
            self._host,
            self.config.libmelee_port,
            self.config.libmelee_opponent_port,
            recorder,
            DEFAULT_DIFFICULTY,
        )
        if self._melee is None:
            agent = build()
        else:
            with _shared_framedata(self._melee, self.framedata):
                agent = build()
        if getattr(agent, "framedata", self.framedata) is not self.framedata:
            raise RuntimeError(
                "SmashBot's ESAgent did not take the shared FrameData; upstream's construction "
                "changed and one instance per Dolphin would cost 54 MB and 800 ms each"
            )
        return agent

    def reset(self, index: int) -> None:
        """Forget one environment: a fresh agent (strategy, tactic, lockouts) and a neutral controller."""

        self._recorders[index] = PersistentControllerRecorder()
        self._agents[index] = self._build_agent(index)
        self._invuln[index].reset()

    def reset_all(self) -> None:
        for index in range(self.num_envs):
            self.reset(index)
        self._rng.reset()

    # -- one frame ---------------------------------------------------------

    def rows(self, gamestates: Sequence[Any], needs_reset: torch.Tensor) -> list[ControllerRow]:
        """One frame for every environment: reset, guard, act, read the controller back.

        A row is produced for every environment.  An environment that is not live, is in a menu, or has
        no SmashBot port in its gamestate holds its controller's *current* state rather than acting,
        which is what a real controller does when nobody touches it.
        """

        if len(gamestates) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} gamestates, got {len(gamestates)}")
        if int(needs_reset.shape[0]) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset flags, got {int(needs_reset.shape[0])}")

        flags = needs_reset.to(dtype=torch.bool, device="cpu").tolist()
        live: list[int] = []
        for index, gamestate in enumerate(gamestates):
            if flags[index]:
                self.reset(index)
            if self._actionable(gamestate):
                live.append(index)
        if live:
            self._rng.invoke(lambda: self._act(live, gamestates))
        rows = [recorder.row() for recorder in self._recorders]
        # ``DolphinEnv.step`` flushes every controlled port every frame, menu frame or not, and
        # libmelee's ``flush`` is what advances ``prev``.  Do the same here so a SmashBot that skipped
        # a frame still sees an accurate "what did I send last frame".
        for recorder in self._recorders:
            recorder.commit()
        return rows

    def _actionable(self, gamestate: Any) -> bool:
        """Upstream's own two guards: ``menu_state is IN_GAME`` and ``port in gamestate.players``."""

        if gamestate is None:
            return False
        if getattr(gamestate, "menu_state", None) not in self._in_game:
            return False
        return self.config.libmelee_port in getattr(gamestate, "players", {})

    def _act(self, live: Sequence[int], gamestates: Sequence[Any]) -> None:
        for index in live:
            agent = self._agents[index]
            if self.config.compat:
                # The countdown 0.47.3 stopped computing; part of the compatibility restoration, so the
                # calibration A/B (compat = false) measures the whole drift including this.
                self._invuln[index].apply(gamestates[index])
            try:
                agent.act(gamestates[index])
            except Exception as error:
                self._record_exception(index, error)
                if self.config.strict:
                    raise
                # Upstream's handler, so a half-written frame cannot stick as a held input.
                self._recorders[index].empty_input()
            self.frames_seen += 1
            self.tactics[self._tactic_label(agent)] += 1
            if self._recorders[index].row() == neutral_row():
                self.neutral_frames += 1

    def _record_exception(self, index: int, error: BaseException) -> None:
        self.exceptions += 1
        self.exception_envs.add(index)
        if not self.first_error:
            self.first_error = f"env {index}: {type(error).__name__}: {error}\n" + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        LOG.warning("SmashBot env %d raised %s: %s", index, type(error).__name__, error)

    @staticmethod
    def _tactic_label(agent: Any) -> str:
        """``"<Tactic>/<Chain>"`` -- the calibration gate's read on whether SmashBot is really playing.

        Upstream's ``str(strategy)`` concatenates ``repr`` of the classes; the objects are read here
        instead so the histogram keys stay short.
        """

        tactic = getattr(getattr(agent, "strategy", None), "tactic", None)
        chain = getattr(tactic, "chain", None)
        name = type(tactic).__name__ if tactic is not None else "-"
        return f"{name}/{type(chain).__name__ if chain is not None else '-'}"

    # -- health ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What the calibration gate reads: exceptions must be zero and the tactics must not be idle."""

        return {
            "envs": self.num_envs,
            "frames": self.frames_seen,
            "neutral_frames": self.neutral_frames,
            "neutral_share": self.neutral_frames / self.frames_seen if self.frames_seen else 0.0,
            "exceptions": self.exceptions,
            "exception_envs": sorted(self.exception_envs),
            "first_error": self.first_error,
            "tactics": dict(self.tactics.most_common()),
            "revision": SMASHBOT_REVISION,
            "delay_frames": self.config.delay_frames,
            "compat": self.config.compat,
        }


__all__ = [
    "DEFAULT_DIFFICULTY",
    "DEFAULT_SOURCE_DIR",
    "ENTRY_MODULE",
    "SMASHBOT_LICENSE",
    "SMASHBOT_REPO_URL",
    "SMASHBOT_REVISION",
    "PersistentControllerRecorder",
    "SmashBotApi",
    "SmashBotConfig",
    "SmashBotPlayer",
    "load_smashbot_api",
]
