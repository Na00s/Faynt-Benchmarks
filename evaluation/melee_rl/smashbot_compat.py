"""The libmelee compatibility shim SmashBot needs (P25, 9 Sep 2026).

`altf4/SmashBot <https://github.com/altf4/SmashBot>`_ (GPL-3.0) is a hand-written expert system --
Strategies over Tactics over Chains -- and the strongest public Melee agent that is not a neural
network.  Its HEAD (``67c4120``, 15 May 2024, "Fix stage API bugs with libmelee 0.41.0") is written
against **altf4/libmelee 0.41.1**, published the same day.  Our pin is ``melee==0.47.3``, which is
**vladfi1/libmelee**: PyPI's ``melee`` name moved to that fork at 0.45.0 (20 Feb 2026), and it is the
fork that added ``use_exi_inputs`` / ``blocking_input`` / ``polling_mode`` -- the flags
:mod:`melee_rl.env.dolphin` runs on.

The drift comes in two kinds, and the second is the surprising one.

**Kind 1 -- SmashBot calls libmelee differently than 0.47.3 expects.**  Found by diffing the two sdists
at signature level; everything else that changed is ``Console`` / ``Controller`` / ``MenuHelper``, which
the integration never uses (our environment owns the console and the menus, and upstream
``smashbot.py``'s main loop is not part of it -- we drive ``esagent.ESAgent.act`` directly).

===========================================  ======================  ======================
site                                         0.41.1                  0.47.3
===========================================  ======================  ======================
``stages.top_platform_position``             ``(gamestate)``         ``(stage)``
``stages.left_platform_position``            ``(gamestate)``         ``(stage)``
``stages.right_platform_position``           ``(gamestate)``         ``(stage)``
``stages.side_platform_position``            ``(right, gamestate)``  ``(right, stage)``
``FrameData.roll_end_position``              ``(cs, gamestate)``     ``(cs, stage)``
``FrameData.project_hit_location``           ``(cs, gamestate, n)``  ``(cs, stage, n)``
``Projectile.x`` / ``.y``                    present                 ``.position.x`` / ``.y``
===========================================  ======================  ======================

**Kind 2 -- libmelee 0.47.3 is broken against itself.**  The fork deleted things and left its own
``FrameData`` still naming them, so two of its methods raise for *every* caller:

===============================  =============================================================
site                             what
===============================  =============================================================
``framedata.py:162``             ``is_bmove`` compares against ``Action.UNKNOWN_ANIMATION``,
                                 which was ``0xffff`` in 0.41.1 and is gone in 0.47.3 (replaced
                                 by the class ``gamestate.UnknownAnimation``) -> ``AttributeError``
``framedata.py:325``             ``in_range`` reads ``defender.y``, a flat alias 0.41.1's parser
                                 wrote on every frame (``console.py:733-734``) and 0.47.3 dropped
                                 along with the field -> ``AttributeError``
===============================  =============================================================

Nothing in ``melee_rl`` calls ``FrameData`` at all, which is why these were invisible to us; SmashBot
calls both, and each one cost a Dolphin boot on Modal to find.  ``tests/rl/test_smashbot_compat.py``
now exercises all eighteen ``FrameData`` methods SmashBot uses, and audits the installed libmelee for
dangling enum members and reads of deleted fields, so the next pin bump surfaces its own bugs locally.

One further 0.41 -> 0.47 behaviour change is **recorded but not patched**: ``UnknownAnimation`` is an
unfrozen dataclass and therefore unhashable, where 0.41.1's sentinel was a hashable enum member, so
``framedata[character][action]`` raises ``TypeError`` rather than ``KeyError`` for an unknown
animation.  Only ``is_bmove`` guards that case in either version, it has not been observed to matter in
a real game, and changing libmelee's behaviour on a guess is worse than writing it down.

**Why this matters more than a signature mismatch usually does.**  Two of the four helpers fail
*silently*: ``top_platform_position(gamestate)`` falls through every ``if stage == ...`` branch and
returns ``(None, None, None)``, which reads as "this stage has no platforms" -- so on Battlefield,
Dreamland, Yoshi's Story and Fountain of Dreams SmashBot's juggle, retreat, challenge and
platform-boarding logic quietly stops seeing the platforms.  The other two raise
``TypeError: unhashable type: 'GameState'`` out of ``EDGE_GROUND_POSITION[stage]``, which escapes
``roll_end_position``'s own ``except KeyError``.  Upstream then swallows it: ``smashbot.py`` wraps every
``act`` in ``except Exception: controller.empty_input()``.  A SmashBot broken this way does not crash --
it looks like a working but strangely passive one, which is exactly how a "modern Dolphin makes SmashBot
just barely miss its recoveries" report would present.  :mod:`melee_rl.smashbot` therefore counts and
reports exceptions instead of swallowing them, and this module removes their cause.

**Why patching the installed libmelee is safe.**  The wrappers are *additive and backward compatible*:
each coerces its stage argument with :func:`as_stage` (``getattr(argument, "stage", argument)``), which
turns a ``GameState`` into its ``Stage`` and leaves a ``Stage`` -- which has no ``.stage`` attribute --
exactly as it was.  New-style calls are therefore bit-for-bit unchanged, and ``melee_rl`` calls none of
these functions anywhere in the first place.  The two kind-2 sites are restorations rather than
                                 by the class ``gamestate.UnknownAnimation``) ->
                                 ``AttributeError``
parameter it is about to coerce is still *named* ``stage`` in the installed libmelee, so a future
release that reverts the signature is skipped rather than double-coerced.  Nothing in SmashBot's own
tree is edited. The separately supplied upstream source is GPL-3.0 at a pinned revision.
See ``THIRD_PARTY_NOTICES.md`` for provenance and the scope of the release review.

Call :func:`install` before the first ``ESAgent.act``; :class:`melee_rl.smashbot.SmashBotPlayer` does.
It is idempotent and process-wide.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, Final

LOG: Final[logging.Logger] = logging.getLogger(__name__)

COMPAT_SITES: Final[tuple[str, ...]] = (
    "Action.UNKNOWN_ANIMATION",
    "stages.top_platform_position",
    "stages.left_platform_position",
    "stages.right_platform_position",
    "stages.side_platform_position",
    "FrameData.roll_end_position",
    "FrameData.project_hit_location",
    "Projectile.x",
    "Projectile.y",
    "Projectile.x_speed",
    "Projectile.y_speed",
    "PlayerState.x",
    "PlayerState.y",
    "stages.fountain_of_dreams",
)
"""Every site this module touches.  ``tests/rl/test_smashbot_compat.py`` pins the set exactly."""

SMASHBOT_LIBMELEE: Final[str] = "0.41.1"
"""The libmelee SmashBot ``67c4120`` was written against (altf4/libmelee, 16 May 2024)."""

_MARKER: Final[str] = "_melee_rl_smashbot_compat"
"""Set on every wrapper this module installs, so :func:`install` can be called any number of times."""

_STAGE_PARAMETER: Final[str] = "stage"
"""The parameter the wrappers coerce.  Installation is skipped unless it still carries this name."""

_PLATFORM_HELPERS: Final[tuple[tuple[str, int], ...]] = (
    ("top_platform_position", 0),
    ("left_platform_position", 0),
    ("right_platform_position", 0),
    ("side_platform_position", 1),
)
"""``melee.stages`` helpers and the positional index of their stage argument."""

_FRAMEDATA_METHODS: Final[tuple[tuple[str, int], ...]] = (
    ("roll_end_position", 2),
    ("project_hit_location", 2),
)
"""``FrameData`` methods and the index of their stage argument (index 0 is ``self``)."""

_FIELD_ALIASES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("Projectile", "x", "position", "x"),
    ("Projectile", "y", "position", "y"),
    ("Projectile", "x_speed", "speed", "x"),
    ("Projectile", "y_speed", "speed", "y"),
    ("PlayerState", "x", "position", "x"),
    ("PlayerState", "y", "position", "y"),
)
"""``(class, removed field, container, attribute)``.

0.41.1 carried flat aliases beside the structured fields, and *its parser wrote them on every frame*
(``console.py:733-734``: ``playerstate.x = playerstate.position.x``).  0.47.3 dropped both the fields
and the writes.  ``Projectile.x`` is read by SmashBot (``Tactics/retreat.py:81,83,140,142``);
``PlayerState.y`` is read by **libmelee itself** -- ``framedata.py:325`` in ``FrameData.in_range`` --
which is the second place 0.47.3 kept a reference to something it had deleted.
"""

_FOD_EDGES: Final[dict[bool, tuple[float, float]]] = {
    False: (-49.5, -21.0),
    True: (49.5, 21.0),
}
"""0.41.1's Fountain of Dreams side-platform edges (``stages.py:102,126``), verbatim.

Note the right platform's ``(49.5, 21)``: ``left`` greater than ``right``, the opposite of every other
stage.  That is a quirk of the version SmashBot was written against, and it is reproduced rather than
corrected -- making our SmashBot better than the one everybody else runs would be the same error as
leaving it worse.
"""

_UNKNOWN_ANIMATION: Final[str] = "UNKNOWN_ANIMATION"
"""The enum member 0.41.1 had as ``0xffff`` and 0.47.3 replaced with a class -- but still names."""


def as_stage(argument: Any) -> Any:
    """A ``GameState``'s stage, or ``argument`` itself when it is already a ``Stage``.

    ``melee.enums.Stage`` is an ``Enum`` with no ``stage`` member, so this is the identity for a stage
    and the projection for a gamestate -- which is the whole of the shim's semantics.
    """

    return getattr(argument, _STAGE_PARAMETER, argument)


def _accepts_a_stage_at(function: Callable[..., Any], index: int) -> bool:
    """True when parameter ``index`` of ``function`` is still called ``stage`` in this libmelee."""

    try:
        parameters = list(inspect.signature(function).parameters)
    except (TypeError, ValueError):  # pragma: no cover - a C function would land here
        return False
    return len(parameters) > index and parameters[index] == _STAGE_PARAMETER


def _fod_side_platform(argument: Any, right: bool) -> tuple[Any, float, float] | None:
    """0.41.1's live Fountain of Dreams side platform, or ``None`` when it is not knowable.

    FoD's side platforms move.  0.41.1 read their height straight off the gamestate
    (``stages.py:102,126`` -> ``gamestate._fod_platform_left`` / ``_fod_platform_right``, parsed from
    the frame's event bytes in ``console.py:935-941``).  0.47.3 renamed the field to
    ``gamestate.fod_platforms`` -- it still parses it -- but left ``stages.py`` answering
    ``(None, None, None) #TODO``, and by changing the signature from a gamestate to a stage it made the
    live height unreachable even in principle.

    SmashBot passes a gamestate, so the 0.41.1 answer is recoverable for exactly the caller that needs
    it.  Without one there is nothing to recover and ``None`` is returned, which is why ``melee_rl``'s
    own ``Stage``-shaped calls are completely unaffected.
    """

    platforms = getattr(argument, "fod_platforms", None)
    if platforms is None:
        return None
    stage = getattr(argument, "stage", None)
    if getattr(stage, "name", None) != "FOUNTAIN_OF_DREAMS":
        return None
    height = getattr(platforms, "right" if right else "left", None)
    if height is None:
        return None
    left_edge, right_edge = _FOD_EDGES[right]
    return (height, left_edge, right_edge)


def _coerce_stage_at(function: Callable[..., Any], index: int) -> Callable[..., Any]:
    """``function`` with its stage argument passed through :func:`as_stage`, positionally or by name."""

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _STAGE_PARAMETER in kwargs:
            kwargs = {**kwargs, _STAGE_PARAMETER: as_stage(kwargs[_STAGE_PARAMETER])}
        elif len(args) > index:
            args = (*args[:index], as_stage(args[index]), *args[index + 1 :])
        return function(*args, **kwargs)

    setattr(wrapper, _MARKER, True)
    return wrapper


_FOD_SIDES: Final[dict[str, bool | None]] = {
    "left_platform_position": False,
    "right_platform_position": True,
    "side_platform_position": None,  # which side comes from the call's own first argument
}
"""The helpers that must consult :func:`_fod_side_platform` before coercing (``top_`` need not: 0.47.3
already answers FoD's *top* platform, which does not move)."""


def _side_platform_wrapper(function: Callable[..., Any], index: int, side: bool | None) -> Callable[..., Any]:
    """A stage-coercing wrapper that answers Fountain of Dreams from the gamestate when it can."""

    coerced = _coerce_stage_at(function, index)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        argument = (
            kwargs.get(_STAGE_PARAMETER)
            if _STAGE_PARAMETER in kwargs
            else (args[index] if len(args) > index else None)
        )
        right = side
        if right is None:  # side_platform_position(right_platform, stage)
            right = bool(kwargs["right_platform"]) if "right_platform" in kwargs else bool(args[0])
        restored = _fod_side_platform(argument, right)
        return restored if restored is not None else coerced(*args, **kwargs)

    setattr(wrapper, _MARKER, True)
    return wrapper


def _alias_property(container: str, attribute: str) -> property:
    """``self.<container>.<attribute>`` as a read-only property (0.41.1's flat field, live not snapshot)."""

    def getter(self: Any) -> Any:
        return getattr(getattr(self, container), attribute)

    getter.__name__ = f"{container}_{attribute}"
    getter.__doc__ = f"libmelee 0.41.1 compatibility: ``self.{container}.{attribute}``."
    setattr(getter, _MARKER, True)
    return property(getter)


def _import_melee() -> Any:
    from melee_rl.env.libmelee_frames import import_melee

    return import_melee()


def installed(melee_module: Any | None = None) -> bool:
    """True when :func:`install` has already run against this ``melee`` in this process."""

    melee = _import_melee() if melee_module is None else melee_module
    return bool(getattr(melee.stages.top_platform_position, _MARKER, False)) and hasattr(
        melee.Action, _UNKNOWN_ANIMATION
    )


def install(melee_module: Any | None = None) -> tuple[str, ...]:
    """Patch every site of :data:`COMPAT_SITES`; return the ones this call actually changed.

    Idempotent: a second call returns ``()``.  A site whose signature no longer matches what the shim
    expects (libmelee reverting a name, or restoring a removed field) is left alone and logged, so the
    shim can never double-coerce.
    """

    melee = _import_melee() if melee_module is None else melee_module
    changed: list[str] = []

    for name, index in _PLATFORM_HELPERS:
        function = getattr(melee.stages, name)
        if getattr(function, _MARKER, False):
            continue
        if not _accepts_a_stage_at(function, index):
            LOG.warning("smashbot compat: melee.stages.%s no longer takes a stage at %d", name, index)
            continue
        # ``in``, not ``.get() is None``: ``side_platform_position``'s entry *is* None, meaning "the
        # side comes from the call's own first argument".
        wrapper = (
            _side_platform_wrapper(function, index, _FOD_SIDES[name])
            if name in _FOD_SIDES
            else _coerce_stage_at(function, index)
        )
        setattr(melee.stages, name, wrapper)
        # ``melee/__init__.py`` does ``from melee.stages import *``, so the package holds its own
        # binding -- and SmashBot calls that one (``Tactics/juggle.py:85``).
        setattr(melee, name, wrapper)
        changed.append(f"stages.{name}")

    frame_data = melee.framedata.FrameData
    for name, index in _FRAMEDATA_METHODS:
        function = getattr(frame_data, name)
        if getattr(function, _MARKER, False):
            continue
        if not _accepts_a_stage_at(function, index):
            LOG.warning("smashbot compat: FrameData.%s no longer takes a stage at %d", name, index)
            continue
        setattr(frame_data, name, _coerce_stage_at(function, index))
        changed.append(f"FrameData.{name}")

    # libmelee 0.47.3 replaced ``Action.UNKNOWN_ANIMATION`` (``0xffff`` in 0.41.1) with the separate
    # class ``gamestate.UnknownAnimation`` -- which ``PlayerState.action`` *defaults* to -- but left
    # ``framedata.py:162`` still naming the enum member, so ``FrameData.is_bmove`` raises
    # ``AttributeError`` for any caller.  That is a bug in libmelee, not in SmashBot: nothing in
    # ``melee_rl`` calls ``FrameData``, so only SmashBot reaches it (it did, on the first real frame of
    # the first Modal smoke).  Binding the name to an ``UnknownAnimation`` instance restores exactly the
    # 0.41.1 semantics the branch was written for: true for an unknown animation, false for every real
    # ``Action``, since the dataclass compares equal to its own kind and to nothing else.
    action = melee.Action
    unknown = getattr(melee.gamestate, "UnknownAnimation", None)
    if not hasattr(action, _UNKNOWN_ANIMATION):
        if unknown is None:
            LOG.warning(
                "smashbot compat: libmelee has neither Action.%s nor UnknownAnimation", _UNKNOWN_ANIMATION
            )
        else:
            setattr(action, _UNKNOWN_ANIMATION, unknown())
            changed.append(f"Action.{_UNKNOWN_ANIMATION}")

    for class_name, alias, container, attribute in _FIELD_ALIASES:
        owner = getattr(melee.gamestate, class_name, None)
        if owner is None:
            LOG.warning("smashbot compat: libmelee has no gamestate.%s", class_name)
            continue
        existing = getattr(owner, alias, None)
        if existing is not None:
            if not getattr(getattr(existing, "fget", None), _MARKER, False):
                LOG.warning("smashbot compat: %s.%s exists already; leaving it alone", class_name, alias)
            continue
        setattr(owner, alias, _alias_property(container, attribute))
        changed.append(f"{class_name}.{alias}")

    if changed:
        LOG.info("smashbot compat: patched %s for SmashBot's libmelee %s", changed, SMASHBOT_LIBMELEE)
    return tuple(changed)


__all__ = ["COMPAT_SITES", "SMASHBOT_LIBMELEE", "as_stage", "install", "installed"]
