"""The libmelee compatibility shim SmashBot needs (P25, 9 Sep 2026).

SmashBot's HEAD (``67c4120``, 15 May 2024) is written against **altf4/libmelee 0.41.1**, published the
same day.  Our pin is ``melee==0.47.3``, which is **vladfi1/libmelee** -- PyPI's ``melee`` name moved to
that fork at 0.45.0.  Diffing the two sdists at signature level, the whole drift that touches SmashBot's
*decision* code is six functions and one field:

===========================================  =====================  ====================
site                                         0.41.1                 0.47.3
===========================================  =====================  ====================
``stages.top_platform_position``             ``(gamestate)``        ``(stage)``
``stages.left_platform_position``            ``(gamestate)``        ``(stage)``
``stages.right_platform_position``           ``(gamestate)``        ``(stage)``
``stages.side_platform_position``            ``(right, gamestate)`` ``(right, stage)``
``FrameData.roll_end_position``              ``(cs, gamestate)``    ``(cs, stage)``
``FrameData.project_hit_location``           ``(cs, gamestate, n)`` ``(cs, stage, n)``
``Projectile.x`` / ``.y``                    present                ``.position.x`` / ``.y``
===========================================  =====================  ====================

Two of those fail *silently*: a ``GameState`` where a ``Stage`` is expected makes the platform helpers
fall through to ``(None, None, None)`` -- "this stage has no platforms" -- rather than raise.  The other
two raise ``TypeError: unhashable type: 'GameState'`` from ``EDGE_GROUND_POSITION[stage]``, which escapes
``roll_end_position``'s own ``except KeyError``.  Upstream ``smashbot.py`` wraps every ``act`` in a blanket
``except Exception: controller.empty_input()``, so a SmashBot broken this way looks like a working but
passive one.  These tests pin that the shim removes all four failure modes without changing what the
*new* call style does -- which is the only style ``melee_rl`` itself uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from melee_rl import smashbot_compat

melee = pytest.importorskip("melee", reason="the compat shim patches the installed libmelee")

from melee.enums import Action, Character, Stage  # noqa: E402
from melee.gamestate import GameState, PlayerState, Projectile  # noqa: E402

STAGES = (
    Stage.FINAL_DESTINATION,
    Stage.BATTLEFIELD,
    Stage.POKEMON_STADIUM,
    Stage.DREAMLAND,
    Stage.FOUNTAIN_OF_DREAMS,
    Stage.YOSHIS_STORY,
)
"""The six legal stages -- the set our match configs cycle over and the set SmashBot supports."""


@pytest.fixture(scope="module", autouse=True)
def _installed() -> None:
    """The shim is process-wide and idempotent; every test in this module runs with it on."""

    smashbot_compat.install()


def gamestate_on(stage: Stage) -> GameState:
    state = GameState()
    state.stage = stage
    return state


def rolling_fox(x: float = 30.0) -> PlayerState:
    """A Fox mid forward-roll on the ground: enough state for ``roll_end_position`` to reach the stage."""

    player = PlayerState()
    player.character = Character.FOX
    player.action = Action.ROLL_FORWARD
    player.action_frame = 2
    player.facing = True
    player.position.x = x
    player.position.y = 0.0
    return player


def launched_fox() -> PlayerState:
    """A Fox in hitstun: enough state for ``project_hit_location`` to reach the stage."""

    player = PlayerState()
    player.character = Character.FOX
    player.action = Action.DAMAGE_HIGH_1
    player.speed_x_attack = 2.0
    player.speed_y_attack = 3.0
    player.speed_y_self = -1.0
    player.position.x = 0.0
    player.position.y = 20.0
    player.hitstun_frames_left = 20
    return player


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------


def test_install_is_idempotent_and_reports_only_the_first_pass() -> None:
    """The first call in a process patches; every later call is a no-op that changes nothing."""

    assert smashbot_compat.installed()
    assert smashbot_compat.install() == ()
    assert smashbot_compat.install() == ()
    assert smashbot_compat.installed()


def test_every_documented_site_is_patched() -> None:
    """:data:`COMPAT_SITES` is the contract; nothing may be patched that is not listed."""

    assert set(smashbot_compat.COMPAT_SITES) == {
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
    }


def test_the_package_reexport_is_patched_too() -> None:
    """``melee/__init__.py`` does ``from melee.stages import *``, so ``melee.f`` is a *separate* binding
    from ``melee.stages.f`` -- and SmashBot calls the package one (``Tactics/juggle.py:85``)."""

    for name in (
        "top_platform_position",
        "left_platform_position",
        "right_platform_position",
        "side_platform_position",
    ):
        assert getattr(melee, name) is getattr(melee.stages, name), name


# ---------------------------------------------------------------------------
# the four silent platform helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_top_platform_position_accepts_a_gamestate(stage: Stage) -> None:
    assert melee.stages.top_platform_position(gamestate_on(stage)) == (
        melee.stages.top_platform_position(stage)
    )


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("right", [True, False])
def test_side_platform_position_accepts_a_gamestate(stage: Stage, right: bool) -> None:
    assert melee.stages.side_platform_position(right, gamestate_on(stage)) == (
        melee.stages.side_platform_position(right, stage)
    )


@pytest.mark.parametrize("stage", STAGES)
def test_left_and_right_platform_position_accept_a_gamestate(stage: Stage) -> None:
    for helper in (melee.stages.left_platform_position, melee.stages.right_platform_position):
        assert helper(gamestate_on(stage)) == helper(stage)


def test_the_platform_helpers_used_to_answer_no_platforms_for_a_gamestate() -> None:
    """The regression this shim exists for: unpatched, Battlefield's top platform reads as absent.

    ``__wrapped__`` is the 0.47.3 original, so this pins the *old* behaviour we are fixing rather than
    merely asserting the new one.
    """

    original = melee.stages.top_platform_position.__wrapped__
    assert original(gamestate_on(Stage.BATTLEFIELD)) == (None, None, None)
    assert melee.stages.top_platform_position(gamestate_on(Stage.BATTLEFIELD))[0] is not None


# ---------------------------------------------------------------------------
# the two raising FrameData methods
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_roll_end_position_accepts_a_gamestate(stage: Stage) -> None:
    framedata = melee.framedata.FrameData()
    player = rolling_fox()
    assert framedata.roll_end_position(player, gamestate_on(stage)) == (
        framedata.roll_end_position(player, stage)
    )


@pytest.mark.parametrize("stage", STAGES)
def test_project_hit_location_accepts_a_gamestate(stage: Stage) -> None:
    framedata = melee.framedata.FrameData()
    player = launched_fox()
    assert framedata.project_hit_location(player, gamestate_on(stage)) == (
        framedata.project_hit_location(player, stage)
    )


def test_project_hit_location_keeps_its_frames_argument() -> None:
    """SmashBot passes ``frames`` positionally (``Tactics/juggle.py:121``) and by keyword."""

    framedata = melee.framedata.FrameData()
    player = launched_fox()
    expected = framedata.project_hit_location(player, Stage.BATTLEFIELD, 5)
    assert framedata.project_hit_location(player, gamestate_on(Stage.BATTLEFIELD), 5) == expected
    assert framedata.project_hit_location(player, gamestate_on(Stage.BATTLEFIELD), frames=5) == expected


def test_the_framedata_methods_used_to_raise_for_a_gamestate() -> None:
    """The other regression: a ``TypeError`` that escapes ``roll_end_position``'s ``except KeyError``."""

    framedata = melee.framedata.FrameData()
    with pytest.raises(TypeError):
        melee.framedata.FrameData.roll_end_position.__wrapped__(
            framedata, rolling_fox(), gamestate_on(Stage.BATTLEFIELD)
        )
    with pytest.raises(TypeError):
        melee.framedata.FrameData.project_hit_location.__wrapped__(
            framedata, launched_fox(), gamestate_on(Stage.BATTLEFIELD)
        )


# ---------------------------------------------------------------------------
# the removed Projectile fields
# ---------------------------------------------------------------------------


def test_projectile_position_and_speed_aliases() -> None:
    """``Tactics/retreat.py:81,83,140,142`` still read ``projectile.x``."""

    projectile = Projectile()
    projectile.position.x, projectile.position.y = 3.0, 4.0
    projectile.speed.x, projectile.speed.y = 5.0, 6.0
    assert (projectile.x, projectile.y) == (3.0, 4.0)
    assert (projectile.x_speed, projectile.y_speed) == (5.0, 6.0)


def test_projectile_aliases_follow_the_live_object() -> None:
    """They are properties, not a snapshot: libmelee rewrites ``position`` in place every frame."""

    projectile = Projectile()
    projectile.position.x = 1.0
    assert projectile.x == 1.0
    projectile.position.x = -2.5
    assert projectile.x == -2.5


# ---------------------------------------------------------------------------
# our own call style is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", STAGES)
def test_the_new_call_style_is_bit_for_bit_unchanged(stage: Stage) -> None:
    """``melee_rl`` never calls these six, but the shim is process-wide, so prove it changes nothing.

    Each wrapper is compared against its own ``__wrapped__`` original on the *new* argument style.
    """

    framedata = melee.framedata.FrameData()
    assert melee.stages.top_platform_position(stage) == (
        melee.stages.top_platform_position.__wrapped__(stage)
    )
    for right in (True, False):
        assert melee.stages.side_platform_position(right, stage) == (
            melee.stages.side_platform_position.__wrapped__(right, stage)
        )
    for helper in (melee.stages.left_platform_position, melee.stages.right_platform_position):
        assert helper(stage) == helper.__wrapped__(stage)
    assert framedata.roll_end_position(rolling_fox(), stage) == (
        melee.framedata.FrameData.roll_end_position.__wrapped__(framedata, rolling_fox(), stage)
    )
    assert framedata.project_hit_location(launched_fox(), stage) == (
        melee.framedata.FrameData.project_hit_location.__wrapped__(framedata, launched_fox(), stage)
    )


def test_a_stage_has_no_stage_attribute_so_coercion_is_a_no_op_for_it() -> None:
    """The whole coercion is ``getattr(argument, "stage", argument)``; this is why it is safe."""

    for stage in STAGES:
        assert smashbot_compat.as_stage(stage) is stage
        assert smashbot_compat.as_stage(gamestate_on(stage)) is stage


# ---------------------------------------------------------------------------
# the eighth site: a dangling reference inside libmelee itself
# ---------------------------------------------------------------------------


def test_action_unknown_animation_is_restored() -> None:
    """0.41.1 had ``Action.UNKNOWN_ANIMATION = 0xffff``; 0.47.3 replaced it with a separate
    ``gamestate.UnknownAnimation`` class -- but left ``framedata.py:162`` still naming the enum member.

    That is a bug in libmelee, not in SmashBot: any caller of ``FrameData.is_bmove`` raises
    ``AttributeError``.  ``melee_rl`` never calls ``FrameData`` so it never sees it; SmashBot calls
    ``is_bmove`` and hit it on the first real frame of the first Modal smoke.
    """

    from melee.gamestate import UnknownAnimation

    assert UnknownAnimation() == melee.Action.UNKNOWN_ANIMATION
    assert melee.Action.STANDING != melee.Action.UNKNOWN_ANIMATION


def test_is_bmove_works_for_a_real_action_and_for_an_unknown_one() -> None:
    """The regression the smoke found, exactly as SmashBot triggers it."""

    from melee.gamestate import UnknownAnimation

    framedata = melee.framedata.FrameData()
    assert framedata.is_bmove(Character.FOX, UnknownAnimation()) is False
    assert isinstance(framedata.is_bmove(Character.FOX, Action.LASER_GUN_PULL), bool)
    assert framedata.is_bmove(Character.FOX, Action.STANDING) is False


def test_a_default_player_state_can_be_asked_about() -> None:
    """``PlayerState.action`` *defaults* to ``UnknownAnimation()``, which is why this fires on frame one."""

    framedata = melee.framedata.FrameData()
    assert framedata.is_bmove(Character.FOX, PlayerState().action) is False


def test_libmelee_has_no_dangling_enum_references_left() -> None:
    """A self-consistency audit of the installed libmelee, so a future pin bump surfaces its own
    ``UNKNOWN_ANIMATION``-shaped bugs here rather than forty minutes into a Modal job."""

    import ast
    import os

    enums = [
        "Action",
        "Character",
        "Button",
        "ProjectileType",
        "Menu",
        "Stage",
        "AttackState",
        "ControllerStatus",
        "ControllerType",
        "Submenu",
    ]
    root = os.path.dirname(melee.__file__)
    dangling: set[tuple[str, int, str]] = set()
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse((Path(root) / name).read_text())
        except SyntaxError:  # pragma: no cover - libmelee ships only valid modules
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts: list[str] = []
            cursor: object = node
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            chain = list(reversed(parts))
            for index, part in enumerate(chain[:-1]):
                if part in enums:
                    enum = getattr(melee, part, None)
                    member = chain[index + 1]
                    if enum is not None and member not in ("value", "name") and not hasattr(enum, member):
                        dangling.add((name, node.lineno, f"{part}.{member}"))
                    break
    assert dangling == set(), f"libmelee names enum members that do not exist: {sorted(dangling)}"


def test_player_state_position_aliases() -> None:
    """0.41.1's parser wrote ``playerstate.x = playerstate.position.x`` on every frame
    (``console.py:733-734``); 0.47.3 dropped the field *and* the write, but ``framedata.py:325`` still
    reads it."""

    player = PlayerState()
    player.position.x, player.position.y = -12.5, 33.0
    assert (player.x, player.y) == (-12.5, 33.0)


def test_in_range_works() -> None:
    """The second libmelee bug the Modal smoke found: ``FrameData.in_range`` reads the removed ``.y``
    and raises ``AttributeError`` for every caller.  SmashBot calls it from ``Tactics/defend.py:106``."""

    framedata = melee.framedata.FrameData()
    attacker = rolling_fox()
    attacker.action = Action.DASH_ATTACK
    defender = rolling_fox(x=40.0)
    assert isinstance(framedata.in_range(attacker, defender, Stage.FINAL_DESTINATION), int)


def test_libmelee_does_not_read_fields_it_has_removed() -> None:
    """The sibling of the enum audit, for the flat aliases 0.41.1's parser used to write.

    Both libmelee bugs this shim works around have this shape: 0.47.3 deleted a field but kept a
    reference to it in ``framedata.py``, so the method raises for every caller.  ``melee_rl`` never
    calls ``FrameData``, so only SmashBot reaches them -- and both cost a Modal run to find.  This
    test makes the next one cost nothing.
    """

    import ast
    import os

    from melee.gamestate import Position

    removed = {"x", "y", "cursor_x", "cursor_y", "x_speed", "y_speed"}
    structured = {"position", "speed", "cursor", "ecb", "top", "bottom", "left", "right"}
    root = os.path.dirname(melee.__file__)
    dangling: list[str] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".py"):
            continue
        text = (Path(root) / name).read_text()
        lines = text.split("\n")
        for node in ast.walk(ast.parse(text)):
            if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)):
                continue
            if node.attr not in removed:
                continue
            receiver = node.value
            if isinstance(receiver, ast.Attribute) and receiver.attr in structured:
                continue
            if isinstance(receiver, ast.Name) and receiver.id in ("self", "np", "math"):
                continue
            dangling.append(f"{name}:{node.lineno}  {lines[node.lineno - 1].strip()[:90]}")
    # framedata.py:325 is the known one; the shim's PlayerState aliases make it work, so it is allowed
    # to remain -- what must not happen is a *new* one appearing on a libmelee bump.
    assert len(dangling) <= 1, f"libmelee reads fields it has removed: {dangling}"
    assert Position is not None


# ---------------------------------------------------------------------------
# the whole FrameData surface SmashBot calls
# ---------------------------------------------------------------------------

SMASHBOT_FRAMEDATA_METHODS: tuple[str, ...] = (
    "attack_state",
    "dj_height",
    "first_hitbox_frame",
    "frame_count",
    "frames_until_dj_apex",
    "hitbox_count",
    "iasa",
    "in_range",
    "is_attack",
    "is_bmove",
    "is_grab",
    "is_roll",
    "last_roll_frame",
    "project_hit_location",
    "range_backward",
    "range_forward",
    "roll_end_position",
    "slide_distance",
)
"""Every ``FrameData`` method SmashBot calls, collected from its tree (18 of them)."""


def test_every_framedata_method_smashbot_calls_actually_runs() -> None:
    """The test that would have saved two Modal runs.

    Both libmelee bugs this shim works around live in ``FrameData`` methods that libmelee's own type
    surface still advertises: ``is_bmove`` names a deleted enum member and ``in_range`` reads a deleted
    field.  Nothing in ``melee_rl`` calls ``FrameData`` at all, so neither shows up anywhere else --
    each one cost a Dolphin boot on Modal to find.  Calling the whole surface here costs a second.
    """

    framedata = melee.framedata.FrameData()
    attacker = rolling_fox()
    attacker.action = Action.DASH_ATTACK
    defender = rolling_fox(x=40.0)
    stage = Stage.BATTLEFIELD
    calls = {
        "attack_state": (Character.FOX, Action.DASH_ATTACK, 3),
        "dj_height": (attacker,),
        "first_hitbox_frame": (Character.FOX, Action.DASH_ATTACK),
        "frame_count": (Character.FOX, Action.DASH_ATTACK),
        "frames_until_dj_apex": (attacker,),
        "hitbox_count": (Character.FOX, Action.DASH_ATTACK),
        "iasa": (Character.FOX, Action.DASH_ATTACK),
        "in_range": (attacker, defender, stage),
        "is_attack": (Character.FOX, Action.DASH_ATTACK),
        "is_bmove": (Character.FOX, Action.DASH_ATTACK),
        "is_grab": (Character.FOX, Action.GRAB),
        "is_roll": (Character.FOX, Action.ROLL_FORWARD),
        "last_roll_frame": (Character.FOX, Action.ROLL_FORWARD),
        "project_hit_location": (launched_fox(), stage),
        "range_backward": (Character.FOX, Action.DASH_ATTACK, 3),
        "range_forward": (Character.FOX, Action.DASH_ATTACK, 3),
        "roll_end_position": (rolling_fox(), stage),
        "slide_distance": (attacker, 1.0, 10),
    }
    assert set(calls) == set(SMASHBOT_FRAMEDATA_METHODS)
    broken: list[str] = []
    for name, args in calls.items():
        try:
            getattr(framedata, name)(*args)
        except Exception as error:
            broken.append(f"{name}: {type(error).__name__}: {error}")
    assert broken == [], "libmelee methods SmashBot depends on are broken: " + "; ".join(broken)


def test_only_is_bmove_guards_the_unknown_animation() -> None:
    """``PlayerState.action`` defaults to ``UnknownAnimation()``, and exactly one method copes with it.

    ``is_bmove`` has an explicit ``if action == Action.UNKNOWN_ANIMATION: return False`` -- which is the
    branch the missing enum member broke.  The others index ``framedata[character][action]`` with no
    guard, so an unknown animation raises in *both* libmelee versions: ``KeyError`` in 0.41.1, where the
    sentinel was a hashable enum member, and ``TypeError`` in 0.47.3, where it is an unfrozen dataclass
    and therefore unhashable.  That difference is recorded here rather than papered over: it has not
    been observed to matter in a real game, and inventing a fix for it without evidence would be
    changing libmelee's behaviour on a guess.
    """

    framedata = melee.framedata.FrameData()
    blank = PlayerState()
    blank.character = Character.FOX
    assert framedata.is_bmove(Character.FOX, blank.action) is False
    with pytest.raises(TypeError):
        framedata.is_attack(Character.FOX, blank.action)


# ---------------------------------------------------------------------------
# Fountain of Dreams: the platforms 0.47.3 marks #TODO
# ---------------------------------------------------------------------------


def fod_gamestate(left: float = 22.0, right: float = 26.0) -> GameState:
    """A Fountain of Dreams gamestate with its side platforms at a given height."""

    from melee.gamestate import FoDPlatforms

    state = gamestate_on(Stage.FOUNTAIN_OF_DREAMS)
    state.fod_platforms = FoDPlatforms(left=left, right=right)
    return state


def test_fountain_of_dreams_side_platforms_are_restored_from_the_gamestate() -> None:
    """0.41.1 read FoD's *moving* platforms live off the gamestate
    (``stages.py:102,126`` -> ``gamestate._fod_platform_left/right``, parsed in ``console.py:935-941``).
    0.47.3 renamed the field to ``fod_platforms`` and left ``stages.py`` returning
    ``(None, None, None) #TODO`` -- and, by changing the signature from a gamestate to a stage, made the
    live height unreachable at all.  SmashBot passes a gamestate, so 0.41.1's answer is recoverable.
    """

    state = fod_gamestate(left=22.0, right=26.0)
    assert melee.stages.left_platform_position(state) == (22.0, -49.5, -21)
    assert melee.stages.right_platform_position(state) == (26.0, 49.5, 21)
    assert melee.stages.side_platform_position(False, state) == (22.0, -49.5, -21)
    assert melee.stages.side_platform_position(True, state) == (26.0, 49.5, 21)


def test_the_inverted_right_edges_are_reproduced_verbatim() -> None:
    """0.41.1 returns ``(height, 49.5, 21)`` for FoD's right platform -- ``left`` greater than
    ``right``, unlike every other stage.  It is a quirk of the version SmashBot was written against, so
    it is reproduced rather than corrected: making our SmashBot better than the real one would be the
    same error as leaving it worse.
    """

    height, left_edge, right_edge = melee.stages.right_platform_position(fod_gamestate())
    assert left_edge > right_edge
    for stage in (Stage.BATTLEFIELD, Stage.DREAMLAND, Stage.YOSHIS_STORY, Stage.POKEMON_STADIUM):
        _, other_left, other_right = melee.stages.right_platform_position(stage)
        assert other_left < other_right, stage
    assert height is not None


def test_a_bare_stage_still_cannot_know_the_moving_platforms() -> None:
    """Without a gamestate the live height is genuinely unknowable; ``melee_rl``'s own call style
    (a ``Stage``) must therefore be completely unchanged."""

    for helper in (melee.stages.left_platform_position, melee.stages.right_platform_position):
        assert helper(Stage.FOUNTAIN_OF_DREAMS) == (None, None, None)
    assert melee.stages.side_platform_position(True, Stage.FOUNTAIN_OF_DREAMS) == (None, None, None)


def test_a_gamestate_without_platform_data_is_not_invented() -> None:
    state = gamestate_on(Stage.FOUNTAIN_OF_DREAMS)
    state.fod_platforms = None
    assert melee.stages.left_platform_position(state) == (None, None, None)


@pytest.mark.parametrize("stage", [s for s in STAGES if s is not Stage.FOUNTAIN_OF_DREAMS])
def test_other_stages_are_untouched_by_the_fod_path(stage: Stage) -> None:
    state = gamestate_on(stage)
    for helper in (melee.stages.left_platform_position, melee.stages.right_platform_position):
        assert helper(state) == helper(stage)


def test_smashbots_board_side_platform_bounds_exist_on_fountain_of_dreams() -> None:
    """The concrete consequence: ``Chains/boardsideplatform.py:33`` compares against
    ``platform_left``/``platform_right`` without the ``is not None`` guard it applied two lines
    earlier, so an unrestored FoD raises there -- 68 times in one 18-match six-stage arm."""

    height, left_edge, right_edge = melee.stages.side_platform_position(False, fod_gamestate())
    assert None not in (height, left_edge, right_edge)
