"""``DummyMeleeEnv``: a deterministic toy Melee behind ``EnvProtocol`` (PLAN.md §6 P1).

Two players on a flat stage with Melee-like bookkeeping so that every field the encoder
reads varies and the reward semantics (``melee_rl.reward``) have something to measure:

* the main stick walks (``x += WALK_SPEED * (2 sx - 1)``, half speed in the air), X/Y jump
  (edge-triggered, ``JUMPS`` per airtime), A attacks (``ATTACK_FRAMES`` frames; a hit within
  ``ATTACK_RANGE_X/Y`` adds ``ATTACK_DAMAGE`` percent, ``HITSTUN_FRAMES`` of hitstun and a
  percent-scaled knockback), L/R (or the shoulder) shield on the ground (blocks hits while
  ``shield_strength > 0``, decays and regenerates);
* ``percent > percent_threshold`` starts a death: ``DYING_FRAMES`` frames in a dying state
  (``action <= 0xA``, the reward's death edge), then ``RESPAWN_FRAMES`` frames on the halo
  (``0x0C``) at the spawn point with ``percent = 0`` and respawn invulnerability; a stock
  is removed when the death starts; when the last stock is lost the next output is the
  first frame of a new game (``frame_index == -123``, ``needs_reset``); a game also ends
  after ``game_frames`` frames;
* the scripted ``cpu`` player walks towards its opponent, attacks every ``cpu_attack_every``
  frames when in range and jumps every ``cpu_jump_every`` frames (phases drawn per game);
* ``learnable_signal`` adds ``signal_reward`` to a controlled port's reward whenever the
  *executed* buttons label equals ``target_buttons`` (the P3 learning-signal test);
* every controlled port sees the game from its own perspective (``p0`` = itself).

Everything is vectorised over the ``B`` environments with element-wise tensor ops, so the
dynamics are bitwise deterministic; randomness (spawn jitter, cpu phases) comes from one
``torch.Generator`` per environment, seeded ``seed * 1000 + env``, and is drawn only when a
game starts -- environment ``b`` therefore evolves identically for any ``num_envs``.
All values stay inside the ranges ``SlippiEncoder`` accepts (``melee_rl.frames``).
The ``controller`` field of each player echoes the controller executed on the transition
into the frame (on the first frame of a game it was sent but not applied).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import torch

from controller_codec import ButtonState, ControllerState, CustomV1Codec, StickState
from melee_rl.env.protocol import INITIAL_FRAME_INDEX, EnvOutput, swap_perspective
from melee_rl.frames import FrameTree, neutral_labels
from model import SlippiEncoder
from tensor_batch import BUTTON_ORDER, MAX_ITEMS

# libmelee ``melee.enums.Action`` ids used by the toy.
ACTION_DEAD_UP: Final[int] = 0x03
ACTION_ON_HALO_DESCENT: Final[int] = 0x0C
ACTION_STANDING: Final[int] = 0x0E
ACTION_WALK_MIDDLE: Final[int] = 0x10
ACTION_JUMPING_FORWARD: Final[int] = 0x19
ACTION_FALLING: Final[int] = 0x1D
ACTION_LANDING: Final[int] = 0x2A
ACTION_NEUTRAL_ATTACK_1: Final[int] = 0x2C
ACTION_DAMAGE_HIGH_1: Final[int] = 0x4B
ACTION_SHIELD: Final[int] = 0xB3

# Dynamics constants (toy units: Melee-ish coordinates).
WALK_SPEED: Final[float] = 1.2
AIR_DRIFT: Final[float] = 0.6
STICK_DEADZONE: Final[float] = 0.2
JUMP_VELOCITY: Final[float] = 3.4
GRAVITY: Final[float] = 0.32
JUMPS: Final[int] = 2
ATTACK_RANGE_X: Final[float] = 10.0
ATTACK_RANGE_Y: Final[float] = 6.0
ATTACK_DAMAGE: Final[int] = 12
ATTACK_FRAMES: Final[int] = 4
HITSTUN_FRAMES: Final[int] = 5
KNOCKBACK: Final[float] = 6.0
DYING_FRAMES: Final[int] = 5
RESPAWN_FRAMES: Final[int] = 3
RESPAWN_INVULNERABLE_FRAMES: Final[int] = 30
STAGE_EDGE: Final[float] = 68.0
SPAWN_X: Final[float] = 30.0
SPAWN_JITTER: Final[float] = 5.0
SHIELD_MAX: Final[float] = 60.0
SHIELD_DECAY: Final[float] = 0.5
SHIELD_REGEN: Final[float] = 0.1
SHOULDER_SHIELD_THRESHOLD: Final[float] = 0.3
CPU_APPROACH_DISTANCE: Final[float] = 4.0

PLAYER_KINDS: Final[tuple[str, ...]] = ("policy", "cpu")


@dataclass(frozen=True)
class DummyEnvConfig:
    """Configuration of :class:`DummyMeleeEnv` (``[env]`` table of the dummy-env TOML configs, P3)."""

    num_envs: int = 2
    seed: int = 0
    players: tuple[str, str] = ("policy", "cpu")
    stocks: int = 4
    game_frames: int = 28_800
    percent_threshold: int = 100
    cpu_attack_every: int = 7
    cpu_jump_every: int = 41
    learnable_signal: bool = False
    target_buttons: int = 3
    signal_reward: float = 1.0
    stage: int = 32
    characters: tuple[int, int] = (2, 20)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if len(self.players) != 2 or any(kind not in PLAYER_KINDS for kind in self.players):
            raise ValueError(f"players must be two of {PLAYER_KINDS}, got {self.players!r}")
        if self.stocks < 1:
            raise ValueError("stocks must be >= 1")
        if self.game_frames < 1:
            raise ValueError("game_frames must be >= 1")
        if self.percent_threshold < 0:
            raise ValueError("percent_threshold must be >= 0")
        if self.cpu_attack_every < 0 or self.cpu_jump_every < 0:
            raise ValueError("cpu_attack_every and cpu_jump_every must be >= 0 (0 disables)")
        if not 0 <= self.target_buttons < 728:
            raise ValueError("target_buttons must be a buttons label in [0, 728)")
        if not 0 <= self.stage < SlippiEncoder.STAGE_SIZE:
            raise ValueError(f"stage must be in [0, {SlippiEncoder.STAGE_SIZE})")
        if len(self.characters) != 2 or any(
            not 0 <= character < SlippiEncoder.CHARACTER_SIZE for character in self.characters
        ):
            raise ValueError(f"characters must be two ids in [0, {SlippiEncoder.CHARACTER_SIZE})")

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return tuple(port for port, kind in enumerate(self.players) if kind == "policy")


def _stack_sticks(controls: list[ControllerState]) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([control.main_stick.x.to(torch.float32) for control in controls])
    y = torch.stack([control.main_stick.y.to(torch.float32) for control in controls])
    return x, y


def _stack_button(controls: list[ControllerState], name: str) -> torch.Tensor:
    return torch.stack([getattr(control.buttons, name).to(torch.bool) for control in controls])


class DummyMeleeEnv:
    """Deterministic toy Melee for ``B`` environments; see the module docstring."""

    def __init__(self, config: DummyEnvConfig) -> None:
        self.config = config
        self._codec = CustomV1Codec()
        batch = config.num_envs
        self._generators = [
            torch.Generator().manual_seed(config.seed * 1000 + env) for env in range(batch)
        ]
        zeros_f = torch.zeros((2, batch), dtype=torch.float32)
        zeros_l = torch.zeros((2, batch), dtype=torch.long)
        falses = torch.zeros((2, batch), dtype=torch.bool)
        self._x, self._y, self._vy, self._shield = zeros_f, zeros_f, zeros_f, zeros_f.clone()
        self._percent, self._action, self._jumps_left, self._stocks = zeros_l, zeros_l, zeros_l, zeros_l
        self._dying, self._respawn, self._hitstun, self._attack, self._invulnerable = (zeros_l,) * 5
        self._facing, self._on_ground, self._jump_held = falses, falses, falses
        self._cpu_attack_phase = torch.zeros(batch, dtype=torch.long)
        self._cpu_jump_phase = torch.zeros(batch, dtype=torch.long)
        self._frame_index = torch.zeros(batch, dtype=torch.long)
        self._game_over = torch.zeros(batch, dtype=torch.bool)
        self._neutral = self._codec.decode(neutral_labels((batch,)))
        self._controls: list[ControllerState] = [self._neutral, self._neutral]
        self._constant_leaves = self._make_constant_leaves(batch)
        self._pending = self.reset()

    # -- protocol facts ---------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self.config.controlled_ports

    def current(self) -> EnvOutput:
        return self._pending

    def close(self) -> None:
        return None

    # -- games --------------------------------------------------------------------

    def reset(self) -> EnvOutput:
        """Start a new game in every environment; the pending state is its first frame."""

        batch = self.num_envs
        self._start_games(torch.ones(batch, dtype=torch.bool))
        self._frame_index = torch.full((batch,), INITIAL_FRAME_INDEX, dtype=torch.long)
        self._game_over = torch.zeros(batch, dtype=torch.bool)
        self._controls = [self._neutral, self._neutral]
        zero = torch.zeros(batch, dtype=torch.float32)
        self._pending = self._output({port: zero for port in self.controlled_ports})
        return self._pending

    def _start_games(self, starting: torch.Tensor) -> None:
        """Fresh game state for the environments where ``starting`` is true (draws per-env randomness)."""

        batch = self.num_envs
        offsets = torch.zeros((2, batch), dtype=torch.float32)
        attack_phase = torch.zeros(batch, dtype=torch.long)
        jump_phase = torch.zeros(batch, dtype=torch.long)
        attack_every = max(self.config.cpu_attack_every, 1)
        jump_every = max(self.config.cpu_jump_every, 1)
        for env in starting.nonzero().flatten().tolist():
            generator = self._generators[env]
            jitter = (torch.rand(2, generator=generator) * 2.0 - 1.0) * SPAWN_JITTER
            offsets[:, env] = jitter
            attack_phase[env] = int(torch.randint(0, attack_every, (1,), generator=generator))
            jump_phase[env] = int(torch.randint(0, jump_every, (1,), generator=generator))
        mask = starting.unsqueeze(0).expand(2, batch)
        sides = torch.tensor([-1.0, 1.0], dtype=torch.float32).unsqueeze(1)
        spawn_x = sides * SPAWN_X + offsets
        facing = torch.tensor([True, False]).unsqueeze(1).expand(2, batch)

        def put(old: torch.Tensor, new: torch.Tensor | float | int | bool) -> torch.Tensor:
            return torch.where(mask, torch.as_tensor(new, dtype=old.dtype).expand_as(old), old)

        self._x = put(self._x, spawn_x)
        self._y = put(self._y, 0.0)
        self._vy = put(self._vy, 0.0)
        self._shield = put(self._shield, SHIELD_MAX)
        self._percent = put(self._percent, 0)
        self._action = put(self._action, ACTION_STANDING)
        self._jumps_left = put(self._jumps_left, JUMPS)
        self._stocks = put(self._stocks, self.config.stocks)
        self._dying = put(self._dying, 0)
        self._respawn = put(self._respawn, 0)
        self._hitstun = put(self._hitstun, 0)
        self._attack = put(self._attack, 0)
        self._invulnerable = put(self._invulnerable, 0)
        self._facing = put(self._facing, facing)
        self._on_ground = put(self._on_ground, True)
        self._jump_held = put(self._jump_held, False)
        self._cpu_attack_phase = torch.where(starting, attack_phase, self._cpu_attack_phase)
        self._cpu_jump_phase = torch.where(starting, jump_phase, self._cpu_jump_phase)

    # -- stepping -----------------------------------------------------------------

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        batch = self.num_envs
        if tuple(sorted(actions)) != self.controlled_ports:
            raise ValueError(
                f"step needs controllers for ports {self.controlled_ports}, got {tuple(actions)}"
            )
        controls: list[ControllerState] = []
        for port, kind in enumerate(self.config.players):
            if kind == "policy":
                control = actions[port]
                if tuple(control.shoulder.shape) != (batch,):
                    raise ValueError(f"port {port} controller must have batch shape ({batch},)")
                controls.append(control)
            else:
                controls.append(self._script(port))

        starting = self._game_over
        if bool(starting.any()):
            self._start_games(starting)
        live = ~starting
        self._physics(controls, live)
        self._frame_index = torch.where(starting, INITIAL_FRAME_INDEX, self._frame_index + 1)
        timed_out = self._frame_index == self.config.game_frames - 124
        self._game_over = live & (self._game_over | timed_out)
        self._controls = controls

        rewards: dict[int, torch.Tensor] = {}
        for port in self.controlled_ports:
            reward = torch.zeros(batch, dtype=torch.float32)
            if self.config.learnable_signal:
                labels = self._codec.encode(controls[port])
                hit = labels.buttons == self.config.target_buttons
                reward = torch.where(live & hit, self.config.signal_reward, reward)
            rewards[port] = reward
        self._pending = self._output(rewards)
        return self._pending

    def _script(self, port: int) -> ControllerState:
        """The scripted cpu controller for ``port`` from the pending state."""

        other = 1 - port
        batch = self.num_envs
        dx = self._x[other] - self._x[port]
        dy = self._y[other] - self._y[port]
        counter = self._frame_index - INITIAL_FRAME_INDEX
        stick_x = torch.where(
            dx > CPU_APPROACH_DISTANCE,
            1.0,
            torch.where(dx < -CPU_APPROACH_DISTANCE, 0.0, torch.full((batch,), 0.5)),
        )
        half = torch.full((batch,), 0.5, dtype=torch.float32)
        falses = torch.zeros(batch, dtype=torch.bool)
        attack = falses
        if self.config.cpu_attack_every > 0:
            due = (counter + self._cpu_attack_phase) % self.config.cpu_attack_every == 0
            near = (dx.abs() <= ATTACK_RANGE_X + 2.0) & (dy.abs() <= ATTACK_RANGE_Y)
            attack = due & near
        jump = falses
        if self.config.cpu_jump_every > 0:
            jump = (counter + self._cpu_jump_phase) % self.config.cpu_jump_every == 0
        return ControllerState(
            main_stick=StickState(x=stick_x.to(torch.float32), y=half),
            c_stick=StickState(x=half, y=half),
            shoulder=torch.zeros(batch, dtype=torch.float32),
            buttons=ButtonState(
                A=attack, B=falses, X=falses, Y=jump, Z=falses, L=falses, R=falses, D_UP=falses
            ),
        )

    def _physics(self, controls: list[ControllerState], live: torch.Tensor) -> None:
        """One frame of dynamics for both players; envs with ``live`` false keep their fresh state."""

        live2 = live.unsqueeze(0).expand(2, self.num_envs)
        stick_x, _ = _stack_sticks(controls)
        press_a = _stack_button(controls, "A")
        press_jump = _stack_button(controls, "X") | _stack_button(controls, "Y")
        shoulder = torch.stack([control.shoulder.to(torch.float32) for control in controls])
        press_shield = _stack_button(controls, "L") | _stack_button(controls, "R")
        press_shield = press_shield | (shoulder > SHOULDER_SHIELD_THRESHOLD)

        x, y, vy, shield = self._x, self._y, self._vy, self._shield
        percent, jumps_left, stocks = self._percent, self._jumps_left, self._stocks
        dying, respawn, hitstun, attack, invulnerable = (
            self._dying,
            self._respawn,
            self._hitstun,
            self._attack,
            self._invulnerable,
        )
        facing, on_ground = self._facing, self._on_ground

        # Respawn placement on the first halo frame.
        placing = respawn == RESPAWN_FRAMES
        sides = torch.tensor([-1.0, 1.0], dtype=torch.float32).unsqueeze(1)
        spawn_x = (sides * SPAWN_X).expand_as(x)
        x = torch.where(placing, spawn_x, x)
        y = torch.where(placing, 0.0, y)
        vy = torch.where(placing, 0.0, vy)
        percent = torch.where(placing, 0, percent)
        on_ground = torch.where(placing, True, on_ground)
        jumps_left = torch.where(placing, JUMPS, jumps_left)
        shield = torch.where(placing, SHIELD_MAX, shield)
        hitstun = torch.where(placing, 0, hitstun)
        attack = torch.where(placing, 0, attack)
        invulnerable = torch.where(placing, RESPAWN_FRAMES + RESPAWN_INVULNERABLE_FRAMES, invulnerable)

        # Deaths begin when the percent crossed the threshold on the previous frame.
        dead_phase = (dying > 0) | (respawn > 0)
        start_dying = (percent > self.config.percent_threshold) & ~dead_phase
        dying = torch.where(start_dying, DYING_FRAMES, dying)
        stocks = torch.where(start_dying, stocks - 1, stocks)
        hitstun = torch.where(start_dying, 0, hitstun)
        attack = torch.where(start_dying, 0, attack)
        dead_phase = dead_phase | start_dying

        stunned = hitstun > 0
        attacking = attack > 0
        can_act = ~dead_phase & ~stunned & ~attacking

        # Shield.
        shielding = can_act & press_shield & on_ground
        shield = torch.where(
            shielding,
            torch.clamp(shield - SHIELD_DECAY, min=0.0),
            torch.clamp(shield + SHIELD_REGEN, max=SHIELD_MAX),
        )
        shield_up = shielding & (shield > 0.0)

        # Attacks: hit detection on the positions of the previous frame.
        start_attack = can_act & ~shielding & press_a
        attack = torch.where(start_attack, ATTACK_FRAMES, attack)
        dx = x.flip(0) - x
        dy = y.flip(0) - y
        in_range = (dx.abs() <= ATTACK_RANGE_X) & (dy.abs() <= ATTACK_RANGE_Y)
        target_open = ~(dead_phase | (invulnerable > 0) | shield_up).flip(0)
        lands_hit = start_attack & in_range & target_open
        got_hit = lands_hit.flip(0)
        percent = torch.where(got_hit, percent + ATTACK_DAMAGE, percent)
        hitstun = torch.where(got_hit, HITSTUN_FRAMES, hitstun)
        attack = torch.where(got_hit, 0, attack)
        push = torch.where(dx >= 0.0, -1.0, 1.0) * KNOCKBACK * (1.0 + percent.to(torch.float32) / 100.0)
        x = torch.where(got_hit, x + push, x)
        shielding = shielding & ~got_hit

        # Jumps (edge-triggered) and horizontal movement.
        jump_edge = press_jump & ~self._jump_held
        do_jump = can_act & ~shielding & ~got_hit & jump_edge & (jumps_left > 0)
        vy = torch.where(do_jump, JUMP_VELOCITY, vy)
        on_ground = torch.where(do_jump, False, on_ground)
        jumps_left = torch.where(do_jump, jumps_left - 1, jumps_left)
        drive = 2.0 * stick_x - 1.0
        moving = can_act & ~shielding & ~got_hit & (drive.abs() > STICK_DEADZONE)
        speed = torch.where(on_ground, WALK_SPEED, AIR_DRIFT)
        x = torch.where(moving, x + speed * drive, x)
        facing = torch.where(moving, drive > 0.0, facing)

        # Gravity and landing.
        airborne = ~on_ground & ~dead_phase
        vy = torch.where(airborne, vy - GRAVITY, vy)
        y = torch.where(airborne, y + vy, y)
        landed = airborne & (y <= 0.0)
        y = torch.where(landed, 0.0, y)
        vy = torch.where(landed, 0.0, vy)
        on_ground = on_ground | landed
        jumps_left = torch.where(landed, JUMPS, jumps_left)
        x = torch.clamp(x, -STAGE_EDGE, STAGE_EDGE)
        rising = airborne & ~landed & (vy > 0.0)
        falling = airborne & ~landed & (vy <= 0.0)

        # Action state by priority.
        action = torch.full_like(self._action, ACTION_STANDING)
        action = torch.where(moving & on_ground & ~landed, ACTION_WALK_MIDDLE, action)
        action = torch.where(landed, ACTION_LANDING, action)
        action = torch.where(falling, ACTION_FALLING, action)
        action = torch.where(rising | do_jump, ACTION_JUMPING_FORWARD, action)
        action = torch.where(shielding, ACTION_SHIELD, action)
        action = torch.where(start_attack | attacking, ACTION_NEUTRAL_ATTACK_1, action)
        action = torch.where(got_hit | stunned, ACTION_DAMAGE_HIGH_1, action)
        action = torch.where(respawn > 0, ACTION_ON_HALO_DESCENT, action)
        action = torch.where(dying > 0, ACTION_DEAD_UP, action)

        # End-of-frame timers and transitions.
        dying_ends = dying == 1
        player_out = dying_ends & (stocks <= 0)
        game_over = player_out.any(dim=0)
        hitstun = torch.clamp(hitstun - 1, min=0)
        attack = torch.clamp(attack - 1, min=0)
        invulnerable = torch.clamp(invulnerable - 1, min=0)
        respawn = torch.clamp(respawn - 1, min=0)
        dying = torch.clamp(dying - 1, min=0)
        respawn = torch.where(dying_ends & ~player_out, RESPAWN_FRAMES, respawn)

        def commit(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            return torch.where(live2, new, old)

        self._x, self._y, self._vy = commit(self._x, x), commit(self._y, y), commit(self._vy, vy)
        self._shield, self._percent = commit(self._shield, shield), commit(self._percent, percent)
        self._jumps_left, self._stocks = commit(self._jumps_left, jumps_left), commit(self._stocks, stocks)
        self._dying, self._respawn = commit(self._dying, dying), commit(self._respawn, respawn)
        self._hitstun, self._attack = commit(self._hitstun, hitstun), commit(self._attack, attack)
        self._invulnerable = commit(self._invulnerable, invulnerable)
        self._facing, self._on_ground = commit(self._facing, facing), commit(self._on_ground, on_ground)
        self._jump_held = commit(self._jump_held, press_jump)
        self._action = commit(self._action, action)
        self._game_over = live & game_over

    # -- outputs ------------------------------------------------------------------

    @staticmethod
    def _make_constant_leaves(batch: int) -> dict[str, torch.Tensor]:
        return {
            "zero_f": torch.zeros(batch, dtype=torch.float32),
            "zero_l": torch.zeros(batch, dtype=torch.long),
            "false": torch.zeros(batch, dtype=torch.bool),
            "item_false": torch.zeros((batch, MAX_ITEMS), dtype=torch.bool),
            "item_zero_l": torch.zeros((batch, MAX_ITEMS), dtype=torch.long),
            "item_zero_f": torch.zeros((batch, MAX_ITEMS), dtype=torch.float32),
        }

    def _controller_tree(self, control: ControllerState) -> FrameTree:
        def stick(value: StickState) -> FrameTree:
            return {"x": value.x.to(torch.float32), "y": value.y.to(torch.float32)}

        return {
            "main_stick": stick(control.main_stick),
            "c_stick": stick(control.c_stick),
            "shoulder": control.shoulder.to(torch.float32),
            "buttons": {name: getattr(control.buttons, name).to(torch.bool) for name in BUTTON_ORDER},
        }

    def _player_tree(self, player: int) -> FrameTree:
        const = self._constant_leaves
        batch = self.num_envs
        return {
            "percent": self._percent[player],
            "facing": self._facing[player],
            "x": self._x[player],
            "y": self._y[player],
            "action": self._action[player],
            "invulnerable": self._invulnerable[player] > 0,
            "character": torch.full((batch,), self.config.characters[player], dtype=torch.long),
            "jumps_left": self._jumps_left[player],
            "shield_strength": self._shield[player],
            "on_ground": self._on_ground[player],
            "controller": self._controller_tree(self._controls[player]),
            "nana": {
                "exists": const["false"],
                "percent": const["zero_l"],
                "facing": const["false"],
                "x": const["zero_f"],
                "y": const["zero_f"],
                "action": const["zero_l"],
                "invulnerable": const["false"],
                "character": const["zero_l"],
                "jumps_left": const["zero_l"],
                "shield_strength": const["zero_f"],
                "on_ground": const["false"],
            },
        }

    def _output(self, rewards: Mapping[int, torch.Tensor]) -> EnvOutput:
        const = self._constant_leaves
        batch = self.num_envs
        tree: FrameTree = {
            "p0": self._player_tree(0),
            "p1": self._player_tree(1),
            "stage": torch.full((batch,), self.config.stage, dtype=torch.long),
            "randall": {"x": const["zero_f"], "y": const["zero_f"]},
            "fod_platforms": {"left": const["zero_f"], "right": const["zero_f"]},
            "items": {
                "exists": const["item_false"],
                "type": const["item_zero_l"],
                "state": const["item_zero_l"],
                "x": const["item_zero_f"],
                "y": const["item_zero_f"],
            },
        }
        frames = {port: (tree if port == 0 else swap_perspective(tree)) for port in self.controlled_ports}
        return EnvOutput(
            frames=frames,
            needs_reset=self._frame_index == INITIAL_FRAME_INDEX,
            frame_index=self._frame_index,
            rewards=dict(rewards),
        )


__all__ = [
    "ACTION_DAMAGE_HIGH_1",
    "ACTION_DEAD_UP",
    "ACTION_FALLING",
    "ACTION_JUMPING_FORWARD",
    "ACTION_LANDING",
    "ACTION_NEUTRAL_ATTACK_1",
    "ACTION_ON_HALO_DESCENT",
    "ACTION_SHIELD",
    "ACTION_STANDING",
    "ACTION_WALK_MIDDLE",
    "ATTACK_DAMAGE",
    "ATTACK_FRAMES",
    "ATTACK_RANGE_X",
    "ATTACK_RANGE_Y",
    "DYING_FRAMES",
    "HITSTUN_FRAMES",
    "JUMPS",
    "RESPAWN_FRAMES",
    "SPAWN_X",
    "DummyEnvConfig",
    "DummyMeleeEnv",
]
