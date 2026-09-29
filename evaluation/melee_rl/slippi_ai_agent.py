"""vladfi1/slippi-ai's released agents as live opponents (P27, 10 Sep 2026).

`slippi-ai <https://github.com/vladfi1/slippi-ai>`_ (MIT, pinned at :data:`SLIPPI_AI_REVISION` -- the
same SHA as ``third_party/VERSIONS.txt`` and the E010 canary) is Phillip II: behaviour cloning on
Fizzi's anonymised ranked replays, then PPO self-play on top.  Its author's agents have played Zain,
Amsa, Cody, Moky and Aklo on stream, which makes them the strongest public learned Melee policies and
the only external opponent that has beaten top humans.  ``melee_rl`` already measures itself against
Dolphin's CPU 9, MIMIC and SmashBot; this module adds the rung above them.

The boundary is the one :mod:`melee_rl.mimic` and :mod:`melee_rl.smashbot` already cross.  Upstream's
``eval_lib.Agent`` takes a raw libmelee ``GameState``, runs its own parser, observation filter,
recurrent core, delay queue and categorical sampler, and presses a libmelee controller -- so this
module feeds it ``DolphinEnv.gamestates()`` and reads the controller back through
:class:`~melee_rl.mimic.ControllerRecorder`, and :class:`~melee_rl.opponents.SlippiAIOpponent` returns
a ``ControllerState`` through ``act_state`` rather than labels through ``act``.  Upstream's own
``Agent`` class is *not* used: it hard-codes ``batch_size=1`` and owns the console.  One **batched**
``DelayedAgent`` serves every environment instead, which is how their own evaluator scales.

Five things are worth knowing before reading a number this module produces.

* **Every Dropbox release carries 18-24 frames of policy delay.**  ``policy.delay = d`` means the network is
  trained to emit the controller for frame ``t + d`` from the state at frame ``t``; it is what lets the
  bot play netplay against humans at up to 300 ms of ping.  Our own policy acts on the next frame
  (``action_offset_frames = 1``), so a match between them is *asymmetric in our favour* and the report
  has to say so.  :data:`RELEASED_MODELS` records every delay we found: probing the whole deployed set
  on 9 Sep 2026 for delays 0-24 returned **18, 21 and 24 only** -- the folder holds no low-delay
  release to make the comparison symmetric with.  **The one delay-0 file** (``fox_d0_tx_like_3x512``,
  shared by the author on 22 Sep 2026) is a JAX / flax *imitation* checkpoint -- ``platform: 'jax'``,
  ``counters`` and no ``rl_config`` -- so it runs through upstream's JAX half
  (``slippi_ai.jax.saving``, its own pip layer in the image), gets ``jax.device_get`` on its outputs
  before decoding as upstream's own ``Agent.step`` does (``eval_lib.py:539-541``), stays off the GPU
  through ``JAX_PLATFORMS=cpu``, and is the first symmetric fight our next-frame policy has had.
  ``console_delay`` subtracts from the queue to pay for a
  console's own lag; our environment runs ``online_delay = 0`` with ``blocking_input``, so an input
  sent for frame N lands on frame N and the faithful setting is ``0`` (E010 used ``2`` because it
  replicated upstream's ``eval_two.py`` recipe, which assumes Slippi's default).
* **The analog remap cannot bite them.**  ``melee==0.47.3`` rounds every ``tilt_analog`` value onto
  Melee's 160-unit grid, which is what broke SmashBot's wavedash (P26, ``legacy_analog_ports``).
  slippi-ai's action space *is* that grid (``controller_lib.from_raw_axis`` is ``x / 160 + 0.5``), so
  the remap has nothing to round.  ``tests/rl/test_slippi_ai_agent.py`` measures this against the real
  libmelee for all 161 values rather than trusting the arithmetic.
* **The button set matches ours exactly.**  Upstream's ``LEGAL_BUTTONS`` is A, B, X, Y, Z, L, R, D_UP,
  and ``tensor_batch.BUTTON_ORDER`` is the same eight in the same order, so nothing is dropped at the
  boundary.  Their ``send_controller`` writes *every* field every frame, so a fresh recorder per frame
  is correct here -- unlike SmashBot, which needs a persistent one.
* **Their delay queue is never cleared between games.**  ``DelayedAgent`` primes the queue with dummy
  outputs once, at construction, and ``needs_reset`` only resets the recurrent state
  (``eval_lib.py:128-131``, ``:170-176``).  The first ``d`` frames of a new game therefore apply the
  previous game's in-flight actions -- upstream's own behaviour, absorbed by Melee's 123-frame
  countdown.  That is why an environment sitting in a menu still advances the batch here, repeating its
  last observation: skipping it would desynchronise the queue and the LSTM for that row.
* **A name is part of the policy.**  The imitation models are conditioned on a player tag and the RL
  ones bake in the tags they trained with, so ``fox_d21_ditto_v4`` plays as "Cody" unless told
  otherwise and ``gm`` plays as the anonymised "Master Player".  :attr:`ReleasedModel.default_name` is
  what upstream would pick; the run file can override it, and ``stats()`` records what was used.

Pinned upstream: revision :data:`SLIPPI_AI_REVISION`, licence MIT. Users supply the source tree and
checkpoint files separately, with the sha256 identities in :data:`RELEASED_MODELS`. Access to a
checkpoint follows its provider's availability and permission terms. See ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from melee_rl.env.dolphin import PORTS, ControllerRow, neutral_row
from melee_rl.mimic import ControllerRecorder

LOG: Final[logging.Logger] = logging.getLogger(__name__)

SLIPPI_AI_REPO_URL: Final[str] = "https://github.com/vladfi1/slippi-ai.git"
SLIPPI_AI_REVISION: Final[str] = "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
"""The repository's own pin (``third_party/VERSIONS.txt``), and the revision E010 gated ``medium-v2`` at."""
SLIPPI_AI_LICENSE: Final[str] = "MIT"
SLIPPI_AI_D0_REVISION: Final[str] = "9eca7479a9555f4ee4b7e6800c656a7f5bb55902"
"""The main snapshot the shared delay-0 Fox was trained on (a March-2026 commit; chosen 22 Sep 2026).

The file carries an ``_embed_module._item_linear`` parameter and ``counters`` without ``train_epoch``:
the former was removed upstream on 23 Mar 2026 (``3885040``, and not as a rename -- the items embed
went from ``sum(linear(items))`` to ``sum(where(exists, mlp(items), 0))``, so the file's ``_item_mlp``
weights were dead parameters in its own code), the latter added on 16 Mar (``576c8ab``).  Between
them upstream's policy modules only gained typing and the RL learner's distributions, so any commit
in the bracket reproduces the training code; this one is the last before ``d654f62`` ("Implement
WebDatasets") made ``webdataset`` -- a git-branch dependency the image does not carry -- a top-level
import of ``slippi_ai.data`` (measured: the image's import check).  The pin cannot restore the file
(measured: "key in pure_dict not available in state: ('network', '_embed_module', '_item_linear',
'bias')"), so the image carries this second checkout beside :data:`SLIPPI_AI_REVISION`'s."""
SLIPPI_AI_ROOT: Final[str] = "/opt/slippi-ai"
DEFAULT_SOURCE_DIR: Final[str] = f"{SLIPPI_AI_ROOT}/source"
"""Default location for the separately supplied :data:`SLIPPI_AI_REVISION` source tree."""


def checkout_dir(revision: str, root: str = SLIPPI_AI_ROOT) -> str:
    """The image's checkout of one upstream revision: the pin at ``source``, any other at
    ``source-<12 hex>`` (``modal_app.slippi_ai_image_commands`` clones every revision the registry
    names).  ``modal_app.SLIPPI_AI_D0_SOURCE_DIR`` duplicates the rule; the tests keep them equal."""

    if revision == SLIPPI_AI_REVISION:
        return f"{root}/source"
    return f"{root}/source-{revision[:12]}"


DEFAULT_MODEL_DIR: Final[str] = "/runs/external/slippi-ai"
"""Where the released checkpoints live on the ``melee-rl-runs`` Volume."""
ENTRY_MODULE: Final[str] = "slippi_ai/eval_lib.py"
"""The file whose absence means the checkout is not a slippi-ai tree."""
INITIAL_FRAME: Final[int] = -123
"""Melee's first frame.  Upstream keys "new game" on it; so do we, alongside our own reset flag."""
IN_GAME_MENU_NAMES: Final[frozenset[str]] = frozenset({"IN_GAME", "SUDDEN_DEATH"})
"""The ``melee.Menu`` members an agent may act in, by name (see :meth:`SlippiAIPlayer._actionable`)."""

CHARACTER_IDS: Final[Mapping[str, int]] = {
    "fox": 1,
    "falco": 22,
    "marth": 18,
    "sheik": 7,
    "jigglypuff": 15,
    "cptfalcon": 2,
    "peach": 9,
    "yoshi": 14,
    "popo": 10,
    "luigi": 17,
    "pikachu": 12,
    "samus": 13,
    "dk": 3,
    "doc": 21,
}
"""libmelee ``Character`` ids for the names the releases use, so importing this needs no libmelee.

Pinned against the installed fork by
``tests/rl/test_slippi_ai_agent.py::test_the_character_table_matches_the_installed_libmelee``.
"""

DEFAULT_IMITATION_NAME: Final[str] = "Master Player"
"""``nametags.DEFAULT_NAME``: the anonymised top-tier tag in Fizzi's ranked dumps.

Every release we measured has it in its ``name_map``, so it is a safe default for the one imitation
model, which would otherwise raise ``Must specify an agent name``.
"""


# ---------------------------------------------------------------------------
# the released models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleasedModel:
    """One file from the author's public ``deployed_models`` Dropbox folder.

    Every field was read out of the checkpoint itself on 9-10 Sep 2026 (``config['policy']['delay']``,
    ``config['network']``, ``config['dataset']['allowed_characters']``, ``rl_config`` /
    ``agent_config``), not from the file name -- the names are a convention, the config is the truth.
    """

    name: str
    byte_length: int
    sha256: str
    delay: int
    characters: tuple[str, ...]
    kind: str
    """``"rl"`` or ``"imitation"``."""
    hidden_size: int
    num_layers: int
    names: tuple[str, ...] = ()
    """The player tags the RL run conditioned on, in the checkpoint's own order."""
    rl_steps: int | None = None
    teacher: str | None = None
    kl_teacher_weight: float | None = None
    opponent: str | None = None
    """For a ``train_two`` release (``<char>_d<N>_vs_<opponent>``), the character it was trained against;
    for an imitation release, ``dataset.allowed_opponents`` when it names one character."""
    note: str = ""
    platform: str = "tf"
    """``config['platform']``: ``"tf"`` (a Sonnet file, the whole Dropbox folder) or ``"jax"`` (flax nnx,
    the shared delay-0 Fox).  Upstream reads a missing key as TensorFlow (``saving.get_platform``)."""
    imitation_steps: int | None = None
    """``counters['step']`` of an imitation checkpoint that carries the trainer's counters."""
    imitation_frames: int | None = None
    """``counters['total_frames']``: batch x unroll x steps of human play the imitation run consumed."""
    revision: str = SLIPPI_AI_REVISION
    """The upstream revision whose code restores and runs the file (its checkout: :func:`checkout_dir`)."""

    @property
    def default_name(self) -> str:
        """The tag the release plays as by default.

        An RL release: the first of the tags it trained with (``build_delayed_agent`` overrides any
        other name with ``rl_name[0]``).  An imitation release: :data:`DEFAULT_IMITATION_NAME`, which
        :class:`SlippiAIPlayer` passes because upstream refuses to build one without a name.
        """

        return self.names[0] if self.names else DEFAULT_IMITATION_NAME

    @property
    def character_ids(self) -> tuple[int, ...]:
        return tuple(CHARACTER_IDS[name] for name in self.characters)


_TWELVE: Final[tuple[str, ...]] = (
    "fox",
    "falco",
    "marth",
    "sheik",
    "jigglypuff",
    "cptfalcon",
    "peach",
    "yoshi",
    "popo",
    "luigi",
    "pikachu",
    "samus",
)
"""``allowed_characters`` of the four 768x3 releases, in the checkpoint's own order."""

_TOP12_TEACHER: Final[str] = "pickled_models/top12_d21_imitation_3x768_v5"

RELEASED_MODELS: Final[Mapping[str, ReleasedModel]] = {
    model.name: model
    for model in (
        # The twelve-character family: one teacher, one architecture, one delay, four amounts of RL on
        # a loosening KL leash.  The file names are the Slippi ranked tiers they are meant to evoke;
        # what actually separates them is `rl_steps` and `kl_teacher_weight`, so they make a strength
        # ladder that can place an outside agent *between* rungs.
        ReleasedModel(
            name="medium-v2",
            byte_length=95_559_707,
            sha256="48dcfd87c52fde9899fb37b0293ce2a597fb2384b7b40baf7fb49c760152a96c",
            delay=21,
            characters=_TWELVE,
            kind="rl",
            hidden_size=768,
            num_layers=3,
            names=("Master Player",) * 12,
            rl_steps=6829,
            teacher=_TOP12_TEACHER,
            kl_teacher_weight=0.05,
            note="the README's twelve-character model; E010 gated this exact sha256",
        ),
        ReleasedModel(
            name="diamond",
            byte_length=95_559_704,
            sha256="7650af84ab02a55f46f6278ec70030f2637c64e3b1538f35691c748455792312",
            delay=21,
            characters=_TWELVE,
            kind="rl",
            hidden_size=768,
            num_layers=3,
            names=("Master Player",) * 12,
            rl_steps=14374,
            teacher=_TOP12_TEACHER,
            kl_teacher_weight=0.02,
        ),
        ReleasedModel(
            name="master",
            byte_length=95_559_704,
            sha256="0c326f2ab772a926e9b13adac9ec78fe166a8d7b9350f75c5b8ed9e2170cc249",
            delay=21,
            characters=_TWELVE,
            kind="rl",
            hidden_size=768,
            num_layers=3,
            names=("Master Player",) * 12,
            rl_steps=26765,
            teacher=_TOP12_TEACHER,
            kl_teacher_weight=0.01,
        ),
        ReleasedModel(
            name="gm",
            byte_length=95_559_733,
            sha256="f03b1ae18a0cce1ed7b7a5b6db6f8a1e49a35344d72e39463780adbba43b8e27",
            delay=21,
            characters=_TWELVE,
            kind="rl",
            hidden_size=768,
            num_layers=3,
            names=("Master Player",) * 12,
            rl_steps=32344,
            teacher=_TOP12_TEACHER,
            kl_teacher_weight=0.005,
            note="the most RL and the loosest leash of the twelve-character family",
        ),
        # The Fox specialists: half the width, one character, and the ditto agents are the strongest
        # Fox the author has released.  Note the steps rise with the delay, so a d18/d21/d24 comparison
        # is not a clean delay sweep.
        ReleasedModel(
            name="fox_d18_imitation_v3",
            byte_length=42_078_960,
            sha256="d0ec155079b55795b97dcec5e17bfa695bcbac9d5f9d533e11d8b3cdde260684",
            delay=18,
            characters=("fox",),
            kind="imitation",
            hidden_size=512,
            num_layers=3,
            note="behaviour cloning only: the like-for-like comparison with our BC checkpoints",
        ),
        ReleasedModel(
            name="fox_d18_ditto_v4",
            byte_length=42_332_052,
            sha256="fdb96e96fed9f4320c28649351bd7318e9e7f2d1ab8a376af19595e9efd76f32",
            delay=18,
            characters=("fox",),
            kind="rl",
            hidden_size=512,
            num_layers=3,
            names=("Cody", "Hax", "Aklo", "SFAT"),
            rl_steps=19551,
            teacher="pickled_models/fox_d18_imitation_v3.1_masked",
            kl_teacher_weight=0.003,
        ),
        ReleasedModel(
            name="fox_d21_ditto_v4",
            byte_length=42_332_034,
            sha256="f4854fc783246b42d4bfc9309d6964c174a388f3425bb21f3f79355a4562e727",
            delay=21,
            characters=("fox",),
            kind="rl",
            hidden_size=512,
            num_layers=3,
            names=("Cody", "Aklo", "Hax", "SFAT"),
            rl_steps=26078,
            teacher="pickled_models/fox_d21_imitation_v4",
            kl_teacher_weight=0.003,
        ),
        ReleasedModel(
            name="fox_d24_ditto_v4",
            byte_length=42_332_052,
            sha256="38ff0851e920f30d15e9974b621aa730e4a13f0e117132596afa3ecd3161c4bb",
            delay=24,
            characters=("fox",),
            kind="rl",
            hidden_size=512,
            num_layers=3,
            names=("Cody", "Hax", "Aklo", "SFAT"),
            rl_steps=37693,
            teacher="pickled_models/fox_d24_imitation_v4",
            kl_teacher_weight=0.003,
        ),
        # A train_two release: a Falco trained specifically against Fox, i.e. the hardest opponent the
        # deployed set holds for a Fox.
        ReleasedModel(
            name="falco_d18_vs_fox_v4",
            byte_length=42_330_517,
            sha256="87983edf03d1e7133ef863821845217a29a489cc3fe0b3820c3e7982e87244f6",
            delay=18,
            characters=("falco",),
            kind="rl",
            hidden_size=512,
            num_layers=3,
            names=("Ginger", "KJH", "Frenzy", "BBB"),
            rl_steps=7530,
            opponent="fox",
            note="trained against Fox with rl/train_two.py; its config lives in agent_config",
        ),
        # Three more imitation releases, added 10 Sep 2026 for the BC-versus-BC campaign.  From v4 on a
        # release masks the opponent's tech for its first frames (``observation.animation.mask`` ->
        # ``observations.AnimationFilter``); a v3 file has no observation config and upstream upgrades it
        # to the null one, so it runs unmasked exactly as it trained.
        ReleasedModel(
            name="falco_d18_imitation_v3",
            byte_length=42_330_281,
            sha256="9d48e885d5591a630002116eca038bb1c6574988d5897f748c8bac614457524e",
            delay=18,
            characters=("falco",),
            kind="imitation",
            hidden_size=512,
            num_layers=3,
            note="behaviour cloning only; unmasked (v3)",
        ),
        ReleasedModel(
            name="sheik_d18_imitation_v4",
            byte_length=42_330_241,
            sha256="4e032eecc54ae94b0918d147764c987b68b7604a993c4805146122aa480c1311",
            delay=18,
            characters=("sheik",),
            kind="imitation",
            hidden_size=512,
            num_layers=3,
            note="behaviour cloning only; tech-masked observation (v4)",
        ),
        ReleasedModel(
            name="ics_d21_imitation_v5",
            byte_length=45_534_828,
            sha256="845930e86babe16b297dd80ff58799056dd5e06438d34aa8de7d0043c012329d",
            delay=21,
            characters=("popo",),
            kind="imitation",
            hidden_size=512,
            num_layers=3,
            note="behaviour cloning only; tech-masked (v5); 145 player tags, hence the larger file",
        ),
        # The one delay-0 file, shared by the author with the team on 22 Sep 2026 (it is not in the
        # Dropbox folder).  Read with the stub unpickler: ``platform: 'jax'``, ``counters`` /
        # ``dataset_metrics`` and no ``rl_config`` -- the JAX imitation trainer's save
        # (``slippi_ai/jax/train_lib.py:414-435``) -- so it is behaviour cloning only; ``tx_like``
        # 3 x 512 (LSTM, FFW x2, GELU), the ``enhanced`` embed, the autoregressive head (residual 128,
        # depth 2), tech mask on, 32 name ids over 52 tags; Fox-ditto data (``allowed_characters`` and
        # ``allowed_opponents`` both ``fox``, swap + mirror), 532 176 steps of 1024 x 80 frames
        # (43.6 B frames, 44.5 h), best eval loss 0.688.  The first symmetric fight for our next-frame
        # Fox.
        ReleasedModel(
            name="fox_d0_tx_like_3x512",
            byte_length=50_676_427,
            sha256="2e085fca35ae2405595577d15b30fefcc13289ffab7c863951d75fbb6ad703b7",
            delay=0,
            characters=("fox",),
            kind="imitation",
            hidden_size=512,
            num_layers=3,
            opponent="fox",
            note=(
                "shared by the author on 22 Sep 2026, not in the Dropbox folder; behaviour cloning only "
                "(no rl_config), JAX / flax, Fox-ditto data, delay 0: the first symmetric fight"
            ),
            platform="jax",
            imitation_steps=532_176,
            imitation_frames=43_595_857_920,
            revision=SLIPPI_AI_D0_REVISION,
        ),
    )
}
"""The twelve Dropbox releases we downloaded and read, by file name, plus the shared delay-0 Fox.

The full folder holds more (a rich Falco matchup set, ``dk``/``doc``/``medium-v1``, ...); these are the
ones P27 measures.  Everything here was verified byte-for-byte: ``medium-v2``'s sha256 is the one E010
pinned, which is what makes the download route trustworthy.
"""


def released_model(name: str) -> ReleasedModel:
    """:data:`RELEASED_MODELS` lookup that names the file it could not find."""

    try:
        return RELEASED_MODELS[name]
    except KeyError:
        raise KeyError(f"{name!r} is not a known slippi-ai release; have {sorted(RELEASED_MODELS)}") from None


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlippiAIConfig:
    """Which release plays, on which port, with which tag, and how its inputs reach the game.

    ``player`` is the *environment* player index (0 = Dolphin port 1, 1 = port 2), as everywhere else
    in ``melee_rl``; :attr:`libmelee_port` is what upstream and ``gamestate.players`` are keyed by.

    ``name`` empty means "whatever upstream would pick" (:attr:`ReleasedModel.default_name`).
    ``character`` 0 means "the only character this release plays", which is an error for the
    twelve-character family -- the campaign plays the Fox ditto and must say so.
    """

    model: str = "fox_d21_ditto_v4"
    model_dir: str = DEFAULT_MODEL_DIR
    model_path_override: str = ""
    source_dir: str = DEFAULT_SOURCE_DIR
    player: int = 1
    character: int = 0
    name: str = ""
    console_delay: int = 0
    sample_temperature: float = 1.0
    use_gpu: bool = False
    jit_compile: bool = False
    async_inference: bool = False
    verify_sha256: bool = True
    strict: bool = True
    torch_graph: bool = False
    """17 Sep 2026: the PyTorch port (:mod:`melee_rl.slippi_ai_torch`, the league's ``slippi:`` mix arms and
    eval opponents) replays its step from a CUDA graph; ``model_dir``, ``verify_sha256`` and
    ``sample_temperature`` apply to the port as well, everything else here is the TensorFlow recorder
    agent's."""

    def __post_init__(self) -> None:
        if self.model not in RELEASED_MODELS:
            raise ValueError(
                f"[slippi_ai] model must be one of {sorted(RELEASED_MODELS)}, got {self.model!r}"
            )
        if self.player not in (0, 1):
            raise ValueError(f"[slippi_ai] player must be 0 or 1, got {self.player}")
        if self.console_delay < 0:
            raise ValueError(f"[slippi_ai] console_delay must be >= 0, got {self.console_delay}")
        if self.console_delay > self.released.delay:
            raise ValueError(
                f"[slippi_ai] console_delay {self.console_delay} exceeds {self.model}'s policy delay "
                f"{self.released.delay}; upstream requires console_delay <= policy delay"
            )
        if self.sample_temperature <= 0.0:
            raise ValueError(f"[slippi_ai] sample_temperature must be > 0, got {self.sample_temperature}")
        if self.character and self.character not in self.released.character_ids:
            raise ValueError(
                f"[slippi_ai] {self.model} cannot play libmelee character {self.character}; "
                f"it plays {self.released.characters}"
            )
        if not self.model_dir and not self.model_path_override:
            raise ValueError("[slippi_ai] model_dir or model_path_override must name the checkpoint")

    @property
    def released(self) -> ReleasedModel:
        return released_model(self.model)

    @property
    def model_path(self) -> str:
        return self.model_path_override or f"{self.model_dir.rstrip('/')}/{self.model}"

    @property
    def libmelee_port(self) -> int:
        """The agent's *libmelee* port (1 or 2), which is what ``gamestate.players`` is keyed by."""

        return PORTS[self.player]

    @property
    def libmelee_opponent_port(self) -> int:
        return PORTS[1 - self.player]

    @property
    def effective_delay(self) -> int:
        """``DelayedAgent.delay``: how many frames its action is held back in our environment."""

        return self.released.delay - self.console_delay

    @property
    def resolved_character(self) -> int:
        """The libmelee character id this release must be started as."""

        if self.character:
            return self.character
        ids = self.released.character_ids
        if len(ids) != 1:
            raise ValueError(
                f"[slippi_ai] {self.model} plays {len(ids)} characters {self.released.characters}; "
                "set [slippi_ai] character to the libmelee id it should play"
            )
        return ids[0]

    @property
    def resolved_name(self) -> str:
        return self.name or self.released.default_name

    @property
    def resolved_source_dir(self) -> str:
        """The checkout this release runs on: ``source_dir`` when the run file set it (an operator's
        override), else the registry revision's own checkout (:func:`checkout_dir`)."""

        if self.source_dir != DEFAULT_SOURCE_DIR:
            return self.source_dir
        return checkout_dir(self.released.revision)


# ---------------------------------------------------------------------------
# the upstream boundary
# ---------------------------------------------------------------------------


def _jax_device_get(value: Any) -> Any:
    import jax  # type: ignore[import-not-found]  # the slippi-ai image's JAX layer, not a dependency here

    return jax.device_get(value)


@dataclass(frozen=True)
class SlippiAIApi:
    """The upstream symbols we use; faked wholesale in ``tests/rl/test_slippi_ai_agent.py``."""

    load_state: Callable[[str], dict[str, Any]]
    """``slippi_ai.saving.load_state_from_disk``."""
    build_agent: Callable[..., Any]
    """``slippi_ai.eval_lib.build_delayed_agent``."""
    parser_factory: Callable[..., Any]
    """``slippi_db.parse_libmelee.Parser(ports=(mine, theirs))`` -- one per environment."""
    observation_filter_factory: Callable[[Any], Any]
    """``slippi_ai.observations.build_observation_filter`` -- also one per environment."""
    send_controller: Callable[[Any, Any], None]
    """``slippi_ai.controller_lib.send_controller``."""
    disable_gpus: Callable[[], None]
    """``slippi_ai.eval_lib.disable_gpus``."""
    map_nt: Callable[..., Any]
    """``slippi_ai.utils.map_nt``: maps over their nested-NamedTuple ``Game``."""
    device_get: Callable[[Any], Any] = _jax_device_get
    """``jax.device_get``: a JAX agent's step returns device arrays (``eval_lib.py:159``); upstream's
    ``Agent.step`` copies them to the host before decoding (``:539-541``) and so does the player.  Only
    ever called for a ``platform == "jax"`` release, so the TensorFlow-only image never imports jax."""


def _loaded_slippi_ai_roots() -> set[Path]:
    """The checkout(s) an already-imported ``slippi_ai`` came from: ``__path__`` for the namespace
    package upstream ships (``__file__`` is ``None`` there), ``__file__`` for a regular one."""

    loaded = sys.modules.get("slippi_ai")
    if loaded is None:
        return set()
    file = getattr(loaded, "__file__", None)
    if file:
        return {Path(file).resolve().parent.parent}
    return {Path(entry).resolve().parent for entry in getattr(loaded, "__path__", [])}


def load_slippi_ai_api(source_dir: str | Path) -> SlippiAIApi:
    """Import the upstream runtime from a slippi-ai checkout, putting it on ``sys.path`` first.

    Importing is all this does; :class:`SlippiAIPlayer` calls ``disable_gpus`` itself, before the first
    device query, so an injected fake API is exercised on the same path as the real one.
    """

    root = Path(source_dir).expanduser().resolve()
    if not (root / ENTRY_MODULE).is_file():
        raise FileNotFoundError(f"slippi-ai checkout has no {ENTRY_MODULE}: {root}")
    loaded_from = _loaded_slippi_ai_roots()
    if loaded_from and loaded_from != {root}:
        # Two checkouts live in the image (the pin and the March snapshot); a process imports one.
        # Upstream's ``slippi_ai`` is a namespace package, so a second root on ``sys.path`` would not
        # even fail loudly -- it would merge, and every submodule would resolve to whichever came first.
        raise RuntimeError(
            f"slippi_ai is already imported from {sorted(map(str, loaded_from))}; a second checkout "
            f"({root}) cannot be loaded into this process -- one release per process"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from slippi_ai import eval_lib, observations, saving, utils  # type: ignore[import-not-found]
    from slippi_ai.controller_lib import send_controller  # type: ignore[import-not-found]
    from slippi_db.parse_libmelee import Parser  # type: ignore[import-not-found]

    return SlippiAIApi(
        load_state=saving.load_state_from_disk,
        build_agent=eval_lib.build_delayed_agent,
        parser_factory=Parser,
        observation_filter_factory=observations.build_observation_filter,
        send_controller=send_controller,
        disable_gpus=eval_lib.disable_gpus,
        map_nt=utils.map_nt,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# the player
# ---------------------------------------------------------------------------


class SlippiAIPlayer:
    """One released agent on one port across ``num_envs`` environments, batched into one step.

    The batch is the reason this class exists rather than a list of upstream ``Agent``s: a 768x3 LSTM
    stepped once per environment per frame is 24 Python-and-TensorFlow round trips per frame at 24
    Dolphins, while one ``step`` on a ``[24]`` batch is one.  Everything per-environment -- the parser,
    the observation filter, the controller -- stays per-environment, exactly as upstream keeps it.
    """

    def __init__(
        self,
        config: SlippiAIConfig,
        num_envs: int,
        *,
        api: SlippiAIApi | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.config = config
        self.num_envs = num_envs
        self.model = config.released
        self._host_copy = self.model.platform == "jax"
        if self._host_copy and not config.use_gpu:
            # JAX reads this at its first backend query, which is the agent build below; an operator's
            # explicit choice is kept.  The recorder's GPU belongs to our policy, as for TensorFlow.
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        self.api = load_slippi_ai_api(config.resolved_source_dir) if api is None else api
        if not config.use_gpu:
            # Before the first device query: TensorFlow must not take the GPU our policy is on.
            self.api.disable_gpus()

        path = Path(config.model_path)
        self.sha256_verified = False
        if config.verify_sha256 and path.is_file():
            digest = _sha256(path)
            if digest != self.model.sha256:
                raise RuntimeError(
                    f"{config.model} at {path} has sha256 {digest}, expected {self.model.sha256}"
                )
            self.sha256_verified = True

        state = self.api.load_state(config.model_path)
        found = self._delay_of(state)
        if found is not None and found != self.model.delay:
            raise RuntimeError(
                f"{config.model} has policy delay {found} on disk but the registry says "
                f"{self.model.delay}; the file is not the release P27 measured"
            )
        platform = self._platform_of(state)
        if platform is not None and platform != self.model.platform:
            # Upstream builds the stack the config names (``saving.get_platform``, a missing key meaning
            # TensorFlow): a JAX file that lost its key, or a TensorFlow file registered as JAX, would
            # build the wrong one and die -- or worse, run -- after a pool of Dolphins has booted.
            raise RuntimeError(
                f"{config.model} says platform {platform!r} on disk but the registry says "
                f"{self.model.platform!r}; the file is not the one the registry describes"
            )

        self._agent = self.api.build_agent(
            state=state,
            batch_size=num_envs,
            console_delay=config.console_delay,
            # Upstream's own CLI default (``eval_lib.build_agent(name=nametags.DEFAULT_NAME)``), not
            # ``None``: an RL release overrides it with the first tag it trained with, and an
            # imitation one *needs* it -- ``build_delayed_agent(name=None)`` raises "Must specify an
            # agent name" for a checkpoint with no RL tags.  P27's ``fox_d18_imitation_v3`` pairing
            # died on construction three times before this was read back out of the source.
            name=config.name or DEFAULT_IMITATION_NAME,
            sample_temperature=config.sample_temperature,
            async_inference=config.async_inference,
            # ``build_basic_agent`` picks the framework dict by the policy's platform, so passing the
            # TensorFlow one is safe whatever the checkpoint is.  Their own evaluator defaults XLA on;
            # E010 gated ``medium-v2`` with it off, which is the default here.
            tf={"jit_compile": config.jit_compile},
        )
        self._agent.warmup()
        self._decode = self._agent.decode_controller
        self._observation_config = getattr(self._agent, "observation_config", None)

        # Parsers are built lazily: one per environment at its first reset, which is the first
        # frame of its first game (``rows`` also builds one for an environment that somehow arrives
        # without a reset, so a missing parser can never be a crash mid-campaign).
        self._parsers: list[Any] = [None] * num_envs
        self._filters: list[Any] = [
            self.api.observation_filter_factory(self._observation_config) for _ in range(num_envs)
        ]
        self._last_game: list[Any] = [None] * num_envs
        self._zero_game: Any | None = None
        self._last_rows: list[ControllerRow] = [neutral_row()] * num_envs

        self.frames_seen = 0
        self.steps = 0
        self.held_frames = 0
        self.neutral_frames = 0
        self.resets = 0
        self.exceptions = 0
        self.first_error: str = ""

    # -- upstream state ----------------------------------------------------

    @staticmethod
    def _delay_of(state: Any) -> int | None:
        """``config['policy']['delay']`` if the state has one; ``None`` for a fake."""

        try:
            return int(state["config"]["policy"]["delay"])
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _platform_of(state: Any) -> str | None:
        """``config['platform']`` as upstream reads it -- a missing key is TensorFlow; ``None`` for a
        fake with no config at all."""

        try:
            config = state["config"]
        except (KeyError, TypeError):
            return None
        if not isinstance(config, Mapping):
            return None
        return str(config.get("platform", "tf"))

    @property
    def delay(self) -> int:
        """The frames its action is held back, as the agent itself reports it where it can."""

        return int(getattr(self._agent, "delay", self.config.effective_delay))

    @property
    def last_rows(self) -> list[ControllerRow]:
        """What each environment's controller currently holds."""

        return list(self._last_rows)

    # -- resets ------------------------------------------------------------

    def _new_parser(self, index: int) -> None:
        self._parsers[index] = self.api.parser_factory(
            ports=(self.config.libmelee_port, self.config.libmelee_opponent_port)
        )
        self._filters[index].reset()
        self._last_game[index] = None

    def reset(self, index: int) -> None:
        """Forget one environment: a new parser, a reset filter, a neutral controller."""

        self._new_parser(index)
        self._last_rows[index] = neutral_row()
        self.resets += 1

    def reset_all(self) -> None:
        for index in range(self.num_envs):
            self.reset(index)

    # -- one frame ---------------------------------------------------------

    def rows(self, gamestates: Sequence[Any], needs_reset: torch.Tensor) -> list[ControllerRow]:
        """One frame for every environment: reset, observe, one batched step, decode, read back.

        An environment that is not in a game holds its controller and repeats its last observation in
        the batch, so its row of the recurrent state and of the delay queue stays aligned with the
        frames it does play.
        """

        if len(gamestates) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} gamestates, got {len(gamestates)}")
        if int(needs_reset.shape[0]) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset flags, got {int(needs_reset.shape[0])}")

        flags = needs_reset.to(dtype=torch.bool, device="cpu").tolist()
        resets: list[bool] = []
        for index, gamestate in enumerate(gamestates):
            first_frame = gamestate is not None and int(getattr(gamestate, "frame", 0)) == INITIAL_FRAME
            reset = bool(flags[index]) or first_frame
            if reset:
                self.reset(index)
            resets.append(reset)

        live = [index for index, gamestate in enumerate(gamestates) if self._actionable(gamestate)]
        games: list[Any] = []
        for index, gamestate in enumerate(gamestates):
            if index in live:
                if self._parsers[index] is None:
                    self._new_parser(index)
                game = self._filters[index].filter(self._parsers[index].get_game(gamestate))
                self._last_game[index] = game
                if self._zero_game is None:
                    self._zero_game = self.api.map_nt(np.zeros_like, game)
            games.append(self._last_game[index])

        if not live and all(game is None for game in games):
            # Nothing has ever been observed: there is no batch to build and nothing to send.
            self.held_frames += self.num_envs
            return list(self._last_rows)

        batch = self.api.map_nt(
            lambda *leaves: np.stack(leaves, axis=0),
            *[game if game is not None else self._zero_game for game in games],
        )
        try:
            sample = self._agent.step(batch, np.asarray(resets, dtype=bool))
            self.steps += 1
            if self._host_copy:
                # A JAX agent returns device arrays; upstream's ``Agent.step`` copies them to the host
                # before decoding (``eval_lib.py:539-541``) and so does this.
                sample = self.api.device_get(sample)
        except Exception as error:  # counted, then re-raised unless the run file says not to
            self._record_exception(error)
            if self.config.strict:
                raise
            self._last_rows = [neutral_row()] * self.num_envs
            return list(self._last_rows)

        for index in range(self.num_envs):
            if index not in live:
                self.held_frames += 1
                continue
            recorder = ControllerRecorder()
            taken = self.api.map_nt(lambda leaf, row=index: leaf[row], sample.controller_state)
            controller = self._decode(taken)
            self.api.send_controller(recorder, controller)
            row = recorder.row()
            self._last_rows[index] = row
            self.frames_seen += 1
            if row == neutral_row():
                self.neutral_frames += 1
        return list(self._last_rows)

    def _actionable(self, gamestate: Any) -> bool:
        """Upstream's own guards: in a game, and our port is in it.

        The menu is compared by *name* rather than against ``melee.Menu`` members, so importing this
        module -- and therefore :mod:`melee_rl.opponents` -- needs no libmelee, matching the discipline
        of :mod:`melee_rl.env.dolphin`.  ``SUDDEN_DEATH`` counts, as it does in ``_DolphinInstance``.
        """

        if gamestate is None:
            return False
        menu = getattr(gamestate, "menu_state", None)
        if str(getattr(menu, "name", menu)) not in IN_GAME_MENU_NAMES:
            return False
        return self.config.libmelee_port in getattr(gamestate, "players", {})

    def _record_exception(self, error: BaseException) -> None:
        self.exceptions += 1
        if not self.first_error:
            self.first_error = f"{type(error).__name__}: {error}\n" + "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            )
        LOG.warning("slippi-ai %s raised %s: %s", self.config.model, type(error).__name__, error)

    # -- health ------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """What a campaign reads to know the opponent was really playing (P25's lesson)."""

        return {
            "model": self.config.model,
            "sha256": self.model.sha256,
            "sha256_verified": self.sha256_verified,
            "revision": self.model.revision,
            "source_dir": self.config.resolved_source_dir,
            "platform": self.model.platform,
            "kind": self.model.kind,
            "rl_steps": self.model.rl_steps,
            "imitation_steps": self.model.imitation_steps,
            "imitation_frames": self.model.imitation_frames,
            "policy_delay": self.model.delay,
            "console_delay": self.config.console_delay,
            "effective_delay": self.delay,
            "name": self.config.resolved_name,
            "character": self.config.resolved_character,
            "libmelee_port": self.config.libmelee_port,
            "sample_temperature": self.config.sample_temperature,
            "num_envs": self.num_envs,
            "frames_seen": self.frames_seen,
            "steps": self.steps,
            "held_frames": self.held_frames,
            "neutral_frames": self.neutral_frames,
            "neutral_share": (self.neutral_frames / self.frames_seen) if self.frames_seen else 0.0,
            "resets": self.resets,
            "exceptions": self.exceptions,
            "first_error": self.first_error,
        }

    def close(self) -> None:
        stop = getattr(self._agent, "stop", None)
        if callable(stop):
            stop()


# ---------------------------------------------------------------------------
# the preflight (22 Sep 2026)
# ---------------------------------------------------------------------------


def synthetic_gamestate(melee: Any, frame: int, *, stage: str = "FINAL_DESTINATION") -> Any:
    """A real libmelee ``GameState`` with two Foxes standing on the stage, for a step without a Dolphin.

    Real objects rather than a stand-in because upstream's parser reads them field by field
    (``slippi_db/parse_libmelee.py:45-60``: ``position``, ``action.value``, ``character.value``,
    ``controller_state.processed_button``, ``nana``, ``stage.value``, ``fod_platforms``,
    ``projectiles``), and what the recorder feeds it is exactly this class.
    """

    gamestate = melee.GameState()
    gamestate.frame = frame
    gamestate.menu_state = melee.Menu.IN_GAME
    gamestate.stage = getattr(melee.Stage, stage)
    for port, x in zip(PORTS, (-20.0, 20.0), strict=True):
        player = melee.PlayerState()
        player.character = melee.Character.FOX
        player.action = melee.Action.STANDING
        player.position.x = x
        player.position.y = 0.0
        player.facing = x < 0
        player.stock = 4
        player.jumps_left = 2
        gamestate.players[port] = player
    return gamestate


def preflight(
    config: SlippiAIConfig,
    *,
    frames: int = 60,
    num_envs: int = 2,
    api: SlippiAIApi | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build one batched agent and play ``frames`` synthetic frames through the real parser path.

    P25's lesson, P32's shape: a pip layer that does not import, a checkout without the module a
    platform needs, a file that will not restore or a step that raises is a ten-second error here and
    a forty-minute one after a pool of Dolphins has booted.  The frames start at Melee's first frame,
    so every environment resets once and the delay queue is exercised from its primed state.
    """

    from melee_rl.env.libmelee_frames import import_melee

    melee = import_melee()
    started = time.perf_counter()
    player = SlippiAIPlayer(config, num_envs, api=api)
    built = time.perf_counter() - started
    if log is not None:
        log(f"slippi-ai {config.model}: agent built in {built:.1f} s ({player.model.platform})")
    flags = torch.zeros(num_envs, dtype=torch.bool)
    previous = player.last_rows
    changed = 0
    stepping = 0.0
    try:
        for offset in range(frames):
            gamestates = [synthetic_gamestate(melee, INITIAL_FRAME + offset) for _ in range(num_envs)]
            tick = time.perf_counter()
            rows = player.rows(gamestates, flags)
            stepping += time.perf_counter() - tick
            changed += sum(1 for before, after in zip(previous, rows, strict=True) if before != after)
            previous = rows
        stats = player.stats()
    finally:
        player.close()
    return {
        "model": config.model,
        "platform": stats["platform"],
        "kind": stats["kind"],
        "sha256": stats["sha256"],
        "sha256_verified": stats["sha256_verified"],
        "revision": stats["revision"],
        "delay": stats["effective_delay"],
        "name": stats["name"],
        "character": stats["character"],
        "num_envs": num_envs,
        "frames": frames,
        "steps": stats["steps"],
        "build_seconds": built,
        "seconds": time.perf_counter() - started,
        "step_ms": 1000.0 * stepping / max(1, frames),
        "rows_changed": changed,
        "neutral_share": stats["neutral_share"],
        "resets": stats["resets"],
        "exceptions": stats["exceptions"],
        "first_error": stats["first_error"],
    }


def format_preflight(report: Mapping[str, Any]) -> str:
    return (
        f"slippi-ai {report['model']}: {report['platform']} {report['kind']}, delay {report['delay']}, "
        f"as {report['name']!r}; sha256 {'verified' if report['sha256_verified'] else 'NOT verified'}; "
        f"built in {report['build_seconds']:.1f} s; {report['steps']} batched steps over "
        f"{report['frames']} frames x {report['num_envs']} envs, {report['step_ms']:.2f} ms per frame; "
        f"controller changed on {report['rows_changed']} env-frames, neutral share "
        f"{report['neutral_share']:.2f}; {report['exceptions']} exceptions"
    )
