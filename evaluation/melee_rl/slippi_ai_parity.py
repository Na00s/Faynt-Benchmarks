"""The PyTorch port against slippi-ai's own TensorFlow agent, in real games (17 Sep 2026, LEAGUE-PLAN.md
§3.4).

The league run trains against :mod:`melee_rl.slippi_ai_torch`; a subtly wrong port -- swapped LSTM matrices,
an epsilon in a LayerNorm, the approximate GELU, a slip in the head's order, the tech mask, the name code --
would train our Fox against a weaker fake. Shapes cannot catch most of those, numbers can. This harness plays
our fork on port 1 against the release's TensorFlow agent exactly as the recorder does (``SlippiAIPlayer``:
upstream's parser, observation filter, batched ``DelayedAgent``), and runs the port alongside it on *our*
frame tree, teacher-forced on the TensorFlow agent's own fresh sample every frame (:class:`ParityOpponent`).
Both see the same game; the TensorFlow agent's controller is what Dolphin receives. Per component the harness
records the largest absolute logit difference (and the logit it sat on), the largest difference relative to
the logit's size, the mean and largest KL(TensorFlow || port) and the argmax agreement.

**The gate:** |Δ logit| <= 1e-3 * max(1, |logit|) and mean KL <= 1e-6 for every component of every case, over
>= 4 000 in-game frames per file (float32 drift of ~1e-5 is expected at unit scale; upstream's own TF-to-JAX
check is ``scripts/convert_tf_checkpoint_to_jax.py:327-385``). The bar was absolute until the first run
(17 Sep 2026): see :data:`MAX_REL_LOGIT`. A failure is a port bug: no training run launches until
it passes. The cases (:data:`DEFAULT_CASES`): gm as Fox, Ice Climbers (Nana), Peach and Samus (items) on
Fountain of Dreams and Yoshi's Story (the platforms, Randall); ``fox_d18_ditto_v4`` and
``falco_d18_vs_fox_v4`` (version-4 files: the upgrade and the tech mask)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import torch
from torch.nn import functional

from controller_codec import ControllerState
from melee_rl import checkpoint as checkpoint_lib
from melee_rl.actor import RolloutWorker
from melee_rl.env.dummy import DummyMeleeEnv
from melee_rl.env.protocol import EnvProtocol
from melee_rl.frames import FrameTree
from melee_rl.slippi_ai_agent import RELEASED_MODELS
from melee_rl.slippi_ai_torch import COMPONENTS, SlippiAIAgent, load_network
from tensor_batch import BUTTON_ORDER

if TYPE_CHECKING:  # the config module imports the opponents and the evaluator; typing only
    from melee_rl.config import RLConfig

MAX_REL_LOGIT: Final[float] = 1e-3
"""The logit bar, relative to the logit's own size: |d| <= 1e-3 * max(1, |logit|) (17 Sep 2026). The first
run's absolute 1e-3 bar failed the Falco's D_UP at 2.6e-3 with mean KL 1e-12 and full argmax agreement, and
two correct float32 runs of the port (CPU vs GPU) differ by 1e-2 at a logit of -494: an absolute bar reads
rounding as a bug wherever a component is never used."""
MEAN_KL: Final[float] = 1e-6
FINAL_DESTINATION, FOUNTAIN_OF_DREAMS, YOSHIS_STORY = 25, 8, 6
"""libmelee internal stage ids."""


@dataclass(frozen=True)
class ParityCase:
    """One release in real games: the opponent's character on each Dolphin (their count is the batch) and the
    stages cycled over them."""

    release: str
    characters: tuple[int, ...]
    stages: tuple[int, ...]
    frames: int = 4096


DEFAULT_CASES: Final[tuple[ParityCase, ...]] = (
    ParityCase(
        "gm",
        characters=(
            1,
            1,
            10,
            10,
            9,
            9,
            13,
            13,
        ),  # Fox, Ice Climbers (Nana), Peach, Samus (items): each on both stages
        stages=(FOUNTAIN_OF_DREAMS, YOSHIS_STORY),
        frames=6144,
    ),
    ParityCase(
        "fox_d18_ditto_v4", characters=(1,) * 4, stages=(FOUNTAIN_OF_DREAMS, YOSHIS_STORY), frames=4096
    ),
    ParityCase(
        "falco_d18_vs_fox_v4", characters=(22,) * 4, stages=(FOUNTAIN_OF_DREAMS, YOSHIS_STORY), frames=4096
    ),
)


def select_cases(text: str) -> tuple[ParityCase, ...]:
    """``"gm,fox_d18_ditto_v4"`` -> those of :data:`DEFAULT_CASES`; empty -> all of them."""

    if not text:
        return DEFAULT_CASES
    wanted = [name.strip() for name in text.split(",") if name.strip()]
    known = {case.release: case for case in DEFAULT_CASES}
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise ValueError(f"no parity case for {unknown}; have {sorted(known)}")
    return tuple(known[name] for name in wanted)


# ---------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------


def component_kl(reference: torch.Tensor, shadow: torch.Tensor) -> torch.Tensor:
    """KL(reference || shadow) per row for one component's logits: Bernoulli for ``[B, 1]``, categorical
    else."""

    reference = reference.to(torch.float64)
    shadow = shadow.to(torch.float64)
    if reference.shape[-1] == 1:
        a, b = reference[..., 0], shadow[..., 0]
        on_p, off_p = functional.logsigmoid(a), functional.logsigmoid(-a)
        on_q, off_q = functional.logsigmoid(b), functional.logsigmoid(-b)
        return torch.exp(on_p) * (on_p - on_q) + torch.exp(off_p) * (off_p - off_q)
    log_p = torch.log_softmax(reference, dim=-1)
    log_q = torch.log_softmax(shadow, dim=-1)
    return (torch.exp(log_p) * (log_p - log_q)).sum(dim=-1)


class ParityRecorder:
    """Per-component running statistics of reference-vs-port logits."""

    def __init__(self) -> None:
        self.rows = 0
        self._stats: dict[str, dict[str, float]] = {
            name: {
                "rows": 0.0,
                "max_abs_logit": 0.0,
                "logit_at_max": 0.0,
                "max_rel_logit": 0.0,
                "kl_sum": 0.0,
                "max_kl": 0.0,
                "agree": 0.0,
            }
            for name in COMPONENTS
        }

    def add(self, reference: Sequence[torch.Tensor | np.ndarray], shadow: Sequence[torch.Tensor]) -> None:
        if len(reference) != len(COMPONENTS) or len(shadow) != len(COMPONENTS):
            raise ValueError(f"expected {len(COMPONENTS)} components, got {len(reference)} and {len(shadow)}")
        rows = int(np.shape(reference[0])[0])
        for name, theirs, mine in zip(COMPONENTS, reference, shadow, strict=True):
            left = torch.as_tensor(np.asarray(theirs) if isinstance(theirs, np.ndarray) else theirs)
            left = left.to("cpu", torch.float64).reshape(rows, -1)
            right = mine.to("cpu", torch.float64).reshape(rows, -1)
            if left.shape != right.shape:
                raise ValueError(f"{name}: reference logits {tuple(left.shape)} vs port {tuple(right.shape)}")
            kl = component_kl(left, right)
            if left.shape[-1] == 1:
                agree = ((left[:, 0] > 0) == (right[:, 0] > 0)).sum()
            else:
                agree = (left.argmax(dim=-1) == right.argmax(dim=-1)).sum()
            stats = self._stats[name]
            stats["rows"] += rows
            difference = (left - right).abs()
            worst = int(difference.argmax())
            if float(difference.flatten()[worst]) > stats["max_abs_logit"]:
                stats["max_abs_logit"] = float(difference.flatten()[worst])
                stats["logit_at_max"] = float(left.flatten()[worst])
            relative = difference / left.abs().clamp_min(1.0)
            stats["max_rel_logit"] = max(stats["max_rel_logit"], float(relative.max()))
            stats["kl_sum"] += float(kl.sum())
            stats["max_kl"] = max(stats["max_kl"], float(kl.max()))
            stats["agree"] += float(agree)
        self.rows += rows

    def summary(self, *, max_rel_logit: float = MAX_REL_LOGIT, mean_kl: float = MEAN_KL) -> dict[str, Any]:
        components: dict[str, dict[str, float]] = {}
        passed = self.rows > 0
        for name, stats in self._stats.items():
            rows = max(stats["rows"], 1.0)
            entry = {
                "rows": stats["rows"],
                "max_abs_logit": stats["max_abs_logit"],
                "logit_at_max": stats["logit_at_max"],
                "max_rel_logit": stats["max_rel_logit"],
                "mean_kl": stats["kl_sum"] / rows,
                "max_kl": stats["max_kl"],
                "top1": stats["agree"] / rows,
            }
            entry["pass"] = float(entry["max_rel_logit"] <= max_rel_logit and entry["mean_kl"] <= mean_kl)
            passed = passed and bool(entry["pass"])
            components[name] = entry
        return {
            "frames": self.rows,
            "pass": passed,
            "thresholds": {"max_rel_logit": max_rel_logit, "mean_kl": mean_kl},
            "components": components,
        }


def controller_to_indices(controller: Any) -> torch.Tensor:
    """An upstream ``Controller`` of class indices (numpy or TensorFlow tensors) -> ``[B, 13]`` in the head's
    order."""

    columns = [np.asarray(getattr(controller.buttons, name)).astype(np.int64) for name in BUTTON_ORDER]
    for stick in (controller.main_stick, controller.c_stick):
        columns += [np.asarray(stick.x).astype(np.int64), np.asarray(stick.y).astype(np.int64)]
    columns.append(np.asarray(controller.shoulder).astype(np.int64))
    return torch.from_numpy(np.stack(columns, axis=1))


def sample_outputs_to_forced(outputs: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Upstream's ``SampleOutputs`` (``controller_state`` and ``logits``, both ``Controller``-shaped) -> the
    port's ``[B, 13]`` class indices and the thirteen logits in the head's order, read field by field by
    name."""

    controller, logits = outputs.controller_state, outputs.logits
    forced = controller_to_indices(controller)
    rows = forced.shape[0]
    arrays = [getattr(logits.buttons, name) for name in BUTTON_ORDER]
    arrays += [logits.main_stick.x, logits.main_stick.y, logits.c_stick.x, logits.c_stick.y, logits.shoulder]
    return forced, [
        torch.from_numpy(np.asarray(array, dtype=np.float32)).reshape(rows, -1) for array in arrays
    ]


# ---------------------------------------------------------------------------
# the reference and the shadow
# ---------------------------------------------------------------------------


class ReferenceSource(Protocol):
    """What plays the frame (its controller goes to Dolphin) and hands over its fresh sample with logits."""

    def act_state(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState: ...

    def fresh(self) -> tuple[torch.Tensor, list[torch.Tensor]] | None: ...

    def initial_prev(self) -> torch.Tensor | None: ...

    def reset_state(self) -> None: ...


class PortSource:
    """A port agent as the reference (the harness's own tests; a port-against-port sanity run)."""

    def __init__(self, agent: SlippiAIAgent) -> None:
        self.agent = agent
        self._fresh: tuple[torch.Tensor, list[torch.Tensor]] | None = None

    def act_state(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState:
        control, fresh, logits = self.agent.step_with_logits(frames, needs_reset)
        self._fresh = (fresh, logits)
        return control

    def fresh(self) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        return self._fresh

    def initial_prev(self) -> torch.Tensor | None:
        return self.agent.prev.to("cpu", copy=True)

    def reset_state(self) -> None:
        self.agent.reset_all()


class TensorFlowSource:
    """slippi-ai's own agent as the recorder plays it (a ``SlippiAIOpponent``), with its ``DelayedAgent.push``
    wrapped so the fresh ``SampleOutputs`` of every frame -- the one queued, not the delayed one sent -- is
    kept."""

    def __init__(self, opponent: Any) -> None:
        self.opponent = opponent
        agent = opponent.player._agent
        original = agent.push
        self._outputs: Any = None

        def push(game: Any, needs_reset: Any) -> None:
            original(game, needs_reset)
            self._outputs = agent._output_queue[-1]

        agent.push = push

    def act_state(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState:
        self._outputs = None
        control: ControllerState = self.opponent.act_state(frames, needs_reset)
        return control

    def fresh(self) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
        return None if self._outputs is None else sample_outputs_to_forced(self._outputs)

    def initial_prev(self) -> torch.Tensor | None:
        """``BasicAgent._prev_controller`` as it stands: ``warmup()`` stepped the agent once on a dummy game
        at construction (``SlippiAIPlayer``, upstream's own ``Agent`` too, ``eval_lib.py:497``), and a reset
        never clears it -- so the first real frame's controller input is the warm-up's sample, not zeros."""

        return controller_to_indices(self.opponent.player._agent._agent._prev_controller)

    def reset_state(self) -> None:
        self.opponent.reset_state()

    def stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = self.opponent.player.stats()
        return stats


class ParityOpponent:
    """Port 1: the reference plays; the port shadows it teacher-forced on the reference's fresh sample."""

    def __init__(self, reference: ReferenceSource, agent: SlippiAIAgent, recorder: ParityRecorder) -> None:
        self.reference = reference
        self.agent = agent
        self.recorder = recorder
        self.skipped = 0
        seed = reference.initial_prev()
        if seed is not None:  # the reference's own starting controller input (its warm-up sample)
            agent.prev.copy_(seed.to(agent.prev.device))

    @property
    def port(self) -> int:
        return 1

    @property
    def controls_port(self) -> bool:
        return True

    def act(self, frames: Any, needs_reset: torch.Tensor) -> Any:
        raise NotImplementedError(
            "the parity opponent emits a ControllerState; the worker must call act_state"
        )

    def act_state(self, frames: FrameTree, needs_reset: torch.Tensor) -> ControllerState:
        control = self.reference.act_state(frames, needs_reset)
        fresh = self.reference.fresh()
        if fresh is None:  # the reference did not step (nothing observed yet): neither does the shadow
            self.skipped += 1
            return control
        forced, logits = fresh
        self.recorder.add(logits, self.agent.step_forced(frames, needs_reset, forced))
        return control

    def refresh(self, student: Any) -> None:
        return None

    def reset_state(self) -> None:
        self.reference.reset_state()
        self.agent.reset_all()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

ReferenceFactory = Callable[[ParityCase, EnvProtocol, SlippiAIAgent], ReferenceSource]


def _case_env(config: RLConfig, case: ParityCase) -> EnvProtocol:
    count = len(case.characters)
    if config.env.type == "dummy":
        return DummyMeleeEnv(replace(config.env.dummy, num_envs=count, players=("policy", "policy")))
    from melee_rl.env.dolphin_mp import build_dolphin_env

    dolphin = config.env.dolphin
    return build_dolphin_env(
        replace(
            dolphin,
            num_envs=count,
            players=("policy", "policy"),
            characters=(dolphin.characters[0], case.characters[0]),
            opponent_characters=tuple(case.characters),
            stages=tuple(case.stages),
            worker_processes=0,  # the TensorFlow agent reads the raw gamestates: this process only
            shared_memory=False,
            env_index_offset=0,
            save_replays=False,
        )
    )


def _tensorflow_reference(config: RLConfig, case: ParityCase, env: EnvProtocol) -> ReferenceSource:
    from melee_rl.opponents import SlippiAIOpponent
    from melee_rl.slippi_ai_agent import SlippiAIPlayer

    gamestates = getattr(env, "gamestates", None)
    if gamestates is None:
        raise ValueError(
            "the TensorFlow reference needs an in-process Dolphin environment (its raw gamestates)"
        )
    settings = replace(config.slippi_ai, model=case.release, character=case.characters[0], player=1)
    player = SlippiAIPlayer(settings, len(case.characters))
    return TensorFlowSource(SlippiAIOpponent(player, gamestates, port=1))


def run_slippi_parity(
    config: RLConfig,
    checkpoint: str,
    cases: Sequence[ParityCase],
    *,
    device: torch.device | str,
    output: str | Path | None = None,
    reference_factory: ReferenceFactory | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Every case: our ``checkpoint`` (at its own trained delay) on port 0 against the reference on port 1
    with the port shadowing it; one report, written to ``output`` as JSON when given."""

    from melee_rl.adapter import MeleePolicyAdapter

    policy = checkpoint_lib.load_policy(checkpoint, device=device)
    policy.eval()
    policy.requires_grad_(False)
    if isinstance(policy, MeleePolicyAdapter):
        policy.fast_step = config.policy.fast_step
        policy.fast_attention = config.policy.fast_attention
    delay = checkpoint_lib.trained_delay(checkpoint)
    actor = replace(
        config.actor,
        context_mode="ring",
        context_frames=0,
        delay_frames=delay,
        batch_steps=1,
        packed_frames=False,
        decision_stride=1,
        interleave_arms=False,
    )
    settings = config.slippi_ai
    records: list[dict[str, Any]] = []
    for case in cases:
        released = RELEASED_MODELS[case.release]
        wrong = sorted(set(case.characters) - set(released.character_ids))
        if wrong:
            raise ValueError(f"{case.release} cannot play libmelee characters {wrong}")
        network = load_network(
            Path(settings.model_dir) / case.release,
            name=case.release,
            sha256=released.sha256 if settings.verify_sha256 else None,
        )
        spec = network.spec
        count = len(case.characters)
        agent = SlippiAIAgent(network, count, names=(spec.default_name,) * count, device=device, seed=0)
        log(
            f"parity {case.release}: {count} Dolphins, characters {case.characters}, stages {case.stages}, "
            f"delay {spec.delay}, name {spec.default_name!r}, tech mask {spec.tech_mask}; our delay {delay}"
        )
        started = time.perf_counter()
        env = _case_env(config, case)
        try:
            if reference_factory is None:
                reference = _tensorflow_reference(config, case, env)
            else:
                reference = reference_factory(case, env, agent)
            recorder = ParityRecorder()
            opponent = ParityOpponent(reference, agent, recorder)
            worker = RolloutWorker(policy, env, opponent, actor)
            resets = 0
            while worker.frames_seen < case.frames:
                trajectory = worker.rollout()
                resets += int(trajectory.is_resetting[:, trajectory.context_frames :].sum())
                log(f"parity {case.release}: {worker.frames_seen} frames")
        finally:
            env.close()
        summary = recorder.summary()
        record: dict[str, Any] = {
            "release": case.release,
            "sha256": released.sha256,
            "characters": list(case.characters),
            "stages": list(case.stages),
            "names": [spec.default_name] * count,
            "delay": spec.delay,
            "tech_mask": spec.tech_mask,
            "frames": worker.frames_seen,
            "resets": resets,
            "skipped": opponent.skipped,
            "seconds": time.perf_counter() - started,
            **summary,
        }
        stats = getattr(reference, "stats", None)
        if callable(stats):
            record["reference"] = stats()
        worst = max(summary["components"].items(), key=lambda item: item[1]["max_rel_logit"])
        log(
            f"parity {case.release}: pass {summary['pass']}; worst component {worst[0]} "
            f"relative {worst[1]['max_rel_logit']:.2e} (|Δ logit| {worst[1]['max_abs_logit']:.2e} at "
            f"{worst[1]['logit_at_max']:.1f}), mean KL {worst[1]['mean_kl']:.2e}"
        )
        records.append(record)
    report = {
        "checkpoint": checkpoint,
        "our_delay": delay,
        "device": str(device),
        "cases": records,
        "pass": bool(records) and all(record["pass"] for record in records),
        "thresholds": {"max_rel_logit": MAX_REL_LOGIT, "mean_kl": MEAN_KL},
    }
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1) + "\n")
    return report


__all__ = [
    "DEFAULT_CASES",
    "MAX_REL_LOGIT",
    "MEAN_KL",
    "ParityCase",
    "ParityOpponent",
    "ParityRecorder",
    "PortSource",
    "ReferenceSource",
    "TensorFlowSource",
    "component_kl",
    "controller_to_indices",
    "run_slippi_parity",
    "sample_outputs_to_forced",
    "select_cases",
]
