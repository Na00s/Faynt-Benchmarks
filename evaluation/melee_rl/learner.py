"""``PPOLearner``: slippi-ai's PPO semantics in PyTorch (PLAN.md §3.6, R4).

One learner step takes ``num_batches`` trajectories (PLAN.md §2.3) and, per trajectory:

1. **teacher pass** (no grad): ``teacher.unroll_window`` on the policy window, logits of the
   decision rows ``[C, C+T-D)`` (``rl/learner.py:219-224``);
2. **value pass**: ``ValueNet`` on the value window, returns ``G_t = r_t + gamma_t G_{t+1}``
   bootstrapped with ``V(s_T)``, advantages ``G - V`` (detached), one Adam step of the value
   optimizer per trajectory (``rl/learner.py:194-203``); advantages ``[D, T)`` pair with the
   decisions (``rl/learner.py:225``);
3. **behaviour log-probs** ``log pi_old(a)`` from the stored actor logits (exact in ``reprime``
   fp32 mode) or, ``behaviour_logits = "learner"`` (forced by ``"auto"`` in ring mode), from a
   no-grad pre-pass of the current policy (deviation 2 of PLAN.md §3.6; ``loss/actor_kl_pre``
   is ``KL(actor || pre-pass)``).

In ``context_mode = "prefix"`` (INTERFACE.md patch #1, session 39) the policy passes forward only the
decision rows ``[C, C+T-D)`` against the actor's stored cache (``Trajectory.initial_cache`` through
``unroll_window(..., prefix=)``) -- one row per trained decision, and exactly what the actor computed,
so the stored logits serve as in ``reprime`` mode and ``loss/actor_kl_pre`` (the first epoch's actor KL
at unchanged weights) is the exactness monitor.  The teacher and the value net keep reading the
``C + T`` window.

The teacher's own delay (``[teacher] delay_frames``, 11 Sep 2026, ``/delay-finetune``): with
``teacher_delay_frames = D_T`` different from the trajectory's ``D`` the teacher forwards
``Trajectory.teacher_window(D_T)`` -- row ``k + D - D_T`` with the pairing of a ``D_T``-delayed policy, whose
decision executes at the same frame as the student's -- eagerly through ``unroll_window``, and its logits are
read at ``teacher_decision_rows``; ``D_T = 0`` is the hindsight teacher (the frozen delay-0 expert where the
decision lands).  Decisions whose teacher row lies in another game (``Trajectory.teacher_keep``) leave the
teacher terms: ``loss/teacher_kl`` / ``loss/reverse_teacher_kl`` average over the kept rows,
``hindsight/masked_fraction`` says how many were dropped, ``hindsight/top1`` (``pre/top1`` before the update)
is the share of decisions whose button argmax agrees with the teacher's, and ``hindsight/teacher_entropy`` the
teacher's entropy beside ``loss/entropy``.  ``D_T = D`` (the default) is today's path bit for bit.

Then ``num_epochs`` training epochs (``rl/learner.py:263-296``): ``log_rhos = log pi_new - log
pi_old``, clipped in **log space** to ``[-epsilon, epsilon]`` (``epsilon = 1e-2``),
``ppo_objective = min(rho A, clip(rho) A)``, loss = mean of ``-pg_weight * ppo_objective + beta *
KL(pi_old || pi_new) + kl_teacher_weight * KL(pi_new || teacher) + reverse_kl_teacher_weight *
KL(teacher || pi_new) - entropy_weight * H(pi_new)``, advantages not normalised, gradients
accumulated over trajectories and env microbatches (deviation 3: loss scaled by ``mb / B /
num_batches``, or ``mb / rows of the step`` with ``loss_weighting = "row"``, 17 Sep 2026) and applied once
per epoch; then one **evaluation epoch** (no update) whose
mean actor KL is compared with ``max_mean_actor_kl``: above it, the policy parameters *and*
the policy optimizer state are **reverted** to their pre-epoch values (``rl/learner.py:451-468``;
the JAX twin raises instead, ``jax/rl/learner.py:629-635``).  After the guard (15 Sep 2026,
``STRENGTH-LEVERS.md`` §2): an EMA teacher (``teacher_ema_tau``) moves toward the accepted policy, the
target-entropy controller (``entropy_target``) moves the live ``entropy_weight``, and the KL-targeted rule
(``actor_kl_target``) moves the live ``policy_lr``; a ``reference`` policy's KL is logged at weight 0.

Burn-in ladder (``jax/rl/learner.py:613-618``, folded into the learner): steps
``[0, value_burnin_steps)`` train only the value net (0 epochs); the next
``optimizer_burnin_steps`` run one epoch with policy learning rate 0 (Adam moments warm up,
parameters unchanged); afterwards the full ``ppo.num_epochs``.  The value net is trained on
every step.  Distribution math is ``melee_rl.distributions`` (closed form per component from
teacher-forced logits, PLAN.md §3.1); the policy runs in ``train()`` (Ali's gradient
checkpointing engages, ``model.py:1440``; dropout is 0), the teacher stays frozen in ``eval()``.

Metrics (flat ``dict[str, float]``; names fixed for ``logging.py``, PLAN.md §8): ``loss/*`` and
``ppo/*`` come from the **post-update evaluation epoch** (what the guard sees), ``pre/*`` from
the first pass (pre-update), ``ppo_step/{i}/*`` from every training epoch, ``value/*`` from the
value pass, ``weights/drift_step`` / ``weights/drift_total`` the relative L2 movement of the policy
parameters over this step and since the learner was built (a reverted step reports exactly 0 for
both), ``opt/*`` and ``timing/*`` from this step.  After :meth:`PPOLearner.step` the
parameters' ``.grad`` fields hold the last training epoch's accumulated gradients (diagnostics).
"""

from __future__ import annotations

import contextlib
import copy
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import torch
from torch import nn

from controller_codec import ControllerLabels
from melee_rl import distributions
from melee_rl.fast_forward import ATTENTION_MODES, COMPILE_BACKENDS, CacheStager, FastForward
from melee_rl.frames import labels_index
from melee_rl.kv_cache import cache_to
from melee_rl.profiling import NULL_PROFILER, SectionProfiler
from melee_rl.protocol import PolicyProtocol
from melee_rl.reward import RewardConfig
from melee_rl.rl_lib import ReturnsConfig, unexplained_variance
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig, ValueNet, value_targets
from model import MeleePolicy

BEHAVIOUR_LOGITS: Final[tuple[str, ...]] = ("auto", "actor", "learner")
MATMUL_PRECISIONS: Final[tuple[str, ...]] = ("highest", "high", "medium")
LOSS_WEIGHTINGS: Final[tuple[str, ...]] = ("trajectory", "row")
"""``[learner] loss_weighting`` (17 Sep 2026). ``"trajectory"``: each trajectory's row-mean loss weighs
1 / trajectories, so a mix's arms train at equal shares whatever their row counts. ``"row"``: every decision
row of the step weighs the same, so the arms train at their row shares."""
STAGES: Final[tuple[str, ...]] = ("value_burnin", "optimizer_burnin", "ppo")

_EPOCH_KEYS: Final[dict[str, str]] = {
    "total": "loss/total",
    "ppo_objective": "loss/ppo_objective",
    "teacher_kl": "loss/teacher_kl",
    "reverse_teacher_kl": "loss/reverse_teacher_kl",
    "actor_kl_mean": "loss/actor_kl_mean",
    "actor_kl_max": "loss/actor_kl_max",
    "entropy": "loss/entropy",
    "log_rho_mean": "ppo/log_rho_mean",
    "log_rho_abs_max": "ppo/log_rho_abs_max",
    "clip_fraction": "ppo/clip_fraction",
    "top1": "hindsight/top1",
    "reference_kl": "loss/reference_kl",
    "reverse_reference_kl": "loss/reverse_reference_kl",
}
_TEACHER_TERMS: Final[tuple[str, ...]] = ("teacher_kl", "reverse_teacher_kl", "top1")
"""Per-row terms that average over the decisions the teacher keeps (``Trajectory.teacher_keep``)."""


class TrainablePolicy(PolicyProtocol, Protocol):
    """:class:`PolicyProtocol` plus the two ``nn.Module`` methods the learner needs."""

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def train(self, mode: bool = True) -> Any: ...


@dataclass(frozen=True)
class PPOConfig:
    """``[learner.ppo]`` (slippi-ai ``PPOConfig``, ``rl/learner.py:21-29``; defaults PLAN.md §5)."""

    num_epochs: int = 2
    num_batches: int = 8
    epsilon: float = 1.0e-2
    beta: float = 0.0
    max_mean_actor_kl: float = 1.0e-4
    behaviour_logits: str = "auto"

    def __post_init__(self) -> None:
        if self.num_epochs < 0:
            raise ValueError("num_epochs must be >= 0")
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if not self.epsilon > 0.0 or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive and finite")
        if self.beta < 0.0 or self.max_mean_actor_kl < 0.0:
            raise ValueError("beta and max_mean_actor_kl must be >= 0")
        if self.behaviour_logits not in BEHAVIOUR_LOGITS:
            raise ValueError(f"behaviour_logits must be one of {BEHAVIOUR_LOGITS}")


@dataclass(frozen=True)
class LearnerConfig:
    """``[learner]`` (slippi-ai ``LearnerConfig``, ``rl/learner.py:31-44``; defaults PLAN.md §5).

    ``loss_weighting`` (:data:`LOSS_WEIGHTINGS`) sets how a step's trajectories share the policy loss; the
    value net still takes one optimizer step per trajectory either way.  ``concat_arms`` (a mix only, read
    by the trainer, 17 Sep 2026) joins each rollout's arm trajectories into one learner trajectory: fewer,
    larger microbatches (``microbatch_envs`` divides the joined rows) and one value step per rollout.
    ``value_learning_rate`` ``None`` means ``learning_rate`` (slippi-ai shares one variable,
    ``rl/learner.py:117-118``); ``microbatch_envs`` ``None`` means the whole batch; ``reward``
    ``None`` trusts the worker's rewards, a config recomputes them (``rl/learner.py:98-105``).
    ``teacher_prefix`` (``context_mode = "prefix"`` only) lets the frozen teacher attend to the actor's
    stored cache like the policy does, so its KL measures weight drift alone: with the ``C + T`` window
    the teacher sees ``C + t`` frames where the policy saw its full ring, and that context mismatch
    read 4.95e-3 nats at unchanged weights on the 10m smoke of 3 Sep 2026 (exact at the start of
    training, where teacher and policy share their weights; the prefix keys drift with the policy
    afterwards, a second-order effect bounded by the very KL term).
    ``matmul_precision`` (5 Sep 2026, ``/rl-speedup``) is ``torch.set_float32_matmul_precision`` applied to
    the **policy's** learner passes only (the behaviour pre-pass, the PPO epochs and the post pass; the
    teacher and the value net stay at ``"highest"``): ``"high"`` lets the fp32 GEMMs run on TF32 tensor
    cores, ``"medium"`` on bf16 ones.  Weights and activations stay fp32; the cost is measured, not
    assumed -- at unchanged weights ``loss/actor_kl_pre`` (vs the fp32 actor) and ``pre/teacher_kl`` (vs
    the fp32 teacher) read exactly the precision loss; the bar is 1e-6 nats (PLAN.md §3.9).  The setting
    is restored after every block.
    ``prefix_on_host`` (``context_mode = "prefix"`` only, the 75m run of 4 Sep 2026) keeps every stored
    prefix in host memory across the PPO epochs and moves each microbatch's slice to the learner device
    on demand: the 16 x 168 fp32 prefixes of one update are 3.3 GiB at 10m but 10.8 GiB at 75m (11 layers
    x 3 kv heads x 64 x 256 slots), which cannot sit beside the activations on a 24 GB L4.  The forward
    is the same tensors either way; only the copies move.
    ``fast_forward`` (5 Sep 2026, ``/rl-learner-speedup``; ``context_mode = "prefix"`` only) routes the
    policy's and the teacher's prefix-window passes through ``melee_rl.fast_forward.FastForward`` -- Ali's
    modules and weights re-orchestrated without host reads, with a static prefix length, traffic-optimal depth
    reads and attention without the GQA expansion (the same fp32 math, re-associated; ``loss/actor_kl_pre``
    and ``pre/teacher_kl`` at unchanged weights measure it) -- and stages host-resident prefixes through
    pinned buffers; ``fast_forward_attention`` picks ``"sdpa"`` (a fused fp32 kernel where the device offers
    one) or ``"bmm"`` (explicit batched matmuls per kv head).  ``compile`` (lever 2) runs the lean pass
    through ``torch.compile`` (``compile_backend``: ``"inductor"`` on the GPU, ``"aot_eager"`` for the CPU
    tests), one graph per grad mode at the run's fixed chunk shape; it needs ``fast_forward`` and the model's
    ``gradient_checkpointing`` off (``policy.model_overrides``); inductor's fusion rounds differently --
    measured like the rest.
    The two controllers of 15 Sep 2026 (``STRENGTH-LEVERS.md`` §2.3, §2.5) act once per PPO-stage step, after
    the guard.  ``entropy_target`` arms a target-entropy controller on the entropy bonus: on every accepted
    step ``w <- clamp(w + entropy_gain * (target - H), 0, entropy_weight_max)`` with ``H`` the post-update
    mean entropy per decision, so the bonus is zero while the policy is above its floor and grows only below
    it; ``entropy_weight`` is the starting value.  ``actor_kl_target`` arms a KL-targeted policy learning rate
    (rsl_rl's rule): after each step -- reverted ones included, their KL is the attempted one -- the rate is
    divided by ``lr_adapt_factor`` when the post-update actor KL is above twice the target and multiplied by
    it when below half, clamped to ``[lr_min, lr_max]`` (defaults: a quarter and four times
    ``learning_rate``); ``learning_rate`` is the starting value.  Both live values are learner state
    (``opt/entropy_weight``, ``opt/lr_policy``; saved and restored with the checkpoint).
    """

    learning_rate: float = 1.0e-4
    value_learning_rate: float | None = None
    policy_gradient_weight: float = 1.0
    kl_teacher_weight: float = 0.1
    reverse_kl_teacher_weight: float = 0.0
    entropy_weight: float = 0.0
    entropy_target: float | None = None
    entropy_gain: float = 1.0e-3
    entropy_weight_max: float = 0.02
    actor_kl_target: float | None = None
    lr_min: float | None = None
    lr_max: float | None = None
    lr_adapt_factor: float = 1.5
    value_burnin_steps: int = 0
    optimizer_burnin_steps: int = 0
    microbatch_envs: int | None = None
    loss_weighting: str = "trajectory"
    concat_arms: bool = False
    teacher_prefix: bool = False
    prefix_on_host: bool = False
    matmul_precision: str = "highest"
    fast_forward: bool = False
    fast_forward_attention: str = "sdpa"
    compile: bool = False
    compile_backend: str = "inductor"
    reward: RewardConfig | None = None
    returns: ReturnsConfig = field(default_factory=ReturnsConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        rates = {"learning_rate": self.learning_rate, "value_learning_rate": self.value_learning_rate}
        for name, rate in rates.items():
            if rate is not None and (rate < 0.0 or not math.isfinite(rate)):
                raise ValueError(f"{name} must be finite and >= 0")
        weights = {
            "policy_gradient_weight": self.policy_gradient_weight,
            "kl_teacher_weight": self.kl_teacher_weight,
            "reverse_kl_teacher_weight": self.reverse_kl_teacher_weight,
            "entropy_weight": self.entropy_weight,
        }
        for name, weight in weights.items():
            if weight < 0.0 or not math.isfinite(weight):
                raise ValueError(f"{name} must be finite and >= 0")
        if self.value_burnin_steps < 0 or self.optimizer_burnin_steps < 0:
            raise ValueError("burn-in step counts must be >= 0")
        if self.microbatch_envs is not None and self.microbatch_envs < 1:
            raise ValueError("microbatch_envs must be >= 1 (or None for the whole batch)")
        if self.loss_weighting not in LOSS_WEIGHTINGS:
            raise ValueError(f"loss_weighting must be one of {LOSS_WEIGHTINGS}, got {self.loss_weighting!r}")
        if self.matmul_precision not in MATMUL_PRECISIONS:
            raise ValueError(
                f"matmul_precision must be one of {MATMUL_PRECISIONS}, got {self.matmul_precision!r}"
            )
        if self.fast_forward_attention not in ATTENTION_MODES:
            raise ValueError(
                f"fast_forward_attention must be one of {ATTENTION_MODES}, "
                f"got {self.fast_forward_attention!r}"
            )
        if self.compile_backend not in COMPILE_BACKENDS:
            raise ValueError(
                f"compile_backend must be one of {COMPILE_BACKENDS}, got {self.compile_backend!r}"
            )
        if self.compile and not self.fast_forward:
            raise ValueError(
                "learner.compile = true needs learner.fast_forward = true (it compiles the lean pass)"
            )
        target = self.entropy_target
        if target is not None and not (target > 0.0 and math.isfinite(target)):
            raise ValueError(f"entropy_target must be positive and finite or None, got {target}")
        if self.entropy_gain < 0.0 or not math.isfinite(self.entropy_gain):
            raise ValueError(f"entropy_gain must be finite and >= 0, got {self.entropy_gain}")
        if self.entropy_weight_max < 0.0 or not math.isfinite(self.entropy_weight_max):
            raise ValueError(f"entropy_weight_max must be finite and >= 0, got {self.entropy_weight_max}")
        if self.entropy_target is not None and self.entropy_weight > self.entropy_weight_max:
            raise ValueError(
                f"entropy_weight {self.entropy_weight} is above entropy_weight_max "
                f"{self.entropy_weight_max}: the controller clamps the bonus to [0, entropy_weight_max]"
            )
        kl_target = self.actor_kl_target
        if kl_target is not None and not (kl_target > 0.0 and math.isfinite(kl_target)):
            raise ValueError(f"actor_kl_target must be positive and finite or None, got {kl_target}")
        if not (self.lr_adapt_factor > 1.0 and math.isfinite(self.lr_adapt_factor)):
            raise ValueError(f"lr_adapt_factor must be finite and > 1, got {self.lr_adapt_factor}")
        for name, bound in (("lr_min", self.lr_min), ("lr_max", self.lr_max)):
            if bound is not None and (bound <= 0.0 or not math.isfinite(bound)):
                raise ValueError(f"{name} must be positive and finite or None, got {bound}")
        if self.lr_min is not None and self.lr_min > self.learning_rate:
            raise ValueError(f"lr_min {self.lr_min} is above learning_rate {self.learning_rate}")
        if self.lr_max is not None and self.lr_max < self.learning_rate:
            raise ValueError(f"lr_max {self.lr_max} is below learning_rate {self.learning_rate}")

    @property
    def effective_value_learning_rate(self) -> float:
        return self.learning_rate if self.value_learning_rate is None else self.value_learning_rate

    @property
    def effective_lr_min(self) -> float:
        """The KL-targeted rate's floor: ``lr_min``, or a quarter of ``learning_rate``."""

        return self.learning_rate / 4.0 if self.lr_min is None else self.lr_min

    @property
    def effective_lr_max(self) -> float:
        """The KL-targeted rate's ceiling: ``lr_max``, or four times ``learning_rate``."""

        return self.learning_rate * 4.0 if self.lr_max is None else self.lr_max


@dataclass
class _Batch:
    """One trajectory prepared for the PPO epochs (detached tensors on the learner device)."""

    trajectory: Trajectory
    chunks: list[slice]
    subs: list[Trajectory]
    labels: ControllerLabels
    log_old: torch.Tensor
    old_logits: dict[str, torch.Tensor]
    teacher_logits: dict[str, torch.Tensor] | None
    advantages: torch.Tensor
    batch_metrics: dict[str, float]
    value_grad_norm: float
    actor_kl_pre: float | None
    teacher_keep: torch.Tensor | None = None
    reference_logits: dict[str, torch.Tensor] | None = None


class _Accumulator:
    """Element-weighted sums of the per-row PPO quantities over one epoch.

    Everything stays on the device until :meth:`summary` reads it in one round trip (5 Sep 2026,
    ``/rl-learner-speedup``: the three per-chunk host reads of the old accumulator were 3 of the ≈ 40 sync
    points of a learner pass).  The values are the ones the per-chunk reads gave: maxima and integer counts
    are exact, and the sums are the same fp32 tensors read once instead of per epoch.  With a ``keep`` mask
    (the hindsight teacher, 11 Sep 2026) the teacher terms are summed over the kept rows and averaged over
    their count; without one every sum and count is what it was.
    """

    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0
        self._kept = 0
        self._kept_extra: torch.Tensor | None = None
        self._actor_kl_max: torch.Tensor | None = None
        self._log_rho_abs_max: torch.Tensor | None = None
        self._clipped: torch.Tensor | None = None

    def add(self, elements: Mapping[str, torch.Tensor], keep: torch.Tensor | None = None) -> None:
        count = int(elements["total"].numel())
        self._count += count
        if keep is None:
            self._kept += count
        else:
            kept = keep.detach().sum()
            self._kept_extra = kept if self._kept_extra is None else self._kept_extra + kept
        for name, value in elements.items():
            value = value.detach().float()
            if keep is not None and name in _TEACHER_TERMS:
                value = value * keep.to(value.dtype)
            total = value.sum()
            self._sums[name] = total if name not in self._sums else self._sums[name] + total
        actor_kl_max = elements["actor_kl"].detach().max()
        self._actor_kl_max = (
            actor_kl_max if self._actor_kl_max is None else torch.maximum(self._actor_kl_max, actor_kl_max)
        )
        log_rho = elements["log_rho"].detach().abs()
        log_rho_max = log_rho.max()
        previous_max = self._log_rho_abs_max
        self._log_rho_abs_max = (
            log_rho_max if previous_max is None else torch.maximum(previous_max, log_rho_max)
        )
        clipped = (log_rho > self._epsilon).sum()
        self._clipped = clipped if self._clipped is None else self._clipped + clipped

    def summary(self) -> dict[str, float]:
        maxima = (self._actor_kl_max, self._log_rho_abs_max, self._clipped)
        if self._count == 0 or any(value is None for value in maxima):
            raise RuntimeError("no PPO rows were accumulated")
        assert self._actor_kl_max is not None and self._log_rho_abs_max is not None
        assert self._clipped is not None
        names = list(self._sums)
        # One device-to-host read: the sums, the two maxima, the clip count and the kept-row count (fp32 ->
        # double is exact, so the doubles are what ``float(tensor)`` per value would have given).
        extra = [] if self._kept_extra is None else [self._kept_extra.double()]
        stacked = torch.stack(
            [self._sums[name].double() for name in names]
            + [self._actor_kl_max.double(), self._log_rho_abs_max.double(), self._clipped.double()]
            + extra
        )
        read = stacked.tolist()
        kept = self._kept + (read[-1] if extra else 0.0)
        rows = max(kept, 1.0)  # every decision masked: the teacher terms read 0
        means = {
            name: read[index] / (rows if name in _TEACHER_TERMS else self._count)
            for index, name in enumerate(names)
        }
        actor_kl_max, log_rho_abs_max, clipped = read[len(names) : len(names) + 3]
        return {
            "total": means["total"],
            "ppo_objective": means["ppo_objective"],
            "teacher_kl": means["teacher_kl"],
            "reverse_teacher_kl": means["reverse_teacher_kl"],
            "actor_kl_mean": means["actor_kl"],
            "actor_kl_max": actor_kl_max,
            "entropy": means["entropy"],
            "log_rho_mean": means["log_rho"],
            "log_rho_abs_max": log_rho_abs_max,
            "clip_fraction": clipped / self._count,
            "top1": means["top1"],
            # The weight-0 reference (15 Sep 2026): 0 when the learner has none.
            "reference_kl": means.get("reference_kl", 0.0),
            "reverse_reference_kl": means.get("reverse_reference_kl", 0.0),
        }


def _tree_cpu(value: Any) -> Any:
    """Detached CPU copies of every tensor in a nested container (for checkpoints)."""

    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, Mapping):
        return {key: _tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_tree_cpu(item) for item in value)
    return value


def _decision_logits(logits: Mapping[str, torch.Tensor], context: int) -> dict[str, torch.Tensor]:
    """Window logits ``[B, L, V]`` restricted to the decision rows ``[C, L)``."""

    return {name: value[:, context:] for name, value in logits.items()}


def _squared_sums(tensors: Sequence[torch.Tensor]) -> list[float]:
    """Each tensor's fp32 sum of squares, read from the device in one round trip (5 Sep 2026).

    The per-tensor kernels and the host-side summation order of the callers are those of the old
    one-read-per-tensor code, so every norm below is bit-identical to what it was; only the 205 syncs per
    call became one.
    """

    partial = [(tensor.detach().float() ** 2).sum() for tensor in tensors]
    if not partial:
        return []
    values: list[float] = torch.stack(partial).tolist()
    return values


def _parameter_norm(parameters: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(_squared_sums(parameters)))


def _relative_distance(
    current: Sequence[torch.Tensor], reference: Sequence[torch.Tensor], scale: float
) -> float:
    """``||current - reference|| / scale`` -- 0.0 when ``scale`` is 0 (an all-zero reference)."""

    if scale <= 0.0:
        return 0.0
    differences = [
        now.detach().float() - before.detach().float() for now, before in zip(current, reference, strict=True)
    ]
    return math.sqrt(sum(_squared_sums(differences))) / scale


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    total = 0.0
    for value in _squared_sums(grads):
        total += value
    return math.sqrt(total)


@contextlib.contextmanager
def matmul_precision(setting: str) -> Iterator[None]:
    """``torch.set_float32_matmul_precision(setting)`` for the block, restored afterwards.

    ``"highest"`` leaves the setting as it is.
    """

    if setting == "highest":
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(setting)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


class PPOLearner:
    """PPO with a frozen teacher and a separate value net; see the module docstring."""

    def __init__(
        self,
        policy: TrainablePolicy,
        teacher: PolicyProtocol | None,
        value_net: ValueNet,
        config: LearnerConfig,
        *,
        profiler: SectionProfiler | None = None,
        teacher_delay_frames: int | None = None,
        teacher_ema_tau: float | None = None,
        reference: PolicyProtocol | None = None,
    ) -> None:
        self._profiler: SectionProfiler = NULL_PROFILER if profiler is None else profiler
        if teacher is None and (config.kl_teacher_weight != 0.0 or config.reverse_kl_teacher_weight != 0.0):
            raise ValueError("teacher KL weights must be 0 when no teacher is given")
        if teacher_delay_frames is not None:
            if teacher is None:
                raise ValueError("teacher_delay_frames needs a teacher")
            if teacher_delay_frames < 0:
                raise ValueError("teacher_delay_frames must be >= 0")
        if teacher_ema_tau is not None:
            if teacher is None:
                raise ValueError("teacher_ema_tau needs a teacher (the EMA anchor starts as a copy of it)")
            if not (0.0 < teacher_ema_tau <= 1.0):
                raise ValueError(f"teacher_ema_tau must be in (0, 1], got {teacher_ema_tau}")
        self.teacher_delay_frames = teacher_delay_frames
        """The teacher's own delay (``[teacher] delay_frames``); ``None`` = each trajectory's ``D``."""
        self.teacher_ema_tau = teacher_ema_tau
        """``[teacher] ema_tau``: the anchor moves ``tau`` of the way to the policy after every accepted PPO
        step (15 Sep 2026); ``None`` = a fixed teacher."""
        self.policy = policy
        self.teacher = teacher
        self.reference = reference
        """A frozen policy whose forward / reverse KL are logged at weight 0 (``[teacher] reference_path``,
        15 Sep 2026)."""
        self.value_net = value_net
        self.config = config
        self.entropy_weight = config.entropy_weight
        """The live entropy bonus: ``config.entropy_weight`` until the target-entropy controller moves it."""
        self.policy_lr = config.learning_rate
        """The live PPO-stage learning rate: ``config.learning_rate`` until the KL-targeted rule moves it."""
        self._policy_parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        if not self._policy_parameters:
            raise ValueError("the policy has no trainable parameters")
        self.device = self._policy_parameters[0].device
        if value_net.device != self.device:
            raise ValueError(f"value_net is on {value_net.device}, the policy on {self.device}")
        self.policy_optimizer = torch.optim.Adam(self._policy_parameters, lr=config.learning_rate)
        self.value_optimizer = torch.optim.Adam(
            value_net.parameters(),
            lr=config.effective_value_learning_rate,
        )
        self._steps = 0
        self.last_metrics: dict[str, float] = {}
        # 5 Sep 2026: the lean prefix-window forward, one binding per model, and the pinned prefix staging.
        self._fast: dict[int, FastForward] = {}
        self._stager: CacheStager | None = None
        if config.fast_forward:
            for model in (policy, teacher, reference):
                if model is not None:
                    self._bind_fast(model)
            self._stager = CacheStager(self.device)
        self._ema_pairs = self._ema_parameter_pairs()
        self._reference = [parameter.detach().clone() for parameter in self._policy_parameters]
        self._reference_norm = _parameter_norm(self._reference)

    def _bind_fast(self, model: PolicyProtocol) -> None:
        if not isinstance(model, MeleePolicy):
            raise ValueError("learner.fast_forward needs MeleePolicy-based policies, teachers and references")
        config = self.config
        self._fast[id(model)] = FastForward(
            model,
            attention=config.fast_forward_attention,
            compile=config.compile_backend if config.compile else None,
        )

    def _ema_parameter_pairs(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """``(anchor, policy)`` parameter pairs of the EMA update, aligned by registration order."""

        if self.teacher_ema_tau is None:
            return []
        if not isinstance(self.teacher, nn.Module):
            raise ValueError("the EMA anchor needs an nn.Module teacher (its parameters move in place)")
        anchor = list(self.teacher.parameters())
        policy = list(self.policy.parameters())
        if len(anchor) != len(policy) or any(a.shape != p.shape for a, p in zip(anchor, policy, strict=True)):
            raise ValueError("the EMA anchor's parameters do not line up with the policy's")
        return list(zip(anchor, policy, strict=True))

    def set_teacher(self, teacher: PolicyProtocol | None) -> None:
        """Swap the teacher (a resume rebuilds it from the checkpoint's record): rebinds the lean forward
        and the EMA pairs, which are keyed on the object."""

        previous = self.teacher
        if previous is not None:
            self._fast.pop(id(previous), None)
        self.teacher = teacher
        if teacher is not None and self.config.fast_forward:
            self._bind_fast(teacher)
        self._ema_pairs = self._ema_parameter_pairs()

    def _ema_update(self) -> None:
        """``theta_T <- theta_T + tau * (theta - theta_T)`` in place (``torch.lerp`` per tensor)."""

        tau = self.teacher_ema_tau
        assert tau is not None
        with torch.no_grad():
            for anchor, parameter in self._ema_pairs:
                anchor.lerp_(parameter.detach(), tau)

    # -- facts --------------------------------------------------------------------------

    @property
    def steps_done(self) -> int:
        """Number of completed :meth:`step` calls (drives the burn-in ladder)."""

        return self._steps

    @property
    def policy_parameters(self) -> list[nn.Parameter]:
        return list(self._policy_parameters)

    @property
    def discount(self) -> float:
        return self.config.returns.discount

    def stage(self, step: int | None = None) -> tuple[str, int, float]:
        """``(stage, num_epochs, policy_learning_rate)`` for learner step ``step`` (default: the next)."""

        index = self._steps if step is None else step
        config = self.config
        if index < config.value_burnin_steps:
            return "value_burnin", 0, 0.0
        if index < config.value_burnin_steps + config.optimizer_burnin_steps:
            return "optimizer_burnin", 1, 0.0
        return "ppo", config.ppo.num_epochs, self.policy_lr

    # -- one learner step ----------------------------------------------------------------

    def step(self, trajectories: Sequence[Trajectory]) -> dict[str, float]:
        """Teacher + value passes, ``num_epochs`` PPO epochs, the evaluation epoch and the guard."""

        started = time.perf_counter()
        if not trajectories:
            raise ValueError("step needs at least one trajectory")
        stage, num_epochs, policy_lr = self.stage()
        _set_learning_rate(self.policy_optimizer, policy_lr)
        _set_learning_rate(self.value_optimizer, self.config.effective_value_learning_rate)
        self.policy.train()
        self.value_net.train()
        mode = self._behaviour_mode(trajectories)

        batches = [self._prepare(trajectory, mode) for trajectory in trajectories]
        prepared = time.perf_counter()

        with self._profiler.window("learner/snapshot"):
            snapshot = self._snapshot()
        entropy_weight_used = self.entropy_weight
        epochs = [self._epoch(batches, train=True) for _ in range(num_epochs)]
        post = self._epoch(batches, train=False)
        reverted = post["actor_kl_mean"] > self.config.ppo.max_mean_actor_kl
        if reverted:
            self._restore(snapshot)
        if stage == "ppo":
            self._adapt(post, reverted)
        self._steps += 1
        finished = time.perf_counter()

        first = epochs[0] if epochs else post
        pre_kls = [batch.actor_kl_pre for batch in batches if batch.actor_kl_pre is not None]
        actor_kl_pre = first["actor_kl_mean"] if mode == "actor" else _mean(pre_kls)
        metrics: dict[str, float] = {key: post[name] for name, key in _EPOCH_KEYS.items()}
        metrics["loss/actor_kl_pre"] = actor_kl_pre
        metrics["pre/total"] = first["total"]
        metrics["pre/teacher_kl"] = first["teacher_kl"]
        metrics["pre/reverse_teacher_kl"] = first["reverse_teacher_kl"]
        metrics["pre/entropy"] = first["entropy"]
        metrics["pre/top1"] = first["top1"]
        metrics["pre/reference_kl"] = first["reference_kl"]
        for index, epoch in enumerate(epochs):
            for name, value in epoch.items():
                metrics[f"ppo_step/{index}/{name}"] = value
        metrics["ppo/reverted"] = float(reverted)
        metrics["ppo/epochs"] = float(num_epochs)
        metrics["ppo/stage"] = float(STAGES.index(stage))
        for name in batches[0].batch_metrics:
            metrics[name] = _mean([batch.batch_metrics[name] for batch in batches])
        metrics["opt/lr_policy"] = policy_lr
        metrics["opt/lr_value"] = self.config.effective_value_learning_rate
        metrics["opt/entropy_weight"] = entropy_weight_used
        metrics["teacher/ema_tau"] = 0.0 if self.teacher_ema_tau is None else self.teacher_ema_tau
        with self._profiler.window("learner/drift"):
            metrics["weights/drift_step"] = _relative_distance(
                self._policy_parameters, snapshot[0], self._reference_norm
            )
            metrics["weights/drift_total"] = _relative_distance(
                self._policy_parameters, self._reference, self._reference_norm
            )
        metrics["opt/grad_norm_policy"] = epochs[-1]["grad_norm"] if epochs else 0.0
        metrics["opt/grad_norm_value"] = _mean([batch.value_grad_norm for batch in batches])
        metrics["timing/learner_s"] = finished - started
        metrics["timing/learner_prepare_s"] = prepared - started
        metrics["timing/learner_ppo_s"] = finished - prepared
        metrics["learner/step"] = float(self._steps)
        self.last_metrics = metrics
        return metrics

    def _adapt(self, post: Mapping[str, float], reverted: bool) -> None:
        """The post-guard updates of a PPO-stage step (15 Sep 2026): the EMA anchor and the entropy
        controller on accepted steps, the KL-targeted learning rate on every step."""

        config = self.config
        if not reverted:
            if self.teacher_ema_tau is not None:
                with self._profiler.window("learner/ema"):
                    self._ema_update()
            if config.entropy_target is not None:
                weight = self.entropy_weight + config.entropy_gain * (config.entropy_target - post["entropy"])
                self.entropy_weight = min(max(weight, 0.0), config.entropy_weight_max)
        target = config.actor_kl_target
        if target is not None:
            actor_kl = post["actor_kl_mean"]
            if actor_kl > 2.0 * target:
                self.policy_lr = max(self.policy_lr / config.lr_adapt_factor, config.effective_lr_min)
            elif actor_kl < target / 2.0:
                self.policy_lr = min(self.policy_lr * config.lr_adapt_factor, config.effective_lr_max)

    # -- preparation: teacher, value net, behaviour log-probs --------------------------------

    def _behaviour_mode(self, trajectories: Sequence[Trajectory]) -> str:
        mode = self.config.ppo.behaviour_logits
        if mode == "auto":
            ring = any(trajectory.context_mode == "ring" for trajectory in trajectories)
            return "learner" if ring else "actor"
        return mode

    def _chunks(self, batch_size: int) -> list[slice]:
        microbatch = self.config.microbatch_envs
        size = batch_size if microbatch is None else min(microbatch, batch_size)
        return [slice(start, min(start + size, batch_size)) for start in range(0, batch_size, size)]

    def _decision_forward(
        self, model: PolicyProtocol, sub: Trajectory, *, prefix: bool
    ) -> dict[str, torch.Tensor]:
        """``model``'s logits over the decision rows ``[C, C+T-D)`` of ``sub``.

        With ``prefix`` the rows are forwarded alone against the actor's stored cache
        (``unroll_window(..., prefix=)``, INTERFACE.md patch #1); otherwise the ``C + T - D`` window is
        re-forwarded and its last rows kept.  The caller sets the grad mode.
        """

        if prefix:
            window = sub.prefix_window()
            assert window.targets is not None
            cache = sub.initial_cache
            assert cache is not None
            device = window.reset_mask.device
            fast = self._fast.get(id(model))
            if fast is not None:
                if cache.keys.device != device:  # prefix_on_host: pinned, non-blocking staging
                    assert self._stager is not None
                    with self._profiler.section("learner/prefix_copy"):
                        cache = self._stager.stage(cache)
                return fast.logits(
                    window.frames,
                    window.controller_prev,
                    window.targets,
                    window.reset_mask,
                    window.padding_mask,
                    cache,
                )
            if cache.keys.device != device:  # prefix_on_host: this microbatch's slice rides to the device
                with self._profiler.section("learner/prefix_copy"):
                    cache = cache_to(cache, device)
            output = model.unroll_window(
                window.frames,
                window.controller_prev,
                window.targets,
                window.reset_mask,
                window.padding_mask,
                prefix=cache,
            )
            return output.logits
        window = sub.policy_window()
        assert window.targets is not None
        output = model.unroll_window(
            window.frames,
            window.controller_prev,
            window.targets,
            window.reset_mask,
            window.padding_mask,
        )
        return _decision_logits(output.logits, sub.context_frames)

    def _policy_logits(self, sub: Trajectory) -> dict[str, torch.Tensor]:
        """The current policy's decision-row logits: the prefix path in ``prefix`` mode, else the window."""

        return self._decision_forward(self.policy, sub, prefix=sub.context_mode == "prefix")

    def teacher_delay_for(self, trajectory: Trajectory) -> int:
        """The teacher's delay for ``trajectory``: ``teacher_delay_frames``, or the trajectory's own ``D``."""

        return trajectory.delay_frames if self.teacher_delay_frames is None else self.teacher_delay_frames

    def _teacher_logits(self, sub: Trajectory) -> dict[str, torch.Tensor]:
        """The frozen teacher's decision-row logits.

        At its own delay (``teacher_delay_for(sub) != sub.delay_frames``) the teacher forwards
        ``sub.teacher_window`` eagerly and its logits are read at ``sub.teacher_decision_rows`` (the hindsight
        teacher); otherwise the prefix path with ``teacher_prefix``, else the student's window.
        """

        assert self.teacher is not None
        teacher_delay = self.teacher_delay_for(sub)
        if teacher_delay != sub.delay_frames:
            window = sub.teacher_window(teacher_delay)
            assert window.targets is not None
            output = self.teacher.unroll_window(
                window.frames,
                window.controller_prev,
                window.targets,
                window.reset_mask,
                window.padding_mask,
            )
            rows = sub.teacher_decision_rows(teacher_delay)
            return {name: value[:, rows] for name, value in output.logits.items()}
        use_prefix = self.config.teacher_prefix and sub.context_mode == "prefix"
        return self._decision_forward(self.teacher, sub, prefix=use_prefix)

    def _prepare(self, trajectory: Trajectory, mode: str) -> _Batch:
        profiler = self._profiler
        with profiler.section("learner/prepare/validate"):
            trajectory = trajectory.to(self.device, cache=not self.config.prefix_on_host)
            trajectory.validate()
            if self.config.reward is not None:
                trajectory = trajectory.recompute_rewards(self.config.reward)
        delay = trajectory.delay_frames
        batch = trajectory.batch_size
        chunks = self._chunks(batch)
        # The hindsight teacher (11 Sep 2026): the decisions whose teacher row lies in another game.
        teacher_keep: torch.Tensor | None = None
        if self.teacher is not None and self.teacher_delay_for(trajectory) != delay:
            if self.config.teacher_prefix:
                raise ValueError(
                    "learner.teacher_prefix needs the teacher at the trajectory's delay (the stored prefix "
                    "keys hold the student's pairing); a teacher at its own delay forwards its own window"
                )
            teacher_keep = trajectory.teacher_keep(self.teacher_delay_for(trajectory))
        # The per-chunk sub-trajectories once per step (every epoch re-sliced them before 5 Sep 2026).
        subs = [trajectory.index_envs(chunk) if len(chunks) > 1 else trajectory for chunk in chunks]
        teacher_parts: list[dict[str, torch.Tensor]] = []
        reference_parts: list[dict[str, torch.Tensor]] = []
        pre_parts: list[dict[str, torch.Tensor]] = []
        returns_parts: list[torch.Tensor] = []
        values_parts: list[torch.Tensor] = []
        advantage_parts: list[torch.Tensor] = []
        self.value_optimizer.zero_grad(set_to_none=True)
        for chunk, sub in zip(chunks, subs, strict=True):
            with profiler.window("learner/chunks/prepare"):
                scale = (chunk.stop - chunk.start) / batch
                with torch.no_grad():
                    if self.teacher is not None:
                        with profiler.section("learner/prepare/teacher"):
                            teacher_parts.append(self._teacher_logits(sub))
                    if self.reference is not None:
                        # The weight-0 reference reads the student's window at the run's delay (no prefix:
                        # its keys are another model's), like a teacher with ``teacher_prefix = false``.
                        with profiler.section("learner/prepare/reference"):
                            reference_parts.append(self._decision_forward(self.reference, sub, prefix=False))
                    if mode == "learner":
                        with (
                            profiler.section("learner/prepare/behaviour"),
                            matmul_precision(self.config.matmul_precision),
                        ):
                            pre_parts.append(self._policy_logits(sub))
                with profiler.section("learner/prepare/value_forward"):
                    values = self.value_net.forward_window(sub.value_window())
                with profiler.section("learner/prepare/value_backward"):
                    outputs = value_targets(sub, values, self.config.returns)
                    (outputs.loss() * scale).backward()
                returns_parts.append(outputs.returns)
                values_parts.append(outputs.rollout_values.detach())
                advantage_parts.append(outputs.advantages)
        with profiler.window("learner/prepare/tail"):
            with profiler.section("learner/prepare/value_step"):
                value_grad_norm = _grad_norm(list(self.value_net.parameters()))
                self.value_optimizer.step()

            with profiler.section("learner/prepare/value_metrics"):
                returns = torch.cat(returns_parts)
                rollout_values = torch.cat(values_parts)
                squared = (returns - rollout_values) ** 2
                value_metrics = {
                    "value/loss": float(squared.mean()),
                    "value/uev": float(unexplained_variance(squared, returns)),
                    "value/return_mean": float(returns.mean()),
                    "value/value_mean": float(rollout_values.mean()),
                }
            advantages = torch.cat(advantage_parts)[:, delay:]

            with profiler.section("learner/prepare/log_old"):
                labels = trajectory.actor_labels()
                actor_logits = trajectory.actor_logits()
                actor_kl_pre: float | None = None
                if mode == "learner":
                    old_logits = {
                        name: torch.cat([part[name] for part in pre_parts]) for name in actor_logits
                    }
                    actor_kl_pre = float(distributions.kl(actor_logits, old_logits).mean())
                else:
                    old_logits = actor_logits
                log_old = distributions.log_prob(old_logits, labels).detach()
                teacher_logits = None
                if teacher_parts:
                    teacher_logits = {
                        name: torch.cat([part[name] for part in teacher_parts]) for name in old_logits
                    }
                reference_logits = None
                if reference_parts:
                    reference_logits = {
                        name: torch.cat([part[name] for part in reference_parts]) for name in old_logits
                    }
            batch_metrics = dict(value_metrics)
            batch_metrics["hindsight/masked_fraction"] = (
                0.0 if teacher_keep is None else 1.0 - float(teacher_keep.float().mean())
            )
            teacher_entropy = 0.0
            if teacher_logits is not None:
                with torch.no_grad():
                    entropy_rows = distributions.entropy(teacher_logits)
                    if teacher_keep is None:
                        teacher_entropy = float(entropy_rows.mean())
                    else:
                        weight = teacher_keep.to(entropy_rows.dtype)
                        teacher_entropy = float((entropy_rows * weight).sum() / weight.sum().clamp_min(1.0))
            batch_metrics["hindsight/teacher_entropy"] = teacher_entropy
        return _Batch(
            trajectory=trajectory,
            chunks=chunks,
            subs=subs,
            labels=labels,
            log_old=log_old,
            old_logits={name: value.detach() for name, value in old_logits.items()},
            teacher_logits=teacher_logits,
            advantages=advantages.detach(),
            batch_metrics=batch_metrics,
            value_grad_norm=value_grad_norm,
            actor_kl_pre=actor_kl_pre,
            teacher_keep=teacher_keep,
            reference_logits=reference_logits,
        )

    # -- PPO epochs ------------------------------------------------------------------------

    def _loss(
        self,
        logits: Mapping[str, torch.Tensor],
        batch: _Batch,
        chunk: slice,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        config = self.config
        labels = labels_index(batch.labels, chunk)
        log_old = batch.log_old[chunk]
        advantages = batch.advantages[chunk]
        old_logits = {name: value[chunk] for name, value in batch.old_logits.items()}
        teacher_logits = (
            None
            if batch.teacher_logits is None
            else {name: value[chunk] for name, value in batch.teacher_logits.items()}
        )
        # One log_softmax per logits tensor (commit A of 5 Sep 2026); the labels were range-checked by
        # ``Trajectory.validate`` in ``_prepare``, so the per-pass host read is skipped.
        terms = distributions.ppo_terms(logits, labels, old_logits, teacher_logits, validate_labels=False)
        log_new = terms["log_prob"]
        log_rhos = log_new - log_old
        rhos = torch.exp(log_rhos)
        clipped_rhos = torch.exp(torch.clamp(log_rhos, -config.ppo.epsilon, config.ppo.epsilon))
        ppo_objective = torch.minimum(rhos * advantages, clipped_rhos * advantages)
        actor_kl = terms["actor_kl"]
        entropy = terms["entropy"]
        teacher_kl = terms["teacher_kl"]
        reverse_teacher_kl = terms["reverse_teacher_kl"]
        keep = None if batch.teacher_keep is None else batch.teacher_keep[chunk]
        if keep is not None:  # the hindsight teacher: decisions whose teacher row is in another game drop out
            weight = keep.to(teacher_kl.dtype)
            teacher_kl = teacher_kl * weight
            reverse_teacher_kl = reverse_teacher_kl * weight
        if teacher_logits is None:
            top1 = torch.zeros_like(entropy)
        else:
            agree = logits["buttons"].argmax(dim=-1) == teacher_logits["buttons"].argmax(dim=-1)
            top1 = agree.to(entropy.dtype)
        total = (
            -config.policy_gradient_weight * ppo_objective
            + config.ppo.beta * actor_kl
            + config.kl_teacher_weight * teacher_kl
            + config.reverse_kl_teacher_weight * reverse_teacher_kl
            - self.entropy_weight * entropy
        )
        elements = {
            "total": total.detach(),
            "ppo_objective": ppo_objective.detach(),
            "teacher_kl": teacher_kl.detach(),
            "reverse_teacher_kl": reverse_teacher_kl.detach(),
            "actor_kl": actor_kl.detach(),
            "entropy": entropy.detach(),
            "log_rho": log_rhos.detach(),
            "top1": top1.detach(),
        }
        if batch.reference_logits is not None:
            # Logged only (weight 0), so no graph: two log-softmaxes per component under no_grad.
            with torch.no_grad():
                reference = {name: value[chunk] for name, value in batch.reference_logits.items()}
                detached = {name: value.detach() for name, value in logits.items()}
                elements["reference_kl"] = distributions.kl(detached, reference)
                elements["reverse_reference_kl"] = distributions.kl(reference, detached)
        return total.mean(), elements

    def _epoch(self, batches: Sequence[_Batch], *, train: bool) -> dict[str, float]:
        profiler = self._profiler
        chunk_window = "learner/chunks/train" if train else "learner/chunks/post"
        accumulator = _Accumulator(self.config.ppo.epsilon)
        if train:
            self.policy_optimizer.zero_grad(set_to_none=True)
        rows = sum(batch.trajectory.batch_size for batch in batches)
        by_row = self.config.loss_weighting == "row"
        for batch in batches:
            trajectory = batch.trajectory
            size = trajectory.batch_size
            for chunk, sub in zip(batch.chunks, batch.subs, strict=True):
                with profiler.window(chunk_window):
                    if by_row:  # every decision row of the step weighs the same
                        scale = (chunk.stop - chunk.start) / rows
                    else:  # every trajectory weighs the same
                        scale = (chunk.stop - chunk.start) / size / len(batches)
                    with torch.set_grad_enabled(train), matmul_precision(self.config.matmul_precision):
                        with profiler.section("learner/epoch/forward"):
                            logits = self._policy_logits(sub)
                        with profiler.section("learner/epoch/loss"):
                            loss, elements = self._loss(logits, batch, chunk)
                        if train:
                            with profiler.section("learner/epoch/backward"):
                                (loss * scale).backward()
                    with profiler.section("learner/epoch/accumulate"):
                        keep = None if batch.teacher_keep is None else batch.teacher_keep[chunk]
                        accumulator.add(elements, keep)
        with profiler.window("learner/optimizer" if train else "learner/post_summary"):
            summary = accumulator.summary()
            if train:
                summary["grad_norm"] = _grad_norm(self._policy_parameters)
                self.policy_optimizer.step()
        return summary

    # -- guard: snapshot and revert ---------------------------------------------------------------

    def _snapshot(self) -> tuple[list[torch.Tensor], dict[str, Any]]:
        parameters = [parameter.detach().clone() for parameter in self._policy_parameters]
        return parameters, copy.deepcopy(self.policy_optimizer.state_dict())

    def _restore(self, snapshot: tuple[list[torch.Tensor], dict[str, Any]]) -> None:
        parameters, optimizer_state = snapshot
        with torch.no_grad():
            for parameter, saved in zip(self._policy_parameters, parameters, strict=True):
                parameter.copy_(saved)
        self.policy_optimizer.load_state_dict(optimizer_state)

    # -- state -------------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Policy and value weights, both optimizers, the step counter, the live entropy bonus and learning
        rate, and -- with an EMA anchor -- the anchor's weights (CPU copies)."""

        state: dict[str, Any] = {
            "policy": self.policy.get_state(),
            "value": _tree_cpu(self.value_net.state_dict()),
            "policy_optimizer": _tree_cpu(self.policy_optimizer.state_dict()),
            "value_optimizer": _tree_cpu(self.value_optimizer.state_dict()),
            "step": self._steps,
            "entropy_weight": float(self.entropy_weight),
            "policy_lr": float(self.policy_lr),
        }
        if self.teacher_ema_tau is not None:
            assert self.teacher is not None
            state["teacher"] = self.teacher.get_state()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.policy.set_state(state["policy"])
        self.value_net.load_state_dict(dict(state["value"]), strict=True)
        self.policy_optimizer.load_state_dict(copy.deepcopy(dict(state["policy_optimizer"])))
        self.value_optimizer.load_state_dict(copy.deepcopy(dict(state["value_optimizer"])))
        self._steps = int(state["step"])
        # The live controller values (15 Sep 2026); a checkpoint from before them keeps the config's.
        entropy_weight = state.get("entropy_weight")
        self.entropy_weight = self.config.entropy_weight if entropy_weight is None else float(entropy_weight)
        policy_lr = state.get("policy_lr")
        self.policy_lr = self.config.learning_rate if policy_lr is None else float(policy_lr)
        teacher_state = state.get("teacher")
        if teacher_state is not None:
            if self.teacher is None:
                raise ValueError("the checkpoint carries an EMA anchor but this learner has no teacher")
            self.teacher.set_state(teacher_state)
        # Re-base the drift reference: ``weights/drift_total`` means "since this launch started", so a
        # resumed run reports movement from its own starting weights rather than from a random init it
        # never had.  Movement across launches comes from diffing two ``step_<n>.pt`` snapshots.
        self._reference = [parameter.detach().clone() for parameter in self._policy_parameters]
        self._reference_norm = _parameter_norm(self._reference)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "BEHAVIOUR_LOGITS",
    "LOSS_WEIGHTINGS",
    "MATMUL_PRECISIONS",
    "STAGES",
    "LearnerConfig",
    "PPOConfig",
    "PPOLearner",
    "TrainablePolicy",
    "matmul_precision",
]
