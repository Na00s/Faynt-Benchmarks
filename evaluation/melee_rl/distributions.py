"""Closed-form log-probability, KL and entropy over ``custom_v1`` component logits.

PLAN.md §3.1 (R5).  For taken labels ``a = (a_b, a_m)`` and logits produced under teacher
forcing on ``a_b`` (``model.py:1777-1788``: the main-stick logits are conditioned on the
*taken* buttons):

* ``log pi(a)  = log_softmax(l_b)[a_b] + log_softmax(l_m | a_b)[a_m]`` -- exact; it is
  ``-loss`` of ``AutoregressiveControllerHead.loss`` (``model.py:1860-1869``);
* ``KL(pi || q) = KL(pi_b || q_b) + KL(pi_m(*|a_b) || q_m(*|a_b))`` -- exact first term, a
  one-sample Monte-Carlo estimate of ``E_{a_b~pi_b} KL(pi_m(*|a_b) || q_m(*|a_b))`` for the
  second; ``pi`` is the first argument;
* ``H(pi)     = H(pi_b) + H(pi_m(*|a_b))`` likewise.

This is the per-component surrogate of slippi-ai's TF learner, which sums the
per-component quantities of categorical distributions built from teacher-forced logits
(``slippi_ai/rl/learner.py:164-177``: ``_compute_kl`` / ``_get_log_prob`` /
``_compute_entropy``).  Nothing is ported; the formulas are re-derived in PyTorch.
Every function works in float32 from ``logits.float()`` (PLAN.md §3.9) and returns
per-position tensors shaped like the logits' prefix ``[...]`` (``[B, T]`` in the learner);
masking and reduction are the caller's job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from controller_codec import COMPONENT_ORDER, ControllerLabels

Logits = Mapping[str, torch.Tensor]
"""``{component: float [..., V_component]}`` -- keys exactly :data:`COMPONENT_ORDER`."""

LabelsLike = ControllerLabels | Mapping[str, torch.Tensor]


def _ordered_logits(logits: Logits) -> list[tuple[str, torch.Tensor]]:
    missing = [name for name in COMPONENT_ORDER if name not in logits]
    if missing:
        raise KeyError(f"logits are missing components {missing}; expected {COMPONENT_ORDER!r}")
    ordered: list[tuple[str, torch.Tensor]] = []
    prefix: torch.Size | None = None
    for name in COMPONENT_ORDER:
        value = logits[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} logits must be a tensor, got {type(value).__name__}")
        if value.ndim < 1 or value.shape[-1] < 1:
            raise ValueError(f"{name} logits must have shape [..., V] with V >= 1, got {tuple(value.shape)}")
        if prefix is None:
            prefix = value.shape[:-1]
        elif value.shape[:-1] != prefix:
            raise ValueError(
                f"all component logits must share the prefix shape {tuple(prefix)}; "
                f"{name} has {tuple(value.shape[:-1])}"
            )
        ordered.append((name, value.float()))
    return ordered


def _ordered_labels(
    labels: LabelsLike,
    ordered_logits: list[tuple[str, torch.Tensor]],
    *,
    check_range: bool = True,
) -> dict[str, torch.Tensor]:
    if isinstance(labels, ControllerLabels):
        values: Mapping[str, torch.Tensor] = labels.as_dict()
    else:
        missing = [name for name in COMPONENT_ORDER if name not in labels]
        if missing:
            raise KeyError(f"labels are missing components {missing}; expected {COMPONENT_ORDER!r}")
        values = labels
    result: dict[str, torch.Tensor] = {}
    for name, component_logits in ordered_logits:
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} labels must be a tensor, got {type(value).__name__}")
        if value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} labels must have an integer dtype, got {value.dtype}")
        if value.shape != component_logits.shape[:-1]:
            raise ValueError(
                f"{name} labels must have shape {tuple(component_logits.shape[:-1])}, "
                f"got {tuple(value.shape)}"
            )
        vocabulary = component_logits.shape[-1]
        if check_range and bool(((value < 0) | (value >= vocabulary)).any()):
            raise ValueError(f"{name} labels must be in [0, {vocabulary})")
        result[name] = value.long().to(component_logits.device)
    return result


def sum_components(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Sum per-component tensors in :data:`COMPONENT_ORDER`."""

    total: torch.Tensor | None = None
    for name in COMPONENT_ORDER:
        value = values[name]
        total = value if total is None else total + value
    if total is None:
        raise ValueError("no components to sum")
    return total


def log_probs_by_component(logits: Logits, labels: LabelsLike) -> dict[str, torch.Tensor]:
    """``log_softmax(l_c)[a_c]`` per component, shaped like the logits' prefix."""

    ordered = _ordered_logits(logits)
    taken = _ordered_labels(labels, ordered)
    return {
        name: F.log_softmax(component_logits, dim=-1).gather(-1, taken[name].unsqueeze(-1)).squeeze(-1)
        for name, component_logits in ordered
    }


def log_prob(logits: Logits, labels: LabelsLike) -> torch.Tensor:
    """``log pi(a)`` summed over components (exact under teacher forcing on the taken prefix)."""

    return sum_components(log_probs_by_component(logits, labels))


def entropy_by_component(logits: Logits) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, component_logits in _ordered_logits(logits):
        log_p = F.log_softmax(component_logits, dim=-1)
        result[name] = -(log_p.exp() * log_p).sum(dim=-1)
    return result


def entropy(logits: Logits) -> torch.Tensor:
    """``H(pi_b) + H(pi_m(*|a_b))`` in nats."""

    return sum_components(entropy_by_component(logits))


def kl_by_component(logits_p: Logits, logits_q: Logits) -> dict[str, torch.Tensor]:
    """``KL(p_c || q_c)`` per component; finite logits assumed (no masked actions)."""

    ordered_p = _ordered_logits(logits_p)
    ordered_q = _ordered_logits(logits_q)
    result: dict[str, torch.Tensor] = {}
    for (name, raw_p), (_, raw_q) in zip(ordered_p, ordered_q, strict=True):
        if raw_p.shape != raw_q.shape:
            raise ValueError(
                f"{name} logits of p and q must have the same shape, got {raw_p.shape} and {raw_q.shape}"
            )
        log_p = F.log_softmax(raw_p, dim=-1)
        log_q = F.log_softmax(raw_q, dim=-1)
        result[name] = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    return result


def kl(logits_p: Logits, logits_q: Logits) -> torch.Tensor:
    """Forward KL ``KL(p || q)`` summed over components (``p`` first, e.g. ``kl(policy, teacher)``)."""

    return sum_components(kl_by_component(logits_p, logits_q))


PPO_TERMS: tuple[str, ...] = ("log_prob", "entropy", "actor_kl", "teacher_kl", "reverse_teacher_kl")
"""The keys of :func:`ppo_terms`."""


def ppo_terms(
    logits: Logits,
    labels: LabelsLike,
    old_logits: Logits,
    teacher_logits: Logits | None = None,
    *,
    validate_labels: bool = True,
) -> dict[str, torch.Tensor]:
    """The learner's five per-position quantities from one ``log_softmax`` per logits tensor.

    ``log_prob`` = :func:`log_prob` of the new ``logits``, ``entropy`` = :func:`entropy`, ``actor_kl`` =
    ``kl(old_logits, logits)``, ``teacher_kl`` = ``kl(logits, teacher_logits)`` and ``reverse_teacher_kl`` =
    ``kl(teacher_logits, logits)`` (zeros without a teacher), each summed over the components.  The separate
    functions take ``log_softmax`` of the new logits five times; this takes it once per component and uses the
    very same per-term formulas, so the values are bit-identical (``tests/rl/test_distributions.py``) -- fewer
    kernels, fewer autograd nodes, the same numbers.  ``validate_labels=False`` skips the label range check (a
    host round trip per component) for labels a ``Trajectory.validate`` already checked (5 Sep 2026,
    ``/rl-learner-speedup``).
    """

    ordered = _ordered_logits(logits)
    ordered_old = _ordered_logits(old_logits)
    ordered_teacher = None if teacher_logits is None else _ordered_logits(teacher_logits)
    taken = _ordered_labels(labels, ordered, check_range=validate_labels)
    parts: dict[str, dict[str, torch.Tensor]] = {name: {} for name in PPO_TERMS}
    for index, (name, new) in enumerate(ordered):
        old = ordered_old[index][1]
        if old.shape != new.shape:
            raise ValueError(
                f"{name} logits of old and new must have the same shape, got {old.shape} and {new.shape}"
            )
        log_new = F.log_softmax(new, dim=-1)
        p_new = log_new.exp()
        log_old = F.log_softmax(old, dim=-1)
        parts["log_prob"][name] = log_new.gather(-1, taken[name].unsqueeze(-1)).squeeze(-1)
        parts["entropy"][name] = -(p_new * log_new).sum(dim=-1)
        parts["actor_kl"][name] = (log_old.exp() * (log_old - log_new)).sum(dim=-1)
        if ordered_teacher is None:
            zeros = torch.zeros_like(parts["log_prob"][name])
            parts["teacher_kl"][name] = zeros
            parts["reverse_teacher_kl"][name] = zeros
            continue
        teacher = ordered_teacher[index][1]
        if teacher.shape != new.shape:
            raise ValueError(
                f"{name} logits of the teacher and new must have the same shape, "
                f"got {teacher.shape} and {new.shape}"
            )
        log_teacher = F.log_softmax(teacher, dim=-1)
        parts["teacher_kl"][name] = (p_new * (log_new - log_teacher)).sum(dim=-1)
        parts["reverse_teacher_kl"][name] = (log_teacher.exp() * (log_teacher - log_new)).sum(dim=-1)
    return {name: sum_components(values) for name, values in parts.items()}


def sample(
    logits: Logits,
    *,
    temperature: float = 1.0,
    generator: torch.Generator | None = None,
) -> ControllerLabels:
    """Sample each component from *finalized* logits (``softmax(l / temperature)``; 0 = argmax).

    The components are treated as already conditioned -- the autoregressive conditioning
    is ``AutoregressiveControllerHead.sample``'s job (``model.py:1813-1836``), which the
    actor uses.  This mirrors ``CustomV1Codec.sample`` operation-for-operation so the two
    agree for a shared generator state.
    """

    if not math.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative")
    samples: dict[str, torch.Tensor] = {}
    for name, component_logits in _ordered_logits(logits):
        if temperature == 0.0:
            samples[name] = component_logits.argmax(dim=-1)
            continue
        probabilities = torch.softmax(component_logits / temperature, dim=-1)
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        samples[name] = torch.multinomial(flat, 1, generator=generator).reshape(component_logits.shape[:-1])
    return ControllerLabels(buttons=samples["buttons"], main_stick=samples["main_stick"])


__all__ = [
    "PPO_TERMS",
    "LabelsLike",
    "Logits",
    "entropy",
    "entropy_by_component",
    "kl",
    "kl_by_component",
    "log_prob",
    "log_probs_by_component",
    "ppo_terms",
    "sample",
    "sum_components",
]
