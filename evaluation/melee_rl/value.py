"""``ValueNet``: a separate value function for PPO (PLAN.md §3.7, R3).

``ValueNet = SlippiEncoder(cfg) + CausalTransformer(cfg') + Linear(d_model, 1)`` where ``cfg'``
is the policy's ``ModelConfig`` with a small backbone (defaults: the 3m profile at 2 layers,
``d_model 128, n_heads 2, n_kv_heads 1, head_dim 64, d_ff 304``, ``config.yaml:84``), float32
compute, no gradient checkpointing and ``context_length >= W = C + T + 1`` -- about 2.3 M
parameters, most of them the fresh encoder's embedding tables.  It reads the trajectory's
value window (rows ``[0, C+T]`` with the *undelayed* pairing ``(frame, controller_exec)``,
PLAN.md §2.3) and is trained inside the learner step with its own Adam (slippi-ai
``rl/learner.py:194-203``), on squared errors to the discounted returns
``G_t = r_t + gamma_t G_{t+1}`` bootstrapped with ``V(s_T)`` (``tf/value_function.py:70-93``),
lambda = 1.  :func:`value_targets` builds those targets from a :class:`Trajectory`;
:class:`ValueOutputs` carries values, returns, advantages and the metrics
(``value/loss``, ``value/uev``, ``value/return_mean``, ``value/value_mean``).

Nothing in ``model.py`` is modified: ``SlippiEncoder`` and ``CausalTransformer`` are used
as library modules with a derived ``ModelConfig`` (``dataclasses.replace``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from controller_codec import ControllerLabels, CustomV1Codec
from melee_rl.rl_lib import ReturnsConfig, discounted_returns, unexplained_variance, value_discounts
from melee_rl.trajectory import Trajectory, Window
from model import CausalTransformer, ModelConfig, SlippiEncoder


@dataclass(frozen=True)
class ValueConfig:
    """``[learner.value]``: the value backbone (PLAN.md §3.7 defaults).

    ``context_length`` is the longest window the backbone accepts; ``None`` means the policy's
    ``context_length + 1`` (enough for any ``W = C + T + 1`` with ``C + T <= context_length``).
    """

    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    n_kv_heads: int = 1
    head_dim: int = 64
    d_ff: int = 304
    context_length: int | None = None

    def __post_init__(self) -> None:
        if min(self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.head_dim, self.d_ff) <= 0:
            raise ValueError("value backbone dimensions must be positive")
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("value d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("value n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("value head_dim must be even (RoPE)")
        if self.context_length is not None and self.context_length < 1:
            raise ValueError("value context_length must be >= 1")


def value_model_config(policy_config: ModelConfig, config: ValueConfig) -> ModelConfig:
    """The policy config with the value backbone dimensions, fp32 compute and no checkpointing."""

    context = policy_config.context_length + 1 if config.context_length is None else config.context_length
    return replace(
        policy_config,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        d_ff=config.d_ff,
        context_length=context,
        compute_dtype="float32",
        cache_dtype="float32",
        gradient_checkpointing=False,
    )


def _as_mask(value: Any, shape: tuple[int, ...], name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a bool tensor of shape {shape}")
    return value.to(device=device, dtype=torch.bool)


class ValueNet(nn.Module):
    """Encoder + small causal transformer + linear head; ``forward`` returns ``[B, L]`` values."""

    def __init__(self, policy_config: ModelConfig, config: ValueConfig | None = None) -> None:
        super().__init__()
        self.config = ValueConfig() if config is None else config
        self.model_config = value_model_config(policy_config, self.config)
        self.codec = CustomV1Codec()
        self.encoder = SlippiEncoder(self.model_config, self.codec)
        self.backbone = CausalTransformer(self.model_config, input_dim=self.encoder.output_dim)
        self.head = nn.Linear(self.model_config.d_model, 1, bias=True)
        # Zero head: the value starts at 0 everywhere (advantages == returns on the first step).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def context_length(self) -> int:
        return self.model_config.context_length

    @property
    def device(self) -> torch.device:
        return self.head.weight.device

    def forward(
        self,
        frames: Mapping[str, Any],
        controller: ControllerLabels,
        reset_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Values ``[B, L]`` (float32) for ``L <= context_length`` rows; padded rows give the head bias."""

        if not isinstance(controller, ControllerLabels) or controller.buttons.ndim != 2:
            raise ValueError("controller must be ControllerLabels with [B, L] components")
        batch, length = controller.buttons.shape
        if length > self.context_length:
            raise ValueError(f"window length {length} exceeds the value context_length {self.context_length}")
        device = self.device
        labels = ControllerLabels(
            buttons=controller.buttons.to(device=device, dtype=torch.long),
            main_stick=controller.main_stick.to(device=device, dtype=torch.long),
        )
        reset = _as_mask(reset_mask, (batch, length), "reset_mask", device)
        padding = _as_mask(padding_mask, (batch, length), "padding_mask", device)
        encoded = self.encoder(frames, labels)
        hidden, _ = self.backbone(encoded, reset_mask=reset, padding_mask=padding)
        values: torch.Tensor = self.head(hidden.float()).squeeze(-1)
        return values

    def forward_window(self, window: Window) -> torch.Tensor:
        """:meth:`forward` on a :class:`Window` (``Trajectory.value_window()``)."""

        return self.forward(window.frames, window.controller_prev, window.reset_mask, window.padding_mask)


@dataclass(frozen=True)
class ValueOutputs:
    """Values over the rollout rows and their targets.

    ``values`` is ``[B, T+1]`` (rows ``C..C+T``, the last one the bootstrap state) *with*
    gradients; ``returns``, ``advantages`` and ``discounts`` are detached ``[B, T]``.
    """

    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    discounts: torch.Tensor

    @property
    def rollout_values(self) -> torch.Tensor:
        """``values[:, :T]`` -- the rows with a target."""

        return self.values[:, :-1]

    @property
    def bootstrap(self) -> torch.Tensor:
        return self.values[:, -1]

    def loss(self) -> torch.Tensor:
        """Mean squared error to the returns over ``[B, T]`` (``tf/value_function.py:90``)."""

        return ((self.returns - self.rollout_values) ** 2).mean()

    def metrics(self) -> dict[str, float]:
        with torch.no_grad():
            squared = (self.returns - self.rollout_values) ** 2
            return {
                "value/loss": float(squared.mean()),
                "value/uev": float(unexplained_variance(squared, self.returns)),
                "value/return_mean": float(self.returns.mean()),
                "value/value_mean": float(self.rollout_values.mean()),
            }


def value_targets(trajectory: Trajectory, values_window: torch.Tensor, config: ReturnsConfig) -> ValueOutputs:
    """Returns and advantages of ``trajectory`` from the value window's values ``[B, W]``.

    Bootstraps with the detached value of row ``C+T``; discounts per :func:`value_discounts`
    (zero into reset rows, ``discount_on_death`` on respawns); advantages ``G - V`` (detached).
    """

    context, length = trajectory.context_frames, trajectory.rollout_length
    if tuple(values_window.shape) != (trajectory.batch_size, trajectory.window_length):
        raise ValueError(
            f"values_window must be [{trajectory.batch_size}, {trajectory.window_length}], "
            f"got {tuple(values_window.shape)}"
        )
    values = values_window[:, context:]
    rewards = trajectory.rewards.to(values.device, torch.float32)
    frames = trajectory.rollout_frames() if config.discount_on_death is not None else None
    discounts = value_discounts(
        config,
        rewards,
        is_resetting=trajectory.is_resetting[:, context:].to(values.device),
        frames=frames,
    )
    returns = discounted_returns(rewards, discounts, values[:, length].detach()).detach()
    advantages = (returns - values[:, :length].detach()).detach()
    return ValueOutputs(values=values, returns=returns, advantages=advantages, discounts=discounts)


__all__ = ["ValueConfig", "ValueNet", "ValueOutputs", "value_model_config", "value_targets"]
