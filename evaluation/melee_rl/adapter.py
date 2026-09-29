"""``MeleePolicyAdapter``: Ali's ``MeleePolicy`` behind ``PolicyProtocol`` (INTERFACE.md §4-5).

This is the only module of ``melee_rl`` that touches ``MeleePolicy`` internals
(``encoder`` / ``backbone`` / ``controller_head``).  Nothing in ``model.py`` is
modified: the adapter is a subclass that adds no parameters, so its ``state_dict`` keys
are exactly ``MeleePolicy``'s (BC checkpoints load with ``strict=True`` and the export of
PLAN.md §3.3 is trivial).

Two paths (PLAN.md §2.2):

* actor -- ``encoder -> backbone.step -> controller_head.sample`` under ``torch.no_grad()``
  in ``eval()`` mode: ``MeleePolicy.forward(use_cache=True)`` returns argmax labels
  (``model.py:1785-1786``), so it is bypassed for sampling; ``prime`` is
  ``backbone.forward(..., cache=cache, use_cache=True)`` (``model.py:1586-1603``) after
  clearing every slot;
* learner -- ``MeleePolicy.forward(game_state, controller_t=controller_prev, reset_mask,
  padding_mask, controller_t_plus_1=targets)`` so the main-stick logits are conditioned on
  the *taken* buttons (``model.py:1777-1788``).

Exactness (PLAN.md §3.2, tested in ``tests/rl/test_adapter.py``): ``prime(C frames)`` +
``T x step_sample`` reproduces ``unroll_window(C + T frames)`` in float32 -- same RoPE
positions (``model.py:1384-1390`` vs ``:1505, :1552``), same segment and padding semantics
(``:1412-1416`` vs ``:1165-1173, :1504``).  Padded rows neither write the cache nor
advance positions (``:1507, :1539-1552``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from melee_rl.fast_step import FastStepper
    from melee_rl.graph_step import GraphedStep

from controller_codec import ControllerLabels
from melee_rl.frames import frame_tree_to_game_state, game_state_to_tree, leading_shape, tree_index, tree_to
from melee_rl.packed import StaticFrameTree
from melee_rl.protocol import ActionSpec, Frames, PolicyProtocol, StepOutput, WindowOutput
from model import KVCache, MeleePolicy, ModelConfig, _autocast_context
from tensor_batch import GameStateBatch

RL_PRECISION: Mapping[str, str] = {"compute_dtype": "float32", "cache_dtype": "float32"}
"""RL default precision (PLAN.md §3.9): Ali's bf16 autocast also applies on CPU (``model.py:421-426``)."""


def rl_model_config(config: ModelConfig, **changes: Any) -> ModelConfig:
    """``config`` with the RL precision defaults, then ``changes`` (``dataclasses.replace``)."""

    return replace(config, **{**RL_PRECISION, **changes})


def _as_mask(value: Any, shape: tuple[int, ...], name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a bool tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    return value.to(device=device, dtype=torch.bool)


def _as_labels(value: Any, ndim: int, name: str, device: torch.device) -> ControllerLabels:
    if not isinstance(value, ControllerLabels):
        raise TypeError(f"{name} must be ControllerLabels, got {type(value).__name__}")
    if value.buttons.shape != value.main_stick.shape:
        raise ValueError(f"{name} component shapes differ: {value.buttons.shape} vs {value.main_stick.shape}")
    if value.buttons.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} leading dimension(s), got shape {tuple(value.buttons.shape)}"
        )
    for component in value.values():
        if component.is_floating_point() or component.is_complex():
            raise TypeError(f"{name} must have integer labels, got {component.dtype}")
    return ControllerLabels(
        buttons=value.buttons.to(device=device, dtype=torch.long),
        main_stick=value.main_stick.to(device=device, dtype=torch.long),
    )


class MeleePolicyAdapter(MeleePolicy):
    """``MeleePolicy`` with the ``PolicyProtocol`` methods; ``from_policy`` wraps an existing instance."""

    fast_step: bool = False
    """Route :meth:`step_sample` through the vectorised ring step (``melee_rl.fast_step``,
    P7-PLAN.md 3).  Plain attribute (never in ``state_dict``); ``from_policy`` copies it, so
    ``clone_frozen`` teachers, self-play opponents and eval clones inherit the setting."""

    fast_attention: str = "sdpa"
    """The fast step's attention: ``"sdpa"`` (Ali's) or ``"lean"`` (``melee_rl.fast_step``, the copy-free
    per-kv-head matmuls; 5 Sep 2026).  Plain attribute like :attr:`fast_step`; ``from_policy`` copies it."""

    fast_encoder: bool = False
    """Encode the actor's frames with ``melee_rl.fast_encode.fast_encode`` (bit-identical to Ali's encoder,
    no host round trips; 5 Sep 2026).  Plain attribute; ``from_policy`` copies it."""

    graph_step: bool = False
    """Replay the frame step from a CUDA graph (``melee_rl.graph_step``, 5 Sep 2026): captured per cache the
    first time :meth:`step_sample` sees a :class:`~melee_rl.packed.StaticFrameTree` on a CUDA device with
    ``fast_step`` and ``fast_encoder`` on; every other call takes the eager fast path.  Plain attribute;
    ``from_policy`` copies it."""

    _fast_stepper: FastStepper | None = None

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._graphs: dict[int, GraphedStep] = {}

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml", profile: str | None = None) -> MeleePolicyAdapter:
        return cls(ModelConfig.from_yaml(path, profile=profile))

    @classmethod
    def from_policy(cls, policy: MeleePolicy) -> MeleePolicyAdapter:
        """A new adapter with ``policy``'s config and a copy of its weights (same device and mode)."""

        adapter = cls(policy.config)
        adapter.to(next(policy.parameters()).device)
        adapter.load_state_dict(policy.state_dict(), strict=True)
        adapter.train(policy.training)
        adapter.fast_step = bool(getattr(policy, "fast_step", False))
        adapter.fast_attention = str(getattr(policy, "fast_attention", "sdpa"))
        adapter.fast_encoder = bool(getattr(policy, "fast_encoder", False))
        adapter.graph_step = bool(getattr(policy, "graph_step", False))
        return adapter

    # -- static facts -------------------------------------------------------

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(tuple(self.codec.component_order), tuple(self.codec.vocabulary_sizes))

    @property
    def delay_offset(self) -> int:
        return self.config.action_offset_frames

    @property
    def context_length(self) -> int:
        return self.config.context_length

    @property
    def device(self) -> torch.device:
        return self.backbone.input_projection.weight.device

    # -- helpers --------------------------------------------------------------

    @contextlib.contextmanager
    def _inference(self) -> Iterator[None]:
        """``eval()`` + ``no_grad`` + Ali's autocast for the cache path; restores the mode after.

        Idempotent: when the adapter is already in ``eval()`` (the rollout loop hoists the mode
        switch around the whole rollout, P7-PLAN.md 3.2), the 100+ module walks of ``eval()`` /
        ``train()`` are skipped.
        """

        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad(), _autocast_context(self.device, self.config.torch_compute_dtype):
                yield
        finally:
            if was_training:
                self.train(True)

    def _fast(self, cache: KVCache) -> FastStepper:
        """The :class:`FastStepper` bound to ``cache`` (rebuilt when the cache object changes)."""

        stepper = self._fast_stepper
        if stepper is None or stepper.cache is not cache or stepper.attention != self.fast_attention:
            from melee_rl.fast_step import FastStepper

            stepper = FastStepper(self.backbone, cache, attention=self.fast_attention)
            self._fast_stepper = stepper
        return stepper

    def _graphed(self, cache: KVCache, frame: StaticFrameTree) -> GraphedStep:
        """The captured step for ``cache`` (one graph per cache; re-captured if its static tree changes)."""

        graph = self._graphs.get(id(cache))
        if graph is None or not graph.matches(cache, frame):
            from melee_rl.graph_step import GraphedStep

            graph = GraphedStep(self, cache, frame)
            self._graphs[id(cache)] = graph
        return graph

    def _encode(self, frame: Frames, labels: ControllerLabels) -> torch.Tensor:
        """The encoder forward of the actor's step: Ali's, or the sync-free re-orchestration."""

        if self.fast_encoder:
            from melee_rl.fast_encode import fast_encode

            return fast_encode(self.encoder, frame, labels)
        return self.encoder(frame, labels)

    def _float_logits(self, logits: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: logits[name].float() for name in self.codec.component_order}

    # -- protocol ---------------------------------------------------------------

    def encode_observation(self, obs: Mapping[str, Any]) -> GameStateBatch:
        return frame_tree_to_game_state(tree_to(obs, self.device))

    def init_cache(self, batch_size: int) -> KVCache:
        return self.backbone.init_cache(batch_size, device=self.device, dtype=self.config.torch_cache_dtype)

    def reset_slots(self, cache: KVCache, reset_mask: torch.Tensor) -> None:
        cache.reset_slots(_as_mask(reset_mask, (cache.batch_size,), "reset_mask", cache.keys.device))

    def prime(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        cache: KVCache,
    ) -> None:
        labels = _as_labels(controller_prev, 2, "controller_prev", self.device)
        batch, length = labels.buttons.shape
        if batch != cache.batch_size:
            raise ValueError(
                f"controller_prev batch {batch} does not match the cache batch {cache.batch_size}"
            )
        if length > self.context_length:
            raise ValueError(f"prime length {length} exceeds context_length {self.context_length}")
        reset = _as_mask(reset_mask, (batch, length), "reset_mask", self.device)
        padding = _as_mask(padding_mask, (batch, length), "padding_mask", self.device)
        cache.reset_slots(torch.ones(batch, dtype=torch.bool, device=cache.keys.device))
        if length == 0:
            return
        with self._inference():
            encoded = self.encoder(frames, labels)
            self.backbone(encoded, reset_mask=reset, padding_mask=padding, cache=cache, use_cache=True)

    def step_sample(
        self,
        frame: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        cache: KVCache,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> StepOutput:
        labels = _as_labels(controller_prev, 1, "controller_prev", self.device)
        (batch,) = labels.buttons.shape
        reset = _as_mask(reset_mask, (batch,), "reset_mask", self.device)
        with self._inference():
            if (
                self.graph_step
                and self.fast_step
                and self.fast_encoder
                and isinstance(frame, StaticFrameTree)
                and self.device.type == "cuda"
            ):
                return self._graphed(cache, frame).step(
                    labels, reset, temperature=temperature, generator=generator
                )
            encoded = self._encode(frame, labels)
            if self.fast_step:
                from melee_rl.fast_step import fast_sample

                hidden = self._fast(cache).step(encoded, reset)
                sampled, logits = fast_sample(
                    self.controller_head,
                    hidden,
                    labels,
                    temperature=temperature,
                    generator=generator,
                )
                return StepOutput(labels=sampled, logits=logits)
            hidden, _ = self.backbone.step(encoded, cache, reset)
            outputs = self.controller_head.sample(
                hidden,
                labels,
                temperature=temperature,
                generator=generator,
            )
        return StepOutput(labels=outputs.labels, logits=self._float_logits(outputs.logits))

    def step_sample_multi(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        reset_mask: torch.Tensor,
        cache: KVCache,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[StepOutput]:
        """``b`` frames through the cache in one inference context (P8-PLAN.md 3.2).

        Bit-identical to ``b`` sequential :meth:`step_sample` calls (labels, logits, cache,
        generator state -- ``tests/rl/test_multi_step.py``).  The encoder consumes each frame's
        previous-action labels, so the frames are encoded one by one; the win is a single
        eval / autocast context and packed ``[B, b]`` inputs, not fused FLOPs.
        ``controller_prev`` is raw: frame ``i``'s input is the running chain (the caller's
        labels for ``i = 0``, else frame ``i - 1``'s sample) masked at that frame's reset rows.
        """

        labels = _as_labels(controller_prev, 1, "controller_prev", self.device)
        (batch,) = labels.buttons.shape
        if not isinstance(reset_mask, torch.Tensor) or reset_mask.ndim != 2:
            raise ValueError(
                f"reset_mask must be a [B, steps] bool tensor, got "
                f"{tuple(reset_mask.shape) if isinstance(reset_mask, torch.Tensor) else type(reset_mask)}"
            )
        steps = int(reset_mask.shape[1])
        if steps < 1:
            raise ValueError("step_sample_multi needs at least one frame")
        reset = _as_mask(reset_mask, (batch, steps), "reset_mask", self.device)
        tree = game_state_to_tree(frames) if isinstance(frames, GameStateBatch) else frames
        shape = leading_shape(tree)
        if tuple(shape) != (batch, steps):
            raise ValueError(f"frames must have leading shape {(batch, steps)}, got {tuple(shape)}")
        outputs: list[StepOutput] = []
        chain = labels
        with self._inference():
            for index in range(steps):
                keep = ~reset[:, index]
                prev = ControllerLabels(
                    buttons=torch.where(keep, chain.buttons, torch.zeros_like(chain.buttons)),
                    main_stick=torch.where(keep, chain.main_stick, torch.zeros_like(chain.main_stick)),
                )
                frame = tree_index(tree, (slice(None), index))
                encoded = self._encode(frame, prev)
                if self.fast_step:
                    from melee_rl.fast_step import fast_sample

                    hidden = self._fast(cache).step(encoded, reset[:, index])
                    sampled, logits = fast_sample(
                        self.controller_head, hidden, prev, temperature=temperature, generator=generator
                    )
                    step = StepOutput(labels=sampled, logits=logits)
                else:
                    hidden, _ = self.backbone.step(encoded, cache, reset[:, index])
                    head = self.controller_head.sample(
                        hidden, prev, temperature=temperature, generator=generator
                    )
                    step = StepOutput(labels=head.labels, logits=self._float_logits(head.logits))
                outputs.append(step)
                chain = step.labels
        return outputs

    def unroll_window(
        self,
        frames: Frames,
        controller_prev: ControllerLabels,
        targets: ControllerLabels,
        reset_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        prefix: KVCache | None = None,
    ) -> WindowOutput:
        """Teacher-forced logits over ``[B, L]`` rows; with ``prefix`` (a snapshot of the actor's ring,
        ``melee_rl.kv_cache``) the rows continue that history exactly (``model.py`` patch #1)."""

        labels = _as_labels(controller_prev, 2, "controller_prev", self.device)
        taken = _as_labels(targets, 2, "targets", self.device)
        batch, length = labels.buttons.shape
        if taken.buttons.shape != labels.buttons.shape:
            raise ValueError(f"targets must have shape {(batch, length)}, got {tuple(taken.buttons.shape)}")
        if length > self.context_length:
            raise ValueError(f"window length {length} exceeds context_length {self.context_length}")
        reset = _as_mask(reset_mask, (batch, length), "reset_mask", self.device)
        padding = _as_mask(padding_mask, (batch, length), "padding_mask", self.device)
        if prefix is not None and prefix.batch_size != batch:
            raise ValueError(f"prefix batch {prefix.batch_size} does not match the window batch {batch}")
        outputs, _ = self(
            frames,
            labels,
            reset_mask=reset,
            padding_mask=padding,
            controller_t_plus_1=taken,
            prefix=prefix,
        )
        return WindowOutput(logits=self._float_logits(outputs.logits), hidden=outputs.hidden)

    def get_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().to("cpu", copy=True) for name, value in self.state_dict().items()}

    def set_state(self, state: Mapping[str, torch.Tensor]) -> None:
        self.load_state_dict(dict(state), strict=True)

    def clone_frozen(self) -> MeleePolicyAdapter:
        """A weight copy in ``eval()`` with gradients disabled (teacher / self-opponent).

        Built as a fresh instance plus ``load_state_dict`` rather than ``copy.deepcopy``:
        ``MeleePolicy`` is not deep-copyable because ``CustomV1Codec`` stores its vocabulary
        sizes in a ``MappingProxyType`` (``controller_codec.py:598``), which ``copy`` cannot
        pickle (INTERFACE.md §6, patch #7).
        """

        clone = type(self).from_policy(self)
        clone.eval()
        clone.requires_grad_(False)
        return clone


def as_policy(adapter: MeleePolicyAdapter) -> PolicyProtocol:
    """Static proof that the adapter satisfies :class:`PolicyProtocol` (checked by mypy)."""

    return adapter


__all__ = ["RL_PRECISION", "MeleePolicyAdapter", "as_policy", "rl_model_config"]
