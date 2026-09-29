"""Adapter tests: shapes, sampling, actor/learner parity (PLAN.md §3.2), ring-mode caveat."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from controller_codec import ControllerLabels, CustomV1Codec
from melee_rl import distributions
from melee_rl.adapter import MeleePolicyAdapter, as_policy
from melee_rl.frames import (
    FrameTree,
    labels_index,
    labels_map,
    labels_stack,
    neutral_frame,
    neutral_labels,
    random_valid_frames,
    tree_index,
    tree_map,
)
from melee_rl.kv_cache import snapshot_cache
from melee_rl.protocol import ActionSpec, PolicyProtocol, StepOutput, WindowOutput
from model import KVCache, MeleePolicy, ModelConfig

B = 3
C = 16
T = 8


# ---------------------------------------------------------------------------
# Synthetic streams and a minimal actor loop driven through the protocol
# ---------------------------------------------------------------------------


@dataclass
class Streams:
    """Per-env synthetic data for ``rollouts x T`` global rows.

    ``history_start[b, r]`` is the first global row held by env ``b``'s history ring at
    the start of rollout ``r``; earlier rows are left padding (``padding_mask`` false,
    neutral frame) in that rollout's window.
    """

    frames: FrameTree
    is_resetting: torch.Tensor
    history_start: torch.Tensor


@dataclass
class ActorRecord:
    prev: ControllerLabels
    samples: ControllerLabels
    logits: dict[str, torch.Tensor]


def _make_streams(rollouts: int, seed: int) -> Streams:
    total = rollouts * T
    frames = random_valid_frames((B, total), torch.Generator().manual_seed(seed))
    is_resetting = torch.zeros((B, total), dtype=torch.bool)
    is_resetting[:, 0] = True  # every env starts with a game start
    is_resetting[1, 11] = True  # env 1: a new game begins inside rollout 1 (window row C + 3)
    is_resetting[2, 8] = True  # env 2: restarted at the first row of rollout 1 ...
    history_start = torch.zeros((B, rollouts), dtype=torch.long)
    history_start[2, 1:] = 8  # ... and its history ring was cleared: rows < 8 become padding
    return Streams(frames=frames, is_resetting=is_resetting, history_start=history_start)


def _record(
    prev_rows: list[ControllerLabels],
    sample_rows: list[ControllerLabels],
    logit_rows: list[dict[str, torch.Tensor]],
) -> ActorRecord:
    if not prev_rows:
        empty = neutral_labels((B, 1))
        return ActorRecord(prev=empty, samples=empty, logits={})
    return ActorRecord(
        prev=labels_stack(prev_rows, dim=1),
        samples=labels_stack(sample_rows, dim=1),
        logits={name: torch.stack([row[name] for row in logit_rows], dim=1) for name in logit_rows[0]},
    )


def _window(
    streams: Streams,
    record: ActorRecord,
    rollout: int,
    *,
    lo: int,
    hi: int,
) -> tuple[FrameTree, ControllerLabels, ControllerLabels, torch.Tensor, torch.Tensor]:
    """Global rows ``[lo, hi)`` as a left-padded window: frames, prev, targets, reset, padding."""

    rows = torch.arange(lo, hi)
    length = hi - lo
    available = (rows[None, :] >= 0) & (rows[None, :] >= streams.history_start[:, rollout, None])
    index = rows.clamp(min=0)

    def pad(selected: torch.Tensor, neutral: torch.Tensor) -> torch.Tensor:
        mask = available.reshape(B, length, *([1] * (selected.ndim - 2)))
        return torch.where(mask, selected, neutral)

    frames = tree_map(pad, tree_index(streams.frames, (slice(None), index)), neutral_frame((B, length)))
    prev = labels_map(pad, labels_index(record.prev, (slice(None), index)), neutral_labels((B, length)))
    targets = labels_map(pad, labels_index(record.samples, (slice(None), index)), neutral_labels((B, length)))
    reset = streams.is_resetting[:, index] & available
    return frames, prev, targets, reset, available


def _run_actor(
    adapter: MeleePolicyAdapter,
    streams: Streams,
    rollouts: int,
    *,
    reprime: bool,
    seed: int,
    snapshots: list[KVCache] | None = None,
) -> ActorRecord:
    """``prime`` (exact mode) or nothing (ring mode) at each boundary, then ``T x step_sample``.

    ``snapshots`` collects a copy of the cache at every rollout boundary, before the first step
    (prefix mode's ``Trajectory.initial_cache``).
    """

    generator = torch.Generator().manual_seed(seed)
    cache = adapter.init_cache(B)
    prev_rows: list[ControllerLabels] = []
    sample_rows: list[ControllerLabels] = []
    logit_rows: list[dict[str, torch.Tensor]] = []
    last = neutral_labels((B,))
    for rollout in range(rollouts):
        start = rollout * T
        if reprime:
            record = _record(prev_rows, sample_rows, logit_rows)
            frames, prev, _, reset, padding = _window(streams, record, rollout, lo=start - C, hi=start)
            adapter.prime(frames, prev, reset, padding, cache)
        if snapshots is not None:
            snapshots.append(snapshot_cache(cache))
        for step in range(T):
            row = start + step
            reset_row = streams.is_resetting[:, row]
            prev_row = ControllerLabels(
                buttons=torch.where(reset_row, 0, last.buttons),
                main_stick=torch.where(reset_row, 0, last.main_stick),
            )
            output = adapter.step_sample(
                tree_index(streams.frames, (slice(None), row)),
                prev_row,
                reset_row,
                cache,
                generator=generator,
            )
            prev_rows.append(prev_row)
            sample_rows.append(output.labels)
            logit_rows.append(output.logits)
            last = output.labels
    return _record(prev_rows, sample_rows, logit_rows)


def _learner(
    adapter: MeleePolicyAdapter,
    streams: Streams,
    record: ActorRecord,
    rollout: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Learner logits and log-probs for the rollout rows of window ``[rT - C, rT + T)``."""

    start = rollout * T
    frames, prev, targets, reset, padding = _window(streams, record, rollout, lo=start - C, hi=start + T)
    output = adapter.unroll_window(frames, prev, targets, reset, padding)
    logits = {name: value[:, C:].detach() for name, value in output.logits.items()}
    taken = labels_index(targets, (slice(None), slice(C, None)))
    return logits, distributions.log_prob(logits, taken)


def _learner_prefix(
    adapter: MeleePolicyAdapter,
    streams: Streams,
    record: ActorRecord,
    rollout: int,
    prefix: KVCache,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Learner logits and log-probs for the rollout rows ``[rT, rT + T)`` alone, continuing ``prefix``."""

    start = rollout * T
    frames, prev, targets, reset, padding = _window(streams, record, rollout, lo=start, hi=start + T)
    output = adapter.unroll_window(frames, prev, targets, reset, padding, prefix=prefix)
    logits = {name: value.detach() for name, value in output.logits.items()}
    return logits, distributions.log_prob(logits, targets)


def _actor_slice(record: ActorRecord, rollout: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    rows = slice(rollout * T, rollout * T + T)
    logits = {name: value[:, rows] for name, value in record.logits.items()}
    return logits, distributions.log_prob(logits, labels_index(record.samples, (slice(None), rows)))


def _last_reset_before(streams: Streams, env: int, row: int) -> int:
    resets = streams.is_resetting[env, : row + 1].nonzero()
    return int(resets[-1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_protocol_conformance_and_static_facts(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
) -> None:
    assert isinstance(tiny_adapter, PolicyProtocol)
    assert isinstance(tiny_adapter, MeleePolicy)
    assert as_policy(tiny_adapter) is tiny_adapter
    assert tiny_adapter.action_spec == ActionSpec(("buttons", "main_stick"), (728, 85))
    assert tiny_adapter.delay_offset == 1
    assert tiny_adapter.context_length == 24
    assert tiny_adapter.device.type == "cpu"
    assert tiny_adapter.config.compute_dtype == "float32" and tiny_adapter.config.cache_dtype == "float32"

    plain = MeleePolicy(tiny_config)
    copied = MeleePolicyAdapter.from_policy(plain)
    assert copied.get_state().keys() == plain.state_dict().keys()
    for name, value in plain.state_dict().items():
        assert torch.equal(copied.get_state()[name], value), name
    assert copied.training == plain.training


def test_adapter_shapes_3m_profile(three_m_config: ModelConfig) -> None:
    # Keeps Ali's gradient_checkpointing=True: the first torch.utils.checkpoint(use_reentrant=False)
    # call of a process lazily imports ~880 modules (~19 s on the container's Windows-mounted venv,
    # ENV.md section 4); every later call is milliseconds.
    torch.manual_seed(3)
    adapter = MeleePolicyAdapter(three_m_config)
    assert adapter.context_length == 256
    assert adapter.config.gradient_checkpointing
    generator = torch.Generator().manual_seed(0)
    batch, history = 2, 8
    frames = random_valid_frames((batch, history + T), generator)
    cache = adapter.init_cache(batch)
    assert tuple(cache.keys.shape) == (3, batch, 256, 1, 64)
    assert cache.keys.dtype == torch.float32

    prev = neutral_labels((batch, history))
    reset = torch.zeros((batch, history), dtype=torch.bool)
    reset[:, 0] = True
    history_frames = tree_index(frames, (slice(None), slice(0, history)))
    adapter.prime(history_frames, prev, reset, torch.ones_like(reset), cache)
    assert cache.valid_length.tolist() == [history] * batch
    assert cache.next_position.tolist() == [history] * batch

    last = neutral_labels((batch,))
    step_labels: list[ControllerLabels] = []
    for step in range(T):
        output = adapter.step_sample(
            tree_index(frames, (slice(None), history + step)),
            last,
            torch.zeros(batch, dtype=torch.bool),
            cache,
            generator=generator,
        )
        assert isinstance(output, StepOutput)
        assert output.labels.buttons.shape == (batch,) and output.labels.buttons.dtype == torch.long
        assert output.logits["buttons"].shape == (batch, 728)
        assert output.logits["buttons"].dtype == torch.float32
        assert output.logits["main_stick"].shape == (batch, 85)
        step_labels.append(output.labels)
        last = output.labels
    assert cache.next_position.tolist() == [history + T] * batch

    adapter.train()
    targets = labels_stack([*([neutral_labels((batch,))] * history), *step_labels], dim=1)
    window_reset = torch.zeros((batch, history + T), dtype=torch.bool)
    window_reset[:, 0] = True
    padding = torch.ones_like(window_reset)
    # History rows were primed with neutral prev labels; each rollout row's prev is the previous sample.
    window_prev = labels_stack(
        [*([neutral_labels((batch,))] * history), neutral_labels((batch,)), *step_labels[:-1]],
        dim=1,
    )
    window = adapter.unroll_window(frames, window_prev, targets, window_reset, padding)
    assert isinstance(window, WindowOutput)
    assert window.logits["buttons"].shape == (batch, history + T, 728)
    assert window.logits["main_stick"].shape == (batch, history + T, 85)
    assert window.hidden.shape == (batch, history + T, 128)
    assert window.logits["buttons"].requires_grad

    loss = -distributions.log_prob(window.logits, targets)[:, history:].mean()
    loss.backward()
    grad = adapter.backbone.input_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert adapter.get_state().keys() == MeleePolicy(three_m_config).state_dict().keys()


def test_teacher_forced_logits_condition_on_taken_buttons(tiny_adapter: MeleePolicyAdapter) -> None:
    generator = torch.Generator().manual_seed(4)
    batch, length = 2, 6
    frames = random_valid_frames((batch, length), generator)
    prev = neutral_labels((batch, length))
    reset = torch.zeros((batch, length), dtype=torch.bool)
    reset[:, 0] = True
    padding = torch.ones_like(reset)
    targets_a = ControllerLabels(
        buttons=torch.randint(0, 728, (batch, length), generator=generator),
        main_stick=torch.randint(0, 85, (batch, length), generator=generator),
    )
    targets_b = ControllerLabels(buttons=(targets_a.buttons + 1) % 728, main_stick=targets_a.main_stick)

    with torch.no_grad():
        out_a = tiny_adapter.unroll_window(frames, prev, targets_a, reset, padding)
        out_b = tiny_adapter.unroll_window(frames, prev, targets_b, reset, padding)
    torch.testing.assert_close(out_a.logits["buttons"], out_b.logits["buttons"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(out_a.hidden, out_b.hidden, rtol=0.0, atol=0.0)
    difference = (out_a.logits["main_stick"] - out_b.logits["main_stick"]).abs().amax(dim=-1)
    assert bool((difference > 1.0e-3).all()), difference


def test_actor_learner_parity_exact(tiny_adapter: MeleePolicyAdapter) -> None:
    rollouts = 3
    streams = _make_streams(rollouts, seed=21)
    record = _run_actor(tiny_adapter, streams, rollouts, reprime=True, seed=8)
    assert record.samples.buttons.shape == (B, rollouts * T)

    tiny_adapter.train()  # the learner path runs in training mode with gradients enabled
    for rollout in range(rollouts):
        learner_logits, learner_log_prob = _learner(tiny_adapter, streams, record, rollout)
        actor_logits, actor_log_prob = _actor_slice(record, rollout)
        for name in ("buttons", "main_stick"):
            torch.testing.assert_close(learner_logits[name], actor_logits[name], rtol=1.0e-4, atol=1.0e-5)
        assert float((learner_log_prob - actor_log_prob).abs().max()) < 1.0e-5
        assert torch.isfinite(learner_log_prob).all()
    # The windows really exercised what they claim: padding, a reset inside a rollout, partial history.
    assert bool(streams.is_resetting[1, 11]) and int(streams.history_start[2, 2]) == 8


def test_ring_mode_mismatch_documented(tiny_adapter: MeleePolicyAdapter) -> None:
    """Without re-priming, actor == learner iff the segment at the first rollout row began inside the window.

    The ring keeps up to ``context_length`` frames with absolute RoPE positions; the learner
    re-forwards the window from a reset state, so the two agree exactly when the last reset at
    or before row ``rT`` lies at or after the window start ``rT - C`` (e.g. the window starts at
    a game start), and differ otherwise (PLAN.md §3.2, ``context_mode = "ring"``).
    """

    rollouts = 5
    streams = _make_streams(rollouts, seed=22)
    record = _run_actor(tiny_adapter, streams, rollouts, reprime=False, seed=9)
    exact_pairs = 0
    mismatch_pairs = 0
    for rollout in range(rollouts):
        learner_logits, _ = _learner(tiny_adapter, streams, record, rollout)
        actor_logits, _ = _actor_slice(record, rollout)
        for env in range(B):
            expected_exact = _last_reset_before(streams, env, rollout * T) >= rollout * T - C
            difference = max(
                float((learner_logits[name][env] - actor_logits[name][env]).abs().max())
                for name in ("buttons", "main_stick")
            )
            assert torch.isfinite(learner_logits["buttons"][env]).all()
            if expected_exact:
                exact_pairs += 1
                assert difference < 1.0e-4, (env, rollout, difference)
            else:
                mismatch_pairs += 1
                assert difference > 1.0e-4, (env, rollout, difference)
    assert exact_pairs == 11 and mismatch_pairs == 4


def test_prefix_mode_is_exact_where_ring_mode_is_not(tiny_adapter: MeleePolicyAdapter) -> None:
    """The ring actor plus a snapshot of its cache at every boundary (``context_mode = "prefix"``): the
    learner's forward over the ``T`` rollout rows alone, attending to the snapshot, reproduces every
    (rollout, env) pair -- including the four that ``test_ring_mode_mismatch_documented`` shows the
    ``C + T`` re-forward cannot -- with a full ring (env 0 from rollout 3), a reset inside a rollout
    (env 1, row 11) and a reset at a boundary with the history cleared (env 2, row 8)."""

    rollouts = 5
    streams = _make_streams(rollouts, seed=22)
    snapshots: list[KVCache] = []
    record = _run_actor(tiny_adapter, streams, rollouts, reprime=False, seed=9, snapshots=snapshots)
    assert len(snapshots) == rollouts
    assert int(snapshots[3].valid_length[0]) == tiny_adapter.context_length  # env 0's ring is full
    assert bool(streams.is_resetting[1, 11]) and bool(streams.is_resetting[2, 8])

    tiny_adapter.train()  # the learner path runs in training mode with gradients enabled
    ring_mismatches = 0
    for rollout in range(rollouts):
        learner_logits, learner_log_prob = _learner_prefix(
            tiny_adapter, streams, record, rollout, snapshots[rollout]
        )
        actor_logits, actor_log_prob = _actor_slice(record, rollout)
        for name in ("buttons", "main_stick"):
            torch.testing.assert_close(learner_logits[name], actor_logits[name], rtol=1.0e-4, atol=1.0e-5)
        assert float((learner_log_prob - actor_log_prob).abs().max()) < 1.0e-5
        assert torch.isfinite(learner_log_prob).all()
        for env in range(B):
            ring_mismatches += _last_reset_before(streams, env, rollout * T) < rollout * T - C
    assert ring_mismatches == 4


def test_padded_window_forwards_and_backpropagates(tiny_adapter: MeleePolicyAdapter) -> None:
    generator = torch.Generator().manual_seed(6)
    batch, length = 2, 12
    frames = random_valid_frames((batch, length), generator)
    padding = torch.ones((batch, length), dtype=torch.bool)
    padding[0, :5] = False
    padding[1, :2] = False
    reset = torch.zeros((batch, length), dtype=torch.bool)
    reset[0, 5] = True
    reset[1, 2] = True
    def blank_padded_rows(leaf: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
        return torch.where(padding.reshape(batch, length, *([1] * (leaf.ndim - 2))), leaf, blank)

    frames = tree_map(blank_padded_rows, frames, neutral_frame((batch, length)))
    prev = neutral_labels((batch, length))
    targets = ControllerLabels(
        buttons=torch.randint(0, 728, (batch, length), generator=generator),
        main_stick=torch.randint(0, 85, (batch, length), generator=generator),
    )

    tiny_adapter.train()
    output = tiny_adapter.unroll_window(frames, prev, targets, reset, padding)
    for value in output.logits.values():
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(output.hidden[~padding]) == 0
    assert torch.count_nonzero(output.hidden[padding]) > 0
    loss = -(distributions.log_prob(output.logits, targets) * padding).sum() / padding.sum()
    loss.backward()
    for name, parameter in tiny_adapter.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_sampling_decode_reencode_round_trip_and_greedy(tiny_adapter: MeleePolicyAdapter) -> None:
    codec = CustomV1Codec()
    generator = torch.Generator().manual_seed(10)
    batch = 5
    frame = random_valid_frames((batch,), generator)
    cache = tiny_adapter.init_cache(batch)
    output = tiny_adapter.step_sample(
        frame,
        neutral_labels((batch,)),
        torch.ones(batch, dtype=torch.bool),
        cache,
        generator=generator,
    )
    assert output.labels.buttons.dtype == torch.long
    assert bool(((output.labels.buttons >= 0) & (output.labels.buttons < 728)).all())
    assert bool(((output.labels.main_stick >= 0) & (output.labels.main_stick < 85)).all())
    decoded = codec.decode(output.labels)
    reencoded = codec.encode(decoded)
    assert torch.equal(reencoded.buttons, output.labels.buttons)
    assert torch.equal(reencoded.main_stick, output.labels.main_stick)
    log_prob = distributions.log_prob(output.logits, output.labels)
    assert torch.isfinite(log_prob).all() and bool((log_prob <= 0).all())
    assert cache.next_position.tolist() == [1] * batch

    greedy = tiny_adapter.step_sample(
        frame,
        output.labels,
        torch.zeros(batch, dtype=torch.bool),
        cache,
        temperature=0.0,
    )
    assert torch.equal(greedy.labels.buttons, greedy.logits["buttons"].argmax(dim=-1))
    assert torch.equal(greedy.labels.main_stick, greedy.logits["main_stick"].argmax(dim=-1))


def test_step_sample_restores_training_mode(tiny_adapter: MeleePolicyAdapter) -> None:
    frame = random_valid_frames((1,), torch.Generator().manual_seed(1))
    cache = tiny_adapter.init_cache(1)
    tiny_adapter.train()
    tiny_adapter.step_sample(frame, neutral_labels((1,)), torch.ones(1, dtype=torch.bool), cache)
    assert tiny_adapter.training
    tiny_adapter.eval()
    tiny_adapter.step_sample(frame, neutral_labels((1,)), torch.zeros(1, dtype=torch.bool), cache)
    assert not tiny_adapter.training


def test_clone_frozen_teacher_and_state_round_trip(tiny_adapter: MeleePolicyAdapter) -> None:
    teacher = tiny_adapter.clone_frozen()
    assert isinstance(teacher, MeleePolicyAdapter)
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.requires_grad for parameter in tiny_adapter.parameters())

    generator = torch.Generator().manual_seed(12)
    batch, length = 2, 10
    frames = random_valid_frames((batch, length), generator)
    prev = neutral_labels((batch, length))
    targets = ControllerLabels(
        buttons=torch.randint(0, 728, (batch, length), generator=generator),
        main_stick=torch.randint(0, 85, (batch, length), generator=generator),
    )
    reset = torch.zeros((batch, length), dtype=torch.bool)
    reset[:, 0] = True
    padding = torch.ones_like(reset)
    with torch.no_grad():
        student = tiny_adapter.unroll_window(frames, prev, targets, reset, padding)
        frozen = teacher.unroll_window(frames, prev, targets, reset, padding)
    torch.testing.assert_close(
        distributions.kl(student.logits, frozen.logits),
        torch.zeros(batch, length),
        rtol=0.0,
        atol=1.0e-6,
    )

    before = teacher.get_state()
    with torch.no_grad():
        for parameter in tiny_adapter.parameters():
            parameter.add_(0.1)
    after = teacher.get_state()
    assert all(torch.equal(before[name], after[name]) for name in before)
    with torch.no_grad():
        changed = tiny_adapter.unroll_window(frames, prev, targets, reset, padding)
    assert float(distributions.kl(changed.logits, frozen.logits).max()) > 0.0

    tiny_adapter.set_state(teacher.get_state())
    with torch.no_grad():
        restored = tiny_adapter.unroll_window(frames, prev, targets, reset, padding)
    torch.testing.assert_close(restored.logits["buttons"], frozen.logits["buttons"], rtol=0.0, atol=0.0)
    with pytest.raises(RuntimeError):
        tiny_adapter.set_state({"encoder.action_embedding.weight": before["encoder.action_embedding.weight"]})


def test_input_validation(tiny_adapter: MeleePolicyAdapter) -> None:
    generator = torch.Generator().manual_seed(13)
    cache = tiny_adapter.init_cache(2)
    frames = random_valid_frames((2, 4), generator)
    labels = neutral_labels((2, 4))
    good = torch.zeros((2, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="reset_mask must have shape"):
        tiny_adapter.prime(frames, labels, torch.zeros((2, 3), dtype=torch.bool), good, cache)
    with pytest.raises(ValueError, match="padding_mask must have shape"):
        tiny_adapter.prime(frames, labels, good, torch.zeros((3, 4), dtype=torch.bool), cache)
    with pytest.raises(ValueError, match="does not match the cache batch"):
        tiny_adapter.prime(frames, labels, good, good, tiny_adapter.init_cache(3))
    with pytest.raises(TypeError, match="ControllerLabels"):
        tiny_adapter.prime(frames, labels.as_dict(), good, good, cache)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer labels"):
        tiny_adapter.unroll_window(
            frames,
            ControllerLabels(labels.buttons.float(), labels.main_stick),
            labels,
            good,
            ~good,
        )
    with pytest.raises(ValueError, match="targets must have shape"):
        tiny_adapter.unroll_window(frames, labels, neutral_labels((2, 3)), good, ~good)

    long_frames = random_valid_frames((2, 25), generator)
    long_labels = neutral_labels((2, 25))
    long_mask = torch.zeros((2, 25), dtype=torch.bool)
    with pytest.raises(ValueError, match="exceeds context_length"):
        tiny_adapter.unroll_window(long_frames, long_labels, long_labels, long_mask, ~long_mask)
    with pytest.raises(ValueError, match="exceeds context_length"):
        tiny_adapter.prime(long_frames, long_labels, long_mask, ~long_mask, cache)
    with pytest.raises(ValueError, match="reset_mask must have shape"):
        tiny_adapter.step_sample(tree_index(frames, (slice(None), 0)), neutral_labels((2,)), good, cache)
    with pytest.raises(ValueError, match="reset_mask must have shape"):
        tiny_adapter.reset_slots(cache, torch.zeros(3, dtype=torch.bool))

    # Zero-length priming only clears the cache.
    tiny_adapter.prime(
        tree_index(frames, (slice(None), slice(0, 0))),
        neutral_labels((2, 0)),
        torch.zeros((2, 0), dtype=torch.bool),
        torch.zeros((2, 0), dtype=torch.bool),
        cache,
    )
    assert cache.valid_length.tolist() == [0, 0]
