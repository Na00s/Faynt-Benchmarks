"""Checkpoints (PLAN.md §3.3, INTERFACE.md §8): RL round-trip, resume-identical step, BC export, formats."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from melee_rl.adapter import MeleePolicyAdapter
from melee_rl.checkpoint import (
    ALI_TRAINING_FORMAT,
    BC_FORMAT,
    FORMATS,
    RL_FORMAT,
    atomic_save,
    checkpoint_paths,
    detect_format,
    export_bc,
    load_bc_checkpoint,
    load_checkpoint,
    load_policy,
    restore_learner,
    save_bc_checkpoint,
    save_rl_checkpoint,
    sha256_file,
)
from melee_rl.learner import LearnerConfig, PPOConfig, PPOLearner
from melee_rl.protocol import PolicyProtocol
from melee_rl.trajectory import Trajectory
from melee_rl.value import ValueConfig, ValueNet
from model import MeleePolicy, ModelConfig

RolloutFactory = Callable[..., list[Trajectory]]
CONFIG_YAML = Path(__file__).resolve().parents[2] / "config.yaml"


def _learner(
    adapter: MeleePolicyAdapter,
    config: ModelConfig,
    value_config: ValueConfig,
    seed: int,
    teacher: PolicyProtocol | None = None,
) -> PPOLearner:
    torch.manual_seed(seed)
    value_net = ValueNet(config, value_config)
    learner_config = LearnerConfig(
        learning_rate=1.0e-3,
        value_burnin_steps=1,
        ppo=PPOConfig(num_epochs=2, max_mean_actor_kl=1.0),
    )
    frozen = adapter.clone_frozen() if teacher is None else teacher
    return PPOLearner(adapter, frozen, value_net, learner_config)


def _same_params(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    pairs = zip(left.parameters(), right.parameters(), strict=True)
    return all(torch.equal(a.detach(), b.detach()) for a, b in pairs)


def test_rl_checkpoint_round_trip_and_resume_identical_step(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    tmp_path: Path,
) -> None:
    trajectories = rollout_trajectories(tiny_adapter, 3, synthetic_rewards=0.1)
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, seed=0)
    learner.step(trajectories[:1])  # value burn-in step (ladder position 1 afterwards)
    learner.step(trajectories[1:2])  # a real PPO step: Adam has state on both optimizers
    rng = {"torch": torch.get_rng_state(), "worker": torch.Generator().manual_seed(3).get_state()}
    latest, periodic = checkpoint_paths(tmp_path, 2)
    payload = save_rl_checkpoint(
        latest,
        learner,
        tiny_config,
        step=2,
        rl_config={"runtime": {"seed": 1}, "paths": {"root": tmp_path}},
        teacher={"random_init": True, "seed": 1234},
        delay_frames=0,
        context_mode="reprime",
        rng=rng,
        wandb_run_id="run-1",
        config_yaml=CONFIG_YAML,
    )
    assert periodic == tmp_path / "step_00000002.pt" and latest == tmp_path / "latest.pt"
    assert payload["format"] == RL_FORMAT and payload["step"] == 2 and payload["learner_step"] == 2
    assert payload["rl_config"]["paths"]["root"] == str(tmp_path)  # Path -> str
    assert payload["config_yaml_sha256"] == sha256_file(CONFIG_YAML)
    assert payload["codec"] == {"name": "custom_v1", "vocab_sizes": {"buttons": 728, "main_stick": 85}}
    assert payload["created"].endswith("Z") and payload["teacher"] == {"random_init": True, "seed": 1234}

    loaded = load_checkpoint(latest)  # weights_only=True inside
    assert loaded["format"] == RL_FORMAT and torch.equal(loaded["rng"]["torch"], rng["torch"])
    assert loaded["learner_config"]["ppo"]["num_epochs"] == 2 and loaded["value_config"]["d_model"] == 32
    assert loaded["model_config"]["context_length"] == 24 and loaded["delay_frames"] == 0

    # Resume into a fresh policy / value net / optimizers (the teacher is not learner state: the
    # training loop rebuilds it from the checkpoint's ``teacher`` record): the next step is bit-identical.
    fresh_policy = MeleePolicyAdapter(tiny_config)
    assert not _same_params(fresh_policy, tiny_adapter)
    assert learner.teacher is not None
    resumed = _learner(fresh_policy, tiny_config, tiny_value_config, seed=99, teacher=learner.teacher)
    assert restore_learner(loaded, resumed) == 2
    assert resumed.steps_done == 2 and resumed.stage() == ("ppo", 2, 1.0e-3)
    assert _same_params(fresh_policy, tiny_adapter) and _same_params(resumed.value_net, learner.value_net)
    original = learner.step(trajectories[2:])
    continued = resumed.step(trajectories[2:])
    for key in ("loss/total", "loss/entropy", "loss/teacher_kl", "value/loss", "ppo/clip_fraction"):
        assert original[key] == continued[key], key
    assert _same_params(fresh_policy, tiny_adapter) and _same_params(resumed.value_net, learner.value_net)

    # The RL checkpoint is a valid BC checkpoint, and the exporter writes the BC format.
    policy_only = load_bc_checkpoint(latest)
    assert policy_only.format == RL_FORMAT and policy_only.model_config == tiny_config
    export = export_bc(tmp_path / "bc.pt", learner, tiny_config, step=3, config_yaml=CONFIG_YAML)
    assert export["format"] == BC_FORMAT and export["training"]["step"] == 3
    reloaded = torch.load(tmp_path / "bc.pt", weights_only=True)
    model = MeleePolicy(ModelConfig(**reloaded["model_config"]))
    model.load_state_dict(reloaded["state_dict"], strict=True)
    assert _same_params(model, tiny_adapter)
    via_helper = load_policy(tmp_path / "bc.pt")
    assert isinstance(via_helper, MeleePolicyAdapter) and _same_params(via_helper, tiny_adapter)
    assert via_helper.config.compute_dtype == "float32"
    with pytest.raises(ValueError, match="value config mismatch"):
        one_layer = ValueConfig(d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, head_dim=8, d_ff=64)
        other = ValueNet(tiny_config, one_layer)
        restore_learner(loaded, PPOLearner(fresh_policy, None, other, LearnerConfig(kl_teacher_weight=0.0)))


def test_trained_delay_reads_the_delay_a_checkpoint_was_trained_at(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    rollout_trajectories: RolloutFactory,
    tmp_path: Path,
) -> None:
    """11 Sep 2026 (/delay-finetune): one checkpoint means one total delay.  The RL payload has stored
    ``delay_frames`` since P3 with nothing reading it; ``trained_delay`` reads it (a path or a loaded
    payload), the BC ``training`` record and the export carry it too, and a BC file, Ali's files or an RL
    file from before the field read 0 -- so a D = 18 file can never be played at 0 by accident."""

    from melee_rl.checkpoint import trained_delay

    bc = tmp_path / "bc.pt"
    save_bc_checkpoint(bc, tiny_adapter.get_state(), tiny_adapter.config)
    assert trained_delay(bc) == 0
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, seed=0)
    learner.step(rollout_trajectories(tiny_adapter, 1, delay=2))
    rl = tmp_path / "rl.pt"
    payload = save_rl_checkpoint(rl, learner, tiny_config, step=1, delay_frames=2, context_mode="reprime")
    assert trained_delay(rl) == 2 and trained_delay(payload) == 2
    assert payload["training"]["delay_frames"] == 2 and load_bc_checkpoint(rl).metadata["delay_frames"] == 2
    export = export_bc(tmp_path / "export.pt", learner, tiny_config, step=1, delay_frames=2)
    assert export["training"]["delay_frames"] == 2 and trained_delay(tmp_path / "export.pt") == 2
    assert export_bc(tmp_path / "plain.pt", learner, tiny_config, step=1)["training"]["delay_frames"] == 0
    old = dict(load_checkpoint(rl))
    old["delay_frames"] = None
    old["training"] = {"step": 1}
    atomic_save(old, tmp_path / "old.pt")
    assert trained_delay(tmp_path / "old.pt") == 0
    with pytest.raises(ValueError, match="delay"):
        atomic_save({**old, "delay_frames": -3}, tmp_path / "bad.pt")
        trained_delay(tmp_path / "bad.pt")


def test_bc_checkpoint_formats_and_validation(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    path = tmp_path / "policy.pt"
    payload = save_bc_checkpoint(path, tiny_adapter.get_state(), tiny_adapter.config, training={"step": 7})
    assert payload["format"] == BC_FORMAT and payload["training"] == {"step": 7}
    assert payload["slippi_ai_commit"] == "577965a7731dc53e3472ea63d9e9853a4e9d65fa"
    checkpoint = load_bc_checkpoint(path)
    assert checkpoint.model_config == tiny_adapter.config
    assert set(checkpoint.state_dict) == set(tiny_adapter.state_dict())
    # Ali's precision defaults survive a non-RL load; the RL loader applies fp32.
    raw = MeleePolicy(checkpoint.model_config)
    raw.load_state_dict(checkpoint.state_dict, strict=True)
    assert load_policy(path, rl_precision=False).config == tiny_adapter.config
    with pytest.raises(ValueError, match="restore_learner needs"):
        no_teacher = LearnerConfig(kl_teacher_weight=0.0)
        restore_learner(payload, PPOLearner(tiny_adapter, None, ValueNet(tiny_adapter.config), no_teacher))
    bogus = tmp_path / "bogus.pt"
    atomic_save({"format": "something.else", "state_dict": {}}, bogus)
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        load_checkpoint(bogus)
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_checkpoint(tmp_path / "list.pt")
    assert not (tmp_path / "policy.pt.tmp").exists()


def test_load_policy_can_switch_gradient_checkpointing_off(
    tiny_adapter: MeleePolicyAdapter, tmp_path: Path
) -> None:
    """10 Sep 2026 (strength gate 1): a frozen teacher never backpropagates, so the trainer reads a teacher
    file with gradient checkpointing off whatever the file says -- Ali's training files carry
    ``gradient_checkpointing = true`` and the compiled lean forward (``learner.compile``) refuses a
    checkpointed model.  The flag is memory only: the weights and the forward stay the file's."""

    from melee_rl.fast_forward import FastForward

    path = tmp_path / "teacher.pt"
    checkpointed = replace(tiny_adapter.config, gradient_checkpointing=True)
    save_bc_checkpoint(path, tiny_adapter.get_state(), checkpointed)
    as_saved = load_policy(path)
    assert as_saved.config.gradient_checkpointing is True
    switched = load_policy(path, gradient_checkpointing=False)
    assert switched.config == replace(as_saved.config, gradient_checkpointing=False)
    assert _same_params(switched, as_saved)
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        FastForward(as_saved, compile="aot_eager")
    FastForward(switched, compile="aot_eager")  # the compiled lean forward accepts the switched teacher


# ---------------------------------------------------------------------------
# P6a: Ali's training checkpoint (format_version 1, training.py:732-785)
# ---------------------------------------------------------------------------


def test_ali_training_checkpoint_v1_loads_through_every_reader(
    tiny_adapter: MeleePolicyAdapter,
    tiny_config: ModelConfig,
    tiny_value_config: ValueConfig,
    tmp_path: Path,
) -> None:
    """A file written by Ali's ``training.save_checkpoint`` loads (``weights_only=True``) into our readers."""

    import optimizers as ali_optimizers
    import training as ali_training

    torch.manual_seed(5)
    policy = MeleePolicy(tiny_config)
    bundle = ali_optimizers.build_adamw(policy, learning_rate=1e-3)
    for parameter in policy.parameters():  # one optimizer step so the AdamW state holds tensors too
        parameter.grad = torch.zeros_like(parameter)
    bundle.optimizers["adamw"].step()
    state = ali_training.TrainingState(
        processed_target_frames=1234,
        frame_cursor=5,
        optimizer_steps=1,
        microbatches=1,
        wall_clock_seconds=1.5,
    )
    path = tmp_path / "ali.pt"
    written = ali_training.save_checkpoint(
        path,
        policy,
        bundle,
        state,
        resolved_config={"model": asdict(tiny_config), "profile": "3m", "training": {"learning_rate": 1e-3}},
        metadata={"run_name": "bc-test", "run_id": "abc123"},
    )
    assert written == path and path.exists()

    payload = load_checkpoint(path)  # torch.load(weights_only=True) inside
    assert detect_format(payload) == ALI_TRAINING_FORMAT and ALI_TRAINING_FORMAT in FORMATS
    assert payload["format_version"] == 1 and set(payload["model"]) == set(policy.state_dict())
    assert (
        payload["optimizer_bundle"]["kind"] == "adamw" and payload["training_state"]["optimizer_steps"] == 1
    )
    loaded = load_bc_checkpoint(path)
    assert loaded.format == ALI_TRAINING_FORMAT and loaded.model_config == tiny_config
    assert set(loaded.state_dict) == set(policy.state_dict())
    assert loaded.metadata["run_name"] == "bc-test" and loaded.metadata["run_id"] == "abc123"
    assert loaded.metadata["processed_target_frames"] == 1234
    assert loaded.metadata["training_state"]["wall_clock_seconds"] == 1.5
    adapter = load_policy(path)
    assert isinstance(adapter, MeleePolicyAdapter) and _same_params(adapter, policy)
    assert adapter.config.compute_dtype == "float32" and adapter.config.cache_dtype == "float32"
    assert load_policy(path, rl_precision=False).config == tiny_config
    # The file is untouched: Ali's own reader still accepts it afterwards.
    again = MeleePolicy(tiny_config)
    restored = ali_training.load_checkpoint(
        path, again, ali_optimizers.build_adamw(again, learning_rate=1e-3), restore_rng=False
    )
    assert (
        restored.state == state and restored.metadata["run_name"] == "bc-test" and _same_params(again, policy)
    )
    # RL-only operations refuse it (no learner state inside).
    learner = _learner(tiny_adapter, tiny_config, tiny_value_config, seed=0)
    with pytest.raises(ValueError, match="restore_learner needs"):
        restore_learner(payload, learner)


def test_detect_format_and_rejections(tiny_adapter: MeleePolicyAdapter, tmp_path: Path) -> None:
    assert FORMATS == (BC_FORMAT, RL_FORMAT, ALI_TRAINING_FORMAT)
    assert ALI_TRAINING_FORMAT == "melee_policy.training_checkpoint.v1"
    ours = tmp_path / "bc.pt"
    payload = save_bc_checkpoint(ours, tiny_adapter.get_state(), tiny_adapter.config, training={"step": 4})
    assert detect_format(payload) == BC_FORMAT
    assert load_bc_checkpoint(ours).metadata == {"step": 4}  # ours: the ``training`` record
    assert detect_format({"format": RL_FORMAT}) == RL_FORMAT
    ali_like = {"format_version": 1, "model": {}, "resolved_config": {"model": asdict(tiny_adapter.config)}}
    assert detect_format(ali_like) == ALI_TRAINING_FORMAT
    with pytest.raises(ValueError, match="format_version"):
        detect_format({"format_version": 2, "model": {}, "resolved_config": {"model": {}}})
    with pytest.raises(ValueError, match="resolved_config"):
        detect_format({"format_version": 1, "model": {}})
    with pytest.raises(ValueError, match="resolved_config"):
        detect_format({"format_version": 1, "model": {}, "resolved_config": {}})
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        detect_format({"something": 1})
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        detect_format({"format": "melee_policy.bc_checkpoint.v0"})
    atomic_save({"format_version": 2, "model": {}, "resolved_config": {"model": {}}}, tmp_path / "v2.pt")
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(tmp_path / "v2.pt")
    atomic_save({"format_version": 1, "resolved_config": {}}, tmp_path / "v1-incomplete.pt")
    with pytest.raises(ValueError, match="resolved_config"):
        load_bc_checkpoint(tmp_path / "v1-incomplete.pt")


def test_ali_muon_training_checkpoint_survives_weights_only_load(
    tiny_config: ModelConfig, tmp_path: Path
) -> None:
    """Ali's ``muon_with_aux_adamw`` bundle reads with ``torch.load(weights_only=True)``.

    The BC-init run (BC-INIT-PLAN.md, 24 Aug 2026) starts from his best 5m checkpoint, which was
    trained with Muon + an auxiliary AdamW; P6a only ever exercised the AdamW bundle.  Verified on the
    real file the same day (``kind muon_with_aux_adamw``, backend ``torch.optim.Muon``, 77 tensors);
    this keeps that fact under test with a bundle built by his own ``optimizers.py``.
    """

    import optimizers as ali_optimizers
    import training as ali_training

    torch.manual_seed(6)
    policy = MeleePolicy(tiny_config)
    bundle = ali_optimizers.build_muon_with_aux_adamw(
        policy, muon_learning_rate=1e-2, auxiliary_learning_rate=2e-3
    )
    assert bundle.kind == "muon_with_aux_adamw" and set(bundle.optimizers) == {"muon", "aux_adamw"}
    for parameter in policy.parameters():  # one step so both optimizers carry state tensors
        parameter.grad = torch.zeros_like(parameter)
    bundle.step()
    state = ali_training.TrainingState(
        processed_target_frames=41943040,
        frame_cursor=41943040,
        optimizer_steps=640,
        microbatches=20480,
        wall_clock_seconds=3405.2,
    )
    path = tmp_path / "muon.pt"
    ali_training.save_checkpoint(
        path,
        policy,
        bundle,
        state,
        resolved_config={"model": asdict(tiny_config), "profile": None},
        metadata={"run_name": "5m-muon"},
    )
    payload = load_checkpoint(path)  # torch.load(weights_only=True) inside
    assert detect_format(payload) == ALI_TRAINING_FORMAT
    saved = payload["optimizer_bundle"]
    assert saved["kind"] == "muon_with_aux_adamw" and set(saved["optimizers"]) == {"muon", "aux_adamw"}
    assert saved["muon_backend"] in ("torch.optim.Muon", "local_fallback")
    muon_state = saved["optimizers"]["muon"]["state"]
    assert muon_state and all("momentum_buffer" in entry for entry in muon_state.values())
    loaded = load_bc_checkpoint(path)
    assert loaded.model_config == tiny_config and set(loaded.state_dict) == set(policy.state_dict())
    assert loaded.metadata["run_name"] == "5m-muon"
    assert loaded.metadata["processed_target_frames"] == 41943040
    assert _same_params(load_policy(path), policy)
    # The value types of the real file's param groups (probed 24 Aug 2026): a tuple of Newton-Schulz
    # coefficients, a ``None`` adjust_lr_fn, AdamW's flags.  A synthetic payload with exactly those
    # types round-trips as well, whatever Muon backend the test host has.
    muon_group = {
        "params": [0],
        "lr": 1e-2,
        "base_lr": 1e-2,
        "momentum": 0.95,
        "nesterov": True,
        "ns_coefficients": (3.4445, -4.775, 2.0315),
        "ns_steps": 5,
        "eps": 1e-7,
        "weight_decay": 0.01,
        "adjust_lr_fn": None,
    }
    adamw_group = {
        "params": [0],
        "lr": 2e-3,
        "base_lr": 2e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    synthetic = {
        "format_version": 1,
        "model": policy.state_dict(),
        "optimizer_bundle": {
            "kind": "muon_with_aux_adamw",
            "muon_backend": "torch.optim.Muon",
            "optimizers": {
                "muon": {"state": {0: {"momentum_buffer": torch.zeros(4, 4)}}, "param_groups": [muon_group]},
                "aux_adamw": {
                    "state": {
                        0: {
                            "step": torch.tensor(1.0),
                            "exp_avg": torch.zeros(3),
                            "exp_avg_sq": torch.zeros(3),
                        }
                    },
                    "param_groups": [adamw_group],
                },
            },
        },
        "training_state": asdict(state),
        "frame_cursor": 41943040,
        "rng_state": {"python": (3, (1, 2, 3), None), "torch_cpu": torch.get_rng_state(), "torch_cuda": None},
        "resolved_config": {"model": asdict(tiny_config), "profile": None},
        "metadata": {"run_name": "synthetic"},
    }
    atomic_save(synthetic, tmp_path / "synthetic.pt")
    again = load_bc_checkpoint(tmp_path / "synthetic.pt")
    assert again.format == ALI_TRAINING_FORMAT and again.metadata["run_name"] == "synthetic"
    assert again.payload["optimizer_bundle"]["optimizers"]["muon"]["param_groups"][0]["ns_steps"] == 5


# ---------------------------------------------------------------------------
# Ali's final-pretraining runtime schema (1 Sep 2026): the 10m family checkpoint
# ---------------------------------------------------------------------------


def test_ali_runtime_schema_checkpoint_builds_its_model_config_from_the_profile(tmp_path: Path) -> None:
    """``melee-policy.final-pretraining-runtime.v1`` files (Ali's family-pretraining runs, 1 Sep 2026)
    keep ``format_version 1`` and the ``model`` state dict, but ``resolved_config["model"]`` is a
    16-field summary -- ``profile``, the six dimensions, parameter counts -- not the 54-field
    ``ModelConfig``.  The loader rebuilds the config from ``config.yaml``'s profile and checks the
    summary against it, so a wrong profile or a foreign state dict cannot load silently."""

    from melee_rl.checkpoint import RUNTIME_SCHEMA_VERSION, default_config_yaml

    base = ModelConfig.from_yaml(default_config_yaml(), profile="3m")
    torch.manual_seed(7)
    adapter = MeleePolicyAdapter(base)
    state = {name: value.clone() for name, value in adapter.state_dict().items()}
    total = sum(value.numel() for value in state.values())
    summary = {
        "profile": "3m",
        "d_model": base.d_model,
        "n_layers": base.n_layers,
        "n_heads": base.n_heads,
        "n_kv_heads": base.n_kv_heads,
        "head_dim": base.head_dim,
        "d_ff": base.d_ff,
        "encoder_parameters": 0,
        "backbone_parameters": 0,
        "controller_head_parameters": 0,
        "total_trainable_parameters": total,
        "muon_matrix_count": 0,
        "muon_parameters": 0,
        "auxiliary_parameters": 0,
        "gradient_checkpointing": False,
        "prevalidated_inputs": True,
    }
    payload = {
        "format_version": 1,
        "model": state,
        "resolved_config": {
            "model": summary,
            "training": {"context_length": base.context_length, "action_offset_frames": 1},
        },
        "metadata": {"schema_version": RUNTIME_SCHEMA_VERSION, "trial_id": "3m-test"},
        "training_state": {"processed_target_frames": 4096, "optimizer_steps": 2},
        "frame_cursor": 4096,
    }
    path = tmp_path / "runtime.pt"
    atomic_save(payload, path)
    assert detect_format(load_checkpoint(path)) == ALI_TRAINING_FORMAT
    loaded = load_bc_checkpoint(path)
    assert loaded.format == ALI_TRAINING_FORMAT and loaded.model_config == base
    assert loaded.metadata["processed_target_frames"] == 4096 and loaded.metadata["trial_id"] == "3m-test"
    assert loaded.metadata["schema_version"] == RUNTIME_SCHEMA_VERSION
    policy = load_policy(path)
    assert sum(parameter.numel() for parameter in policy.parameters()) == total
    assert all(torch.equal(policy.state_dict()[name], value) for name, value in state.items())
    assert policy.config.context_length == base.context_length  # RL precision only changes the dtypes

    def variant(name: str, **changes: object) -> Path:
        summary_changed = {**summary, **changes}
        target = tmp_path / name
        atomic_save({**payload, "resolved_config": {"model": summary_changed}}, target)
        return target

    with pytest.raises(ValueError, match="total_trainable_parameters"):
        load_bc_checkpoint(variant("bad-count.pt", total_trainable_parameters=total + 1))
    with pytest.raises(ValueError, match="d_model"):
        load_bc_checkpoint(variant("bad-dim.pt", d_model=base.d_model + 1))
    with pytest.raises(ValueError, match="profile"):
        load_bc_checkpoint(variant("bad-profile.pt", profile="no-such-profile"))
    # A summary without a profile is not loadable at all: nothing says which architecture it is.
    with pytest.raises(ValueError, match="profile"):
        load_bc_checkpoint(variant("no-profile.pt", profile=None))


def test_extra_profiles_fill_in_what_ali_s_config_yaml_lacks() -> None:
    """Ali's 75m family checkpoint (``75m-muon-low``, ``step-86016.pt``; 4 Sep 2026) names profile ``75m``,
    which ``config.yaml`` on this branch does not define (his ``pretraining`` / ``benchmark-launcher``
    branches do: d_model 768, 11 layers, 12 heads, 3 kv heads, d_ff 1920, 75 305 709 parameters).
    ``model_config_from_yaml`` injects the missing entry from ``EXTRA_PROFILES`` -- the YAML's own entry wins
    once it has one -- and ``ali_model_config`` still checks the summary against the state dict."""

    import yaml

    from melee_rl.checkpoint import (
        EXTRA_PROFILE_PARAMETERS,
        EXTRA_PROFILES,
        PROFILE_SUMMARY_FIELDS,
        RUNTIME_SCHEMA_VERSION,
        ali_model_config,
        model_config_from_yaml,
    )

    root = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    profiles = root["scaling"]["profiles"]
    assert set(EXTRA_PROFILES) == {"75m"} and set(EXTRA_PROFILES["75m"]) == set(PROFILE_SUMMARY_FIELDS)
    # Profiles the YAML defines are untouched (the active profile too).
    assert model_config_from_yaml(CONFIG_YAML, "3m") == ModelConfig.from_yaml(CONFIG_YAML, profile="3m")
    assert model_config_from_yaml(CONFIG_YAML) == ModelConfig.from_yaml(CONFIG_YAML)
    config = model_config_from_yaml(CONFIG_YAML, "75m")
    assert config.profile == "75m"
    for name, value in EXTRA_PROFILES["75m"].items():
        assert getattr(config, name) == value, name
    assert config.context_length == 256 and config.action_offset_frames == 1
    if "75m" in profiles:  # Ali's YAML has caught up: it wins, and it must agree with the table
        assert profiles["75m"] == dict(EXTRA_PROFILES["75m"])
    else:
        with pytest.raises(KeyError, match="75m"):
            ModelConfig.from_yaml(CONFIG_YAML, profile="75m")
    # The real 75m: exactly Ali's expected_parameter_counts total on his branches.
    torch.manual_seed(0)
    total = sum(parameter.numel() for parameter in MeleePolicy(config).parameters())
    assert total == EXTRA_PROFILE_PARAMETERS["75m"] == 75_305_709
    # A runtime-schema summary naming 75m gets past the profile lookup and is still checked against the
    # state dict: a foreign (tiny) state dict fails on the count, a wrong dimension on the dimension, an
    # unknown profile on the profile.
    tiny = ModelConfig.from_yaml(CONFIG_YAML, profile="3m")
    state = {name: value.clone() for name, value in MeleePolicyAdapter(tiny).state_dict().items()}
    summary = {"profile": "75m", **EXTRA_PROFILES["75m"], "total_trainable_parameters": 75_305_709}
    payload = {
        "format_version": 1,
        "model": state,
        "resolved_config": {"model": summary},
        "metadata": {"schema_version": RUNTIME_SCHEMA_VERSION},
    }
    with pytest.raises(ValueError, match="total_trainable_parameters"):
        ali_model_config(payload)
    with pytest.raises(ValueError, match="d_model"):
        ali_model_config({**payload, "resolved_config": {"model": {**summary, "d_model": 512}}})
    with pytest.raises(ValueError, match="not a profile"):
        ali_model_config({**payload, "resolved_config": {"model": {**summary, "profile": "750m"}}})


def test_ali_curriculum_checkpoint_with_a_frozenset_loads_weights_only(tmp_path: Path) -> None:
    """Ali's curriculum checkpoints (5 Sep 2026: ``cur-6-kd/10m/B-mix10/steps/step-195248.pt`` and
    ``cur-3/75m/B-mix10/steps/step-127214.pt``) keep the runtime schema but pickle a ``frozenset`` inside
    ``resolved_config["curriculum"]`` and ``metadata["curriculum"]`` (the secondary stage's ``ranks``), and
    their ``metadata`` carries no ``schema_version``.  ``torch.load(weights_only=True)`` refuses every global
    outside its allow-list, ``frozenset`` included, so the loader allow-lists exactly that builtin
    (:data:`SAFE_GLOBALS`; no code can hide in a ``frozenset``) and the file reads through every reader with
    the set intact."""

    import pickle

    from melee_rl.checkpoint import SAFE_GLOBALS, default_config_yaml

    assert len(SAFE_GLOBALS) == 1 and frozenset in SAFE_GLOBALS
    base = ModelConfig.from_yaml(default_config_yaml(), profile="3m")
    torch.manual_seed(11)
    adapter = MeleePolicyAdapter(base)
    state = {name: value.clone() for name, value in adapter.state_dict().items()}
    total = sum(value.numel() for value in state.values())
    summary = {
        "profile": "3m",
        "d_model": base.d_model,
        "n_layers": base.n_layers,
        "n_heads": base.n_heads,
        "n_kv_heads": base.n_kv_heads,
        "head_dim": base.head_dim,
        "d_ff": base.d_ff,
        "total_trainable_parameters": total,
    }
    secondary = {"name": "diamond-winners", "ranks": frozenset({"diamond"}), "secondary": None}
    stage = {"name": "master-winners+10pct-diamond-winners", "ranks": ["master"], "secondary": secondary}
    curriculum = {"schema_version": "melee-policy.curriculum.v1", "arm": "B-mix10", "stages": [stage]}
    payload = {
        "format_version": 1,
        "model": state,
        "resolved_config": {"model": summary, "curriculum": curriculum, "launch": {"launch_id": "cur-test"}},
        "metadata": {"curriculum": curriculum},  # no schema_version, as in the real files
        "training_state": {"processed_target_frames": 8192, "optimizer_steps": 4},
        "frame_cursor": 8192,
    }
    path = tmp_path / "curriculum.pt"
    atomic_save(payload, path)
    with pytest.raises(pickle.UnpicklingError, match="frozenset"):
        torch.load(path, map_location="cpu", weights_only=True)  # the default allow-list: the trap
    raw = load_checkpoint(path)
    assert detect_format(raw) == ALI_TRAINING_FORMAT
    assert raw["resolved_config"]["curriculum"]["stages"][0]["secondary"]["ranks"] == frozenset({"diamond"})
    loaded = load_bc_checkpoint(path)
    assert loaded.format == ALI_TRAINING_FORMAT and loaded.model_config == base
    assert loaded.metadata["processed_target_frames"] == 8192 and "schema_version" not in loaded.metadata
    assert loaded.metadata["curriculum"]["arm"] == "B-mix10"
    policy = load_policy(path)
    assert sum(parameter.numel() for parameter in policy.parameters()) == total
    assert all(torch.equal(policy.state_dict()[name], value) for name, value in state.items())
