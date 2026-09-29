"""RL post-training for Ali's ``MeleePolicy`` -- pure PyTorch.

Phase P0 (21 Aug 2026): ``protocol`` (what the RL package requires from a policy),
``adapter`` (``MeleePolicy`` behind that protocol), ``distributions`` (closed-form
log-prob / KL / entropy over ``custom_v1`` component logits) and ``frames`` (frame
trees and label helpers).  Phase P1 (21 Aug 2026): ``env.protocol`` / ``env.dummy``
(batched environment contract and the deterministic toy Melee), ``reward`` (slippi-ai
reward semantics), ``trajectory`` (the batch-major rollout record and its learner
slices), ``actor`` (``RolloutWorker`` with delay queue, history ring and re-priming)
and ``opponents`` (``cpu`` / ``self``).  Phase P2 (21 Aug 2026): ``rl_lib`` (discounts and
returns), ``value`` (``ValueNet`` and value targets), ``learner`` (``PPOLearner`` with the
teacher KL, the separate value Adam, the actor-KL guard and the burn-in ladder) and
``checkpoint`` (RL checkpoints and the proposed BC export format).  Phase P3 (21 Aug 2026):
``config`` (TOML run files -> frozen dataclasses), ``logging`` (metric names, the W&B wrapper)
and ``train`` (``Trainer`` / ``python -m melee_rl.train``, the slippi-ai ``run_lib`` loop shape
with checkpoint resume); run files live in ``configs/rl/``.  Phase P4 (21 Aug 2026): ``modal_app``
(the Modal harness: one CPU image, ``run_tests`` / ``smoke_train`` functions, ``tests`` / ``train``
entrypoints, environment ``smash_melee_bot``; every cloud run needs an estimate and a yes).  Phase P5
(22 Aug 2026): ``env.libmelee_frames`` (libmelee ``GameState`` -> frame tree, a port of slippi-ai's
``parse_libmelee``) and ``env.dolphin`` (``DolphinEnv``: headless ExiAI Dolphins through libmelee 0.47.3
behind ``EnvProtocol``, relaunch on faults, optional keepalive), ``[env] type = "dolphin"`` run files
(``configs/rl/dolphin_{cpu,self}.toml``) and the Modal Dolphin image / ``dolphin_train`` / ``dolphin_check``.
Phase P6a (22 Aug 2026): two-port self-play (``opponent.train``: ``RolloutWorker(ports=(0, 1))`` trains
both perspectives, port-major rows), the ``other`` opponent from a checkpoint (``opponents.OtherOpponent``),
the remaining slippi-ai reward terms (ledge grabs, stalling, approach), Ali's training-checkpoint format
in ``checkpoint`` (``detect_format``), ``DolphinEnv`` two-phase boot + threaded stepping
(``step_threads`` / ``boot``), the Modal GPU Dolphin image / ``dolphin_train_gpu`` (L4, ``--gpu``) and the
8-Dolphin run files ``configs/rl/dolphin_{cpu8,self8,other8}.toml``.  Phase P6b item 0 (22 Aug 2026):
``evaluate`` (the in-loop evaluator: ``[eval]`` plays a frozen copy of the student against ``cpu<level>`` /
``best`` / ``other:<path>`` opponents every ``interval_seconds`` and logs ``eval/<opponent>/<stat>`` to the
same W&B run; ``best.pt`` + ``eval_history.json`` next to ``latest.pt``; the run file
``configs/rl/dolphin_cpu8_eval.toml``).  Phase P7 (25 Aug 2026): ``env.dolphin_mp``
(``WorkerDolphinEnv``: the Dolphins split over OS worker processes behind the same protocol) and
``fast_step`` (``FastStepper`` / ``fast_sample``: the vectorised ring-attention policy step,
``[policy] fast_step``).  Phase P8 (25 Aug 2026): the delay pipeline -- ``env.async_env``
(``AsyncEnvProtocol`` / ``AsyncEnvAdapter``: push / pop / peek with the initial state in transit),
``pipeline`` (``PipelinedRolloutWorker``: ``batch_steps`` frames per policy call,
``chunk_frames`` frames per worker message, runahead ``delay - (batch_steps - 1)``, bit-identical
to the lockstep loop on the dummy env), ``adapter.step_sample_multi``, ``frames.tree_to_packed``
and the slack rule ``(b - 1) + (k - 1) <= delay_frames`` in ``config.finalize_config``.
Phase P9 (26 Aug 2026): qualitative evaluation -- ``video`` (``[video]``: on demand, for a selected
checkpoint, ``clips`` clips of ``seconds`` seconds of ``policy`` vs ``cpu<level>`` / the BC checkpoint
and of the BC checkpoint vs the CPUs; each clip keeps its ``.slp`` and a row of the evaluator's
statistics, and the BC clips are reused per BC sha256) and ``render`` (``[video.render]``: a second
Dolphin in playback mode dumps frames, ffmpeg burns in the caption and ffprobe verifies every file --
the training Dolphins cannot render, their fast-forward gecko codes switch Melee's rendering off), plus
the Modal video image and the ``video`` / ``render`` / ``video_check`` entrypoints and the run file
``configs/rl/video_20m_bc.toml``.
Phase P10 (26 Aug 2026): the mixed opponent curriculum -- ``[opponent] type = "mix"`` with specs
``["self*32", "cpu9*64"]`` (``opponents.MixGroup`` / ``parse_mix_spec`` / ``make_group_opponent``):
one environment and one rollout worker per arm feeding the same learner step, per-arm statistics
under ``arm/<name>/*``, ``mix_workers`` splitting the worker processes, and ``[policy] init_value``
warm-starting the value net from the RL checkpoint the policy starts from; single-group mixes are
bit-identical to the plain opponent types.
Design: ``docs/rl-post-training/PLAN.md`` and ``INTERFACE.md``.  Nothing in ``model.py``,
``controller_codec.py`` or ``tensor_batch.py`` is modified by this package.

The package deliberately imports nothing at module level so that ``import melee_rl``
is cheap; import the submodules you need.
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "0.0.1"
