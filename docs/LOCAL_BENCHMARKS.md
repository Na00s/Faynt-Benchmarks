# Local benchmark schedules and execution

This guide covers the initial **Base (pretrained)** and **Expert (supervised post-trained)** benchmarks and the expanded **Arena (reinforcement-learned)** panel. Expert includes curricula for both sizes and distillation for 10M. The separate [evaluation runtime](../evaluation/README.md) runs the Phillip and private zero-delay comparisons. The historical post-RL continuation uses 10M step 1,318 and 75M step 222; the final Arena endpoints are the second-run 10M step 632 and 75M step 980.

| Entry point | Scope |
| --- | --- |
| `scripts/run_final_winner_benchmark.py` | Frozen Base schedule, 309 physical games |
| `scripts/run_posttraining_winner_benchmark.py` | Frozen Expert schedule, 329 physical games |
| `scripts/run_post_rl_benchmark.py` | Earlier RL continuation: 10M step 1,318 and 75M step 222 |
| `scripts/slippi_panel_plan.py` | Original expanded schedule for the earlier RL checkpoints, 2,624 games against 14 public releases |
| `scripts/prepare_faynt_d0_benchmark.py` | Exact checkpoint-only transformation of that panel for final Arena 632/980 |
| `scripts/prepare_faynt_d0_cloud.py` | Build its immutable cloud queue locally from a qualified template |
| `scripts/modal_panel_cloud_app.py` | Cloud queue, workers, retries, ownership, budget and durable result handling |
| `scripts/report_slippi_public_panel.py` | Paired descriptive results with separate supported/transfer strata |

The initial reported score uses 152 games per Base or Expert model. Its replay-fidelity audits are complete. The full schedules also contain five shared cross-size games; the Expert schedule adds ten ancestor games per model. Expanded coverage per model is 244 supported-mirror, 534 extended-roster and 534 forced-mirror games. Each matched cell runs both physical ports with the same seed, stage and characters. Public opponent decision delays and native name conditioning stay in the frozen metadata. The final Arena contract is FP32, temperature 1, no added Faynt policy delay, one-frame action offset and a continuous 256-frame ring cache. The recorded actor trajectory-context setting is 128 frames; cache capacity and the trained sequence length are both 256. The `d0` names describe Faynt's delay; the public Slippi-AI opponents retain their recorded 21- or 24-frame queues.

## Required local inputs

Use Python 3.12 and the checked-in `requirements-e010.lock` plus shared runtime setup described in the repository. The validated local environment contained Torch 2.7.1, TensorDict 0.9.1, melee 0.47.3, NumPy 2.2.6, tf-nightly 2.21.0.dev20260203, py-ubjson 0.16.1 and pytest 9.1.1. These are observed research environment versions; the cloud Linux lock records its own 94-distribution environment, including Torch 2.7.1+cpu.

Supply the native Faynt checkpoint bytes at the exact relative paths and SHA-256 identities declared in `final_benchmark_suite.py`, `post_rl_checkpoints.py` and `faynt_d0_checkpoints.py`. HF inference-only weights and complete training checkpoints can have different file hashes. Supply third-party model files, the pinned Slippi-AI source checkout, a compatible prebuilt Dolphin package and a legally obtained game image at the configured paths. None of those assets are distributed here. The three owned Faynt model-source files are bundled and verified against their original hashes.

The expanded opponent canary uses the replay named `.e000-cache/raw/039d70ae59bc37fc1fe5b72c1dadcd95d2b2877b9ad1964bf978aaa6662962ad.slp`. That external replay is required to regenerate the same 64-frame command comparison. Live scoring additionally retains raw replay files and validates physical ports, characters, stage, stocks and natural termination.

## Initial schedules

From the repository root, with dependencies and assets supplied:

```sh
export PYTHONPATH=src:scripts
python scripts/run_final_winner_benchmark.py --preflight-only
python scripts/run_final_winner_benchmark.py --maximum-games 1
python scripts/run_final_winner_benchmark.py
python scripts/run_posttraining_winner_benchmark.py --preflight-only
python scripts/run_posttraining_winner_benchmark.py
```

`--label`, `--group` and `--maximum-games` restrict a launch. `--status-only` reconciles the retained state. Each invocation verifies the frozen model/opponent identities and preserves unresolved postgame validation records. A completed game with an unresolved audit remains distinct from an accepted result.

## Earlier RL expanded local panel

The published frozen manifest and audit summary preserve public metadata and every scientific game field. Personal path prefixes were removed; `docs/metadata-provenance.json` records original and release hashes.

```sh
export PYTHONPATH=src:scripts
python scripts/slippi_panel_canary.py --all
python scripts/slippi_panel_canary.py --release medium-v2 --reference
python -m pytest tests/integration/test_final_benchmark_suite.py tests/integration/test_posttraining_benchmark_variant.py tests/integration/test_faynt_d0_benchmark_plan.py tests/integration/test_slippi_public_panel.py --junitxml=artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/tests.junit.xml
python scripts/slippi_panel_deploy.py --freeze
python scripts/run_slippi_public_panel.py
```

The freeze step requires successful checkpoint canaries, native adapter agreement and at least 100 passing tests. The foreground runner uses the earlier 10M step 1,318 and 75M step 222 endpoints. The final Arena 632/980 run uses the separate preparation and cloud queue below. Its release readiness gate checks the selected full schedule instead of requiring completion of an unrelated earlier campaign. The optional historical `slippi_panel_deploy.py --install` command installs macOS launchd services and starts their supervisor; use it only when that persistent operation is intended.

## Final Arena cloud panel

This is the path for the paper's **1,312 games per final RL checkpoint**, using 10M leash step 632 and 75M step 980. Its reported results are [the frozen Arena ledger](../results/rl_current_scores.json).

```sh
export PYTHONPATH=src:scripts
python scripts/prepare_faynt_d0_benchmark.py
python scripts/prepare_faynt_d0_cloud.py --template /absolute/path/to/fresh-qualified-template/plan.json --template-sha256 SHA256_OF_THAT_PLAN
```

The first command verifies both selected checkpoint files and writes a fresh 2,624-game schedule/state. The second command packages an immutable queue after validating a full-game template and its input hashes. It uploads nothing and starts no game. `--reference-policy` permits an alternate local file location for the exact policy source hash already bound by that template. Original historical defaults remain identifiable in source; a new release run needs a newly qualified template.

The recorded queue uses six phases, 64 logical worker slots, paired port assignments and a historical $150 cap with a $10 reserve. Those numbers describe the saved experiment. Operator authorization and current provider account limits govern a new launch. Budget activation, failure history and accepted results are separate from game selection; retry handling preserves native-start evidence and prohibits replacing an already consumed result with a new sample.

## Provided cloud image and fresh qualification

All cloud constructors require `FAYNT_BENCHMARK_IMAGE` containing a registry reference pinned by `@sha256:`. The image must already contain:

- Ubuntu 22.04 x86_64 and the runtime libraries recorded by `modal_panel_live_context.runtime_spec()`.
- The exact Python environment at `/work/runtime-pilot/venv/bin/python` and the verified emulator package at `/work/runtime-pilot/package`.
- A clean Slippi-AI checkout at `/opt/prepared/slippi-ai`, revision `577965a7731dc53e3472ea63d9e9853a4e9d65fa`. `FAYNT_PREPARED_SLIPPI_SOURCE` can select another canonical path inside the image.

The release performs no package download, emulator compilation, game acquisition or opponent-source fetch. It verifies installed package files, Python distribution versions, compiled UBJSON, source identity, loader state and fresh Xvfb ownership before admitting gameplay. Original lock and package audit metadata are included unchanged, with hashes in `docs/runtime-metadata-provenance.json`.

Fresh runtime qualification retains the existing CLI:

```sh
python scripts/modal_panel_runtime_pilot.py --prepare artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/research/modal-compat-v1/release-runtime-qualification
```

This creates a local plan. Running that plan in Modal is a separate explicit operation documented by `--help` and requires its reviewed SHA. Native policy qualification remains in `modal_panel_policy_pilot.py`; its existing canary runner checks 384 policy steps per case using caller-supplied checkpoints and replay-derived inputs. The controller-tape qualifier, full-game plan builder and durable worker remain in `modal_panel_live_pilot.py`, `modal_panel_game_pilot.py` and `faynt_d0_durable_game.py`.

The retained historical policy comparisons include documented numeric portability limits. Sanitized receipts and template records are in `docs/historical-cloud/`, with original and release SHA-256 values. They describe the past experiment. Source and environment changes require fresh qualification; these archived receipts do not establish numeric admission for the release. A complete new cloud run requires the provided runtime, external policies/replay/game image, fresh qualification records and an operator-created full-game template. No gameplay or cloud execution was performed during this source release.

## Default source validation

From this repository root, `python -m pytest` runs the local integration suite using `pytest.ini`. The final release check passed 2,225 tests and explicitly skipped 141 tests whose external assets, historical records or optional dependencies were unavailable. The skip remains conditional, so supplying the exact fixture enables its validation. The remote `evaluation/` package has its own environment and tests. No emulator game or cloud job ran during this validation.
