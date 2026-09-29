# Faynt evaluation runtime

This is the source-only export of the game execution code on `bicro/research_playground` at `d4673486dcf6bbb1c9970a793c9a85f2cb735af8` (`rl_post_training`, September 22, 2026). It includes the Phillip and Slippi-AI comparison paths, the privately supplied zero-delay checkpoint interface, mirrored controller handling, replay collection, terminal-result parsing, policy timing and checkpoint loading used by the research runs.

The local benchmark code in the parent export provides the initial and expanded panels. This directory supplies the later game recorder and its configuration files. `SOURCE_MANIFEST.json` identifies the source files and modifications. The game loop, policy timing and result interpretation retain their recorded implementations.

The native `checkpoint.pt` files in the [Faynt model family](https://huggingface.co/collections/frisson-labs/faynt) load through this runtime. The six published checkpoints were checked with strict state loading: 10,163,629 parameters for 10M and 75,305,709 for 75M, with zero added delay. Obtain a checkpoint separately and pass its local path.

## Prepare a runtime

Use Python 3.12 for the main runtime. From the benchmark repository root, enter `evaluation/` and install its Python package in your own environment:

```bash
cd evaluation
python -m pip install -e '.[runtime,dev]'
```

Run the commands below from this directory. The package installs Python dependencies. Obtain the emulator, game image, opponent source checkouts and checkpoint files separately. Supply their paths in a copy of the selected TOML configuration or with repeated `--override table.key=value` arguments.

The historical Linux game engine was the mainline ExiAI 14.1 build with libmelee 0.47.3. The configuration files preserve its expected runtime paths as examples. Phillip uses its pinned source `114aea4c446b31359c5637b271db52b200440caa` with Python 3.11 and TensorFlow CPU 2.13.1 in a separate interpreter. MIMIC, Slippi-AI and the zero-delay Slippi-AI checkpoint have their own source and checkpoint identity checks. Keep those checks enabled.

## Inspect a protocol

```bash
python -m melee_rl.release_cli plan --config video_phillip
python -m melee_rl.release_cli plan --config video_phillip_stages
python -m melee_rl.release_cli plan --config video_slippi_ai
python -m melee_rl.release_cli plan --config video_slippi_ai_stages
```

The commands read the stored settings without launching games. Final Destination configurations use 16 games per opponent. Six-stage configurations use 18 games per opponent. The seven Phillip delay-zero checkpoints cover 112 and 126 games respectively. Phillip still acts every two, three or four frames according to its checkpoint configuration. Faynt's added delay remains zero in these configurations.

## Run games

```bash
python -m melee_rl.release_cli run \
  --config video_phillip --device cpu \
  --checkpoint /absolute/path/to/faynt-checkpoint.pt \
  --dolphin /absolute/path/to/dolphin-emu \
  --game /absolute/path/to/game.iso \
  --output /absolute/path/to/results \
  --override 'phillip.source_dir="/absolute/path/to/phillip"' \
  --override 'phillip.python="/absolute/path/to/phillip-venv/bin/python"' \
  --override 'phillip.agent="delay0/FoxFD"'
```

Use the character pairing corresponding to the selected opponent. `melee_rl/phillip.py` lists the seven releases and performs the compatibility checks. Each game produces a replay and a result record. A failed game remains visible in the report. The release command also writes `release-run.json` and returns a failing status when a clip reports an error.

The private zero-delay comparison uses `video_slippi_ai` and `video_slippi_ai_stages`, with `slippi_ai.model="fox_d0_tx_like_3x512"` and the recorded `slippi_ai.name` settings, `Master Player` and `Cody`. The selected model's source checkout and checkpoint must be supplied by someone authorized to provide them. The two name settings cover 68 games per Faynt checkpoint.

## Cloud execution

`melee_rl/modal_app.py` retains the research match-launch functions and resource assignments. Its automatic emulator installation, opponent cloning, checkpoint acquisition and playback-download fallback have been disabled. The source package is mounted through a frozen, hash-checked text-file allowlist.

Set `FAYNT_RUNTIME_IMAGE` to a digest-pinned private OCI image that you have prepared separately. Per-runtime overrides use `FAYNT_CPU_IMAGE`, `FAYNT_GPU_IMAGE`, `FAYNT_MIMIC_IMAGE`, `FAYNT_SLIPPI_IMAGE`, `FAYNT_PHILLIP_CPU_IMAGE`, `FAYNT_PHILLIP_GPU_IMAGE`, `FAYNT_SMASHBOT_CPU_IMAGE`, `FAYNT_SMASHBOT_GPU_IMAGE`, and `FAYNT_PLAYBACK_IMAGE`. Set `FAYNT_MODAL_ENVIRONMENT`, `FAYNT_RUNS_VOLUME`, `FAYNT_GAME_VOLUME`, and `FAYNT_WANDB_SECRET` to resources you control. The game volume must already exist. Runtime images and external assets remain private resources supplied by the operator.

Install the optional Modal client in the same environment with `python -m pip install -e '.[modal]'`. With those prerequisites available, the retained command interface is:

```bash
modal run -m melee_rl.modal_app::video --phillip \
  --config video_phillip --checkpoint /runs/checkpoints/faynt.pt \
  --output-dir /runs/results/phillip --spawn
```

The source export does not execute cloud jobs during installation or validation. Running this command starts paid compute. See `THIRD_PARTY_NOTICES.md` for dependency provenance.
