# Faynt evaluation runtime

This is the source-only export of the game execution code on `bicro/research_playground` at `d4673486dcf6bbb1c9970a793c9a85f2cb735af8` (`rl_post_training`, September 22, 2026). It includes the Phillip and Slippi-AI comparison paths, the privately supplied zero-delay checkpoint interface, mirrored controller handling, replay collection, terminal-result parsing, policy timing and checkpoint loading used by the research runs.

The local benchmark code in the parent export provides the initial and expanded panels. This directory supplies the later game recorder and its configuration files. `SOURCE_MANIFEST.json` identifies the source files and modifications. The game loop, policy timing and result interpretation retain their recorded implementations.

## Model stages and recorded evaluations

The native `checkpoint.pt` files for all six models in the [Faynt family](https://huggingface.co/collections/frisson-labs/faynt) load through this runtime. Base is pretrained; Expert adds supervised curricula and, for 10M, distillation; Arena adds Fox-only reinforcement learning. [Recorded compatibility checks](CHECKPOINT_COMPATIBILITY.json) verified strict state loading, 10,163,629 parameters for 10M and 75,305,709 for 75M, and zero added delay.

| Native checkpoint | Selected step | Paper evaluation coverage |
|---|---:|---|
| `Faynt-10M-Base/checkpoint.pt` | 122,064 | Initial 152-game panel in the parent runtime |
| `Faynt-75M-Base/checkpoint.pt` | 86,016 | Initial 152-game panel in the parent runtime |
| `Faynt-10M-Expert/checkpoint.pt` | 195,248 | Initial panel; 68 private zero-delay games |
| `Faynt-75M-Expert/checkpoint.pt` | 127,214 | Initial panel; 68 private zero-delay games |
| `Faynt-10M-Arena/checkpoint.pt` | Second RL run, 632 | Expanded 1,312-game panel; 68 private zero-delay games; 238 Phillip games |
| `Faynt-75M-Arena/checkpoint.pt` | RL run, 980 | Expanded 1,312-game panel; 68 private zero-delay games; 238 Phillip games |

The four evaluation configurations below accept any of these six native checkpoints through `--checkpoint`. The loader reads the architecture from the checkpoint and checks the actor context and delay against it; the configuration's default `policy.profile="10m"` is a fallback. Passing a different model creates a new evaluation under that protocol. The table identifies which stage results the paper reports. Its Phillip `Init` records retain their original unresolved stage labels, and the earlier 75M RL step 222 comparison remains separate. Obtain the checkpoint independently and pass its local path.

## Prepare a runtime

Use Python 3.12 for the main runtime. From the benchmark repository root, enter `evaluation/` and install its Python package in your own environment:

```bash
cd evaluation
python -m pip install -e '.[runtime,dev]'
```

Run the commands below from this directory. The package installs Python dependencies. Obtain the emulator, game image, opponent source checkouts and checkpoint files separately. Supply their paths in a copy of the selected TOML configuration or with repeated `--override table.key=value` arguments.

The historical Linux game engine was the mainline ExiAI 14.1 build with libmelee 0.47.3. The configuration files preserve its expected runtime paths as examples. The package extras above install the Faynt core and game transport. Prepare the selected opponent's environment as well:

| Opponent | Required source and runtime |
|---|---|
| Phillip | Source `114aea4c446b31359c5637b271db52b200440caa`, with Python 3.11 and TensorFlow CPU 2.13.1 in a separate interpreter |
| Public Slippi-AI releases | Source `577965a7731dc53e3472ea63d9e9853a4e9d65fa` and the recorded TensorFlow/Sonnet dependency stack in the main environment |
| Private zero-delay Slippi-AI | Source `9eca7479a9555f4ee4b7e6800c656a7f5bb55902`, the Slippi-AI dependency stack, and JAX 0.11.1, jaxlib 0.11.1, Flax 0.12.9 in the main environment |

The exact historical dependency lists are retained as `SLIPPI_AI_PIP_PACKAGES` and `SLIPPI_AI_JAX_PIP_PACKAGES` in [modal_app.py](melee_rl/modal_app.py). The private model uses its own source revision; an explicit `slippi_ai.source_dir` must select that checkout. MIMIC, Slippi-AI and Phillip also validate their model or asset identities. Keep those checks enabled.

## Inspect a protocol

```bash
python -m melee_rl.release_cli plan --config video_phillip
python -m melee_rl.release_cli plan --config video_phillip_stages
python -m melee_rl.release_cli plan --config video_slippi_ai
python -m melee_rl.release_cli plan --config video_slippi_ai_stages
```

The commands read the stored settings without launching games. Final Destination configurations use 16 games per opponent. Six-stage configurations use 18 games per opponent, three each on Final Destination, Battlefield, Pokémon Stadium, Dream Land, Fountain of Dreams, and Yoshi's Story. `video_slippi_ai` and `video_slippi_ai_stages` default to a public delayed Fox release; select the private zero-delay model explicitly as shown below. The seven Phillip delay-zero checkpoints cover 112 and 126 games respectively. Phillip still acts every two, three or four frames according to its checkpoint configuration. Faynt's added delay remains zero in these configurations.

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

The example runs one 16-game Fox-mirror block with the supplied checkpoint. For the six-stage block, select `--config video_phillip_stages`. The character pair is ordered `[Faynt, Phillip]`; changing `phillip.agent` does not change that pair automatically.

For native-character mirrors, set `--override 'env.dolphin.characters=[ID,ID]'` using the specialist's ID below. For Fox against a specialist, set `--override 'env.dolphin.characters=[1,ID]'`. For example, `delay0/FalcoFD` uses `[22,22]` for Falco mirrors and `[1,22]` for Faynt Fox against Phillip Falco. Keep the selected character pair with the run's results so these protocols remain distinguishable.

| `phillip.agent` | Fighter | ID |
|---|---|---:|
| `delay0/FoxFD`, `FoxFD0` | Fox | 1 |
| `delay0/FalcoFD` | Falco | 22 |
| `MarthFD0` | Marth | 18 |
| `PeachFD` | Peach | 9 |
| `SheikFD` | Sheik | 7 |
| `FalconFalconBF` | Captain Falcon | 2 |

Use distinct output directories for each checkpoint, opponent and stage condition. `melee_rl/phillip.py` lists the seven releases and performs the compatibility checks. Each game produces a replay and a result record. A failed game remains visible in the report. The release command also writes `release-run.json` and returns a failing status when a clip reports an error.

For the private zero-delay Fox-mirror comparison, provide its pinned source checkout and checkpoint through the declared local directories, then run:

```bash
python -m melee_rl.release_cli run \
  --config video_slippi_ai --device cpu \
  --checkpoint /absolute/path/to/Faynt-10M-Arena/checkpoint.pt \
  --dolphin /absolute/path/to/dolphin-emu \
  --game /absolute/path/to/game.iso \
  --output /absolute/path/to/results/10m-arena-master-fd \
  --override 'slippi_ai.source_dir="/absolute/path/to/slippi-ai-9eca7479a955"' \
  --override 'slippi_ai.model_dir="/absolute/path/to/slippi-ai-models"' \
  --override 'slippi_ai.model="fox_d0_tx_like_3x512"' \
  --override 'slippi_ai.name="Master Player"' \
  --override 'env.dolphin.characters=[1,1]'
```

Repeat with `video_slippi_ai_stages`, then with `slippi_ai.name="Cody"` under both configurations, changing the output directory for every run. These four blocks give 68 games per Faynt checkpoint. The paper compares Expert and Arena at both sizes. The opponent is identified and hash-checked by `melee_rl/slippi_ai_agent.py`; its source and weights are supplied by someone authorized to provide them.

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
