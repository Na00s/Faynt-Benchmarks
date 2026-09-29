# Faynt benchmarks

This repository provides the benchmark source, execution runtimes, and reported results for **Faynt: Scaling and Optimizing Policies for Competitive Melee**. The six released policies cover two model sizes and three training stages: Base, Expert, and Arena. Each checkpoint supports all 26 characters. Browse the [Faynt model family](https://huggingface.co/collections/frisson-labs/faynt) for model cards and inference examples.

| Size | Base | Expert | Arena |
|---|---|---|---|
| 10M | [Pretrained](https://huggingface.co/frisson-labs/Faynt-10M-Base) | [Curriculum and distillation](https://huggingface.co/frisson-labs/Faynt-10M-Expert) | [Reinforcement learning](https://huggingface.co/frisson-labs/Faynt-10M-Arena) |
| 75M | [Pretrained](https://huggingface.co/frisson-labs/Faynt-75M-Base) | [Curriculum](https://huggingface.co/frisson-labs/Faynt-75M-Expert) | [Reinforcement learning](https://huggingface.co/frisson-labs/Faynt-75M-Arena) |

## Initial panel: Base and Expert

The four policies share a 152-game schedule, including opponent checkpoints, character assignments, policy sampling seeds, and physical ports. Games use Final Destination, four stocks, and an eight-minute timer. Dolphin's game RNG is recorded separately. The schedule combines MIMIC, CPU9, Slippi-AI, and roster-extension games.

| Checkpoint | Selected step | Wins / games |
|---|---:|---:|
| 10M Base | 122,064 | 49/152 |
| 10M Expert | 195,248 | 106/152 |
| 75M Base | 86,016 | 69/152 |
| 75M Expert | 127,214 | 123/152 |

The report retains 13 outcomes at 10M Base and seven at 75M Base awaiting replay-fidelity audits. Faynt and MIMIC use zero added policy delay. Slippi-AI opponents retain their configured delays of 18 or 21 frames.

Data: [benchmark_fixed_panel.json](results/benchmark_fixed_panel.json). Each row stores `[wins, losses, stocks_taken, stocks_conceded]`; `games_per_block` gives the block denominators. Source: Section 5.1 and Figure 8; Appendices F.2, F.3, and F.7, pages 38-42.

## Expanded panel: Arena

Each Arena checkpoint plays 1,312 games against 14 frozen Slippi-AI releases. These are 2,624 distinct games across three conditions:

| Condition | Meaning | 10M Arena | 75M Arena |
|---|---|---:|---:|
| Supported mirrors | Both sides use a fighter in the opponent's deployed roster. | 240/244 | 149/244 |
| Extended roster | Faynt uses an outside-roster fighter; the opponent retains a supported fighter. | 427/534 | 198/534 |
| Forced mirrors | Both sides use the same outside-roster fighter. | 531/534 | 503/534 |

Faynt uses zero added policy delay. Opponents retain 21- or 24-frame action queues. The report does not isolate the effect of that difference. Each outside-roster character contributes two games per release and condition. Supported specialist mirrors contribute 12 games per release.

The selected 10M policy follows Expert step 195,248, first-run RL step 1,318, then second-run RL step 632. The 75M follows Expert step 127,214 and RL step 980. Both receive Fox-only RL experience, with distinct training schedules. Some releases also informed RL run or checkpoint selection.

Data: [rl_current_scores.json](results/rl_current_scores.json) contains game definitions, results, replay hashes, terminal evidence, and source labels. It combines 2,613 frozen-ledger games and 11 separately rerun configurations; the replacement labels remain recorded. Source: Sections 5.2-5.3, Figure 9, and Appendices F.5-F.11, pages 40-44.

## Separate zero-delay panels

One privately supplied Slippi-AI checkpoint is evaluated under Master Player and Cody conditioning. Each setting contributes 16 Final Destination games and 18 games across six stages, giving 68 games per Faynt checkpoint. Both policies use zero added action delay.

| Policy | 10M | 75M |
|---|---:|---:|
| Expert | 23/68 | 25/68 |
| Arena | 68/68 | 58/68 |

The 10M Arena wins 61 games without losing a stock. The [private Slippi-AI record](results/benchmark_private_slippi_zero_delay.json) is a transcription of the supplied result table. The checkpoint was privately provided by its developers. Source: Sections 5.4 and 6.5, Figure 15, Appendix F.13 and Table F.8, page 45.

Seven Phillip checkpoints cover six specialist characters, including two Fox specialists:

| Arena evaluation | 10M | 75M |
|---|---:|---:|
| Final Destination | 112/112 | 108/112 |
| Six-stage panel | 126/126 | 126/126 |

Phillip's released configurations use zero added action delay and make decisions every two, three, or four frames; Faynt makes a decision each frame. [Results](results/benchmark_zero_delay.json) and [release timing settings](results/benchmark_zero_delay_timing.json) retain the underlying records, including the earlier 75M RL checkpoint comparison. The source's `Init` rows have no verified mapping to the released Base or Expert models. Source: Appendix F.12 and Table F.7, pages 44-45.

## Inspect the records

The files in [results](results) are copied byte for byte from the supplied paper source bundle. [manifest.json](results/manifest.json) records their byte lengths and SHA-256 hashes. The initial and zero-delay records contain reported aggregate results; the expanded record includes one entry per accepted game. Referenced replay files and source screenshots remain separate research artifacts.

From this directory, the following standard-library Python command recomputes the expanded win counts without launching games:

```bash
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

ledger = json.loads(Path("results/rl_current_scores.json").read_text())
totals = defaultdict(lambda: [0, 0])
for game in ledger["games"].values():
    definition, result = game["definition"], game["result"]
    row = totals[(definition["profile"], definition["block"])]
    row[0] += int(result["win"])
    row[1] += 1
for (model, condition), (wins, games) in sorted(totals.items()):
    print(model, condition, f"{wins}/{games}")
PY
```

## Run the suites

The source is organized into two runtime paths that preserve the recorded experiments:

| Suite | Source and setup |
|---|---|
| Initial 152-game Base and Expert panels | `scripts/run_final_winner_benchmark.py`, `scripts/run_posttraining_winner_benchmark.py` |
| Expanded Arena panel | `scripts/faynt_d0_benchmark_plan.py`, `scripts/prepare_faynt_d0_benchmark.py`, and the retained local/cloud queue tools |
| Phillip and private zero-delay Slippi-AI panels | [Evaluation runtime](evaluation/README.md), including all four 16-game / 18-game configurations |
| Native local matches, controller and replay checks | [Local match runtime](LOCAL_MATCHES.md) |

Read the [benchmark runtime guide](BENCHMARK_RUNTIME.md) for initial/expanded commands, asset identities, fresh qualification and cloud prerequisites. The [evaluation guide](evaluation/README.md) provides the later match recorder, cloud entry points, and a local plan/run CLI. All six published native Faynt checkpoints pass strict loading in that evaluation runtime.

HAL implementation and adapters, game assets, emulator binaries, third-party source trees and checkpoint payloads are excluded. Users provide the declared local inputs. Cloud execution uses an explicitly supplied prepared image. Source revisions, per-file hashes and dependency notices accompany the export.

This repository starts with a clean source export and contains no internal research Git or LFS history. The companion [Faynt Tournament](https://github.com/Na00s/Faynt-Tournament) repository contains the standalone match runtime and mirrored baseline tournament scheduler. A new game run needs its own qualification and accepted-result evidence. The paper's historical results remain separately identified in `results/`.

## License

Faynt-owned code and accompanying documentation are available under the [MIT License](LICENSE), copyright 2026 Frisson Labs. Third-party portions retain their original notices. The Slippi Dolphin source patch is licensed under GPL-2.0-or-later. See the [component licensing and runtime notices](THIRD_PARTY_NOTICES.md) for the scope and retained license texts.
