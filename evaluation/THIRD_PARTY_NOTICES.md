# Third-party notices

The export contains first-party policy and evaluation code from the source revision recorded in `SOURCE_MANIFEST.json`.

The state representation, controller codec, reward functions, RL utilities, frame conversion and PyTorch Slippi-AI compatibility implementation follow or adapt Slippi-AI at revision `577965a7731dc53e3472ea63d9e9853a4e9d65fa`. Its MIT copyright and permission notice is retained in `LICENSES/slippi-ai-MIT.txt`. The MIMIC integration is attributed to Erick Martinez, with its MIT notice retained in `LICENSES/MIMIC-MIT.txt`.

Users provide the following dependencies separately and retain their applicable notices and source obligations:

| Dependency | Use in this export |
|---|---|
| libmelee | Installed game-state and controller transport library |
| Dolphin and compatible builds | Separately supplied emulator executable |
| MIMIC | Separately supplied source and checkpoint bundle |
| Slippi-AI | Separately supplied source and checkpoint files |
| Phillip | Separately supplied source, checkpoints and sidecar interpreter |
| SmashBot | Optional separately supplied source loaded by the historical calibration interface |

HAL implementations and checkpoints are excluded. Game images, game art, upstream source checkouts and third-party checkpoint payloads are excluded. The private zero-delay checkpoint remains a local path prerequisite; its availability is controlled by its provider.

These notices record code provenance. Publication licensing and any obligations arising from a particular dependency combination require review by the repository owners.
