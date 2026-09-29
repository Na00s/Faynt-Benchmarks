# Third-party notices

Faynt-owned policy and evaluation code and accompanying documentation are covered by the repository's [MIT License](../LICENSE), copyright 2026 Frisson Labs. The source revision is recorded in `SOURCE_MANIFEST.json`. Third-party portions retain the notices and licenses identified here.

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

The retained MIT notices apply to the adapted Slippi-AI and MIMIC portions. Separately supplied dependency source, weights, datasets, game images and emulator binaries remain subject to their providers' terms. The [repository component notices](../THIRD_PARTY_NOTICES.md) also identify the GPL-2.0-or-later Slippi Dolphin patch.
