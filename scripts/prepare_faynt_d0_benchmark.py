#!/usr/bin/env python3
"""Prepare the fresh 2,624-game D0 Slippi panel with its recorded historical $150 budget cap."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from faynt_d0_benchmark_plan import OUTPUT, ROOT, build_manifest, verify_learner_files, write_prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from melee_policy.integration.faynt_d0_checkpoints import CHECKPOINTS

    verify_learner_files(CHECKPOINTS)
    manifest = build_manifest(CHECKPOINTS)
    print(json.dumps(write_prepared(manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
