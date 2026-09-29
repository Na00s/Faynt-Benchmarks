"""Inspect and run the released research match configurations locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tomllib


def config_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.resolve()
    candidate = Path(__file__).resolve().parents[1] / "configs" / "rl" / value
    if candidate.suffix != ".toml":
        candidate = candidate.with_suffix(".toml")
    if not candidate.is_file():
        raise FileNotFoundError(f"Unknown configuration: {value}")
    return candidate


def plan(value: str) -> dict:
    path = config_path(value)
    raw = tomllib.loads(path.read_text())
    return {
        "config": path.name,
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime": raw.get("runtime", {}),
        "policy": raw.get("policy", {}),
        "teacher": raw.get("teacher", {}),
        "opponent": raw.get("opponent", {}),
        "mimic": raw.get("mimic", {}),
        "smashbot": raw.get("smashbot", {}),
        "delay": raw.get("delay", {}),
        "environment": raw.get("env", {}),
        "actor": raw.get("actor", {}),
        "video": raw.get("video", {}),
        "phillip": raw.get("phillip", {}),
        "slippi_ai": raw.get("slippi_ai", {}),
        "launches_games": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Faynt policy evaluation")
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dolphin", type=Path)
    parser.add_argument("--game", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--override", action="append", default=[], help="Repeat table.key=value overrides")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            if args.override or any(getattr(args, name) is not None for name in
                                    ("checkpoint", "dolphin", "game", "output", "device")):
                parser.error("plan reads the stored configuration; run-only flags are unsupported")
            print(json.dumps(plan(args.config), indent=2))
            return 0
        for field in ("checkpoint", "dolphin", "game"):
            path = getattr(args, field)
            if path is None or not path.is_file():
                parser.error(f"--{field} must name an existing user-provided file")
        if args.output is None:
            parser.error("--output is required for run")
        overrides = list(args.override)
        overrides += [
            "env.dolphin.dolphin_path=" + json.dumps(str(args.dolphin.resolve())),
            "env.dolphin.iso_path=" + json.dumps(str(args.game.resolve())),
            "video.wandb=false",
        ]
        from melee_rl.release_runner import run_record_video
        result = run_record_video(
            config_path(args.config), checkpoint=str(args.checkpoint.resolve()),
            output_dir=str(args.output.resolve()), render=False, device=args.device,
            overrides=overrides,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "release-run.json").write_text(json.dumps(result, indent=2) + "\n")
        clips = result.get("clips", [])
        return 1 if not clips or any(clip.get("error") for clip in clips) else 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
