#!/usr/bin/env python3
"""Small replay/encoding compatibility probe, zero live Dolphin games."""
import argparse
import copy
import dataclasses
import hashlib
import json
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from slippi_panel_audit import ROOT, OUTPUT, atomic_json
from slippi_panel_plan import ROSTER, STAGES
from slippi_panel_runtime import manifest, register_release, session
from melee_policy.integration import slippi_ai_policy as boundary
from melee_policy.integration.slippi_compatibility import configure_cpu_tensorflow, _default_console_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if args.all:
        def probe(key):
            command = [sys.executable, __file__, "--release", key]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
            print(key, result.returncode, result.stdout[-1200:], flush=True)
            if result.returncode:
                print(result.stderr[-4000:], flush=True)
            return {"release": key, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(probe, manifest()["releases"]))
        atomic_json(OUTPUT / "canary-run.json", {"passed": all(r["exit_code"] == 0 for r in receipts), "runs": receipts})
        raise SystemExit(0 if all(r["exit_code"] == 0 for r in receipts) else 1)
    if not args.release:
        parser.error("--release or --all is required")
    import melee
    record = manifest()["releases"][args.release]
    register_release(record)
    configure_cpu_tensorflow(20260910)
    config = dataclasses.replace(boundary.SlippiAIPolicyConfig.from_project_root(ROOT, port=1, opponent_port=2, release=args.release),
                                 name=record["selected_name"], console_delay_frames=0)
    replay = ROOT / ".e000-cache/raw/039d70ae59bc37fc1fe5b72c1dadcd95d2b2877b9ad1964bf978aaa6662962ad.slp"
    console = _default_console_factory(replay)
    if not console.connect():
        raise RuntimeError("canary replay connect failed")
    states = [console.step() for _ in range(64)]
    console.stop()
    if [s.frame for s in states] != list(range(-123, -59)):
        raise RuntimeError("unexpected canary replay frames")
    policy = boundary.SlippiAIPolicySession(config) if args.reference else session(config, record=record)
    first_commands, covered = [], []
    started = time.monotonic()
    try:
        policy.start()
        for state in states:
            first_commands.append(policy.step(state).as_dict())
        # Input-encoding coverage only: transplant enum fields on replay states.
        # These synthetic contexts do not establish gameplay competence.
        frame = -59
        for i, character in enumerate(ROSTER):
            for offset in range(32):
                state = copy.deepcopy(states[offset])
                state.frame = frame
                state.stage = melee.Stage[STAGES[i % len(STAGES)]]
                state.players[1].character = melee.Character[character]
                state.players[2].character = melee.Character[character]
                policy.step(state)
                frame += 1
            covered.append(character)
        diagnostics = policy.diagnostics()
        metadata = policy.metadata()
    finally:
        policy.close()
    expected = 64 + len(ROSTER) * 32
    checks = {
        "all_26_character_inputs": covered == list(ROSTER),
        "frame_count": diagnostics["frames_total"] == expected,
        "current_frame_barriers": diagnostics["current_frame_inference_barriers"] == expected,
        "native_decoder_exact": diagnostics["capture_decoder_mismatches"] == 0,
        "native_delay": metadata["runtime_contract"]["effective_policy_delay_frames"] == record["delay"],
        "native_name": metadata["runtime_contract"]["effective_player_name"] == record["selected_name"],
    }
    receipt = {"release": args.release, "reference": args.reference, "checks": checks,
               "passed": all(checks.values()), "diagnostics": diagnostics, "metadata": metadata,
               "first_64_commands_sha256": hashlib.sha256(json.dumps(first_commands, sort_keys=True).encode()).hexdigest(),
               "first_64_commands": first_commands, "encoding_characters": covered,
               "probe_kind": "64 actual replay frames plus synthetic enum-coverage states; no game scores",
               "wall_seconds": time.monotonic() - started}
    suffix = "-reference" if args.reference else ""
    atomic_json(OUTPUT / "canaries" / f"{args.release}{suffix}.json", receipt)
    print(json.dumps({"release": args.release, "passed": receipt["passed"], "checks": checks, "wall_seconds": receipt["wall_seconds"]}))
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
