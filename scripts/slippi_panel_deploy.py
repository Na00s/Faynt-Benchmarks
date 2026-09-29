"""Freeze tested local benchmark code and install two scoped launchd services."""
import argparse
import datetime
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import run_slippi_public_panel as job


def freeze():
    xml = ET.parse(job.OUTPUT / "tests.junit.xml").getroot()
    suites = list(xml.iter("testsuite"))
    totals = {key: sum(int(x.get(key, 0)) for x in suites) for key in ("tests", "failures", "errors", "skipped")}
    if totals["tests"] < 100 or totals["failures"] or totals["errors"]:
        raise RuntimeError(f"deployment tests are incomplete: {totals}")
    canaries = job.read(job.OUTPUT / "canary-run.json")
    panel = job.read(job.OUTPUT / "manifest.json")
    if not canaries["passed"] or len(canaries["runs"]) != len(panel["releases"]):
        raise RuntimeError("canaries incomplete")
    for release in panel["releases"]:
        receipt = job.read(job.OUTPUT / "canaries" / f"{release}.json")
        if not receipt["passed"] or not all(receipt["checks"].values()):
            raise RuntimeError(f"canary failed: {release}")
    reference = job.read(job.OUTPUT / "canaries/medium-v2-reference.json")
    actual = job.read(job.OUTPUT / "canaries/medium-v2.json")
    if actual["first_64_commands_sha256"] != reference["first_64_commands_sha256"]:
        raise RuntimeError("medium-v2 native adapter parity failed")
    tests = {"passed": True, **totals, "junit_sha256": job.sha(job.OUTPUT / "tests.junit.xml"),
             "medium_v2_native_first_64_commands_exact": True,
             "test_files": {str(p.relative_to(job.ROOT)): job.sha(p) for p in sorted((job.ROOT / "tests/integration").glob("test_slippi_public_panel.py"))}}
    job.write(job.OUTPUT / "tests.json", tests)
    evidence = [job.OUTPUT / "tests.json", job.OUTPUT / "tests.junit.xml", job.OUTPUT / "canary-run.json",
                *sorted((job.OUTPUT / "canaries").glob("*.json"))]
    deployment = {"schema": "e011.public_panel.deployment.v1", "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_binding": job.source_binding(), "validation_artifacts": {str(p.relative_to(job.ROOT)): job.sha(p) for p in evidence},
        "tests": tests, "new_games": len(panel["games"]), "local_only": True, "cloud_spend": 0,
        "save_slp": True, "save_video": False, "first_live_panel_game_pending": True,
        "estimated_mean_game_seconds": 260.08403784265886, "estimated_queue_hours": 189.57236536087134,
        "measurement": "104 already completed P21 Slippi games; newer opponents and stages can change duration"}
    if (job.OUTPUT / "deployment.json").exists():
        old = job.read(job.OUTPUT / "deployment.json")
        if old["source_binding"] != deployment["source_binding"] or old["validation_artifacts"] != deployment["validation_artifacts"]:
            raise RuntimeError("deployment already frozen; explicit audited migration required")
        return old
    job.write(job.OUTPUT / "deployment.json", deployment)
    return deployment


def install():
    job.validate_deployment()
    installed = []
    for watch in (False, True):
        label = "com.frisson.slippi-public-panel-p21" + (".watch" if watch else "")
        command = ["/usr/bin/caffeinate", "-ims", str(job.PYTHON), "-u", str(job.ROOT / "scripts/run_slippi_public_panel.py")]
        if watch:
            command.append("--watch")
        config = {"Label": label, "ProgramArguments": command, "WorkingDirectory": str(job.ROOT),
            "EnvironmentVariables": {"PYTHONPATH": f"{job.ROOT}/src:{job.ROOT}/scripts", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            "RunAtLoad": True, "ThrottleInterval": 30,
            "StandardOutPath": str(job.OUTPUT / ("watch.log" if watch else "supervisor.log")),
            "StandardErrorPath": str(job.OUTPUT / ("watch.stderr.log" if watch else "supervisor.stderr.log"))}
        # The independent watcher owns the coordinator's bounded restart budget.
        if watch:
            config["KeepAlive"] = {"SuccessfulExit": False}
        payload = plistlib.dumps(config)
        source = job.OUTPUT / f"{label}.plist"
        source.write_bytes(payload)
        destination = Path.home() / "Library/LaunchAgents" / source.name
        if destination.exists() and destination.read_bytes() != payload:
            raise RuntimeError(f"existing service differs: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        subprocess.run(["plutil", "-lint", str(destination)], check=True)
        target = f"gui/{os.getuid()}/{label}"
        exists = subprocess.run(["launchctl", "print", target], capture_output=True).returncode == 0
        if not exists:
            subprocess.run(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)], check=True)
        installed.append({"label": label, "plist": str(destination), "sha256": job.sha(destination), "target": target})
    job.write(job.OUTPUT / "installed.json", {"services": installed, "installed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return installed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = freeze()
        print(json.dumps({"frozen": True, "games": result["new_games"], "tests": result["tests"]}))
    if args.install:
        print(json.dumps(install()))
