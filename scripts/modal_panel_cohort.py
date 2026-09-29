"""Bounded local coordinator for frozen, explicitly approved Modal pairs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import modal_panel_batch_pilot as batch
import modal_panel_queue as queue
import ingest_modal_panel_pair as importer

SCHEMA = "e011.modal-panel-cohort.v1"
MAX_PAIRS = 64


def validate(value):
    if set(value) != {"schema", "journal", "journal_sha256", "max_workers", "pairs"}:
        raise ValueError("exact cohort keys required")
    if value["schema"] != SCHEMA or type(value["max_workers"]) is not int or not 1 <= value["max_workers"] <= MAX_PAIRS:
        raise ValueError("bounded reviewed concurrency required")
    rows = value["pairs"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAIRS:
        raise ValueError("bounded explicit pair list required")
    for row in rows:
        if set(row) != {"pair_id", "directory", "plan_sha256", "claim_token"}:
            raise ValueError("exact pair binding required")
    for key in ("pair_id", "directory", "claim_token"):
        if len({r[key] for r in rows}) != len(rows): raise ValueError("duplicate pair binding")
    return value


def run(path, sha):
    path = Path(path).absolute()
    if batch.base.wire.digest(path) != sha: raise ValueError("frozen cohort SHA required")
    value = validate(queue.read(path))
    journal = queue.journal_path(value["journal"])
    frozen, manifest = queue.load_plan(journal, value["journal_sha256"])
    with (path.parent / "cohort.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # All identities and durable claims are checked before any submission.
        for row in value["pairs"]:
            directory = Path(row["directory"])
            _, plan = batch.plan_binding(directory / "plan.json", row["plan_sha256"])
            claim = queue.find_claim(journal, row["claim_token"])
            queue.validate_claim(claim, frozen, manifest)
            if (claim["pair_id"] != row["pair_id"] or [a["game"] for a in claim["attempts"]] != plan["games"]
                    or [a["attempt_label"] for a in claim["attempts"]] != plan["artifact_labels"]):
                raise ValueError("cohort claim differs from actual game plan")
            if (directory / "intent.json").exists():
                raise ValueError("existing execution intent requires reconciliation; cohort cannot resubmit")
        outcomes = []
        def launch(row):
            directory = Path(row["directory"])
            with (directory / "coordinator.log").open("x") as log:
                result = subprocess.run([sys.executable, str(batch.base.ROOT / "scripts/modal_panel_batch_pilot.py"),
                    "--run", str(directory / "plan.json"), "--sha", row["plan_sha256"]],
                    cwd=batch.base.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            return row, result.returncode
        with ThreadPoolExecutor(max_workers=value["max_workers"]) as pool:
            futures = [pool.submit(launch, row) for row in value["pairs"]]
            for future in as_completed(futures):
                try:
                    row, code = future.result()
                    if code: raise RuntimeError(f"pair {row['pair_id']} exited {code}; preserve returned files")
                    accepted = importer.ingest_pair(row["directory"], journal=journal,
                        journal_sha=value["journal_sha256"], pair_plan_sha=row["plan_sha256"], claim_token=row["claim_token"])
                    outcome = {"pair_id": row["pair_id"], "status": "ingested", "receipts": accepted}
                except Exception as error:
                    outcome = {"status": "requires-reconciliation", "error": str(error)[:2048]}
                outcomes.append(outcome)
                print(json.dumps(outcome), flush=True)
        queue.new_file(path.parent / "cohort-result.json", {"schema": SCHEMA, "cohort_sha256": sha, "outcomes": outcomes})
        return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    args = parser.parse_args()
    run(args.run, args.sha)
