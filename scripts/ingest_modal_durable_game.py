"""Validate a durably saved single game and publish its one claim exactly once."""
import argparse
import json
from pathlib import Path
import modal_panel_durable_game as durable
import modal_panel_queue as q
from ingest_modal_panel_pair import copy_once


def ingest(directory, journal, journal_sha, plan_sha, claim_token):
    directory = Path(directory).absolute(); journal = q.journal_path(journal)
    bound, plan = durable.plan_binding(directory / "plan.json", plan_sha)
    record = q.read(directory / "durable-result.json"); manifest = record["manifest"]
    durable.check_manifest(manifest, plan_sha, directory / "durable", bound)
    if manifest["status"] != durable.SUCCESS:
        raise ValueError("failed native result retained for reconciliation; no score admitted")
    native = dict(manifest); native["schema"] = native["native_schema"]
    files = {r["name"]: directory / "durable/files" / r["name"] for r in manifest["files"]}
    bound.validate_terminal(native, files, plan_sha, plan)
    submission = q.read(directory / "submission.json")
    if submission["plan_sha256"] != plan_sha or submission["call_id"] != manifest["call_id"]:
        raise ValueError("durable result is from another call")
    history = record["physical_history"]
    if not 1 <= len(history) <= durable.MAX_PHYSICAL or not any(r["execution"] == manifest["execution"] for r in history):
        raise ValueError("accepted execution missing from retained physical history")
    if (any(r["call_id"] != submission["call_id"] for r in history)
            or len({r["execution"]["execution_id"] for r in history}) != len(history)):
        raise ValueError("physical retry provenance differs")
    frozen, panel = q.load_plan(journal, journal_sha)
    claim = q.find_claim(journal, claim_token); q.validate_claim(claim, frozen, panel)
    index = plan["scope"]["selected_game_indices"][0]
    if (len(claim["attempts"]) != 1 or claim["attempts"][0]["game"] != plan["games"][index]
            or claim["attempts"][0]["attempt_label"] != plan["artifact_labels"][index]):
        raise ValueError("only the originally claimed game may be published")
    other = plan["artifact_labels"][1-index]
    if any(n.startswith((f"game-{1-index}/", f"game-command-{1-index}/",
                        f"project/artifacts/integration/frisson_ai/{other}/")) for n in files):
        raise ValueError("unselected peer was executed")
    execution = {"app_id": submission["app_id"], "call_id": submission["call_id"], **manifest["execution"]}
    q.bind_execution(journal, journal_sha, claim_token, execution)
    attempt = claim["attempts"][0]; label = attempt["attempt_label"]
    prefix = f"project/artifacts/integration/frisson_ai/{label}/"
    rows = [{"path": r["name"][len(prefix):], "bytes": r["bytes"], "sha256": r["sha256"]}
            for r in manifest["files"] if r["name"].startswith(prefix)]
    if not rows: raise ValueError("native selected-game artifacts absent")
    target = q.job.ARTIFACTS / label; target.mkdir(exist_ok=True)
    for row in rows: copy_once(files[prefix + row["path"]], target / row["path"], row)
    q.verify_artifacts(label, rows)
    receipt = {"schema": q.SCHEMA, "plan_sha256": journal_sha, "claim_token": claim_token,
               "attempt_token": attempt["attempt_token"], "execution": execution, "files": rows, "exit_code": 0}
    path = directory / "durable-ingestion.json"
    if path.exists():
        if q.read(path) != receipt: raise ValueError("preserved import receipt changed")
    else: q.new_file(path, receipt)
    return q.ingest(journal, journal_sha, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "journal", "journal-sha", "plan-sha", "claim-token"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    print(json.dumps(ingest(a.directory, a.journal, a.journal_sha, a.plan_sha, a.claim_token)))
