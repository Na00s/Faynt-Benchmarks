#!/usr/bin/env python3
"""Package a fresh 2,624-game D0 cloud queue locally, without upload or launch."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import time

import faynt_d0_benchmark_plan as fresh
import modal_panel_budget as budget
import modal_panel_cloud_queue as cloud
import modal_panel_queue as q

ROOT = fresh.ROOT
OUTPUT = fresh.OUTPUT / "research/cloud-v1"
PANEL = fresh.OUTPUT / "expanded-slippi/manifest.json"
PANEL_SHA = "7cd987e1ab91581a45a8b54150d2cc58f31a1a6fc8e6874422e692fe02bc6b06"
TEMPLATE = ROOT / (
    "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1/research/modal-compat-v1/"
    "cloud-dispatch-v1/metadata/plans/sp21-10m-medium-v2-supported-mirror-cptfalcon-dreamland-p1.json"
)
TEMPLATE_SHA = "8256d2ecf4feeb6fa266a180b89145a85c995966afea195d9d2625fe5807c536"
REFERENCE_POLICY = fresh.OUTPUT / "reference-bootstrap/frisson_policy.py"
REFERENCE_POLICY_REMOTE = "/opt/runtime-root/src/melee_policy/integration/frisson_policy.py"
REFERENCE_POLICY_SHA = "fcdba59fc186b104689d20510d7a3d2b8fb4c2515cac6df587028926cf6025ca"
FILE_CAP = 4 * 1024**3
METADATA_TOTAL_CAP = 1024**3
ARCHIVE_CAP = 64 * 1024**2


class FileRecords:
    """Hash each regular input once and reject changes during preparation."""

    def __init__(self):
        self.cache = {}
        self.blobs = {}

    def record(self, path, remote, *, expected=None, include_blob=True):
        path = Path(path).absolute()
        if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"canonical regular input required: {path}")
        info = path.stat()
        if not 0 <= info.st_size <= FILE_CAP:
            raise ValueError("input file exceeds byte cap")
        fingerprint = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        previous = self.cache.get(str(path))
        if previous is not None and previous[0] != fingerprint:
            raise ValueError(f"input changed during preparation: {path}")
        if previous is None:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            if fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("input changed while hashing")
            self.cache[str(path)] = (fingerprint, digest)
        else:
            digest = previous[1]
        row = {"local": str(path), "remote": str(remote), "bytes": info.st_size, "sha256": digest}
        if expected is not None and any(row[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError(f"frozen input identity differs: {remote}")
        if include_blob:
            self.blobs.setdefault(digest, {key: row[key] for key in ("local", "bytes", "sha256")})
        return row

    def verify_unchanged(self):
        for path, (expected, _) in self.cache.items():
            p = Path(path)
            if p.is_symlink() or p.resolve(strict=True) != p:
                raise ValueError("input traversal changed during preparation")
            info = p.stat()
            if expected != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns):
                raise ValueError(f"input changed during preparation: {path}")


def pair_assignments(panel):
    """Retain the queue's exact phase order and whole-pair round-robin slots."""
    if len(panel["games"]) != 2624:
        raise ValueError("exact 2,624-game panel required")
    pairs, seen, last_phase = {}, set(), -1
    for ordinal, game in enumerate(panel["games"], 1):
        phase = q.PHASES.index((game["profile"], game["block"]))
        if (game["ordinal"] != ordinal or phase < last_phase or game["label"] in seen
                or not game["label"].startswith(fresh.RUN_ID + "-")
                or game["save_slp"] is not True or game["save_video"] is not False):
            raise ValueError("fresh ordered SLP-only panel required")
        seen.add(game["label"])
        last_phase = phase
        pairs.setdefault(game["pair_id"], []).append(game)
    assignment, phase_counts = {}, [0] * len(q.PHASES)
    for pair_id, games in pairs.items():
        if len(games) != 2:
            raise ValueError("exact two-game port pair required")
        first, second = games
        fields = ("profile", "block", "release", "stage", "seed", "frisson_character", "slippi_character", "player_name")
        if (any(first[key] != second[key] for key in fields)
                or {g["frisson_port"] for g in games} != {1, 2}
                or any(g["slippi_port"] != 3 - g["frisson_port"] for g in games)):
            raise ValueError("paired context differs")
        phase = q.PHASES.index((first["profile"], first["block"]))
        pair = {"pair_id": pair_id, "phase": phase, "worker_slot": phase_counts[phase] % 64,
                "labels": [g["label"] for g in games]}
        phase_counts[phase] += 1
        for game in games:
            assignment[game["label"]] = pair
    if phase_counts != [122, 267, 267, 122, 267, 267]:
        raise ValueError("frozen phase coverage differs")
    return assignment, phase_counts


def authorization(template, bound, plan_template):
    value = copy.deepcopy(template)
    value.update(scope=bound.fixed_scope(), limits=bound.limits())
    value["bindings"].update(checkpoints=bound.CHECKPOINTS, panel_manifest_sha256=PANEL_SHA)
    if value["limits"]["nonpreemptible"] is not False:
        raise ValueError("standard preemptible worker required")
    if value["limits"]["cpu"] != [4, 4] or value["limits"]["memory_mib"] != [16384, 16384]:
        raise ValueError("reviewed four-CPU 16-GiB worker resources required")
    if value["bindings"]["tape_plan_sha256"] != plan_template["prior_tape_plan_sha256"]:
        raise ValueError("reference tape-plan binding differs")
    return value


def _write(path, value):
    body = value if isinstance(value, bytes) else cloud.encoded(value)
    if len(body) > q.MAX_JSON:
        raise ValueError("metadata file exceeds byte cap")
    path.parent.mkdir(parents=True, exist_ok=True)
    q.new_file(path, body)
    return body


def _archive(metadata, target):
    total = 0
    with target.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(metadata.rglob("*")):
                    if path.is_dir():
                        continue
                    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                        raise ValueError("regular metadata archive entries required")
                    total += path.stat().st_size
                    if total > METADATA_TOTAL_CAP:
                        raise ValueError("metadata archive uncompressed byte cap")
                    info = archive.gettarinfo(str(path), path.relative_to(metadata).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    if target.stat().st_size > ARCHIVE_CAP:
        raise ValueError("compressed metadata archive byte cap")


def prepare(output=OUTPUT, *, reference_policy=None, template_path=TEMPLATE, template_sha256=TEMPLATE_SHA):
    """Create a canonical local deployment; each queue game starts pending."""
    output = Path(output).absolute()
    if output != OUTPUT or output.resolve(strict=False) != output or output.exists():
        raise ValueError("fresh canonical D0 cloud-v1 output required")
    import faynt_d0_durable_game as d

    records = FileRecords()
    panel_record = records.record(PANEL, d.REMOTE / d.PANEL, expected={"bytes": PANEL.stat().st_size, "sha256": PANEL_SHA})
    panel_body = q.read_bytes(PANEL)
    panel = json.loads(panel_body)
    assignment, phase_counts = pair_assignments(panel)
    template_row = records.record(Path(template_path), "/reference/template.json", expected={"bytes": Path(template_path).stat().st_size, "sha256": template_sha256}, include_blob=False)
    template = q.read(Path(template_path))
    prior_rows = {row["remote"]: row for row in template["uploads"]}
    if len(prior_rows) != len(template["uploads"]):
        raise ValueError("duplicate reference upload destination")
    auth_remote = str(d.INPUT / "authorization.json")
    auth_record = prior_rows[auth_remote]
    records.record(auth_record["local"], auth_remote, expected=auth_record, include_blob=False)
    auth_template = q.read(auth_record["local"])
    policy = budget.make_policy(total_usd=150, reserve_usd=10, prior_spend_usd=1)
    if policy["total_usd"] != 150 or policy["reserve_usd"] != 10 or policy["nonpreemptible"] is not False:
        raise ValueError("exact $150 standard-preemptible allocation required")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".cloud-v1-preparation-", dir=output.parent))
    metadata = temporary / "metadata"
    (metadata / "plans").mkdir(parents=True)
    (metadata / "blobs").mkdir()
    common = {}
    reference_overrides = {}
    rows = []
    started = time.time()
    try:
        for ordinal, game in enumerate(panel["games"], 1):
            label = game["label"]
            pair = assignment[label]
            labels = pair["labels"]
            index = labels.index(label)
            bound = d.bind(labels, [f"{name}-a1" for name in labels], index)
            allowed = bound.destinations(template["tape_prerequisite"])
            uploads = []
            for remote in sorted(allowed - {auth_remote}):
                if remote.startswith(str(d.REMOTE) + "/"):
                    relative = Path(remote).relative_to(d.REMOTE)
                    row = records.record(ROOT / relative, remote)
                    expected = bound.CHECKPOINTS.get(relative.as_posix())
                    if expected is not None and any(row[k] != expected[k] for k in ("bytes", "sha256")):
                        raise ValueError("new gameplay checkpoint identity differs")
                    if remote == panel_record["remote"] and row != panel_record:
                        raise ValueError("new panel input changed")
                else:
                    if remote not in common:
                        if remote not in prior_rows:
                            raise ValueError(f"unresolved reference destination: {remote}")
                        original = prior_rows[remote]
                        if remote == REFERENCE_POLICY_REMOTE and reference_policy is not None:
                            row = records.record(reference_policy, remote, expected=original)
                            reference_overrides[remote] = row["local"]
                        else:
                            row = records.record(original["local"], remote, expected=original)
                        common[remote] = row
                    row = common[remote]
                uploads.append(copy.deepcopy(row))
            auth = authorization(auth_template, bound, template)
            auth_body = cloud.encoded(auth)
            auth_sha = hashlib.sha256(auth_body).hexdigest()
            blob = metadata / "blobs" / auth_sha
            if not blob.exists():
                _write(blob, auth_body)
            uploads.append({"local": str(output / "metadata/blobs" / auth_sha), "remote": auth_remote,
                            "bytes": len(auth_body), "sha256": auth_sha})
            plan = bound.definition(uploads, output / "future-results" / label,
                template["prior_tape_plan_sha256"], template["tape_prerequisite"], auth_sha,
                bound.selected_games(panel))
            bound.check_layout(plan)
            bound.authorization(auth, plan)
            body = _write(metadata / "plans" / (label + ".json"), plan)
            rows.append({"label": label, "game_sha256": cloud.digest(game),
                "initial_status": "pending", "initial_row": {"status": "pending", "attempts": []},
                "phase": pair["phase"], "worker_slot": pair["worker_slot"], "pair_id": pair["pair_id"],
                "plan_path": "plans/" + label + ".json", "plan_sha256": hashlib.sha256(body).hexdigest()})
            if ordinal % 328 == 0:
                print(json.dumps({"prepared_games": ordinal, "total_games": 2624}), flush=True)
        records.verify_unchanged()
        counts = {"pending": 2624, "complete": 0, "quarantined": 0}
        snapshot = {"schema": cloud.SCHEMA, "run_id": fresh.RUN_ID, "fresh_run": True,
            "maximum_workers": 64, "maximum_physical_attempts": 3, "created_unix": started,
            "manifest_sha256": PANEL_SHA, "manifest_body": panel_body.decode(), "manifest": panel,
            "games": rows, "initial_counts": counts, "queue_deadline_unix": started + 7 * 86400,
            "phase_order": [list(row) for row in q.PHASES], "phase_pair_counts": phase_counts,
            "source_inputs": sorted(records.blobs.values(), key=lambda row: row["sha256"]),
            "reference_template": template_row, "reference_overrides": reference_overrides,
            "budget": policy, "imported_completed_games": [], "adopted_calls": 0}
        cloud.validate_snapshot(snapshot)
        _write(metadata / "snapshot.json", snapshot)
        _archive(metadata, temporary / "metadata.tar.gz")
        deployment = {"schema": cloud.SCHEMA, "run_id": fresh.RUN_ID,
            "app_name": fresh.RUN_ID, "state_name": "faynt-d0-632-980-state-v1",
            "environment_name": "main", "fresh_run": True,
            "durable_module": "faynt_d0_durable_game", "budget": policy,
            "snapshot_sha256": fresh.sha256(metadata / "snapshot.json"),
            "archive_sha256": fresh.sha256(temporary / "metadata.tar.gz"),
            "archive": str(output / "metadata.tar.gz"), "inputs": snapshot["source_inputs"],
            "counts": counts, "pending": 2624, "adopted_calls": 0,
            "preparation": {"cloud_uploads": 0, "cloud_launches": 0,
                "unique_hashed_files": len(records.cache), "manifest_sha256": PANEL_SHA,
                "compute_policy": "standard-preemptible", "hard_spend_limit_usd": 150,
                "provider_budget_verification_required": False, "prior_spend_usd": 1,
                "source_template_sha256": template_sha256}}
        _write(temporary / "deployment.json", deployment)
        from modal_panel_cloud_app import validate_fresh_deployment
        validate_fresh_deployment(deployment, temporary)
        if output.exists():
            raise ValueError("canonical deployment appeared during preparation")
        temporary.rename(output)
    except BaseException:
        print(json.dumps({"partial_preparation_retained": str(temporary), "cloud_uploads": 0, "cloud_launches": 0}), flush=True)
        raise
    print(json.dumps({key: value for key, value in deployment.items() if key != "inputs"}), flush=True)
    return deployment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-policy", type=Path, help="optional exact-byte source path override bound by template")
    parser.add_argument("--template", type=Path, default=TEMPLATE, help="fresh qualified full-game template plan")
    parser.add_argument("--template-sha256", default=TEMPLATE_SHA, help="reviewed SHA-256 of the template plan")
    args = parser.parse_args()
    prepare(reference_policy=args.reference_policy, template_path=args.template, template_sha256=args.template_sha256)


if __name__ == "__main__":
    main()
