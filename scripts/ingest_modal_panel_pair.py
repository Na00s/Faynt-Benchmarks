#!/usr/bin/env python3
"""Verify a returned successful pair and ingest its exact claimed artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

import modal_panel_batch_pilot as batch
import modal_panel_queue as queue


def copy_once(source, destination, row):
    if batch.base.wire.regular(source, queue.MAX_FILE) != row["bytes"] or batch.base.wire.digest(source) != row["sha256"]:
        raise ValueError("returned artifact changed before copy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent or destination.is_symlink():
        raise ValueError("canonical owned destination required")
    if destination.exists():
        if batch.base.wire.regular(destination, queue.MAX_FILE) != row["bytes"] or batch.base.wire.digest(destination) != row["sha256"]:
            raise ValueError("existing artifact differs; preserve for investigation")
        return
    digest=hashlib.sha256(); size=0
    with source.open("rb") as src, destination.open("xb") as dst:
        while chunk:=src.read(1024**2):
            size+=len(chunk); digest.update(chunk)
            if size>row["bytes"] or dst.write(chunk)!=len(chunk): raise OSError("bounded artifact copy failed")
        dst.flush(); os.fsync(dst.fileno())
    if size!=row["bytes"] or digest.hexdigest()!=row["sha256"]: raise ValueError("copied artifact hash differs")


def ingest_pair(directory, *, journal, journal_sha, pair_plan_sha, claim_token):
    directory=Path(directory).absolute(); journal=queue.journal_path(journal)
    bound,plan=batch.plan_binding(directory/"plan.json",pair_plan_sha)
    result=queue.read(directory/"result.json"); call=queue.read(directory/"call.json")
    if result.get("status")!="full-policy-compatibility-pair-passed":
        raise ValueError("successful complete pair required; preserve failed outputs for scoped recovery")
    execution={"app_id":call["app_id"],"call_id":call["call_id"],**call["execution"]}
    if result.get("execution")!=call["execution"]: raise ValueError("returned physical worker identity differs")
    files={}; identities={}
    for row in result["files"]:
        name=row["name"]
        if not bound.allowed_output(name) or name in files: raise ValueError("exact returned pair inventory required")
        path=directory/"returned"/name
        if batch.base.wire.regular(path,bound.FILE_CAP)!=row["bytes"] or batch.base.wire.digest(path)!=row["sha256"]:
            raise ValueError("returned pair identity differs")
        files[name]=path; identities[name]={k:row[k] for k in ("bytes","sha256")}
    bound.validate_terminal(result,files,pair_plan_sha,plan)
    frozen,manifest=queue.load_plan(journal,journal_sha)
    claim=queue.find_claim(journal,claim_token); queue.validate_claim(claim,frozen,manifest)
    if (len(claim["attempts"])!=2 or [a["game"] for a in claim["attempts"]]!=plan["games"]
            or [a["attempt_label"] for a in claim["attempts"]]!=plan["artifact_labels"]):
        raise ValueError("received pair differs from durable claim")
    queue.bind_execution(journal,journal_sha,claim_token,execution)
    completed=[]
    for attempt in claim["attempts"]:
        label=attempt["attempt_label"]; prefix="project/artifacts/integration/frisson_ai/"+label+"/"
        selected=[{"path":name[len(prefix):],**identity} for name,identity in identities.items() if name.startswith(prefix)]
        if not selected: raise ValueError("native attempt artifact set missing")
        destination=queue.job.ARTIFACTS/label
        if not destination.exists():
            temporary=queue.job.ARTIFACTS/(".modal-"+attempt["attempt_token"]+"-receiving")
            temporary.mkdir(exist_ok=True)
            if temporary.resolve()!=temporary: raise ValueError("owned staging directory required")
            for row in selected: copy_once(files[prefix+row["path"]],temporary/row["path"],row)
            queue.verify_artifacts(temporary.name,selected)
            temporary.rename(destination)
            fd=os.open(destination.parent,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        queue.verify_artifacts(label,selected)
        receipt={"schema":queue.SCHEMA,"plan_sha256":journal_sha,"claim_token":claim_token,
            "attempt_token":attempt["attempt_token"],"execution":execution,"files":selected,"exit_code":0}
        path=directory/("ingestion-"+attempt["attempt_token"]+".json")
        if path.exists():
            if queue.read(path)!=receipt: raise ValueError("preserved ingestion receipt differs")
        else: queue.new_file(path,receipt)
        completed.append(queue.ingest(journal,journal_sha,path))
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True,type=Path)
    parser.add_argument("--journal",required=True,type=Path)
    parser.add_argument("--journal-sha",required=True)
    parser.add_argument("--pair-plan-sha",required=True)
    parser.add_argument("--claim-token",required=True)
    args=parser.parse_args()
    print(json.dumps(ingest_pair(args.directory,journal=args.journal,journal_sha=args.journal_sha,
        pair_plan_sha=args.pair_plan_sha,claim_token=args.claim_token)))


if __name__=="__main__": main()
