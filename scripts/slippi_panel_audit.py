#!/usr/bin/env python3
"""Inventory caller-supplied local Slippi checkpoint files without downloads."""
from __future__ import annotations

import argparse
import concurrent.futures
import enum
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/integration/frisson_ai/slippi-public-panel-p21-v1"
CACHE = ROOT / ".e001-cache/slippi-ai/models"
FOLDER = "https://www.dropbox.com/scl/fo/mg916t9exid4stqmx2bjf/"
KEY = "?rlkey=baqxnfxg2uytvcz62w9o8mwzt&dl=1"
LINK_IDS = {
    "fox_d21_ditto_v4": "AEryQOEBtmjLX_8z_AYdEuI",
    "fox_d24_ditto_v4": "ACHNYVS13AjH-yXICzeNDvA",
    "falco_d21_ditto_v4": "AFkL3Vu_7B-ZXgiRLIkrGOg",
    "fox_d21_ditto_hax_v3": "ANt2maXlxZdGhq5j_mPbyAI",
    "gm": "AI2frIdvITPIA7yUi9sw0O8",
    "gm-v1": "AKZ4_4ghCpCOKNpZQv2q2PE",
    "master": "AIvQqeG0uR55u0qYEKujoIc",
    "diamond": "AM2YJ7U-BDnZ8IObiFxVfJw",
    "plat": "AKCnL4iLfie3MAja2yNS9QQ",
    "gold": "ANGO3DUMe3vBmSf4jVy1yXA",
    "silver": "APDMoX-KJPAucW8l6ASV1uE",
    "SFIL": "AJ3PL8rUzWKZtLuX0MD4cN8",
    "medium": "ABb7w8MZfycLsLynGrMo9MA",
    "medium-v1": "AMC9ihXGkkZ6lngNeyk2uhc",
}


class ItemsType(enum.Enum):
    SKIP = "skip"
    FLAT = "flat"
    MLP = "mlp"


class OpponentType(enum.Enum):
    CPU = "cpu"
    SELF = "self"
    OTHER = "other"


class RestrictedUnpickler(pickle.Unpickler):
    """Only permit the value types actually used by these numeric checkpoints."""
    def find_class(self, module, name):
        if module in {"slippi_ai.embed", "slippi_ai.tf.embed"} and name == "ItemsType":
            return ItemsType
        if module == "slippi_ai.rl.run_lib" and name == "OpponentType":
            return OpponentType
        if module in {"melee.enums", "melee"} and name in {"Character", "Stage"}:
            import melee
            return getattr(melee, name)
        if module == "numpy" and name in {"ndarray", "dtype"}:
            return getattr(np, name)
        if module in {"numpy.core.multiarray", "numpy._core.multiarray"} and name in {"_reconstruct", "scalar"}:
            from numpy.core import multiarray
            return getattr(multiarray, name)
        if module in {"numpy.core.numeric", "numpy._core.numeric"} and name == "_frombuffer":
            from numpy.core.numeric import _frombuffer
            return _frombuffer
        raise pickle.UnpicklingError(f"unapproved checkpoint global: {module}.{name}")


def load_restricted(path):
    with Path(path).open("rb") as stream:
        return RestrictedUnpickler(stream).load()


def portable_metadata_path(value: str) -> str:
    """Replace an upstream runtime-home prefix in descriptive metadata only."""
    parts = Path(value).parts
    if len(parts) > 3 and parts[:2] == ("/", "home") and "SSBM" in parts[2:]:
        return "${UPSTREAM_RUNTIME}/" + "/".join(parts[parts.index("SSBM"):])
    return value


def json_safe(value):
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__name__, "name": value.name, "value": value.value}
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, str):
        return portable_metadata_path(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise TypeError(f"unhandled metadata value {type(value)}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def arrays(value):
    if isinstance(value, np.ndarray):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from arrays(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from arrays(item)


def inspect(name, url):
    path = CACHE / name
    if not path.is_file():
        raise FileNotFoundError(f"Supply the pinned opponent checkpoint locally: {path}")
    state = load_restricted(path)
    policy_arrays = list(arrays(state["state"]["policy"]))
    policy_hash = hashlib.sha256()
    for value in policy_arrays:
        policy_hash.update(str(value.shape).encode())
        policy_hash.update(str(value.dtype).encode())
        policy_hash.update(value.tobytes())
    record = {
        "name": name, "path": str(path), "url": url,
        "sha256": digest(path), "bytes": path.stat().st_size,
        "policy_sha256": policy_hash.hexdigest(),
        "variable_count": len(policy_arrays),
        "parameter_count": sum(int(x.size) for x in policy_arrays),
        "finite": all(bool(np.isfinite(x).all()) for x in policy_arrays),
        "metadata": {k: json_safe(v) for k, v in state.items() if k != "state"},
        "state_keys": list(state["state"]),
    }
    atomic_json(OUTPUT / "audit" / (name + ".json"), record)
    print(json.dumps({k: record[k] for k in ("name", "bytes", "sha256", "parameter_count", "finite")}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append")
    args = parser.parse_args()
    candidates = {name: FOLDER + link + "/" + name + KEY for name, link in LINK_IDS.items()}
    candidates["medium-v2"] = "https://www.dropbox.com/scl/fi/lpi9krfei1knfvfw7up7v/medium-v2?rlkey=qmah3qfz5anwva93x48zcx01k&dl=1"
    if args.name:
        candidates = {name: candidates[name] for name in args.name}
    records, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(inspect, name, url): name for name, url in candidates.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                records[name] = future.result()
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
                print(json.dumps({"name": name, "error": failures[name]}), flush=True)
    atomic_json(OUTPUT / "audit-summary.json", {"created_unix": time.time(), "records": records, "failures": failures})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
