"""Bind the proven full-game host workflow to a frozen native panel pair.

Only declarative game settings, artifact labels, checkpoint identities, entry
path and wall limits vary. The proven function code and native game code remain
unchanged. Each binding owns an isolated globals dictionary in one process.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
import types

import modal_panel_game_pilot as proven

PROVEN_SHA = "ad6c07d0512a110b4727ea31f0a6a03da7909b9ea1719cac1fd78ee226c7b2ae"
SCHEMA = "e011.modal-frozen-panel-pair-binding.v1"


def select_pair(panel, labels):
    if not isinstance(labels, (tuple, list)) or len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("exact two ordered original game labels required")
    lookup = {g["label"]: g for g in panel["games"]}
    if len(lookup) != len(panel["games"]): raise ValueError("duplicate native manifest label")
    games = [copy.deepcopy(lookup[label]) for label in labels]
    a, b = games
    same = ("pair_id", "profile", "block", "release", "frisson_character", "slippi_character", "stage", "seed", "player_name")
    if (any(a[k] != b[k] for k in same) or {a["frisson_port"], b["frisson_port"]} != {1,2}
            or any(g["slippi_port"] != 3-g["frisson_port"] or g["save_slp"] is not True or g["save_video"] is not False for g in games)
            or not a["ordinal"] < b["ordinal"] or not re.fullmatch(r"[0-9a-f]{16}", a["pair_id"])):
        raise ValueError("exact original ordered port pair required")
    return games


def bind(panel, labels, attempt_labels, *, project_root, entry_relative):
    """Create a private function namespace without modifying proven globals."""
    root = Path(project_root)
    if proven.wire.digest(root / "scripts/modal_panel_game_pilot.py") != PROVEN_SHA:
        raise ValueError("proven full-game implementation identity differs")
    if proven.wire.digest(root / proven.PANEL) != proven.PANEL_SHA:
        raise ValueError("frozen native manifest identity differs")
    if proven.policy.bounded_json(root / proven.PANEL) != panel:
        raise ValueError("panel object differs from exact manifest bytes")
    games = select_pair(panel, labels)
    if (not isinstance(attempt_labels, (list, tuple)) or len(attempt_labels) != 2
            or any(not re.fullmatch(re.escape(g["label"]) + r"-a[1-3]", label)
                   for g,label in zip(games, attempt_labels, strict=True))):
        raise ValueError("exact native attempt artifact labels required")
    if entry_relative != "scripts/modal_panel_batch_pilot.py":
        raise ValueError("reviewed benchmark entrypoint required")
    first = games[0]; profile = panel["frisson_checkpoints"][first["profile"]]
    release = panel["releases"][first["release"]]
    release_path = Path(release["path"])
    relative = release_path.as_posix()
    if release_path.is_absolute():
        marker = ".e001-cache/slippi-ai/models/"
        if marker not in relative:
            raise ValueError("approved native checkpoint directory required")
        relative = marker + relative.split(marker, 1)[1]
    if not relative.startswith(".e001-cache/slippi-ai/models/") or ".." in Path(relative).parts:
        raise ValueError("approved native checkpoint directory required")
    checkpoints = {
        profile["relative_path"]: {"bytes":profile["byte_length"], "sha256":profile["sha256"]},
        relative: {"bytes":release["bytes"], "sha256":release["sha256"]},
    }
    if any(not proven.wire.SHA.fullmatch(v["sha256"]) or type(v["bytes"]) is not int or v["bytes"] <= 0 for v in checkpoints.values()):
        raise ValueError("exact immutable native checkpoint identities required")
    wall = panel["configuration"]["wall_limit_seconds"]
    if wall != 7200 or panel["configuration"]["maximum_game_frames"] != 30000:
        raise ValueError("original native wall and frame limits required")
    scope = {"benchmark_games":2, "benchmark_ledger_writes":False,
        "profile":first["profile"], "checkpoint_step":profile["step"], "opponent":first["release"],
        "opponent_name":first["player_name"], "characters":[first["frisson_character"],first["slippi_character"]],
        "stage":first["stage"], "seed":first["seed"], "frisson_physical_ports":[g["frisson_port"] for g in games],
        "native_max_game_frames":30000, "policy_inference":True, "save_slp":True, "save_video":False,
        "numeric_classification":proven.context.LIMITED_CLASSIFICATION,
        "numeric_evidence_scope":"retained V4 reference cases; this checkpoint and trajectory may be unmeasured",
        "cross_host_trajectory_equivalence":"unassessed", "pair_id":first["pair_id"]}
    namespace = dict(vars(proven))
    namespace.update(ROOT=root, COMPAT=root/proven.policy.REFERENCE,
        ENTRY=proven.REMOTE/entry_relative, CHECKPOINTS=checkpoints, LABELS=tuple(attempt_labels),
        SOURCE_PATHS=tuple(dict.fromkeys((*proven.SOURCE_PATHS, "scripts/modal_panel_pair_binding.py", entry_relative))),
        GAME_SECONDS=wall, OWNER_SECONDS=2*wall+1500, FUNCTION_SECONDS=2*wall+1590, SESSION_SECONDS=2*wall+1890)
    # The original functions resolve one another inside this isolated mapping.
    # No assignments are made to the imported proven module or native modules.
    for name, function in vars(proven).items():
        if isinstance(function, types.FunctionType) and function.__module__ == proven.__name__:
            clone = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
            clone.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
            namespace[name] = clone
    def fixed_scope(): return copy.deepcopy(scope)
    def selected_games(value):
        selected = select_pair(value, labels)
        if selected != games: raise ValueError("bound pair specification changed")
        return selected
    def allowed_output(name):
        if not isinstance(name,str): return False
        for original, assigned in zip(proven.LABELS, attempt_labels, strict=True):
            prefix = "project/artifacts/integration/frisson_ai/" + assigned + "/"
            if name.startswith(prefix):
                return proven.allowed_output("project/artifacts/integration/frisson_ai/" + original + "/" + name[len(prefix):])
        return proven.allowed_output(name) and not name.startswith("project/")
    namespace.update(fixed_scope=fixed_scope, selected_games=selected_games, allowed_output=allowed_output)
    result = types.SimpleNamespace(**namespace)
    result.binding = {"schema":SCHEMA, "proven_source_sha256":PROVEN_SHA, "labels":list(labels),
        "attempt_labels":list(attempt_labels), "scope":scope,
        "reused_function_code": [name for name,value in namespace.items() if isinstance(value,types.FunctionType)
                                 and name in vars(proven) and value.__code__ is getattr(proven,name).__code__],
        "native_game_code_changed":False}
    return result
