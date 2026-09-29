"""Isolated D0 gameplay binding over the retained owned-game implementation.

Reference qualification keeps its original /opt/runtime-root files. New model
admission and gameplay sources are staged separately under /opt/d0-root.
Import and binding are local read-only operations.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import inspect
import re
import textwrap
import types
from functools import lru_cache
from pathlib import Path

import faynt_d0_benchmark_plan as fresh
import modal_panel_durable_game as durable
import modal_panel_game_pilot as proven
import modal_panel_pair_binding as pairing

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path("/opt/d0-root")
INPUT = Path("/opt/full-game-input")
PLAN = Path("/opt/faynt-d0-game-plan.json")
PANEL = f"artifacts/integration/frisson_ai/{fresh.RUN_ID}/expanded-slippi/manifest.json"
OLD_PANEL = fresh.SOURCE_FILES["expanded_slippi"][0]
ENTRY = "scripts/faynt_d0_durable_game.py"
SCHEMA = "e011.faynt-d0-durable-game.v1"
VOLUME = "frisson-faynt-d0-game-results-v1"
INDEX = "frisson-faynt-d0-game-index-v1"
MOUNT = durable.MOUNT
MAX_PHYSICAL = durable.MAX_PHYSICAL
SUCCESS = durable.SUCCESS
EXTRA_SOURCES = (
    ENTRY, "scripts/faynt_d0_benchmark_plan.py", "scripts/modal_panel_durable_game.py",
    "scripts/modal_panel_batch_pilot.py", "scripts/modal_panel_pair_binding.py",
    "src/melee_policy/integration/faynt_d0_checkpoints.py", OLD_PANEL,
    *proven.runtime.DATA_ASSETS,
)


def _clone_module(module, **updates):
    namespace = {**vars(module), **updates}
    for name, function in vars(module).items():
        if isinstance(function, types.FunctionType) and function.__module__ == module.__name__:
            clone = types.FunctionType(
                function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__,
            )
            clone.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
            namespace[name] = clone
    return types.SimpleNamespace(**namespace), namespace


def _catalog(root):
    # Loading this value-only module by path leaves the gameplay package fresh
    # until OwnedLinuxContext performs its hash-bound native imports.
    path = root / "src/melee_policy/integration/faynt_d0_checkpoints.py"
    spec = importlib.util.spec_from_file_location("_faynt_d0_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=4)
def _read_panel(root_text, new_stat, old_stat):
    del new_stat, old_stat  # Cache keys track file replacement between bindings.
    root = Path(root_text)
    source = root / OLD_PANEL
    if proven.wire.digest(source) != fresh.SOURCE_FILES["expanded_slippi"][1]:
        raise ValueError("original Slippi schedule bytes differ")
    original = proven.policy.bounded_json(source)
    catalog = _catalog(root)
    provenance = {"expanded_slippi": {
        "path": OLD_PANEL, "sha256": proven.wire.digest(source),
        "byte_length": source.stat().st_size, "selected_field": "whole manifest",
    }}
    expected = fresh.build_manifest(
        catalog.CHECKPOINTS, sources={"expanded_slippi": original}, provenance=provenance,
    )["expanded_slippi"]
    panel = proven.policy.bounded_json(root / PANEL)
    if panel != expected:
        raise ValueError("fresh panel differs from the exact checkpoint-only 2624-game transformation")
    return panel, proven.wire.digest(root / PANEL)


def _panel(root):
    keys = []
    for relative in (PANEL, OLD_PANEL):
        stat = (root / relative).stat()
        keys.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return _read_panel(str(root), *keys)


def _reference_views(plan, *, remote):
    """Resolve original proof paths to identical uploaded bytes without edits."""
    rows = {row["remote"]: row for row in plan["uploads"]}
    mapping = {name: Path(row["remote"] if remote else row["local"]) for name, row in rows.items()}
    for name in (str(proven.live.PLAN), str(proven.runtime.PLAN)):
        nested = proven.policy.bounded_json(mapping[name])
        for row in nested["uploads"]:
            outer = rows.get(row["remote"])
            if outer is None or any(outer[key] != row[key] for key in ("bytes", "sha256")):
                raise ValueError("original reference input was replaced by gameplay bytes")
            mapping[row["local"]] = mapping[row["remote"]]

    def resolve(path):
        return mapping.get(str(path), Path(path))

    wire = types.SimpleNamespace(**{
        **vars(proven.wire),
        "digest": lambda path: proven.wire.digest(resolve(path)),
        "regular": lambda path, cap: proven.wire.regular(resolve(path), cap),
    })
    policy = types.SimpleNamespace(**{
        **vars(proven.policy),
        "bounded_json": lambda path: proven.policy.bounded_json(resolve(path)),
    })
    context = types.SimpleNamespace(**{
        **vars(proven.context),
        "read_bound": lambda path, *args, **kwargs: proven.context.read_bound(resolve(path), *args, **kwargs),
    })
    runtime, _ = _clone_module(proven.runtime, wire=wire, policy=policy, context=context)
    live, _ = _clone_module(proven.live, wire=wire, policy=policy, context=context, runtime=runtime)

    def validate_tape(result, files, sha, prior, **ignored):
        mapped = {name: resolve(path) for name, path in files.items()}
        return live.validate_terminal(result, mapped, sha, prior)

    return {"wire": wire, "policy": policy, "runtime": runtime, "live": live,
            "validate_tape_terminal": validate_tape}


def _run_native_d0(runtime, game, label, project, manifest_path):
    """Register exact new identities in this fresh child, then reuse run_game."""
    from melee_policy.integration import frisson_match as fm
    from melee_policy.integration import frisson_slippi_match as match
    from melee_policy.integration.faynt_d0_checkpoints import CHECKPOINTS, FORMAT, require_identity

    project = Path(project)
    panel = proven.policy.bounded_json(manifest_path)
    if panel["frisson_checkpoints"] != CHECKPOINTS or game not in panel["games"]:
        raise ValueError("fresh native learner or game identity differs")
    if any(spec.checkpoint_format == FORMAT for spec in match.FINAL_CHECKPOINT_SPECS):
        raise RuntimeError("D0 native admission requires one fresh child")

    class D0Spec(match._FinalCheckpointSpec):
        @property
        def display_name(self):
            return f"Faynt {self.profile.upper()} RL D0 step-{self.step}"

    match.FINAL_CHECKPOINT_SPECS += tuple(D0Spec(
        profile=profile, checkpoint_format=FORMAT, relative_path=Path(row["relative_path"]),
        sha256=row["sha256"], byte_length=row["byte_length"], step=row["step"],
        processed_target_frames=None, parameter_count=row["parameter_count"],
        trial_id=f"faynt-d0-{profile}-rl-step-{row['step']}",
        wandb_run_id=row["wandb_run_id"], validation_nll=None,
    ) for profile, row in CHECKPOINTS.items())
    fm._INSPECTED_CHECKPOINT_PARAMETER_COUNTS = {
        **fm._INSPECTED_CHECKPOINT_PARAMETER_COUNTS,
        **{(FORMAT, profile): row["parameter_count"] for profile, row in CHECKPOINTS.items()},
    }
    fm.POST_RL_CHECKPOINTS = {
        **fm.POST_RL_CHECKPOINTS, **{f"d0-{profile}": row for profile, row in CHECKPOINTS.items()},
    }
    original = match._selected_checkpoint_contract

    def checked_identity(identity):
        if identity.get("format") == FORMAT:
            require_identity(identity)
        return original(identity)

    match._selected_checkpoint_contract = checked_identity
    runtime.ROOT, runtime.OUTPUT = project, Path(manifest_path).parent
    return runtime.run_game(game, label)


def _bind_child(namespace):
    tree = ast.parse(textwrap.dedent(inspect.getsource(proven.game_child)))
    replacements = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "slippi_panel_runtime.run_game":
            node.func = ast.Name(id="_run_native_d0", ctx=ast.Load())
            node.args = [ast.Name(id="slippi_panel_runtime", ctx=ast.Load()), *node.args,
                         ast.Name(id="PROJECT", ctx=ast.Load()),
                         ast.parse("PROJECT / PANEL", mode="eval").body]
            replacements += 1
    if replacements != 1:
        raise ValueError("proven child entrypoint shape changed")
    namespace["_run_native_d0"] = _run_native_d0
    exec(compile(ast.fix_missing_locations(tree), "<faynt-d0-owned-child>", "exec"), namespace)


def bind(labels, attempt_labels, index, *, remote=False):
    root = REMOTE if remote else ROOT
    if proven.wire.digest(root / "scripts/modal_panel_game_pilot.py") != pairing.PROVEN_SHA:
        raise ValueError("proven full-game implementation identity differs")
    if type(index) is not int or index not in (0, 1):
        raise ValueError("one original game index required")
    panel, panel_sha = _panel(root)
    games = pairing.select_pair(panel, labels)
    if (not isinstance(attempt_labels, (tuple, list)) or len(attempt_labels) != 2
            or any(not re.fullmatch(re.escape(game["label"]) + r"-a[1-3]", label)
                   for game, label in zip(games, attempt_labels, strict=True))):
        raise ValueError("exact fresh attempt labels required")
    first = games[0]
    learner = panel["frisson_checkpoints"][first["profile"]]
    release = panel["releases"][first["release"]]
    release_path = Path(release["path"])
    relative = release_path.as_posix()
    if release_path.is_absolute():
        marker = ".e001-cache/slippi-ai/models/"
        if marker not in relative:
            raise ValueError("approved immutable opponent directory required")
        relative = marker + relative.split(marker, 1)[1]
    if not relative.startswith(".e001-cache/slippi-ai/models/") or ".." in Path(relative).parts:
        raise ValueError("approved immutable opponent directory required")
    checkpoints = {
        learner["relative_path"]: {"bytes": learner["byte_length"], "sha256": learner["sha256"]},
        relative: {"bytes": release["bytes"], "sha256": release["sha256"]},
    }
    if panel["configuration"]["wall_limit_seconds"] != 7200:
        raise ValueError("original native wall limit required")
    scope = {
        "benchmark_games": 1, "benchmark_ledger_writes": False,
        "profile": first["profile"], "checkpoint_step": learner["step"],
        "opponent": first["release"], "opponent_name": first["player_name"],
        "characters": [first["frisson_character"], first["slippi_character"]],
        "stage": first["stage"], "seed": first["seed"],
        "frisson_physical_ports": [game["frisson_port"] for game in games],
        "native_max_game_frames": 30000, "policy_inference": True,
        "save_slp": True, "save_video": False,
        "numeric_classification": proven.context.LIMITED_CLASSIFICATION,
        "numeric_evidence_scope": "retained V4 reference cases; new weights and trajectories unmeasured",
        "cross_host_trajectory_equivalence": "unassessed", "pair_id": first["pair_id"],
        "selected_game_indices": [index], "durable_per_game": True,
        "durable_volume": VOLUME, "durable_index": INDEX, "max_physical_executions": MAX_PHYSICAL,
        "run_id": fresh.RUN_ID, "trained_delay": 0,
        "reference_runtime_root": str(proven.REMOTE), "gameplay_source_root": str(REMOTE),
    }
    work = Path("/work/research/faynt-d0-game")
    _, namespace = _clone_module(
        proven, ROOT=root, REMOTE=REMOTE, INPUT=INPUT, PLAN=PLAN, PANEL=PANEL, PANEL_SHA=panel_sha,
        COMPAT=root / f"artifacts/integration/frisson_ai/{fresh.RUN_ID}/research",
        WORK=work, PARENT=work / "source", PROJECT=work / "source/melee-policy",
        ENTRY=REMOTE / ENTRY, SCHEMA=SCHEMA, CHECKPOINTS=checkpoints, LABELS=tuple(attempt_labels),
        SOURCE_PATHS=tuple(dict.fromkeys((*proven.SOURCE_PATHS, *EXTRA_SOURCES))),
        GAME_SECONDS=7200, OWNER_SECONDS=8700, FUNCTION_SECONDS=8790, SESSION_SECONDS=9090,
        UPLOAD_CAP=proven.UPLOAD_CAP + 1024**3,
    )

    def fixed_scope():
        return copy.deepcopy(scope)

    def selected_games(value):
        selected = pairing.select_pair(value, labels)
        if selected != games:
            raise ValueError("bound fresh game specification changed")
        return selected

    def allowed_output(name):
        if not isinstance(name, str):
            return False
        for old, new in zip(proven.LABELS, attempt_labels, strict=True):
            prefix = "project/artifacts/integration/frisson_ai/" + new + "/"
            if name.startswith(prefix):
                return proven.allowed_output(
                    "project/artifacts/integration/frisson_ai/" + old + "/" + name[len(prefix):],
                )
        return proven.allowed_output(name) and not name.startswith("project/")

    def verify_plan(path, sha, *, remote=False):
        if proven.wire.digest(path) != sha:
            raise ValueError("frozen D0 game plan differs")
        plan = proven.policy.bounded_json(path)
        namespace["check_layout"](plan)
        views = _reference_views(plan, remote=remote)
        verifier = types.FunctionType(proven.verify_plan.__code__, {**namespace, **views})
        # All reference I/O is redirected to the identical outer inventory.
        # The original verification takes its remote branch to retain nested
        # plan bytes while avoiding reconstruction from today's working tree.
        return verifier(path, sha, remote=True)

    original_limits = namespace["limits"]

    def limits():
        return {**original_limits(), "nonpreemptible": False}

    namespace.update(fixed_scope=fixed_scope, selected_games=selected_games,
                     allowed_output=allowed_output, verify_plan=verify_plan, limits=limits)
    durable.selected_function(proven.owner, namespace, index, owner=True)
    durable.selected_function(proven.validate_terminal, namespace, index)
    _bind_child(namespace)
    namespace["binding"] = {
        "schema": SCHEMA + ".binding", "proven_source_sha256": pairing.PROVEN_SHA,
        "labels": list(labels), "attempt_labels": list(attempt_labels), "scope": scope,
        "native_policy_loop_changed": False, "reference_inputs_separate": True,
        "child_change": "exact checkpoint admission and fresh manifest selection",
    }
    return types.SimpleNamespace(**namespace)


def plan_binding(path, sha, *, remote=False):
    if proven.wire.digest(path) != sha:
        raise ValueError("exact frozen D0 durable plan required")
    plan = proven.policy.bounded_json(path)
    indices = plan.get("scope", {}).get("selected_game_indices")
    if indices not in ([0], [1]) or type(indices[0]) is not int:
        raise ValueError("one selected original game required")
    labels = [game["label"] for game in plan["games"]]
    bound = bind(labels, plan["artifact_labels"], indices[0], remote=remote)
    bound.verify_plan(path, sha, remote=remote)
    return bound, plan


# Reuse persistence, crash recovery and bounded physical-attempt accounting with
# independent volume/index names and the fresh entrypoint. Imported originals
# retain their globals and cannot submit a D0 game by accident.
_base = types.SimpleNamespace(**{**vars(proven), "ROOT": ROOT, "REMOTE": REMOTE, "PLAN": PLAN})
_batch = types.SimpleNamespace(**{**vars(durable.batch), "base": _base})
_durable, _durable_globals = _clone_module(
    durable, batch=_batch, ENTRY=ENTRY, VOLUME=VOLUME, INDEX=INDEX, MOUNT=MOUNT,
    bind=bind, plan_binding=plan_binding,
)
# _clone_module preserves original function code; bind functions have dedicated
# implementations because the old binding deliberately admits only P21.
_durable_globals.update(bind=bind, plan_binding=plan_binding)
for _name in (
    "create_json", "check_manifest", "persist_result", "retryable_startup", "run_durable",
    "remote_durable", "download_result", "recover_saved",
):
    globals()[_name] = _durable_globals[_name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--owner", action="store_true")
    group.add_argument("--game-child", type=int, choices=(0, 1))
    parser.add_argument("--sha")
    parser.add_argument("--expires", type=float)
    parser.add_argument("--image-id")
    args = parser.parse_args()
    sha = args.sha or proven.wire.digest(PLAN)
    bound, plan = plan_binding(PLAN, sha, remote=True)
    if args.owner:
        if not 0 < proven.wire.remaining(args.expires) <= bound.OWNER_SECONDS + 1:
            raise ValueError("bounded D0 owner required")
        return bound.owner(plan, sha, args.expires, args.image_id)
    if args.game_child != plan["scope"]["selected_game_indices"][0]:
        raise ValueError("unselected game refused")
    return bound.game_child(args.game_child)


if __name__ == "__main__":
    main()
