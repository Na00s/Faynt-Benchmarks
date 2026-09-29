"""Pure tape, plan and synthetic receipt checks. No native process or model."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import modal_panel_live_canary as c


def args():
    record = {"bytes": 10, "sha256": "a" * 64}
    return {"label": "test-tape-v1", "created_at": 1000, "expires_at": 1900,
            "source_identities": {p: copy.deepcopy(record) for p in c.SOURCE_PATHS},
            "admissions": {p: copy.deepcopy(record) for p in c.ADMISSION_KEYS},
            "runtime_image": {"image_id": "im-test123", "manifest_sha256": "b" * 64, "platform": "linux_x86_64"},
            "game_image": copy.deepcopy(c.IMAGE), "udp_ports": [51441, 51442]}


def test_tape_exact_repeat_and_native_canonical_semantics():
    from melee_policy.integration.slippi_ai_policy import CanonicalControllerCommand, DIGITAL_BUTTON_ORDER
    tape = c.canonical_tape()
    assert tape == c.canonical_tape() and len(tape["rows"]) == 320
    assert tape["policy_inference_calls"] == 0
    buttons = set()
    for i, row in enumerate(tape["rows"]):
        assert row["index"] == i and row["source_frame"] == -123 + i
        for channel in ("a", "b"):
            value = row[channel]
            assert CanonicalControllerCommand(**value).as_dict() == value
            buttons.update(value["buttons"])
            if i % 16 >= 8:
                assert value == CanonicalControllerCommand.neutral().as_dict()
            if i >= 160:
                assert value == tape["rows"][i - 160][channel]
    assert buttons == set(DIGITAL_BUTTON_ORDER)
    tape["rows"][0]["a"]["main_stick"][0] = 0.25
    assert c.canonical_tape()["rows"][0]["a"]["main_stick"][0] == 0.5


def test_plan_is_pure_and_exact_without_asset_reads(monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("plan opened an asset"))
    plan = c.make_plan(**args())
    assert c.validate_plan(plan, expected_sha256=c.identity(plan)["sha256"], now=1000) == plan
    assert plan["scientific"]["required_observed_and_replay_frame"] == 197
    assert plan["execution"]["retry"] is False
    assert plan["claims"]["gameplay_admitted"] is False
    for index in range(320):
        first, reversed_ports = c.physical_commands(plan, 0, index), c.physical_commands(plan, 1, index)
        assert first[1] == reversed_ports[2] and first[2] == reversed_ports[1]


@pytest.mark.parametrize("mutation", [
    lambda a: a.update(label="../bad"), lambda a: a.update(created_at=True),
    lambda a: a.update(expires_at=1901), lambda a: a["source_identities"].pop(c.SOURCE_PATHS[0]),
    lambda a: a["admissions"].pop("game_image_access"), lambda a: a["admissions"]["policy_parity"].update(bytes=0),
    lambda a: a["runtime_image"].update(platform="darwin_arm64"),
    lambda a: a["game_image"].update(disc_revision=1), lambda a: a["game_image"].update(byte_length=True),
    lambda a: a.update(udp_ports=[51441, 51441]), lambda a: a.update(udp_ports=[True, 51442]),
])
def test_invalid_declarations_fail(mutation):
    value = args(); mutation(value)
    with pytest.raises(ValueError): c.make_plan(**value)


@pytest.mark.parametrize("field,value", [
    ("claims", {"gameplay_admitted": True}), ("scientific", {}), ("execution", {}),
    ("unexpected", 1), ("cases", []),
])
def test_plan_tampering_rejected_even_with_recomputed_outer_digest(field, value):
    plan = c.make_plan(**args()); plan[field] = value
    with pytest.raises((ValueError, KeyError, IndexError)):
        c.validate_plan(plan, expected_sha256=c.identity(plan)["sha256"], now=1000)


@pytest.mark.parametrize("now", [999, 1900, float("nan"), float("inf"), True])
def test_absolute_window_cannot_be_reset(now):
    plan = c.make_plan(**args())
    with pytest.raises(TimeoutError): c.validate_plan(plan, expected_sha256=c.identity(plan)["sha256"], now=now)


def test_exact_new_plan_write_preserves_existing_and_links(tmp_path):
    plan = c.make_plan(**args()); dest = tmp_path.resolve() / "plan.json"
    expected = c.write_new_plan(dest, plan)
    assert expected == {"bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
    with pytest.raises(FileExistsError): c.write_new_plan(dest, plan)
    symlink = dest.parent / "link.json"; symlink.symlink_to(dest)
    with pytest.raises(ValueError): c.write_new_plan(symlink, plan)
    assert json.loads(dest.read_bytes()) == plan


def receipt():
    return {"observed_frames": list(range(-123, 198)), "policy_calls": 0,
            "shutdown_method": "child-sigint-input-drain",
            "transport_checks": dict.fromkeys(c.TRANSPORT_CHECKS, True),
            "controller_audit": {"gate": {"decision": "pass", "checks": dict.fromkeys(c.CONTROLLER_CHECKS, True)},
                                 "alignment": {"steady_state_trace_to_replay_lag_frames": 1,
                                               "permitted_terminal_unobservable_trace_frames": [],
                                               "last_aligned_pair": [196, 197]}}}


def test_receipt_requires_entire_tape_and_last_lag_one_command():
    assert all(c.tape_receipt_checks(**receipt()).values())


@pytest.mark.parametrize("mutation", [
    lambda r: r["observed_frames"].pop(), lambda r: r.update(policy_calls=1),
    lambda r: r.update(shutdown_method="child-double-sigint"),
    lambda r: r["transport_checks"].pop(next(iter(c.TRANSPORT_CHECKS))),
    lambda r: r["controller_audit"]["gate"]["checks"].pop(next(iter(c.CONTROLLER_CHECKS))),
    lambda r: r["controller_audit"]["alignment"].update(permitted_terminal_unobservable_trace_frames=[196]),
    lambda r: r["controller_audit"]["alignment"].update(last_aligned_pair=[195, 196]),
    lambda r: r["controller_audit"]["alignment"].update(steady_state_trace_to_replay_lag_frames=2),
])
def test_incomplete_or_changed_receipt_fails(mutation):
    r = receipt(); mutation(r)
    assert not all(c.tape_receipt_checks(**r).values())


def test_check_names_match_unchanged_native_implementations():
    replay = ast.parse((ROOT / "src/melee_policy/integration/slippi_match.py").read_text())
    function = next(n for n in replay.body if isinstance(n, ast.FunctionDef) and n.name == "_audit_controller_boundary_records")
    checks = next(n.value for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "checks" for t in n.targets))
    assert {ast.literal_eval(k) for k in checks.keys} == c.CONTROLLER_CHECKS
    native = ast.parse((ROOT / "src/melee_policy/integration/match_runtime.py").read_text())
    cls = next(n for n in native.body if isinstance(n, ast.ClassDef) and n.name == "_ControllerPipeLockstep")
    gate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "gate_checks")
    check_return = next(n.value for n in gate.body if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict))
    assert {ast.literal_eval(k) for k in check_return.keys} == c.TRANSPORT_CHECKS


def native_transport_class():
    """Compile the unchanged two native classes with their inert dependencies."""
    source = ROOT / "src/melee_policy/integration/match_runtime.py"
    parsed = ast.parse(source.read_text())
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name in {"_ControllerPipeLockstep", "_ControllerFlushProxy"}]
    scope = {"Any": Any, "cast": cast, "Path": Path, "copy": copy, "os": os, "stat": stat}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), "exec"), scope)
    return scope["_ControllerPipeLockstep"]


class RawController:
    def __init__(self, port, directory):
        self.port = port; self.pipe_path = directory / f"Bot{port}"; os.mkfifo(self.pipe_path)
        self.writes = []; self.actions = []
    def _write(self, text): self.writes.append(text)
    def flush(self): self.writes.append("FLUSH\n")
    def press_button(self, *args): self.actions.append(("press", args))
    def release_button(self, *args): self.actions.append(("release", args))
    def tilt_analog(self, *args): self.actions.append(("tilt", args))
    def press_shoulder(self, *args): self.actions.append(("shoulder", args))


def tape_fixture(tmp_path, monkeypatch, *, frames=None):
    from melee_policy.integration import slippi_ai_policy as dispatch
    import melee
    raw = {p: RawController(p, tmp_path.resolve()) for p in (1, 2)}
    player = SimpleNamespace(character=melee.Character.FOX)
    states = [None, SimpleNamespace(menu_state=melee.Menu.CHARACTER_SELECT)]
    states += [SimpleNamespace(frame=f, menu_state=melee.Menu.IN_GAME, stage=melee.Stage.FINAL_DESTINATION,
                              players={1: player, 2: player}) for f in (frames if frames is not None else range(-123, 198))]
    console = SimpleNamespace(controllers=list(raw.values()))
    def step():
        for controller in console.controllers: controller.flush()
        return states.pop(0)
    console.step = step
    transport = native_transport_class().install(console, raw)
    menus = []
    helpers = {p: SimpleNamespace(menu_helper_simple=lambda *a, **kw: menus.append((a, kw))) for p in (1, 2)}
    def require_ports(state, frame):
        assert set(state.players) == {1, 2}
    native = SimpleNamespace(mimic=SimpleNamespace(_require_exact_player_ports=require_ports), dispatch=dispatch,
                             frisson=SimpleNamespace(_player_state=lambda p: {"character": "FOX"}), melee=melee)
    monkeypatch.setattr(c.time, "time", lambda: 1001)
    return transport, helpers, native, raw, menus


@pytest.mark.parametrize("case_index", [0, 1])
def test_native_transport_sender_tape_loop_and_lag_drain(tmp_path, monkeypatch, case_index):
    transport, helpers, native, raw, menus = tape_fixture(tmp_path, monkeypatch)
    rows = []
    result = c.drive_tape(c.make_plan(**args()), case_index, transport=transport,
                          menu_helpers=helpers, native=native, write_trace=rows.append, check_deadline=lambda: None)
    assert result["observed_frames"] == list(range(-123, 198)) and result["policy_calls"] == 0
    assert len(rows) == 320 and rows[-1]["game_frame"] == 196
    assert all(slot["inference"]["called"] is False for row in rows for slot in row["slots"].values())
    assert [entry[1] for entry in menus] == [dict(cpu_level=0, autostart=False, frozen_stadium=True),
                                           dict(cpu_level=0, autostart=True, frozen_stadium=True)]
    assert all(transport.gate_checks().values())
    assert transport.audit_record()["frame_sync"]["kind_commits"]["gameplay"] == 320
    markers = [[w for w in raw[p].writes if w.startswith("FRAME_SYNC")] for p in (1, 2)]
    assert markers[0] == markers[1] and len(markers[0]) == 322
    for row in rows:
        expected = c.physical_commands(c.make_plan(**args()), case_index, row["tape_index"])
        assert {p: row["slots"][f"p{p}"]["command"] for p in (1, 2)} == expected


@pytest.mark.parametrize("frames", [[-122], [-123, -121], [-123, -123]])
def test_tape_frame_gaps_fail_before_another_transaction(tmp_path, monkeypatch, frames):
    transport, helpers, native, raw, menus = tape_fixture(tmp_path, monkeypatch, frames=frames)
    rows = []
    with pytest.raises(RuntimeError, match="exactly consecutive"):
        c.drive_tape(c.make_plan(**args()), 0, transport=transport, menu_helpers=helpers,
                     native=native, write_trace=rows.append, check_deadline=lambda: None)
    assert len(rows) <= 1


def test_tape_external_deadline_retains_partial_rows(tmp_path, monkeypatch):
    transport, helpers, native, raw, menus = tape_fixture(tmp_path, monkeypatch)
    rows = []
    def deadline():
        if len(rows) == 5: raise TimeoutError("outer deadline")
    with pytest.raises(TimeoutError, match="outer deadline"):
        c.drive_tape(c.make_plan(**args()), 0, transport=transport, menu_helpers=helpers,
                     native=native, write_trace=rows.append, check_deadline=deadline)
    assert len(rows) == 5 and rows[-1]["game_frame"] == -119


def primary_row():
    return {"game_frame": -123, "tape_index": 0, "policy_inference_calls": 0,
            "slots": {f"p{p}": {"port": p, "model": "controller-tape", "agent_kind": "controller-tape",
                        "checkpoint": None, "command": c._command(), "controller_dispatch": {"called": True},
                        "inference": {"called": False, "policy_inference_calls": 0}} for p in (1, 2)}}


def test_projection_preserves_every_command_frame_port_and_zero_inference():
    primary = primary_row(); projected = c.audit_projection(primary)
    assert primary["slots"]["p1"]["model"] == "controller-tape"
    assert projected["game_frame"] == primary["game_frame"]
    for port in (1, 2):
        slot, original = projected["slots"][f"p{port}"], primary["slots"][f"p{port}"]
        assert slot["model"] == "frisson-ai" and slot["model_field_role"] == "canonical-command-schema-selector-only"
        assert slot["agent_kind"] == "controller-tape" and slot["checkpoint"] is None
        for key in ("port", "command", "controller_dispatch", "inference"): assert slot[key] == original[key]
    projected["slots"]["p1"]["command"]["buttons"].append("A")
    assert not primary["slots"]["p1"]["command"]["buttons"]


def test_projection_refuses_policy_inference_claim():
    row = primary_row(); row["slots"]["p1"]["inference"]["called"] = True
    with pytest.raises(ValueError): c.audit_projection(row)


def fake_context(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "research"; root.mkdir()
    context = SimpleNamespace(research_output_root=root)
    context.check_deadline = lambda expiry: None
    def verify(plan, output):
        return {"status": "owned-linux-tape-context-ready", "plan_sha256": c.identity(plan)["sha256"],
                "admissions": plan["required_admissions"], "runtime_image": plan["runtime_image"]}
    context.verify_and_prepare = verify
    monkeypatch.setattr(c.time, "time", lambda: 1001)
    return context


def test_run_boundary_is_sequential_new_intent_and_no_retry(tmp_path, monkeypatch):
    context = fake_context(tmp_path, monkeypatch); plan = c.make_plan(**args()); calls = []
    def run(plan, index, context, directory):
        calls.append(index)
        return {"status": "controller-tape-case-passed"}
    monkeypatch.setattr(c, "_run_prepared_case", run)
    result = c.run_prepared_plan(plan, expected_sha256=c.identity(plan)["sha256"], context=context)
    assert calls == [0, 1] and result["status"] == "controller-tape-local-transport-passed"
    assert result["parent_group_cleanup_required"] and not result["gameplay_admitted"]
    with pytest.raises(FileExistsError): c.run_prepared_plan(plan, expected_sha256=c.identity(plan)["sha256"], context=context)
    assert calls == [0, 1]


def test_failed_first_case_retained_and_second_case_stays_unlaunched(tmp_path, monkeypatch):
    context = fake_context(tmp_path, monkeypatch); plan = c.make_plan(**args()); calls = []
    def fail(plan, index, context, directory):
        calls.append(index); (directory / "partial.slp").write_bytes(b"synthetic partial")
        return {"status": "failed"}
    monkeypatch.setattr(c, "_run_prepared_case", fail)
    result = c.run_prepared_plan(plan, expected_sha256=c.identity(plan)["sha256"], context=context)
    assert calls == [0] and result["status"] == "failed"
    assert (context.research_output_root / plan["label"] / "case-0/partial.slp").read_bytes() == b"synthetic partial"


def test_failed_admission_preserves_intent_and_launches_nothing(tmp_path, monkeypatch):
    context = fake_context(tmp_path, monkeypatch); plan = c.make_plan(**args())
    context.verify_and_prepare = lambda *a: {"status": "failed"}
    monkeypatch.setattr(c, "_run_prepared_case", lambda *a: pytest.fail("launched"))
    result = c.run_prepared_plan(plan, expected_sha256=c.identity(plan)["sha256"], context=context)
    assert result["status"] == "failed" and result["cases"] == []
    assert (context.research_output_root / plan["label"] / "intent.json").is_file()


def test_tape_result_retains_explicit_portability_limitation(tmp_path, monkeypatch):
    context = fake_context(tmp_path, monkeypatch); plan = c.make_plan(**args())
    original = context.verify_and_prepare
    limitation = {"classification": "portability-limited-v4-approved-exact-command-parity",
                  "strict_policy_parity_passed": False, "exact_commands_passed": True,
                  "numeric_failure_case_indices": [5, 7, 9], "numeric_failure_path_counts": [4, 1, 4],
                  "future_closed_loop_trajectory_equivalence": "unproven"}
    context.verify_and_prepare = lambda *a: {**original(*a), "policy_qualification": limitation}
    monkeypatch.setattr(c, "_run_prepared_case", lambda *a: {"status": "controller-tape-case-passed"})
    result = c.run_prepared_plan(plan, expected_sha256=c.identity(plan)["sha256"], context=context)
    assert result["owner"]["policy_qualification"] == limitation
    assert result["status"] == "controller-tape-local-transport-passed"
    assert result["gameplay_admitted"] is False and result["policy_inference_calls"] == 0
    saved = json.loads((context.research_output_root / plan["label"] / "result.json").read_bytes())
    assert saved["owner"]["policy_qualification"] == limitation


@pytest.mark.parametrize("failure", [None, "connect", "drive", "stop", "replay"])
def test_prepared_case_uses_host_factory_native_run_and_finally_shutdown(tmp_path, monkeypatch, failure):
    context = fake_context(tmp_path, monkeypatch); plan = c.make_plan(**args()); events = []
    directory = context.research_output_root / "case"; directory.mkdir()
    console = SimpleNamespace(_process=object(), _modal_panel_linux_host={"target_platform": "linux_x86_64"})
    console.run = lambda **kw: events.append(("native-run", kw))
    console.connect = lambda: failure != "connect"
    transport = SimpleNamespace(seal_benchmark_audit=lambda: ({"scope": "synthetic"}, receipt()["transport_checks"]))
    def stop(console, seconds):
        events.append(("native-stop", seconds))
        if failure == "stop": raise RuntimeError("synthetic shutdown failure")
        return "child-sigint-input-drain"
    def collect(replay_dir, existing, root, first, last, sudden):
        assert (first, last, sudden) == (-123, 197, False)
        replay = replay_dir / "synthetic.slp"; replay.write_bytes(b"synthetic replay")
        return [replay], [{"tournament_result_replay": True}]
    context.native = SimpleNamespace(
        mimic=SimpleNamespace(_assert_udp_port_available=lambda p: events.append(("udp", p)),
            _disable_attested_dolphin_stop_hotkey=lambda c: events.append(("hotkey",)),
            _ControllerPipeLockstep=SimpleNamespace(install=lambda *a: transport), _stop_console=stop,
            _legacy_replay_gate_checks=lambda *a, **kw: {"no_unexpected_auxiliary_replays": failure != "replay"}),
        frisson=SimpleNamespace(_console_options=c.host.native_options, _collect_replays=collect),
        replay=SimpleNamespace(_game_start_transport_proof=lambda t: {"decision": "pass"},
            _audit_controller_boundary_candidate=lambda *a, **kw: receipt()["controller_audit"]),
        melee=SimpleNamespace(Controller=lambda **kw: SimpleNamespace(connect=lambda: True),
                              ControllerType=SimpleNamespace(STANDARD="standard"), MenuHelper=lambda: object()),
        console_module=object())
    context.package_root = tmp_path / "unused-package"; context.package_manifest = tmp_path / "unused-manifest"
    context.package_audit = tmp_path / "unused-audit"; context.project_root = tmp_path
    context.version_lock = object(); context.game_image_path = tmp_path / "UNOPENED.iso"
    context.verify_inherited_process = lambda p, plan: {"synthetic_owned": True}
    context.read_game_start_rng = lambda p: {"random_seed": 123, "source": "synthetic"}
    def create(*a, **kw):
        events.append(("host-factory", kw["console_options"]))
        return console
    monkeypatch.setattr(c.host, "create_console", create)
    def drive(*a, **kw):
        for i in range(320):
            row = primary_row(); row.update(game_frame=-123 + i, tape_index=i)
            for p, value in c.physical_commands(plan, 0, i).items(): row["slots"][f"p{p}"]["command"] = value
            kw["write_trace"](row)
            if failure == "drive": raise RuntimeError("synthetic loop failure")
        return {"observed_frames": list(range(-123, 198)), "policy_calls": 0}
    monkeypatch.setattr(c, "drive_tape", drive)
    value = c._run_prepared_case(plan, 0, context, directory)
    assert ("native-stop", 30.0) in events
    assert ("native-run", {"iso_path": str(context.game_image_path)}) in events
    assert not context.game_image_path.exists()
    assert value["status"] == ("controller-tape-case-passed" if failure is None else "failed")
    assert (directory / "case.json").is_file()
    if failure != "connect":
        trace = json.loads((directory / "controller_tape_trace.jsonl").read_text().splitlines()[0])
        projected = json.loads((directory / "controller_tape_canonical_schema_audit_projection.jsonl").read_text().splitlines()[0])
        assert trace["slots"]["p1"]["model"] == "controller-tape"
        assert projected["slots"]["p1"]["model_field_role"] == "canonical-command-schema-selector-only"


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(tape_index=1), lambda r: r.update(game_frame=-122),
    lambda r: r["slots"]["p1"].update(port=2), lambda r: r["slots"]["p1"].update(model="frisson-ai"),
    lambda r: r["slots"]["p1"]["command"].update(buttons=["A"]),
    lambda r: r["slots"]["p1"]["inference"].update(called=True),
])
def test_trace_writer_identity_rejects_command_frame_port_or_policy_changes(mutate):
    row = primary_row(); mutate(row)
    with pytest.raises(ValueError): c.validate_tape_row(c.make_plan(**args()), 0, 0, row)
