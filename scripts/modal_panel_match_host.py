#!/usr/bin/env python3
"""Five process-local Linux host bindings for the unchanged native match loop.

No CLI or default execution context is provided. A reviewed parent must first
admit and prepare the owned Linux context, then install these bindings before
slippi_panel_runtime.configure_game. Policy and frame-loop functions stay intact.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import types

import modal_panel_linux_host as host
import modal_panel_live_context as context

SCHEMA = "e011.modal-native-match-host.v1"
SOURCE_HASHES = {'src/melee_policy/integration/frisson_slippi_match.py': '2fb00d66886e22f6869f35fa2598298afd638106a98bbbd33ea2c639a762b7b0', 'src/melee_policy/integration/frisson_match.py': '4354dddb6bb50f285aaffb517024172a136ac4971e1896f43d93439eb87b2fc5', 'src/melee_policy/integration/match_runtime.py': 'ee1e2e121877003f37b91a0c737befdcb2cc48129acf621c0705a00c05c10874', 'src/melee_policy/integration/runtime_identity.py': '388bd46e041df6a5ec80ec3d96eedcb22abb2e9e0b56fa1281cff4ec818e17a9', 'scripts/slippi_panel_runtime.py': 'ecff60bf292fcec36e2376fd5333836ec556bb1bb07fd656d416c450f17240fc', 'scripts/modal_panel_live_context.py': '8304bcb4ec2031ae6efa8edb253821aa2d876549f4057d49d30c6d895ac1616b', 'scripts/modal_panel_linux_host.py': 'c9b7e4c4189e26fdbd74667c9b3ffdf7d46f2311a8b294e526922db442b9ddb9'}
HOOKS = ("_emulator_application_identity", "_attested_emulator_release", "_create_attested_dolphin_console",
         "_launch_and_connect_attested_dolphin", "_runtime_reproducibility_record")
PROTECTED = ("_run_console", "_run_exact_frame", "_stop_console", "_console_options", "_load_config",
             "_ControllerPipeLockstep", "send_canonical_controller")


def code_tree(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, types.CodeType): yield from code_tree(value)


def verify_sources(root):
    result = {}
    for relative, sha in SOURCE_HASHES.items():
        body, row = context.read_bound(root / relative)
        if row["sha256"] != sha: raise ValueError(f"native host source changed: {relative}")
        result[relative] = row
    return result


def verify_function(function, root, relative):
    source = root / relative
    if (not isinstance(function, types.FunctionType)
            or Path(function.__code__.co_filename).resolve() != source
            or function.__code__ not in tuple(code_tree(compile(source.read_bytes(), str(source), "exec", dont_inherit=True)))):
        raise ValueError("native function differs from frozen source")


class MatchHostAdapter:
    def __init__(self, *, match, owned_context, preparation, execution_plan, linux_pins_path, runtime_identity):
        self.match, self.ctx = match, owned_context
        self.preparation, self.plan = copy.deepcopy(preparation), copy.deepcopy(execution_plan)
        self.root = Path(owned_context.project_root)
        self.pins = Path(linux_pins_path)
        self.runtime_identity = runtime_identity
        self.originals, self.protected, self.hooks = {}, {}, {}
        self.sources = None
        self.console = self.application = self.composed = None
        self.launched = False
        self.launch_evidence = None

    def _owned(self):
        self.ctx.check_deadline(self.plan["expires_at"])
        if (platform.system() != "Linux" or platform.machine() != "x86_64"
                or os.getpid() != os.getpgid(0) or os.getpid() != os.getsid(0)
                or self.preparation.get("status") != "owned-linux-tape-context-ready"
                or self.preparation.get("plan_sha256") != hashlib.sha256(
                    (json.dumps(self.plan, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()).hexdigest()
                or self.preparation.get("process", {}).get("pid") != os.getpid()
                or self.preparation.get("parent_declaration") != self.ctx.owner
                or self.preparation.get("policy_qualification") != self.ctx.policy_qualification
                or self.preparation.get("display", {}).get("software_renderer_verified") is not True
                or self.preparation.get("runtime", {}).get("pip_check_passed") is not True):
            raise RuntimeError("prepared admitted Linux child context required")

    def _intact(self, *, require_owned=True):
        if require_owned: self._owned()
        if not self.hooks: raise RuntimeError("host adapter is not installed")
        if any(getattr(self.match, name) is not value for name, value in self.protected.items()):
            raise RuntimeError("native policy loop, transport or options changed")
        if (self.composed is None and self.match._runtime_reproducibility_record
                is not self.hooks["_runtime_reproducibility_record"]):
            # run_game performs configure_game internally. Its exact native
            # wrapper is adopted once, without replacing that entrypoint.
            self.after_panel_configuration()
        for name, value in self.hooks.items():
            expected = self.composed if name == "_runtime_reproducibility_record" and self.composed else value
            if getattr(self.match, name) is not expected: raise RuntimeError("host binding changed")

    def install(self):
        if self.hooks: raise RuntimeError("host adapter already installed")
        self._owned()
        self.sources = verify_sources(self.root)
        native = self.ctx.native.mimic
        for name in HOOKS:
            function = getattr(self.match, name)
            if function is not getattr(native, name): raise ValueError("install before panel configuration")
            verify_function(function, self.root, "src/melee_policy/integration/match_runtime.py")
            self.originals[name] = function
        for name in ("_run_console", "_run_exact_frame"):
            verify_function(getattr(self.match, name), self.root, "src/melee_policy/integration/frisson_slippi_match.py")
        verify_function(self.match._console_options, self.root, "src/melee_policy/integration/frisson_match.py")
        for name in ("_stop_console", "_load_config", "_ControllerPipeLockstep"):
            if getattr(self.match, name) is not getattr(native, name): raise ValueError("native shared host alias changed")
        if self.match.send_canonical_controller is not self.ctx.native.dispatch.send_canonical_controller:
            raise ValueError("native decoder sender alias changed")
        verify_function(native._disable_attested_dolphin_stop_hotkey, self.root, "src/melee_policy/integration/match_runtime.py")
        verify_function(self.runtime_identity.runtime_environment_record, self.root, "src/melee_policy/integration/runtime_identity.py")
        self.protected = {name: getattr(self.match, name) for name in PROTECTED}
        _, pins = context.read_bound(self.pins)
        if pins["sha256"] != context.PINS_SHA: raise ValueError("exact Linux94 runtime pins required")
        self.hooks = dict(zip(HOOKS, (self.application_identity, self.release, self.create_console,
                                     self.launch_and_connect, self.reproducibility), strict=True))
        for name, value in self.hooks.items(): setattr(self.match, name, value)
        return {"schema": SCHEMA, "source_identities": self.sources, "hooks": list(HOOKS),
                "policy_loop_changes": False, "parent_terminal_cleanup_verified": False,
                "policy_qualification": copy.deepcopy(self.ctx.policy_qualification)}

    def after_panel_configuration(self):
        """Allow only the existing source-verified reproducibility composition."""
        wrapper = self.match._runtime_reproducibility_record
        verify_function(wrapper, self.root, "scripts/slippi_panel_runtime.py")
        captures = tuple(cell.cell_contents for cell in wrapper.__closure__ or ())
        if not any(value is self.hooks["_runtime_reproducibility_record"] for value in captures):
            raise ValueError("panel wrapper did not capture the installed host binding")
        self.composed = wrapper
        self._intact()

    def restore(self):
        self._intact(require_owned=False)
        for name, value in self.originals.items(): setattr(self.match, name, value)
        self.hooks = {}

    def _root(self, root):
        if Path(root).resolve(strict=True) != self.root: raise ValueError("another project root requested")

    def application_identity(self, config, project_root):
        self._intact(); self._root(project_root)
        identity = host.verify_installed(self.ctx.package_root, self.ctx.package_manifest, self.ctx.package_audit)
        self.application = {**identity, "host_contract": SCHEMA,
                            "original_control_emulator": copy.deepcopy(config["emulator"])}
        return copy.deepcopy(self.application)

    def release(self, application_identity):
        self._intact()
        if self.application is None or application_identity != self.application: raise ValueError("Linux application identity differs")
        return host.RELEASE

    def create_console(self, config, project_root, application_identity, **options):
        self._intact(); self._root(project_root); self.release(application_identity)
        if self.console is not None: raise RuntimeError("one Console per owned match")
        if config["emulator"] != self.application["original_control_emulator"]: raise ValueError("control emulator settings changed")
        if (options["slippi_port"] != config["emulator"]["slippi_port"]
                or options["slippi_port"] not in {case["slippi_port"] for case in self.plan["cases"]}):
            raise ValueError("admitted physical UDP port differs")
        self.console = host.create_console(self.ctx.package_root, self.ctx.package_manifest, self.ctx.package_audit,
            replay_directory=Path(options["replay_dir"]), slippi_port=options["slippi_port"],
            console_module=self.ctx.native.console_module, version_lock=self.ctx.native.mimic._LIBMELEE_VERSION_ATTESTATION_LOCK,
            console_options=options)
        return self.console

    def launch_and_connect(self, console, iso_path):
        self._intact()
        if console is not self.console or self.launched or getattr(console, "_process", None) is not None:
            raise RuntimeError("one launch of this owned Console required")
        if Path(iso_path).resolve(strict=True) != self.ctx.game_image_path: raise ValueError("admitted game image path differs")
        self.launched = True
        host.verify_installed(self.ctx.package_root, self.ctx.package_manifest, self.ctx.package_audit)
        self.ctx.native.mimic._disable_attested_dolphin_stop_hotkey(console)
        console.run(iso_path=str(self.ctx.game_image_path))
        self.launch_evidence = self.ctx.verify_inherited_process(console._process, self.plan)
        return bool(console.connect())

    def reproducibility(self, project_root, config_path, dependency_lock, implementation_paths):
        self._intact(); self._root(project_root)
        native = self.ctx.native.mimic
        environment = self.runtime_identity.runtime_environment_record(self.root, self.pins)
        if environment["dependency_lock_validation"]["environment_lock_gate_passed"] is not True:
            raise ValueError("actual Linux runtime does not satisfy frozen94 pins")
        return {"configuration": native._file_identity(Path(config_path), self.root),
                "dependency_lock": native._file_identity(self.pins, self.root),
                "original_control_dependency_lock": native._file_identity(self.root / dependency_lock, self.root),
                "implementation_files": [native._file_identity(self.root / p, self.root) for p in implementation_paths],
                "runtime_environment": environment,
                "linux_host": {"schema": SCHEMA, "source_identities": self.sources,
                    "preparation": copy.deepcopy(self.preparation), "package": host.verify_installed(
                        self.ctx.package_root, self.ctx.package_manifest, self.ctx.package_audit),
                    "policy_qualification": copy.deepcopy(self.ctx.policy_qualification),
                    "parent_terminal_cleanup_verified": False, "hook_scope": list(HOOKS)}}
