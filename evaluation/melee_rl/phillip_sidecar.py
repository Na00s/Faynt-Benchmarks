#!/usr/bin/env python3
"""The Phillip sidecar: vladfi1/phillip's own ``Agent`` class, run in its own interpreter (P32, 18 Sep 2026).

`vladfi1/phillip <https://github.com/vladfi1/phillip>`_ (GPL-3.0) is pinned to ``tensorflow==2.13``
(``setup.py``: ``python_requires='<3.12'``), which the recorder's Python 3.12 environment cannot run.  So
the agent runs *here*, under ``/opt/phillip/venv/bin/python`` (3.11 + ``tensorflow-cpu``), and the
recorder (:mod:`melee_rl.phillip`) drives it over a line protocol: one JSON object per line on stdin,
one JSON object per line on stdout.  Every other print -- upstream is chatty (``Using device``,
``Restoring from``, the restore report) -- is diverted to stderr, which the recorder keeps a tail of.

This file is launched **by path** by an interpreter that has neither torch nor ``melee_rl``, so it
imports nothing from this repository and nothing beyond the standard library at module level.  Upstream
is reached through four public names only -- ``phillip.agent.Agent``, ``phillip.ssbm.GameMemory``,
``phillip.ssbm.RealControllerState`` (through the pad) and ``phillip.util.load_params`` -- and the
agent's own ``act(state, pad)`` does everything else exactly as ``phillip/cpu.py`` would call it: the
``act_every`` action chain, the decision-level delay queue, the history window, the recurrent state,
the per-character banned moves.  Nothing of the package is copied.

The protocol (``PROTOCOL_VERSION``):

* ``{"op": "hello"}`` -> the interpreter's and TensorFlow's versions and where ``phillip`` imports from.
* ``{"op": "load", "agent": "delay0/FoxFD", "envs": N, "epsilon": e, "seed": s, "swap": 0|1,
  "save_cpu": bool, "source_dir": dir}`` -> one ``Agent`` per environment, restored from
  ``<source_dir>/agents/<agent>/snapshot``; the reply carries upstream's config (``act_every``,
  ``delay``, ``memory``) and the **restore report** -- the ``... not in checkpoint, initializing`` and
  ``... padded from ...`` lines upstream's ``tfl.restore`` prints when a checkpoint does not match the
  graph.  The recorder refuses an agent with either: it would be playing random weights.
* ``{"op": "act", "frames": [{"env": i, "state": {...}}, ...]}`` -> one controller per listed
  environment: the buttons upstream's ``SimpleController.send`` pressed and the sticks it set, with the
  two-decimal rounding of ``Pad.tilt_stick`` (``SET MAIN {:.2f} {:.2f}``) the game upstream trained
  against saw.  A frame that raises is reported *per environment* (``"error"``) so the others play on.
* ``{"op": "reset", "envs": [i, ...]}`` -> those agents rebuilt (a fresh chain, queue, history and
  recurrent state; the old TensorFlow session closed).
* ``{"op": "stats"}`` -> per-environment frames, decisions, sends and an action histogram.
* ``{"op": "quit"}``.
"""

from __future__ import annotations

import os
import sys

# Launched by path, so Python puts this directory -- ``melee_rl/``, with its own ``phillip.py`` and
# ``logging.py`` -- first on ``sys.path``; drop it, so ``import phillip`` finds upstream on PYTHONPATH
# (the launcher also sets PYTHONSAFEPATH=1, which does the same on 3.11+; this is the belt).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != _HERE]

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from collections.abc import Callable, Mapping  # noqa: E402
from typing import Any  # noqa: E402

PROTOCOL_VERSION = "melee_rl.phillip_sidecar.v1"
BUTTONS = ("A", "B", "X", "Y", "Z", "L", "R", "START")
"""The digital buttons of upstream's ``RealControllerState`` (``ssbm.py:42-49``), in its field order."""
PLAYER_FIELDS = (
    "percent",
    "stock",
    "facing",
    "x",
    "y",
    "z",
    "action_state",
    "action_counter",
    "action_frame",
    "character",
    "invulnerable",
    "hitlag_frames_left",
    "hitstun_frames_left",
    "jumps_used",
    "charging_smash",
    "in_air",
    "speed_air_x_self",
    "speed_ground_x_self",
    "speed_y_self",
    "speed_x_attack",
    "speed_y_attack",
    "shield_size",
    "cursor_x",
    "cursor_y",
)
"""The ``PlayerMemory`` fields the recorder fills (``ssbm.py:79-111``, all but the controller)."""
GAME_MENU = 2
"""Upstream's ``state.Menu.Game``."""


def _pipe(value: float) -> float:
    """What ``Pad.tilt_stick`` writes to Dolphin's pipe: ``'{:.2f}'`` of the value."""

    return float(f"{float(value):.2f}")


class RecordingPad:
    """The ``pad`` of ``Agent.act``: keeps the last controller upstream sent, as Dolphin's pipe reader does.

    ``SimpleController.send`` calls ``send_controller`` with a whole ``RealControllerState``;
    ``RepeatController.send`` (the ``custom`` action set's last entry) calls nothing, so the previous
    state stays in force -- which is what a durable pad gives for free.
    """

    def __init__(self) -> None:
        self.buttons: dict[str, bool] = dict.fromkeys(BUTTONS, False)
        self.main = (0.5, 0.5)
        self.c = (0.5, 0.5)
        self.sends = 0

    def send_controller(self, controller: Any) -> None:
        for name in BUTTONS:
            self.buttons[name] = bool(getattr(controller, "button_" + name, False))
        self.main = (_pipe(controller.stick_MAIN.x), _pipe(controller.stick_MAIN.y))
        self.c = (_pipe(controller.stick_C.x), _pipe(controller.stick_C.y))
        self.sends += 1

    def press_button(self, button: Any, buffering: bool = False) -> None:
        name = getattr(button, "name", str(button))
        if name in self.buttons:
            self.buttons[name] = True

    def release_button(self, button: Any, buffering: bool = False) -> None:
        name = getattr(button, "name", str(button))
        if name in self.buttons:
            self.buttons[name] = False

    def tilt_stick(self, stick: Any, x: float, y: float, buffering: bool = False) -> None:
        name = getattr(stick, "name", str(stick))
        if name == "MAIN":
            self.main = (_pipe(x), _pipe(y))
        elif name == "C":
            self.c = (_pipe(x), _pipe(y))

    def flush(self) -> None:
        return None

    def state(self) -> dict[str, Any]:
        return {"buttons": dict(self.buttons), "main": list(self.main), "c": list(self.c)}


class Slot:
    """One environment: its agent, its pad, its reusable state struct and its counters."""

    def __init__(self, index: int, agent: Any, state: Any) -> None:
        self.index = index
        self.agent = agent
        self.pad = RecordingPad()
        self.state = state
        self.frames = 0
        self.decisions = 0
        self.actions: Counter[str] = Counter()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "decisions": self.decisions,
            "sends": self.pad.sends,
            "actions": dict(self.actions),
        }


def _fill_state(state: Any, payload: Mapping[str, Any]) -> None:
    """Assign the recorder's message onto upstream's ctypes ``GameMemory``."""

    state.frame = int(payload.get("frame", 0))
    state.menu = int(payload.get("menu", GAME_MENU))
    state.stage = int(payload.get("stage", 0))
    for index, fields in enumerate(payload["players"][:2]):
        player = state.players[index]
        for name in PLAYER_FIELDS:
            if name in fields:
                setattr(player, name, fields[name])


def _close_agent(agent: Any) -> None:
    session = getattr(getattr(agent, "actor", None), "sess", None)
    close = getattr(session, "close", None)
    if callable(close):
        with contextlib.suppress(Exception):
            close()


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


class Sidecar:
    def __init__(self) -> None:
        self.slots: list[Slot] = []
        self.loaded: dict[str, Any] | None = None
        self._agent_kwargs: dict[str, Any] | None = None
        self._upstream: tuple[Any, Any, Any] | None = None
        self._tensorflow: str | None = None

    # -- upstream --------------------------------------------------------

    def _import_upstream(self) -> tuple[Any, Any, Any]:
        """``(phillip.agent, phillip.ssbm, phillip.util)``, importing TensorFlow in graph mode first.

        ``phillip/run.py`` calls ``tf.disable_eager_execution()`` before anything builds a graph; the
        agent's ``Actor`` builds one in its constructor, so doing it here, once, is equivalent.
        """

        if self._upstream is None:
            try:
                import tensorflow as tf

                self._tensorflow = str(tf.__version__)
                tf.compat.v1.disable_eager_execution()
            except ImportError:
                self._tensorflow = None  # the stub package of the tests carries no TensorFlow
            from phillip import agent as agent_module
            from phillip import ssbm, util

            self._upstream = (agent_module, ssbm, util)
        return self._upstream

    def _build_agent(self) -> tuple[Any, dict[str, Any]]:
        """One ``Agent(**kwargs)`` with its restore report parsed out of what the constructor printed."""

        agent_module, _, _ = self._import_upstream()
        assert self._agent_kwargs is not None
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            agent = agent_module.Agent(**dict(self._agent_kwargs))
        text = buffer.getvalue()
        sys.stderr.write(text)
        sys.stderr.flush()
        lines = text.splitlines()
        missing = [
            line.split(" not in checkpoint")[0].strip() for line in lines if "not in checkpoint" in line
        ]
        padded = [line.strip() for line in lines if "padded from" in line]
        return agent, {"missing": missing, "padded": padded, "lines": lines}

    # -- the operations --------------------------------------------------

    def hello(self, request: Mapping[str, Any]) -> dict[str, Any]:
        phillip_dir: str | None = None
        try:
            import phillip

            phillip_dir = os.path.dirname(os.path.abspath(phillip.__file__))
        except ImportError:
            phillip_dir = None
        return {
            "protocol": PROTOCOL_VERSION,
            "python": platform.python_version(),
            "executable": sys.executable,
            "tensorflow": _version("tensorflow"),
            "numpy": _version("numpy"),
            "phillip": phillip_dir,
        }

    def load(self, request: Mapping[str, Any]) -> dict[str, Any]:
        _, ssbm, util = self._import_upstream()
        name = str(request["agent"])
        envs = int(request["envs"])
        source_dir = str(request["source_dir"])
        path = os.path.join(source_dir, "agents", name)
        params = util.load_params(path, "agent")
        params.update(
            reload=0,  # never re-read the snapshot mid-game (the play recipe's --reload 0)
            epsilon=float(request.get("epsilon", 0.0)),  # the README plays with --epsilon 0
            swap=int(request.get("swap", 0)),  # 1: upstream reads itself at player index 0
            save_cpu=int(bool(request.get("save_cpu", True))),  # one thread per session
            gpu=False,
            verbose=False,
        )
        try:
            import numpy as np

            np.random.seed(int(request.get("seed", 0)))
        except ImportError:
            pass
        self._agent_kwargs = params
        for slot in self.slots:
            _close_agent(slot.agent)
        self.slots = []
        started = time.perf_counter()
        missing: list[str] = []
        padded: list[str] = []
        lines: list[str] = []
        for index in range(envs):
            agent, report = self._build_agent()
            missing += report["missing"]
            padded += report["padded"]
            if index == 0:
                lines = report["lines"]
            self.slots.append(Slot(index, agent, ssbm.GameMemory()))
        config = self.slots[0].agent.actor.config
        self.loaded = {
            "agent": name,
            "path": path,
            "envs": envs,
            "swap": params["swap"],
            "epsilon": params["epsilon"],
            "config": {
                "char": params.get("char"),
                "act_every": int(config.act_every),
                "delay": int(config.delay),
                "memory": int(config.memory),
                "fps": int(config.fps),
            },
            "restore": {"missing": sorted(set(missing)), "padded": sorted(set(padded)), "lines": lines[-40:]},
            "seconds": time.perf_counter() - started,
            "tensorflow": self._tensorflow,
        }
        return dict(self.loaded)

    def act(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self.slots:
            raise RuntimeError("act before load")
        controllers: list[dict[str, Any]] = []
        decisions = 0
        for frame in request["frames"]:
            slot = self.slots[int(frame["env"])]
            try:
                _fill_state(slot.state, frame["state"])
                before = slot.agent.action_chain
                slot.agent.act(slot.state, slot.pad)
            except Exception as error:
                controllers.append(
                    {
                        "env": slot.index,
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                )
                continue
            slot.frames += 1
            after = slot.agent.action_chain
            if after is not before:  # ``Agent.act`` makes a new ActionChain on every decision frame
                slot.decisions += 1
                decisions += 1
                slot.actions[str(int(slot.agent.action))] += 1
            controllers.append({"env": slot.index, **slot.pad.state()})
        return {"controllers": controllers, "decisions": decisions}

    def reset(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if not self.slots:
            raise RuntimeError("reset before load")
        _, ssbm, _ = self._import_upstream()
        rebuilt: list[int] = []
        for raw in request.get("envs", []):
            index = int(raw)
            _close_agent(self.slots[index].agent)
            agent, _ = self._build_agent()
            self.slots[index] = Slot(index, agent, ssbm.GameMemory())
            rebuilt.append(index)
        return {"rebuilt": rebuilt}

    def stats(self, request: Mapping[str, Any]) -> dict[str, Any]:
        loaded = {} if self.loaded is None else {k: self.loaded[k] for k in ("agent", "envs", "config")}
        return {"envs": [slot.to_dict() for slot in self.slots], "loaded": loaded}


def main() -> int:
    # The protocol owns fd 1; everything Python-level that upstream prints goes to stderr instead.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    sys.stdout = sys.stderr

    def write(response: Mapping[str, Any]) -> None:
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()

    sidecar = Sidecar()
    handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
        "hello": sidecar.hello,
        "load": sidecar.load,
        "act": sidecar.act,
        "reset": sidecar.reset,
        "stats": sidecar.stats,
    }
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        op = "?"
        try:
            request = json.loads(line)
            op = str(request.get("op"))
            if op == "quit":
                write({"ok": True, "op": op})
                break
            if op not in handlers:
                raise KeyError(f"unknown op {op!r}")
            response = handlers[op](request)
            response.update(ok=True, op=op)
        except Exception as error:
            response = {
                "ok": False,
                "op": op,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
        write(response)
    for slot in sidecar.slots:
        _close_agent(slot.agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
