"""An ``Agent`` with upstream's construction and ``act`` contract, scripted by its ``params`` file.

Construction prints what upstream's ``tfl.restore`` prints -- ``Restoring from <path>`` and, when the
params ask for it (``"stub_missing": [...]``, ``"stub_padded": [...]``), the ``... not in checkpoint,
initializing`` / ``... padded from ...`` lines the sidecar must report -- so the load report is tested
against the real text.  ``act(state, pad)`` runs an action chain of ``act_every`` frames: on a decision
frame it looks at its own player (index 1, or 0 when ``swap``) and presses ``A`` when that player is on
the right half of the stage, else tilts the stick left; between decisions it re-sends the same
controller, as upstream's ``ActionChain`` does.  ``"stub_raise_at"`` makes ``act`` raise on that frame,
and ``"stub_sleep_s"`` makes the constructor sleep (the timeout test).
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

from . import ssbm


class _Config:
    def __init__(self, act_every: int, delay: int, memory: int) -> None:
        self.act_every = act_every
        self.delay = delay
        self.memory = memory
        self.fps = 60 // act_every


class _Session:
    closed = False

    def close(self) -> None:
        self.closed = True


class _Actor:
    def __init__(self, params: dict[str, Any]) -> None:
        self.config = _Config(
            int(params.get("act_every", 3)), int(params.get("delay", 0)), int(params.get("memory", 0))
        )
        self.path = str(params["path"])
        self.sess = _Session()


class _Chain:
    def __init__(self, controller: ssbm.RealControllerState, length: int) -> None:
        self.controller = controller
        self.length = length
        self.index = 0

    def act(self, pad: Any) -> None:
        pad.send_controller(self.controller)
        self.index += 1

    def done(self) -> bool:
        return self.index >= self.length


class Agent:
    built: ClassVar[list[Agent]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.char = kwargs.get("char")
        self.epsilon = kwargs.get("epsilon")
        self.swap = int(kwargs.get("swap") or 0)
        self.pid = 0 if self.swap else 1
        self.reload = kwargs.get("reload")
        self.actor = _Actor(kwargs)
        self.frame_counter = 0
        self.action_chain: _Chain | None = None
        self.action = 0
        if kwargs.get("stub_sleep_s"):
            time.sleep(float(kwargs["stub_sleep_s"]))
        print("Restoring from", self.actor.path + "/snapshot")
        for name in kwargs.get("stub_missing", ()):
            print(f"{name} not in checkpoint, initializing")
        for name in kwargs.get("stub_padded", ()):
            print(f"Variable {name} of shape [4, 4] padded from (4, 2)")
        Agent.built.append(self)

    def act(self, state: ssbm.GameMemory, pad: Any) -> None:
        self.frame_counter += 1
        raise_at = self.kwargs.get("stub_raise_at")
        if raise_at is not None and self.frame_counter == int(raise_at):
            raise RuntimeError(f"the stub agent was told to fail on frame {self.frame_counter}")
        if self.action_chain is not None and not self.action_chain.done():
            self.action_chain.act(pad)
            return
        me = state.players[self.pid]
        controller = ssbm.RealControllerState()
        if me.x > 0:
            controller.button_A = True
            self.action = 1
        else:
            stick = self.kwargs.get("stub_stick")
            if stick:
                controller.stick_MAIN.x = float(stick[0])
                controller.stick_MAIN.y = float(stick[1])
            else:
                controller.stick_MAIN.x = 0.0
            self.action = 2
        self.action_chain = _Chain(controller, self.actor.config.act_every)
        self.action_chain.act(pad)
