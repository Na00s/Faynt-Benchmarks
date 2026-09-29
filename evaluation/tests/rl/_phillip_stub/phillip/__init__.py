"""A stand-in for the ``phillip`` package, for driving the real sidecar script without TensorFlow.

The sidecar (``melee_rl/phillip_sidecar.py``) reaches upstream through four names only --
``phillip.agent.Agent``, ``phillip.ssbm.GameMemory``, ``phillip.ssbm.RealControllerState`` and
``phillip.util.load_params`` -- and this package provides exactly those with the same shapes: the
ctypes game state, a controller struct, an ``Agent`` that decides every ``act_every`` frames and writes
the pad the way upstream's ``SimpleController.send`` does.  Put ``tests/rl/_phillip_stub`` on the
sidecar's ``PYTHONPATH`` (``PhillipSidecar(source_dir=...)`` does) and the whole line protocol runs
end to end under the test interpreter.
"""
