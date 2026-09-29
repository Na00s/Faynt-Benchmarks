"""``WorkerDolphinEnv``: the Dolphins of a ``DolphinEnv`` split over OS worker processes (P7-PLAN.md 2).

Semantics follow slippi-ai's ``AsyncEnvMP`` / ``AsyncBatchedEnvironmentMP``
(``slippi_ai/envs.py:304-522``): everything process-bound is constructed *inside* the worker from
picklable inputs, only nests of small numpy arrays cross the pipes, an error crosses as text (never a
live exception object), and shutdown is a ``None`` sentinel followed by a drain-while-alive join.  No
code is copied.

* Each of ``worker_processes`` children runs a full in-process :class:`~melee_rl.env.dolphin.DolphinEnv`
  over its contiguous slice of ``num_envs // worker_processes`` Dolphins (:func:`worker_configs`:
  ``num_envs`` and ``slippi_port`` sliced, ``worker_processes = 0``).  The existing two-phase boot,
  fault ladder, per-group step threads and ``stats()`` all run unchanged inside the worker -- and
  libmelee's fork of its enet client happens on the worker's *main* thread by construction
  (``dolphin.py``'s rule).  Frame conversion (the GIL-bound ~0.9 ms per env frame) runs in the worker.
* Start method: ``spawn`` (default) or ``forkserver`` -- never plain ``fork``; the trainer process
  holds CUDA and thread pools.  Workers are ``daemon = False`` (a daemonic process could not start
  libmelee's enet child), import torch as a side effect of the module tree, and immediately hide CUDA
  (``CUDA_VISIBLE_DEVICES = ""``) and pin themselves to one torch thread.  ``forkserver`` preloads
  this module, so the import cost is paid once by the fork server, not per worker.
* Wire protocol (versioned tag tuples; protocol 2 added the time-chunked P8 extension, protocol 3
  the ``timing`` request of lever 0, 5 Sep 2026):

  - parent -> worker: ``("step", {env_port: {13 numpy arrays [G]}})`` (:func:`controller_arrays`
    order: sticks + shoulder float32, then ``BUTTON_ORDER`` bools), ``("step_chunk", [k such
    payloads])`` (P8: the worker steps the k frames back to back and replies once),
    ``("reset", None)``, ``("stats", None)``, ``("timing", None)``, and ``None`` = shutdown.
  - worker -> parent: ``("ready", (leaves, frame_index, version))`` once after boot, then
    ``("state", (leaves, frame_index))`` per step / reset -- ``leaves`` is the worker's
    ``FrameBatch.leaves`` dict (78 arrays ``[G]`` / ``[G, 15]``) -- ``("states", [k x (leaves,
    frame_index)])`` per chunk, ``("stats", rows)``, ``("timing", {"worker": counters, "env":
    counters})`` (:mod:`melee_rl.env.timing`: the worker's idle / unpack / step / reply time and its
    ``DolphinEnv.timing()``), or ``("error", (type name, message, traceback))`` followed by ``None``
    before the worker exits.
  - The parent receives the step replies in arrival order (``multiprocessing.connection.wait``)
    and records, per step, the time until the first and the last reply and each worker's arrival
    (``WorkerDolphinEnv.timing()``): the lockstep tail across workers is ``wait_last - wait_first``.

* The asynchronous surface (P8, ``melee_rl.env.async_env.AsyncEnvProtocol``):
  :meth:`WorkerDolphinEnv.push` buffers ``chunk_frames`` action sets and sends them as one
  ``step_chunk`` per worker, so the Dolphins emulate the whole chunk while the trainer computes;
  :meth:`WorkerDolphinEnv.pop` / ``peek`` receive and un-transpose the ``states`` replies in FIFO
  order.  The boot / reset state counts as in transit (``in_flight`` starts at 1, slippi-ai's
  ``_num_in_transit``); ``pop`` with nothing owed raises instead of blocking.  The synchronous
  ``step()`` stays byte-identical on the wire and refuses to run while pipeline state exists.

* Faults keep today's user-visible semantics: recoverable faults relaunch inside the worker (the
  ``-123`` row simply appears in the next state); ``DolphinLaunchError`` -- or a dead worker -- raises
  :class:`EnvWorkerError` in the parent and ends the run, exactly as the in-process env fails today.
* :func:`build_dolphin_env` is the selection point: ``worker_processes == 0`` returns the in-process
  :class:`DolphinEnv` unchanged.  Tests inject scripted backends through ``backend_factory`` -- a
  module-level callable (picklable by reference) called as ``factory(worker_config, worker_index)``
  inside the child.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing
import os
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from multiprocessing.connection import Connection
from multiprocessing.connection import wait as wait_connections
from multiprocessing.process import BaseProcess
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch

from controller_codec import ControllerState
from melee_rl.env.dolphin import (
    ControllerRow,
    DolphinBackend,
    DolphinEnv,
    DolphinEnvConfig,
    build_env_output,
)
from melee_rl.env.libmelee_frames import FrameBatch
from melee_rl.env.protocol import EnvOutput
from melee_rl.env.shm import ExchangeSpec, FrameExchange
from melee_rl.env.timing import new_parent_timing, new_worker_timing
from tensor_batch import BUTTON_ORDER

LOG = logging.getLogger("melee_rl.env.dolphin_mp")

# 2 (P8): step_chunk / states; 3 (lever 0, 5 Sep 2026): the timing request. The boot handshake
# refuses a mismatch.
_PROTOCOL_VERSION: Final[int] = 3
_PRELOAD_MODULE: Final[str] = "melee_rl.env.dolphin_mp"
_ANALOG_KEYS: Final[tuple[str, ...]] = ("main_x", "main_y", "c_x", "c_y", "shoulder")

ControllerArrays = dict[str, npt.NDArray[Any]]
BackendFactory = Callable[[DolphinEnvConfig, int], DolphinBackend]


class EnvWorkerError(RuntimeError):
    """A worker process failed (its error crossed as text) or died without reporting."""


# ---------------------------------------------------------------------------
# wire codec: ControllerState [B] -> numpy arrays -> per-worker rows
# ---------------------------------------------------------------------------


def controller_arrays(control: ControllerState, batch: int) -> ControllerArrays:
    """A ``[B]`` ``ControllerState`` as 13 numpy views (sticks + shoulder float32, buttons bool)."""

    if tuple(control.shoulder.shape) != (batch,):
        raise ValueError(f"controller must have batch shape ({batch},), got {tuple(control.shoulder.shape)}")
    arrays: ControllerArrays = {
        "main_x": control.main_stick.x.detach().numpy(),
        "main_y": control.main_stick.y.detach().numpy(),
        "c_x": control.c_stick.x.detach().numpy(),
        "c_y": control.c_stick.y.detach().numpy(),
        "shoulder": control.shoulder.detach().numpy(),
    }
    for name in BUTTON_ORDER:
        arrays[name] = getattr(control.buttons, name).detach().numpy()
    return arrays


def slice_arrays(arrays: ControllerArrays, start: int, stop: int) -> ControllerArrays:
    """One worker's contiguous slice of :func:`controller_arrays` (views; pickling copies them)."""

    return {name: array[start:stop] for name, array in arrays.items()}


def rows_from_arrays(arrays: ControllerArrays, batch: int) -> list[ControllerRow]:
    """Per-environment ``ControllerRow`` objects from a worker's array slice (worker-side)."""

    if tuple(arrays["shoulder"].shape) != (batch,):
        raise ValueError(
            f"controller arrays must have batch shape ({batch},), got {tuple(arrays['shoulder'].shape)}"
        )
    buttons = [arrays[name] for name in BUTTON_ORDER]
    return [
        ControllerRow(
            main_x=float(arrays["main_x"][i]),
            main_y=float(arrays["main_y"][i]),
            c_x=float(arrays["c_x"][i]),
            c_y=float(arrays["c_y"][i]),
            shoulder=float(arrays["shoulder"][i]),
            buttons=tuple(bool(column[i]) for column in buttons),
        )
        for i in range(batch)
    ]


# ---------------------------------------------------------------------------
# slicing and assembly
# ---------------------------------------------------------------------------


def worker_configs(config: DolphinEnvConfig) -> list[DolphinEnvConfig]:
    """One in-process config per worker: the env / Slippi-port slice, ``worker_processes = 0``.

    Worker ``w``'s instance ``j`` probes from ``slippi_port + w * G + j`` -- the same candidate its
    global index used in the in-process env -- and carries ``env_index_offset = w * G`` so that
    per-environment replay directories keep the global numbering (P9).
    """

    if config.worker_processes < 1:
        raise ValueError("worker_configs needs worker_processes >= 1 (0 is the in-process mode)")
    group = config.envs_per_worker
    return [
        replace(
            config,
            num_envs=group,
            slippi_port=config.slippi_port + w * group,
            worker_processes=0,
            shared_memory=False,  # the exchange spec travels separately; the slice env steps in-process
            env_index_offset=config.env_index_offset + w * group,
        )
        for w in range(config.worker_processes)
    ]


def assemble_states(
    states: Sequence[tuple[Mapping[str, npt.NDArray[Any]], Sequence[int]]],
    controlled_ports: tuple[int, ...],
) -> EnvOutput:
    """Concatenate per-worker ``(leaves, frame_index)`` payloads into one ``EnvOutput``."""

    if not states:
        raise ValueError("assemble_states needs at least one worker state")
    first = states[0][0]
    leaves = {path: np.concatenate([leaves[path] for leaves, _ in states]) for path in first}
    frame_index = [int(index) for _, indices in states for index in indices]
    return build_env_output(FrameBatch.from_leaves(leaves), frame_index, controlled_ports)


# ---------------------------------------------------------------------------
# the worker process
# ---------------------------------------------------------------------------


def _worker_main(
    conn: Connection,
    config: DolphinEnvConfig,
    cpu_level: int,
    worker_index: int,
    backend_factory: BackendFactory | None,
    exchange_spec: ExchangeSpec | None = None,
) -> None:
    """Serve one worker's Dolphins over ``conn`` until the ``None`` sentinel (or the pipe dies).

    Any failure -- boot or mid-loop -- is reported as ``("error", (type, message, traceback))``
    followed by ``None``; the Dolphins are always stopped on the way out.  With ``exchange_spec`` the
    ``("step", ("shm", slot))`` message reads this worker's controller slice from the shared block and
    converts its frames straight into the slot's shared views (:mod:`melee_rl.env.shm`).
    """

    env: DolphinEnv | None = None
    exchange: FrameExchange | None = None
    timing = new_worker_timing()  # lever 0: idle (recv), unpack, step and reply time per served step
    start = worker_index * config.num_envs
    stop = start + config.num_envs
    try:
        if exchange_spec is not None:
            exchange = FrameExchange(exchange_spec.num_envs, exchange_spec.ports, spec=exchange_spec)
        backend = None if backend_factory is None else backend_factory(config, worker_index)
        env = DolphinEnv(config, cpu_level=cpu_level, backend=backend)
        batch, frame_index = env.collect_rows()
        conn.send(("ready", (batch.leaves, list(frame_index), _PROTOCOL_VERSION)))
        while True:
            tick = time.perf_counter()
            message = conn.recv()
            waited = time.perf_counter() - tick
            if message is None:
                conn.send(None)
                return
            tag, payload = message
            if tag == "step" and isinstance(payload, tuple) and payload[0] == "shm":
                if exchange is None:
                    raise RuntimeError(f"env worker {worker_index}: a shared-memory step without an exchange")
                timing["messages"] += 1
                timing["recv_wait_s"] += waited
                tick = time.perf_counter()
                slot = int(payload[1])
                rows = {
                    int(port): rows_from_arrays(exchange.controller_arrays(port, start, stop), env.num_envs)
                    for port in env.controlled_ports
                }
                unpacked = time.perf_counter()
                _, frame_index = env.step_rows(
                    rows, into=exchange.frame_batch(slot, start, stop), output=False
                )
                exchange.frame_index(slot, start, stop)[:] = frame_index
                stepped = time.perf_counter()
                conn.send(("state", None))
                timing["unpack_s"] += unpacked - tick
                timing["step_s"] += stepped - unpacked
                timing["reply_s"] += time.perf_counter() - stepped
            elif tag == "step":
                timing["messages"] += 1
                timing["recv_wait_s"] += waited
                tick = time.perf_counter()
                rows = {
                    int(port): rows_from_arrays(arrays, env.num_envs)
                    for port, arrays in sorted(payload.items())
                }
                unpacked = time.perf_counter()
                batch, frame_index = env.step_rows(rows)
                stepped = time.perf_counter()
                conn.send(("state", (batch.leaves, list(frame_index))))
                timing["unpack_s"] += unpacked - tick
                timing["step_s"] += stepped - unpacked
                timing["reply_s"] += time.perf_counter() - stepped
            elif tag == "step_chunk":
                timing["messages"] += 1
                timing["recv_wait_s"] += waited
                states = []
                for frame_arrays in payload:
                    tick = time.perf_counter()
                    rows = {
                        int(port): rows_from_arrays(arrays, env.num_envs)
                        for port, arrays in sorted(frame_arrays.items())
                    }
                    unpacked = time.perf_counter()
                    batch, frame_index = env.step_rows(rows)
                    states.append((batch.leaves, list(frame_index)))
                    timing["unpack_s"] += unpacked - tick
                    timing["step_s"] += time.perf_counter() - unpacked
                tick = time.perf_counter()
                conn.send(("states", states))
                timing["reply_s"] += time.perf_counter() - tick
            elif tag == "reset":
                env.reset()
                batch, frame_index = env.collect_rows()
                conn.send(("state", (batch.leaves, list(frame_index))))
            elif tag == "stats":
                conn.send(("stats", env.stats()))
            elif tag == "timing":
                conn.send(("timing", {"worker": dict(timing), "env": env.timing()}))
            else:
                raise RuntimeError(f"env worker {worker_index}: unknown message tag {tag!r}")
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        return  # the parent went away (or interrupted): tear down quietly
    except BaseException as error:
        try:
            conn.send(("error", (type(error).__name__, str(error), traceback.format_exc())))
            conn.send(None)
        except Exception:
            LOG.warning("env worker %d could not report its error: %s", worker_index, error)
    finally:
        if env is not None:
            env.close()
        if exchange is not None:
            exchange.close()
        with contextlib.suppress(OSError):
            conn.close()


def _worker_entry(
    conn: Connection,
    config: DolphinEnvConfig,
    cpu_level: int,
    worker_index: int,
    backend_factory: BackendFactory | None,
    exchange_spec: ExchangeSpec | None = None,
) -> None:
    """The process target: hide CUDA, pin one torch thread, then run :func:`_worker_main`."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any CUDA init (torch initialises CUDA lazily)
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():  # pragma: no cover - spawn/forkserver make this impossible
        raise RuntimeError("env worker inherited an initialised CUDA context")
    _worker_main(conn, config, cpu_level, worker_index, backend_factory, exchange_spec)


# ---------------------------------------------------------------------------
# the parent
# ---------------------------------------------------------------------------


def _boot_deadline_s(config: DolphinEnvConfig) -> float:
    """Wall bound for a worker's boot reply: every launch attempt of its group, plus process start."""

    return config.boot_timeout_s * (1 + config.launch_retries) + 120.0


def _step_deadline_s(config: DolphinEnvConfig) -> float:
    """Wall bound for a step / stats reply: one frame plus a full in-worker relaunch ladder."""

    return config.console_timeout_s + config.boot_timeout_s * (config.launch_retries + 1) + 60.0


def _chunk_deadline_s(config: DolphinEnvConfig) -> float:
    """Wall bound for a ``states`` reply: :func:`_step_deadline_s` generalised to ``k`` frames."""

    return (
        config.console_timeout_s * config.chunk_frames
        + config.boot_timeout_s * (config.launch_retries + 1)
        + 60.0
    )


class WorkerDolphinEnv:
    """``worker_processes`` children, each a :class:`DolphinEnv` slice, behind ``EnvProtocol``."""

    def __init__(
        self,
        config: DolphinEnvConfig,
        *,
        cpu_level: int = 9,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if config.worker_processes < 1:
            raise ValueError("WorkerDolphinEnv needs worker_processes >= 1 (build_dolphin_env dispatches)")
        if not 1 <= cpu_level <= 9:
            raise ValueError("cpu_level must be in [1, 9]")
        self.config = config
        self.cpu_level = cpu_level
        self._closed = False
        self._procs: list[BaseProcess] = []
        self._conns: list[Connection] = []
        self._pending: EnvOutput | None = None
        # The P8 asynchronous surface: buffered pushes, chunks awaiting their reply, received
        # states not yet popped, and whether the boot / reset state is still poppable (the
        # "initial state in transit", slippi-ai envs.py _num_in_transit = 1).
        self._push_buffer: list[dict[int, ControllerArrays]] = []
        self._chunks_in_transit = 0
        self._state_queue: deque[EnvOutput] = deque()
        self._seed_pending = True
        self._timing = new_parent_timing(config.worker_processes)  # lever 0: the parent's side
        # The shared-memory exchange (lever 0 / IPC): controllers out, frames back, tokens on the pipe.
        self._exchange: FrameExchange | None = (
            FrameExchange(config.num_envs, config.controlled_ports) if config.shared_memory else None
        )
        self._slot = 0
        self._step_in_flight: tuple[str, int] | None = None  # lever 1: begin_step / end_step
        # The concrete context classes carry Process / Pipe; typeshed's BaseContext does not.
        context: Any = multiprocessing.get_context(config.mp_start_method)
        if config.mp_start_method == "forkserver":
            # The fork server imports the module tree (torch included) once; workers fork from it.
            context.set_forkserver_preload([_PRELOAD_MODULE])
        atexit.register(self.close)
        try:
            spec = None if self._exchange is None else self._exchange.spec
            for worker, worker_config in enumerate(worker_configs(config)):
                parent_conn, child_conn = context.Pipe()
                process = context.Process(
                    target=_worker_entry,
                    args=(child_conn, worker_config, cpu_level, worker, backend_factory, spec),
                    name=f"melee-rl-env-worker-{worker}",
                    daemon=False,  # a daemonic process could not start libmelee's enet child
                )
                process.start()
                child_conn.close()  # the child owns its end now; EOF detection needs ours closed
                self._procs.append(process)
                self._conns.append(parent_conn)
            deadline = _boot_deadline_s(config)
            states = []
            for worker in range(len(self._conns)):
                tag, payload = self._receive(worker, deadline)
                if tag != "ready":
                    raise EnvWorkerError(f"env worker {worker}: expected the boot reply, got {tag!r}")
                leaves, frame_index, version = payload
                if version != _PROTOCOL_VERSION:
                    raise EnvWorkerError(
                        f"env worker {worker} speaks protocol {version}, expected {_PROTOCOL_VERSION} "
                        "(stale fork server?)"
                    )
                states.append((leaves, frame_index))
            self._pending = assemble_states(states, self.controlled_ports)
        except BaseException:
            self.close()
            raise

    # -- protocol facts -----------------------------------------------------------------------

    @property
    def num_envs(self) -> int:
        return self.config.num_envs

    @property
    def controlled_ports(self) -> tuple[int, ...]:
        return self.config.controlled_ports

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def worker_count(self) -> int:
        return self.config.worker_processes

    @property
    def envs_per_worker(self) -> int:
        return self.config.envs_per_worker

    @property
    def shared_memory(self) -> bool:
        """True when the lockstep step exchanges frames through shared memory (``config.shared_memory``)."""

        return self._exchange is not None

    def current(self) -> EnvOutput:
        assert self._pending is not None
        return self._pending

    def stats(self) -> list[dict[str, Any]]:
        """Per-environment counters, globally indexed; each row also names its ``worker``."""

        self._check_open()
        for worker in range(len(self._conns)):
            self._send(worker, ("stats", None))
        group = self.config.envs_per_worker
        rows: list[dict[str, Any]] = []
        for worker in range(len(self._conns)):
            tag, payload = self._receive(worker, _step_deadline_s(self.config))
            if tag != "stats":
                raise EnvWorkerError(f"env worker {worker}: expected stats, got {tag!r}")
            for row in payload:
                rows.append({**row, "index": int(row["index"]) + worker * group, "worker": worker})
        return rows

    def timing(self) -> dict[str, Any]:
        """The cumulative timing counters (:mod:`melee_rl.env.timing`): ``parent`` (this side's sends,
        waits, receives, assembly and per-worker arrivals) and ``workers`` (each worker's loop counters
        and its ``DolphinEnv.timing()``).  Refuses while a chunk is in flight (the reply would be a
        ``states`` message)."""

        self._check_open()
        if self._push_buffer or self._chunks_in_transit:
            raise RuntimeError("timing() while chunks are in flight; pop every pushed state first")
        workers = range(len(self._conns))
        for worker in workers:
            self._send(worker, ("timing", None))
        payloads = self._collect(workers, _step_deadline_s(self.config), "timing", record=False)
        parent = {**self._timing, "arrival_s": list(self._timing["arrival_s"])}
        return {"parent": parent, "workers": list(payloads)}

    # -- the P8 asynchronous surface (AsyncEnvProtocol) ---------------------------------------

    @property
    def chunk_frames(self) -> int:
        return self.config.chunk_frames

    @property
    def in_flight(self) -> int:
        """States owed to :meth:`pop`: the seed + every push not yet popped (buffered or in transit)."""

        return (
            int(self._seed_pending)
            + len(self._state_queue)
            + self._chunks_in_transit * self.config.chunk_frames
            + len(self._push_buffer)
        )

    def _controller_payload(self, actions: Mapping[int, ControllerState]) -> dict[int, ControllerArrays]:
        ports = tuple(sorted(actions))
        if ports != self.controlled_ports:
            raise ValueError(f"step needs controllers for ports {self.controlled_ports}, got {ports}")
        return {port: controller_arrays(actions[port], self.num_envs) for port in ports}

    def push(self, actions: Mapping[int, ControllerState]) -> None:
        """Buffer one action set; the ``chunk_frames``-th push sends one ``step_chunk`` per worker."""

        self._check_open()
        self._push_buffer.append(self._controller_payload(actions))
        if len(self._push_buffer) == self.config.chunk_frames:
            group = self.config.envs_per_worker
            ports = tuple(sorted(self._push_buffer[0]))
            for worker in range(len(self._conns)):
                start, stop = worker * group, (worker + 1) * group
                chunk = [
                    {port: slice_arrays(arrays[port], start, stop) for port in ports}
                    for arrays in self._push_buffer
                ]
                self._send(worker, ("step_chunk", chunk))
            self._chunks_in_transit += 1
            self._push_buffer.clear()

    def _receive_chunk(self, action: str) -> None:
        """Receive one ``states`` reply per worker and queue the k assembled ``EnvOutput`` frames."""

        if self._chunks_in_transit == 0:
            raise RuntimeError(
                f"{action} with no state in flight ({len(self._push_buffer)} of "
                f"{self.config.chunk_frames} actions buffered): the pipeline slack "
                "(batch_steps - 1) + (chunk_frames - 1) <= delay_frames guarantees availability, "
                "so this is an invariant violation"
            )
        deadline = _chunk_deadline_s(self.config)
        replies = self._collect(range(len(self._conns)), deadline, "states")
        per_worker: list[list[tuple[Mapping[str, npt.NDArray[Any]], Sequence[int]]]] = [
            list(payload) for payload in replies
        ]
        lengths = {len(states) for states in per_worker}
        if lengths != {self.config.chunk_frames}:
            raise EnvWorkerError(
                f"chunk replies must carry {self.config.chunk_frames} states per worker, got {lengths}"
            )
        tick = time.perf_counter()
        for index in range(self.config.chunk_frames):
            frame = [states[index] for states in per_worker]
            self._state_queue.append(assemble_states(frame, self.controlled_ports))
        self._timing["assemble_s"] += time.perf_counter() - tick
        self._chunks_in_transit -= 1

    def pop(self) -> EnvOutput:
        """The next state in FIFO order (receiving a chunk reply if needed); never blocks on nothing."""

        self._check_open()
        if self._seed_pending:
            self._seed_pending = False
            assert self._pending is not None
            return self._pending
        if not self._state_queue:
            self._receive_chunk("pop")
        output = self._state_queue.popleft()
        self._pending = output
        return output

    def peek(self) -> EnvOutput:
        """The state the next :meth:`pop` returns, without consuming it."""

        self._check_open()
        if self._seed_pending:
            assert self._pending is not None
            return self._pending
        if not self._state_queue:
            self._receive_chunk("peek")
        return self._state_queue[0]

    def _drain_in_transit(self) -> None:
        """Receive and discard every outstanding chunk reply (reset while the pipeline is loaded)."""

        deadline = _chunk_deadline_s(self.config)
        while self._chunks_in_transit:
            for worker in range(len(self._conns)):
                tag, _ = self._receive(worker, deadline)
                if tag != "states":
                    raise EnvWorkerError(f"env worker {worker}: expected a chunk of states, got {tag!r}")
            self._chunks_in_transit -= 1
        self._state_queue.clear()
        self._push_buffer.clear()

    # -- protocol actions ---------------------------------------------------------------------

    def step(self, actions: Mapping[int, ControllerState]) -> EnvOutput:
        self.begin_step(actions)
        return self.end_step()

    def begin_step(self, actions: Mapping[int, ControllerState]) -> None:
        """Send one step's controllers to every worker and return at once (lever 1: the Dolphins emulate
        while the caller does other work); :meth:`end_step` collects the frame.  One step in flight at a
        time; ``step`` is ``begin_step`` + ``end_step``."""

        self._check_open()
        if self._push_buffer or self._chunks_in_transit or self._state_queue:
            raise RuntimeError(
                "step() while the asynchronous pipeline holds state (buffered pushes, chunks in "
                "transit or unpopped states); use push/pop, or reset() first"
            )
        if self._step_in_flight is not None:
            raise RuntimeError("begin_step() while a step is in flight; end_step() first")
        arrays = self._controller_payload(actions)
        ports = tuple(sorted(arrays))
        group = self.config.envs_per_worker
        workers = range(len(self._conns))
        exchange = self._exchange
        tick = time.perf_counter()
        if exchange is not None:
            slot = self._slot
            for port in ports:
                exchange.write_controllers(port, arrays[port])
            for worker in workers:
                self._send(worker, ("step", ("shm", slot)))
            self._step_in_flight = ("shm", slot)
        else:
            for worker in workers:
                payload = {
                    port: slice_arrays(arrays[port], worker * group, (worker + 1) * group) for port in ports
                }
                self._send(worker, ("step", payload))
            self._step_in_flight = ("pipe", 0)
        self._timing["send_s"] += time.perf_counter() - tick

    def end_step(self) -> EnvOutput:
        """Collect the replies of the step :meth:`begin_step` started and assemble its ``EnvOutput``."""

        self._check_open()
        in_flight = self._step_in_flight
        if in_flight is None:
            raise RuntimeError("end_step() without a step in flight; begin_step() first")
        workers = range(len(self._conns))
        deadline = _step_deadline_s(self.config)
        try:
            kind, slot = in_flight
            exchange = self._exchange
            if kind == "shm":
                assert exchange is not None
                self._collect(workers, deadline, "state")
                tick = time.perf_counter()
                self._pending = build_env_output(
                    exchange.frame_batch(slot), exchange.frame_index(slot).tolist(), self.controlled_ports
                )
                self._timing["assemble_s"] += time.perf_counter() - tick
                self._slot = (slot + 1) % exchange.slots
            else:
                states = self._collect(workers, deadline, "state")
                tick = time.perf_counter()
                self._pending = assemble_states(states, self.controlled_ports)
                self._timing["assemble_s"] += time.perf_counter() - tick
        finally:
            self._step_in_flight = None
        return self._pending

    def reset(self) -> EnvOutput:
        """Hard reset in every worker; the new boot state is also reseeded as the next :meth:`pop`."""

        self._check_open()
        self._drain_in_transit()
        for worker in range(len(self._conns)):
            self._send(worker, ("reset", None))
        deadline = _boot_deadline_s(self.config)
        states = [self._expect_state(worker, deadline) for worker in range(len(self._conns))]
        self._pending = assemble_states(states, self.controlled_ports)
        self._seed_pending = True
        return self._pending

    def close(self) -> None:
        """Sentinel every worker, drain while it lives, then join / terminate / kill (idempotent)."""

        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            with contextlib.suppress(BrokenPipeError, OSError):
                conn.send(None)
        for worker, (conn, process) in enumerate(zip(self._conns, self._procs, strict=True)):
            deadline = time.monotonic() + 10.0
            while process.is_alive() and time.monotonic() < deadline:
                # The worker may be blocked writing a reply; drain until its final None (envs.py:396).
                if not conn.poll(0.1):
                    continue
                try:
                    if conn.recv() is None:
                        break
                except (EOFError, ConnectionResetError, OSError):
                    break
            process.join(10.0)
            if process.is_alive():
                LOG.warning("env worker %d did not exit; terminating", worker)
                process.terminate()
                process.join(2.0)
            if process.is_alive():  # pragma: no cover - SIGTERM is enough for a python worker
                process.kill()
                process.join()
            with contextlib.suppress(OSError):
                conn.close()
        if self._exchange is not None:
            self._exchange.close()
            self._exchange.unlink()
        atexit.unregister(self.close)

    # -- pipe plumbing ------------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("WorkerDolphinEnv is closed")

    def _send(self, worker: int, message: Any) -> None:
        try:
            self._conns[worker].send(message)
        except (BrokenPipeError, OSError) as error:
            raise EnvWorkerError(
                f"env worker {worker} is gone (exit code {self._procs[worker].exitcode}); "
                f"could not send {message[0] if isinstance(message, tuple) else message!r}"
            ) from error

    def _receive(self, worker: int, deadline_s: float) -> tuple[str, Any]:
        """The next message from ``worker``; raises :class:`EnvWorkerError` on error / death / timeout."""

        conn, process = self._conns[worker], self._procs[worker]
        deadline = time.monotonic() + deadline_s
        while True:
            if conn.poll(0.2):
                try:
                    message = conn.recv()
                except (EOFError, ConnectionResetError, OSError) as error:
                    raise EnvWorkerError(
                        f"env worker {worker} died mid-reply (exit code {process.exitcode})"
                    ) from error
                if message is None:
                    raise EnvWorkerError(f"env worker {worker} shut down unexpectedly")
                tag, payload = message
                if tag == "error":
                    name, text, trace = payload
                    raise EnvWorkerError(f"env worker {worker} failed with {name}: {text}\n{trace}")
                return str(tag), payload
            if not process.is_alive():
                if conn.poll(0.0):
                    continue  # a reply raced the exit; drain it on the next pass
                raise EnvWorkerError(f"env worker {worker} died (exit code {process.exitcode})")
            if time.monotonic() >= deadline:
                raise EnvWorkerError(f"env worker {worker}: no reply within {deadline_s:g} s")

    def _expect_state(
        self, worker: int, deadline_s: float
    ) -> tuple[Mapping[str, npt.NDArray[Any]], Sequence[int]]:
        tag, payload = self._receive(worker, deadline_s)
        if tag != "state":
            raise EnvWorkerError(f"env worker {worker}: expected a state, got {tag!r}")
        leaves, frame_index = payload
        return leaves, frame_index

    def _collect(
        self, workers: Sequence[int] | range, deadline_s: float, expect: str, *, record: bool = True
    ) -> list[Any]:
        """One ``expect``-tagged reply per worker, received in arrival order, returned in worker order.

        The parent no longer blocks on worker 0 while worker 5's reply sits in its pipe: whichever
        connection is readable is served first.  With ``record`` the step's wait until the first and
        the last reply, the receive (unpickling) time and each worker's arrival time go into the
        parent's timing counters (lever 0).
        """

        pending = {self._conns[worker]: worker for worker in workers}
        results: dict[int, Any] = {}
        started = time.perf_counter()
        deadline = time.monotonic() + deadline_s
        first = True
        while pending:
            ready = wait_connections(list(pending), timeout=0.2)
            arrived = time.perf_counter()
            for readable in ready:
                conn = cast(Connection, readable)
                worker = pending.pop(conn)
                if record:
                    if first:
                        self._timing["wait_first_s"] += arrived - started
                        first = False
                    self._timing["arrival_s"][worker] += arrived - started
                tick = time.perf_counter()
                tag, payload = self._receive(worker, deadline_s)  # readable: returns at once
                if record:
                    self._timing["recv_s"] += time.perf_counter() - tick
                if tag != expect:
                    raise EnvWorkerError(f"env worker {worker}: expected {expect!r}, got {tag!r}")
                results[worker] = payload
            if not ready:
                for conn, worker in list(pending.items()):
                    if not self._procs[worker].is_alive() and not conn.poll(0.0):
                        raise EnvWorkerError(
                            f"env worker {worker} died (exit code {self._procs[worker].exitcode})"
                        )
                if time.monotonic() >= deadline:
                    waiting = sorted(pending.values())
                    raise EnvWorkerError(f"env workers {waiting}: no reply within {deadline_s:g} s")
        if record:
            self._timing["wait_last_s"] += time.perf_counter() - started
            self._timing["steps"] += 1
        return [results[worker] for worker in workers]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def build_dolphin_env(
    config: DolphinEnvConfig,
    *,
    cpu_level: int = 9,
    backend: DolphinBackend | None = None,
    backend_factory: BackendFactory | None = None,
) -> DolphinEnv | WorkerDolphinEnv:
    """The Dolphin environment ``config`` selects: in-process (``worker_processes == 0``) or workers.

    ``backend`` injects a live backend into the in-process env (the P5 test seam);
    ``backend_factory`` is its worker-mode counterpart, a module-level callable built inside each
    child.  Passing the wrong one for the mode is an error rather than a silent ignore.
    """

    if config.worker_processes == 0:
        if backend_factory is not None:
            raise ValueError("worker_processes = 0 runs in-process: pass backend, not backend_factory")
        return DolphinEnv(config, cpu_level=cpu_level, backend=backend)
    if backend is not None:
        raise ValueError("a live backend cannot cross process boundaries: pass backend_factory")
    return WorkerDolphinEnv(config, cpu_level=cpu_level, backend_factory=backend_factory)


__all__ = [
    "BackendFactory",
    "ControllerArrays",
    "EnvWorkerError",
    "WorkerDolphinEnv",
    "assemble_states",
    "build_dolphin_env",
    "controller_arrays",
    "rows_from_arrays",
    "slice_arrays",
    "worker_configs",
]
