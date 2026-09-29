"""P26: the analog remap our libmelee pin applies, and the flag that turns it off.

`melee==0.47.3` (vladfi1's fork) added ``Controller(fix_analog_inputs=True)`` -- on by default
(``controller.py:89``) -- which rounds every ``tilt_analog`` value onto Melee's internal [-80, 80]
grid before writing it to the pipe.  altf4's `melee==0.41.1`, the version SmashBot was written
against, has no such remapping: ``tilt_analog`` writes the caller's float straight through
(``0.41.1 controller.py:320-332``).

It matters because SmashBot's wavedash asks for ``tilt_analog(MAIN, 0.5 +/- 0.475, 0.35)`` --
"near perfect wavedash angle" (``Chains/wavedash.py:67``, ``Chains/waveshine.py:148``) -- and 0.35
is *one unit* of 80 clear of Melee's 23/80 stick deadzone.  In 15 recorded games the downward
component arrived at the game as exactly neutral on all 580 airdodges with ``|x| >= 0.95`` and
arrived correctly (-0.30) on the 70 with ``x = 0.5``: a horizontal airdodge instead of a wavedash,
which off a ledge is a self-destruct.

So the port SmashBot plays on can opt out, and nothing else changes: ``legacy_analog_ports`` is
empty by default, which is exactly today's behaviour on every port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from melee_rl.env.dolphin import PORTS, DolphinEnvConfig, LibmeleeBackend

# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------


def test_legacy_analog_ports_is_empty_by_default() -> None:
    assert DolphinEnvConfig().legacy_analog_ports == ()


def test_legacy_analog_ports_accepts_the_two_libmelee_ports() -> None:
    config = DolphinEnvConfig(legacy_analog_ports=(1, 2))
    assert config.legacy_analog_ports == (1, 2)


@pytest.mark.parametrize("port", [0, 3, -1])
def test_legacy_analog_ports_rejects_a_port_that_is_not_played(port: int) -> None:
    with pytest.raises(ValueError, match="legacy_analog_ports"):
        DolphinEnvConfig(legacy_analog_ports=(port,))


def test_legacy_analog_ports_rejects_a_repeat() -> None:
    with pytest.raises(ValueError, match="legacy_analog_ports"):
        DolphinEnvConfig(legacy_analog_ports=(1, 1))


# ---------------------------------------------------------------------------
# the backend passes it per port
# ---------------------------------------------------------------------------


class _RecordingMelee:
    """Just enough ``melee`` for :meth:`LibmeleeBackend.make_controller`."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any]]] = []

        class ControllerType:
            STANDARD = "standard"

        self.ControllerType = ControllerType

    def Controller(self, console: Any, port: int, kind: Any, **kwargs: Any) -> str:
        self.calls.append((port, dict(kwargs)))
        return f"controller-{port}"


def _prepared(backend: LibmeleeBackend, config: DolphinEnvConfig) -> None:
    """``prepare`` needs a real Dolphin; the step of it that reads this config does not."""

    backend.configure_analog(config)


def test_make_controller_keeps_the_remap_on_by_default() -> None:
    melee = _RecordingMelee()
    backend = LibmeleeBackend(melee)
    _prepared(backend, DolphinEnvConfig())
    for port in PORTS:
        backend.make_controller(object(), port)
    assert [kwargs.get("fix_analog_inputs", True) for _, kwargs in melee.calls] == [True] * len(PORTS)


def test_make_controller_turns_the_remap_off_only_on_the_listed_port() -> None:
    melee = _RecordingMelee()
    backend = LibmeleeBackend(melee)
    _prepared(backend, DolphinEnvConfig(legacy_analog_ports=(2,)))
    for port in PORTS:
        backend.make_controller(object(), port)
    assert dict((port, kwargs["fix_analog_inputs"]) for port, kwargs in melee.calls) == {1: True, 2: False}


def test_a_backend_that_was_never_configured_remaps_everything() -> None:
    """The protocol's ``make_controller`` takes no config, so an unconfigured backend must be safe."""

    backend = LibmeleeBackend(_RecordingMelee())
    assert backend.legacy_analog_ports == frozenset()
    backend.make_controller(object(), 1)
    assert backend._melee.calls == [(1, {"fix_analog_inputs": True})]


# ---------------------------------------------------------------------------
# what the flag actually changes, against the real libmelee
# ---------------------------------------------------------------------------


class _FakeConsole:
    is_dolphin = True
    logger = None

    def __init__(self, path: Path) -> None:
        self._path = path

    def get_dolphin_pipes_path(self, port: int) -> str:
        return str(self._path)

    def setup_dolphin_controller(self, port: int, kind: Any) -> None:
        return None


def _written(tmp_path: Path, *, fix: bool, x: float, y: float) -> str:
    melee = pytest.importorskip("melee")
    pipe = tmp_path / f"pipe-{fix}"
    console = _FakeConsole(pipe)
    controller = melee.Controller(console, 1, melee.ControllerType.STANDARD, fix_analog_inputs=fix)
    with pipe.open("w") as handle:
        controller.pipe = handle
        controller.tilt_analog(melee.Button.BUTTON_MAIN, x, y)
        controller.pipe = None
    return pipe.read_text()


def test_the_remap_moves_smashbots_wavedash_angle(tmp_path: Path) -> None:
    """0.35 is what SmashBot asks for; the remap sends something else, and 0.41.1 sent 0.35."""

    remapped = _written(tmp_path, fix=True, x=0.025, y=0.35)
    verbatim = _written(tmp_path, fix=False, x=0.025, y=0.35)
    assert "0.35" in verbatim
    assert "0.35" not in remapped
    assert remapped != verbatim


def test_the_remap_is_a_no_op_on_the_neutral_stick(tmp_path: Path) -> None:
    """Turning it off must not move a stick that is already on the grid, or every run changes."""

    remapped = _written(tmp_path, fix=True, x=0.5, y=0.5)
    verbatim = _written(tmp_path, fix=False, x=0.5, y=0.5)
    assert [float(token) for token in remapped.split()[2:4]] == pytest.approx([0.5, 0.5], abs=2e-3)
    assert [float(token) for token in verbatim.split()[2:4]] == [0.5, 0.5]
