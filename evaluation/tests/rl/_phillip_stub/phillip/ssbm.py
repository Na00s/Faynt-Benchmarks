"""The ctypes state and controller structs, with upstream's field names and types."""

from __future__ import annotations

from ctypes import Structure, c_bool, c_float, c_uint


class Stick(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class RealControllerState(Structure):
    _fields_ = [
        ("button_A", c_bool),
        ("button_B", c_bool),
        ("button_X", c_bool),
        ("button_Y", c_bool),
        ("button_Z", c_bool),
        ("button_L", c_bool),
        ("button_R", c_bool),
        ("button_START", c_bool),
        ("trigger_L", c_float),
        ("trigger_R", c_float),
        ("stick_MAIN", Stick),
        ("stick_C", Stick),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.stick_MAIN.x = 0.5
        self.stick_MAIN.y = 0.5
        self.stick_C.x = 0.5
        self.stick_C.y = 0.5


class PlayerMemory(Structure):
    _fields_ = [
        ("percent", c_uint),
        ("stock", c_uint),
        ("facing", c_float),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("action_state", c_uint),
        ("action_counter", c_uint),
        ("action_frame", c_float),
        ("character", c_uint),
        ("invulnerable", c_bool),
        ("hitlag_frames_left", c_float),
        ("hitstun_frames_left", c_float),
        ("jumps_used", c_uint),
        ("charging_smash", c_bool),
        ("in_air", c_bool),
        ("speed_air_x_self", c_float),
        ("speed_ground_x_self", c_float),
        ("speed_y_self", c_float),
        ("speed_x_attack", c_float),
        ("speed_y_attack", c_float),
        ("shield_size", c_float),
        ("cursor_x", c_float),
        ("cursor_y", c_float),
        ("controller", RealControllerState),
    ]


class GameMemory(Structure):
    _fields_ = [
        ("players", PlayerMemory * 2),
        ("frame", c_uint),
        ("menu", c_uint),
        ("stage", c_uint),
        ("sss_cursor_x", c_float),
        ("sss_cursor_y", c_float),
    ]
