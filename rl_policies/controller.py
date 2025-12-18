#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Marc Duclusaud

import evdev
from typing import Tuple

AXIS_RANGE_MIN = 0
AXIS_RANGE_MAX = 65535

LEFT_MAP = {
    evdev.ecodes.ABS_Y: 0,
    evdev.ecodes.ABS_X: 1,
}

RIGHT_MAP = {
    evdev.ecodes.ABS_RZ: 0,
    evdev.ecodes.ABS_Z: 1,
}

SIGNS = {
    evdev.ecodes.ABS_X: 1,
    evdev.ecodes.ABS_Y: -1,
    evdev.ecodes.ABS_Z: 1,
    evdev.ecodes.ABS_RZ: -1,
}


def normalize(value: float) -> float:
    """Normalize axis value from [0, 65535] to [-1, 1]."""
    axis_center = (AXIS_RANGE_MAX - AXIS_RANGE_MIN) / 2
    return (value - axis_center) / axis_center


def setup_controller() -> evdev.InputDevice:
    """
    Initialize and return the input device.
    
    \return Input device for joystick control.
    \raise SystemExit if no device or multiple devices found.
    """
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]

    if len(devices) > 1:
        print("Multiple input devices detected:")
        for device in devices:
            print(f"Device: {device.name}, Path: {device.path}")
        exit(1)

    if len(devices) == 0:
        print("No input devices found.")
        exit(1)

    return devices[0]


class JoystickState:
    """Track joystick positions."""
    
    def __init__(self):
        self.left_stick = [0.0, 0.0]
        self.right_stick = [0.0, 0.0]
    
    def update(self, event: evdev.event.InputEvent) -> None:
        """
        Update joystick state from input event.
        
        \param event Input event from device.
        """
        if event.type != evdev.ecodes.EV_ABS:
            return
        
        # Left joystick
        if event.code in LEFT_MAP:
            idx = LEFT_MAP[event.code]
            self.left_stick[idx] = normalize(event.value) * SIGNS[event.code]
        
        # Right joystick
        if event.code in RIGHT_MAP:
            idx = RIGHT_MAP[event.code]
            self.right_stick[idx] = normalize(event.value) * SIGNS[event.code]
    
    def get_left_stick(self) -> Tuple[float, float]:
        """Return left stick position as (y, x)."""
        return tuple(self.left_stick)
    
    def get_right_stick(self) -> Tuple[float, float]:
        """Return right stick position as (rz, z)."""
        return tuple(self.right_stick)


if __name__ == "__main__":
    device = setup_controller()
    joystick = JoystickState()
    
    for event in device.read_loop():
        joystick.update(event)
        print("Left stick :", joystick.get_left_stick())
        print("Right stick :", joystick.get_right_stick())