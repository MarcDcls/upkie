#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Marc Duclusaud

import gymnasium as gym
import numpy as np

import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

qdd100 = {
    "kp": 400.0,
    "kd": 2.0,
    "ki": 1.0,
}

mj5208 = {
    "kp": 4.0,
    "kd": 0.3,
    "ki": 0.0,
}

max_lin_vel = 1.0  # m/s
max_ang_vel = 1.5  # rad/s

def create_servo_target(
    position: float = np.nan,
    velocity: float = 0.0,
    max_torque: float = 20.0,
) -> dict:
    r"""!
    Create a servo target dictionary.

    \param position Target position in radians.
    \param velocity Target velocity in radians per second.
    \param max_torque Maximum torque in Newton-meters.
    \return Servo target dictionary.
    """

    target = {
        "position": position,
        "velocity": velocity,
        "maximum_torque": max_torque,
        }
    return target

def get_command_from_controller(spine_observation, deadzone: float = 0.1) -> tuple[float, float]:
    """
    Read joystick axes from spine sensors (SDL2).
    Returns a tuple (lin_vel, ang_vel).
    """
    _, left_y = spine_observation["joystick"]["left_axis"]
    right_x, _ = spine_observation["joystick"]["right_axis"]

    lin = -left_y
    ang = -right_x

    if abs(lin) < deadzone:
        lin = 0.0
    if abs(ang) < deadzone:
        ang = 0.0

    return max_lin_vel * lin, max_ang_vel * ang

def run(
    frequency: float = 50.0,
) -> None:
    r"""!
    Run agent using a spine environment.

    \param frequency Control loop frequency in Hz.
    """
    upkie.envs.register()

    with gym.make("Upkie-Spine-Servos", frequency=frequency, max_gain_scale=100.0) as env:
        _, info = env.reset()
        spine_observation = info["spine_observation"]

        # Set motor gains
        # TODO (si gains sets hors congigure_servos.py)

        while True:

            lin_vel, ang_vel = get_command_from_controller(spine_observation)

            print(f"Command: lin_vel={lin_vel:.2f} m/s, ang_vel={ang_vel:.2f} rad/s")

            agent_action = [0, 0, 0, 0, lin_vel, lin_vel] # for testing purposes
            action = {
                "left_hip": create_servo_target(position=agent_action[0], max_torque=16.0),
                "left_knee": create_servo_target(position=agent_action[1], max_torque=16.0),
                "right_hip": create_servo_target(position=agent_action[2], max_torque=16.0),
                "right_knee": create_servo_target(position=agent_action[3], max_torque=16.0),
                "left_wheel": create_servo_target(velocity=agent_action[4], max_torque=1.7),
                "right_wheel": create_servo_target(velocity=agent_action[5], max_torque=1.7),
            }
            
            for joint in ["left_hip", "left_knee", "right_hip", "right_knee"]:
                action[joint]["kp_scale"] = 8.0 * 2 * np.pi / qdd100["kp"]
                action[joint]["kd_scale"] = 0.1 * 2 * np.pi / qdd100["kd"]

            _, _, terminated, truncated, info = env.step(action)
            spine_observation = info["spine_observation"]

            if terminated or truncated:
                _, info = env.reset()
                spine_observation = info["spine_observation"]


if __name__ == "__main__":
    
    # On Raspberry Pi, configure the process to run on a separate CPU core
    if on_raspi():
        configure_agent_process()

    try:
        run()
    except KeyboardInterrupt:
        logger.info("Terminating in response to keyboard interrupt")
