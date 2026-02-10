#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Marc Duclusaud

import gymnasium as gym
import numpy as np
import time

import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from rl_policies.utils import create_servo_target, get_inputs
from identification.filter import IIRFilter


def run(
    frequency: float = 50.0,
    position: bool = False,
    velocity: bool = False,
) -> None:
    """
    Run and record trajectories to identify delay using a spine environment.

    :param frequency: Control loop frequency in Hz.
    :param position: Record position controlled motor trajectories.
    :param velocity: Record velocity controlled motor trajectories.
    """
    upkie.envs.register()

    data = {"timestamp": [], 
            "target_pos": [], 
            "read_pos": [], 
            "target_vel": [], 
            "read_vel": [],
             }

    with gym.make("Upkie-Spine-Servos", frequency=frequency, max_gain_scale=100.0) as env:
        _, info = env.reset()
        spine_observation = info["spine_observation"]

        last_action = [0.0] * 6
        command = [0.0] * 3

        t = 0
        start_time = time.perf_counter()
        while t < 12.0:
            t = time.perf_counter() - start_time

            position = 0.0
            velocity = 0.0
            if position:
                position = np.sin(t * np.pi) * 0.3
            elif velocity:
                velocity = np.sin(t * np.pi / 2) * 6.0

            action_dict = {
                "left_hip": create_servo_target(position=0.0, max_torque=16.0),
                "left_knee": create_servo_target(position=position, max_torque=16.0),
                "right_hip": create_servo_target(position=0.0, max_torque=16.0),
                "right_knee": create_servo_target(position=0.0, max_torque=16.0),
                "left_wheel": create_servo_target(velocity=0.0, max_torque=1.7),
                "right_wheel": create_servo_target(velocity=velocity, max_torque=1.7),
            }

            observation = get_inputs(last_action, command, spine_observation, frequency)
            
            data["timestamp"].append(t)
            if position:
                data["target"].append(position)
                data["read"].append(float(observation["obs"][0][1]))
            if velocity:
                data["target"].append(velocity)
                data["read"].append(float(observation["obs"][0][5]))

            for joint in ["left_hip", "left_knee", "right_hip", "right_knee"]:
                action_dict[joint]["kp_scale"] = 1.0
                action_dict[joint]["kd_scale"] = 1.0

            _, _, terminated, truncated, info = env.step(action_dict)
            spine_observation = info["spine_observation"]

            if terminated or truncated:
                _, info = env.reset()
                spine_observation = info["spine_observation"]
    
    # Save data
    if position or velocity:
        import json

        filename = "position_trajectories.json" if position else "velocity_trajectories.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run delay identification agent for Upkie.")
    parser.add_argument("-p", "--position", action="store_true", help="Record position controlled motor trajectories")
    parser.add_argument("-v", "--velocity", action="store_true", help="Record velocity controlled motor trajectories")
    args = parser.parse_args()

    # On Raspberry Pi, configure the process to run on a separate CPU core
    if on_raspi():
        configure_agent_process()

    try:
        run(servo=args.servo)
    except KeyboardInterrupt:
        logger.info("Terminating in response to keyboard interrupt")
