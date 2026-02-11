#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Marc Duclusaud

import gymnasium as gym
import numpy as np
import time
import json

import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from rl_policies.utils import create_servo_target, get_inputs, QDD_100
from identification.trajectory import get_position_trajectory


def log_data(data, t, observation, action):
    data["timestamp"].append(t)
    
    data["left_hip_read"].append(float(observation["obs"][0][0]))
    data["left_knee_read"].append(float(observation["obs"][0][1]))
    data["right_hip_read"].append(float(observation["obs"][0][2]))
    data["right_knee_read"].append(float(observation["obs"][0][3]))
    data["left_wheel_read"].append(float(observation["obs"][0][4]))
    data["right_wheel_read"].append(float(observation["obs"][0][5]))
    
    data["left_hip_target"].append(float(action["left_hip"]["position"]))
    data["left_knee_target"].append(float(action["left_knee"]["position"]))
    data["right_hip_target"].append(float(action["right_hip"]["position"]))
    data["right_knee_target"].append(float(action["right_knee"]["position"]))
    data["left_wheel_target"].append(float(action["left_wheel"]["velocity"]))
    data["right_wheel_target"].append(float(action["right_wheel"]["velocity"]))


def run(
    frequency: float = 50.0,
    position: bool = False,
    velocity: bool = False,
    trajectory: bool = False,
) -> None:
    """
    Run and record trajectories to identify delay using a spine environment.

    :param frequency: Control loop frequency in Hz.
    :param position: Record position controlled motor trajectories.
    :param velocity: Record velocity controlled motor trajectories.
    """
    upkie.envs.register()

    data = {
        "timestamp": [],
        "left_hip_read": [],
        "left_knee_read": [],
        "right_hip_read": [],
        "right_knee_read": [],
        "left_wheel_read": [],
        "right_wheel_read": [],
        "left_hip_target": [],
        "left_knee_target": [],
        "right_hip_target": [],
        "right_knee_target": [],
        "left_wheel_target": [],
        "right_wheel_target": [],
    }

    duration = 12.0

    traj = None
    if trajectory:
        traj = get_position_trajectory(6.0, frequency, hold_duration=1.0)
        duration = 7.0

    with gym.make("Upkie-Spine-Servos", frequency=frequency, max_gain_scale=100.0) as env:
        _, info = env.reset()
        spine_observation = info["spine_observation"]

        last_action = [0.0] * 6
        command = [0.0] * 3

        # Going to zero position before starting the trajectory
        for t in range(int(2.0 * frequency)):
            action_dict = {
                "left_hip": create_servo_target(position=0.0, max_torque=16.0),
                "left_knee": create_servo_target(position=0.0, max_torque=16.0),
                "right_hip": create_servo_target(position=0.0, max_torque=16.0),
                "right_knee": create_servo_target(position=0.0, max_torque=16.0),
                "left_wheel": create_servo_target(velocity=0.0, max_torque=1.7),
                "right_wheel": create_servo_target(velocity=0.0, max_torque=1.7),
            }
            _, _, terminated, truncated, info = env.step(action_dict)
            spine_observation = info["spine_observation"]
        
        t = 0
        start_time = time.perf_counter()
        while t < duration:
            t = time.perf_counter() - start_time

            action_dict = {
                "left_hip": create_servo_target(position=0.0, max_torque=16.0),
                "left_knee": create_servo_target(position=0.0, max_torque=16.0),
                "right_hip": create_servo_target(position=0.0, max_torque=16.0),
                "right_knee": create_servo_target(position=0.0, max_torque=16.0),
                "left_wheel": create_servo_target(velocity=0.0, max_torque=1.7),
                "right_wheel": create_servo_target(velocity=0.0, max_torque=1.7),
            }

            if position:
                action_dict["left_knee"]["position"] = np.sin(t * np.pi) * 0.3
            
            if velocity:
                action_dict["right_wheel"]["velocity"] = np.sin(t * np.pi / 2) * 6.0

            if trajectory:
                action_dict["left_hip"]["position"] = -traj["left_hip"][int(t * frequency)]
                action_dict["left_knee"]["position"] = traj["left_knee"][int(t * frequency)]
                action_dict["right_hip"]["position"] = -traj["right_hip"][int(t * frequency)]
                action_dict["right_knee"]["position"] = traj["right_knee"][int(t * frequency)]

            observation = get_inputs(last_action, command, spine_observation, frequency)
            
            log_data(data, t, observation, action_dict)

            for joint in ["left_hip", "left_knee", "right_hip", "right_knee"]:
                action_dict[joint]["kp_scale"] = 8.0 * 2 * np.pi / QDD_100["kp"]
                action_dict[joint]["kd_scale"] = 0.1 * 2 * np.pi / QDD_100["kd"]

            _, _, terminated, truncated, info = env.step(action_dict)
            spine_observation = info["spine_observation"]

            if terminated or truncated:
                _, info = env.reset()
                spine_observation = info["spine_observation"]
    
    # Save data
    if position or velocity or trajectory:
        filename = "position_trajectory.json" 
        if velocity:
            filename = "velocity_trajectory.json"
        if trajectory:
            filename = "sinusoidal_trajectory" + time.strftime("-%Y%m%d-%H%M%S") + ".json"

        with open(filename, "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run delay identification agent for Upkie.")
    parser.add_argument("-p", "--position", action="store_true", help="Record position controlled motor trajectories")
    parser.add_argument("-v", "--velocity", action="store_true", help="Record velocity controlled motor trajectories")
    parser.add_argument("-t", "--trajectory", action="store_true", help="Record complex position trajectories for all joints")
    args = parser.parse_args()

    # On Raspberry Pi, configure the process to run on a separate CPU core
    if on_raspi():
        configure_agent_process()

    try:
        run(position=args.position, velocity=args.velocity, trajectory=args.trajectory)
    except KeyboardInterrupt:
        logger.info("Terminating in response to keyboard interrupt")
