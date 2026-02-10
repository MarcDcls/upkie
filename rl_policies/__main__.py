#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Marc Duclusaud


import gymnasium as gym
import numpy as np
import time
from pathlib import Path

import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from rl_policies.utils import (
    QDD_100,
    create_servo_target, 
    get_command_from_controller, 
    get_inputs, 
    load_onnx_model,
)


def run(
    frequency: float = 50.0,
    command_max_change: float = 0.03, # Check if necessary when running on real robot
    wheel_action_scale: float = 100.0,
) -> None:
    """
    Run agent using a spine environment.

    :param frequency: Control loop frequency in Hz.
    """
    upkie.envs.register()

    with gym.make("Upkie-Spine-Servos", frequency=frequency, max_gain_scale=100.0) as env:
        _, info = env.reset()
        spine_observation = info["spine_observation"]

        model_path = Path(__file__).parent / "agents" / "default.onnx"
        ort_sess = load_onnx_model(str(model_path))

        last_action = [0.0] * 6
        last_command = [0.0] * 3

        agent_is_running = False

        last_time = time.perf_counter()
        while True:
            current_time = time.perf_counter()
            elapsed_time = current_time - last_time
            if elapsed_time < 0.9 / frequency:
                print(f"Warning: Control loop is running slower than expected ({elapsed_time:.3f} seconds per iteration)")
            elif elapsed_time > 1.1 / frequency:
                print(f"Warning: Control loop is running faster than expected ({elapsed_time:.3f} seconds per iteration)")
            last_time = time.perf_counter()

            # Activate agent when A button is pressed
            if (not agent_is_running) and spine_observation["joystick"]["cross_button"]:
                agent_is_running = True
                print("Starting RL agent.", flush=True)

            # Deactivate agent when X button is pressed
            if agent_is_running and spine_observation["joystick"]["square_button"]:
                agent_is_running = False
                print("Stopping RL agent.", flush=True)

            action_dict = {
                "left_hip": create_servo_target(position=0, max_torque=16.0),
                "left_knee": create_servo_target(position=0, max_torque=16.0),
                "right_hip": create_servo_target(position=0, max_torque=16.0),
                "right_knee": create_servo_target(position=0, max_torque=16.0),
                "left_wheel": create_servo_target(velocity=0, max_torque=1.7),
                "right_wheel": create_servo_target(velocity=0, max_torque=1.7),
            }

            if agent_is_running:
                # Smooth command to avoid bang-bang behavior
                lin_vel, ang_vel = get_command_from_controller(spine_observation)
                command = [
                    last_command[0] + min(abs(lin_vel - last_command[0]), command_max_change) * np.sign(lin_vel - last_command[0]),
                    0.0,
                    last_command[2] + min(abs(ang_vel - last_command[2]), command_max_change) * np.sign(ang_vel - last_command[2]),
                ]
                last_command = command
                
                observation = get_inputs(last_action, command, spine_observation, frequency)
                action = ort_sess.run(None, observation)[0][0]
                last_action = action.tolist()

                action_dict = {
                    "left_hip": create_servo_target(position=-action[0], max_torque=16.0),
                    "left_knee": create_servo_target(position=action[1], max_torque=16.0),
                    "right_hip": create_servo_target(position=-action[2], max_torque=16.0),
                    "right_knee": create_servo_target(position=action[3], max_torque=16.0),
                    "left_wheel": create_servo_target(velocity=action[4] * wheel_action_scale, max_torque=1.7),
                    "right_wheel": create_servo_target(velocity=action[5] * wheel_action_scale, max_torque=1.7),
                }
            
            for joint in ["left_hip", "left_knee", "right_hip", "right_knee"]:
                action_dict[joint]["kp_scale"] = 8.0 * 2 * np.pi / QDD_100["kp"]
                action_dict[joint]["kd_scale"] = 0.1 * 2 * np.pi / QDD_100["kd"]

            _, _, terminated, truncated, info = env.step(action_dict)
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
