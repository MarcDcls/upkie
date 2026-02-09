#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Marc Duclusaud

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import time
from pathlib import Path

from scipy.spatial.transform import Rotation as R


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

command_max_change = 0.03 # Check if necessary when running on real robot

wheel_action_scale = 100.0

R_body_to_imu = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
])

last_wheel_positions = None


def load_onnx_model(model_path: str) -> ort.InferenceSession:
    """
    Load an ONNX model and create an inference session.
    
    :param model_path: Path to the ONNX model file.
    :return: ONNX Runtime inference session.
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    ort_sess = ort.InferenceSession(model_path)
    return ort_sess


def create_servo_target(
    position: float = np.nan,
    velocity: float = 0.0,
    max_torque: float = 20.0,
) -> dict:
    """
    Create a servo target dictionary.

    :param position: Target position in radians.
    :param velocity: Target velocity in radians per second.
    :param max_torque: Maximum torque in Newton-meters.
    :return: Servo target dictionary.
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


def get_inputs(last_action, command, spine_observation, frequency) -> dict:
    """Prepare observation dictionary for ONNX model inference."""
    obs = []

    # Joint positions
    obs.append(-spine_observation["servo"]["left_hip"]["position"])
    obs.append(spine_observation["servo"]["left_knee"]["position"])
    obs.append(-spine_observation["servo"]["right_hip"]["position"])
    obs.append(spine_observation["servo"]["right_knee"]["position"])

    # Wheel velocities
    wheel_positions = np.array([spine_observation["servo"]["left_wheel"]["position"], 
                                spine_observation["servo"]["right_wheel"]["position"]])
    
    global last_wheel_positions
    wheel_velocities = [0.0, 0.0]
    if last_wheel_positions is not None:
        wheel_velocities = ((wheel_positions - last_wheel_positions) * frequency).tolist()

    last_wheel_positions = wheel_positions
    obs.extend(wheel_velocities)

    # IMU readings (quaternion in (w, x, y, z) order)
    imu_quat = spine_observation["imu"]["orientation"]
    r_imu = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]])
    
    R_mod_imu = r_imu.as_matrix() @ R_body_to_imu
    euler_mod_imu = R.from_matrix(R_mod_imu).as_euler('xyz', degrees=True) * np.array([-1, 1, -1])
    
    r_mod_imu = R.from_euler('xyz', euler_mod_imu, degrees=True)
    mod_imu_quat = r_mod_imu.as_quat()  # (x, y, z, w) order
    obs.extend([mod_imu_quat[3], mod_imu_quat[0], mod_imu_quat[1], mod_imu_quat[2]])

    # Gyro readings
    angular_velocity = np.array(spine_observation["imu"]["angular_velocity"])
    corrected_angular_velocity = (angular_velocity @ R_body_to_imu) * np.array([-1.0, 1.0, -1.0])
    obs.extend(corrected_angular_velocity.tolist())

    # Last action
    obs.extend(last_action)

    # Command
    obs.extend(command)

    # Debug
    # print(f"joint positions: {np.array(obs[0:4]).round(2)}")
    # print(f"wheel velocities: {np.array(obs[4:6]).round(2)}")
    # print(f"wheel position: {np.array(spine_observation['servo']['left_wheel']['position']):.2f}, {np.array(spine_observation['servo']['right_wheel']['position']):.2f}")
    # print(f"IMU quaternion: {np.array(obs[6:10]).round(2)}")
    # print(f"gyro readings: {np.array(obs[10:13]).round(2)}")
    # print(f"last action: {np.array(obs[13:19]).round(2)}")
    # print(f"command: {np.array(obs[19:22]).round(2)}")
    # print("------------------------")

    return {
        "obs": [
            np.array(obs, dtype=np.float32),
        ]
    }

def run(
    frequency: float = 50.0,
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
                action_dict[joint]["kp_scale"] = 8.0 * 2 * np.pi / qdd100["kp"]
                action_dict[joint]["kd_scale"] = 0.1 * 2 * np.pi / qdd100["kd"]

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
