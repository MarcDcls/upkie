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

wheel_action_scale = 100.0

last_action = [0.0] * 6


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


def get_inputs(last_action, command, spine_observation) -> dict:
    """Prepare observation dictionary for ONNX model inference."""
    obs = []

    # Joint positions
    obs.append(spine_observation["servo"]["left_hip"]["position"])
    obs.append(spine_observation["servo"]["left_knee"]["position"])
    obs.append(spine_observation["servo"]["right_hip"]["position"])
    obs.append(spine_observation["servo"]["right_knee"]["position"])

    # Wheel velocities
    obs.append(spine_observation["servo"]["left_wheel"]["velocity"])
    obs.append(spine_observation["servo"]["right_wheel"]["velocity"])

    # IMU readings (quaternion)
    obs.extend(spine_observation["imu"]["orientation"]) # Check order (w component first or last)

    # Gyro readings
    obs.extend(spine_observation["imu"]["angular_velocity"])

    # Last action
    obs.extend(last_action)

    # Command
    obs.extend(command)

    # Debug
    # print("joint positions:", obs[0:4])
    # print("wheel velocities:", obs[4:6])
    print("IMU quaternion:", obs[6:10])
    print("gyro readings:", obs[10:13])
    print("last action:", obs[13:19])
    print("command:", obs[19:22])
    # print("------------------------")

    return {
        "obs": [
            np.array(obs, dtype=np.float32),
        ]
    }

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

        ort_sess = load_onnx_model("agents/default.onnx")

        # last_time = time.perf_counter()
        while True:
            print("------------------------")
            # current_time = time.perf_counter()
            # print(f"Time since last step: {current_time - last_time:.3f} seconds")
            # if current_time - last_time > 1.1 / frequency:
            #     print("Warning: Control loop is running slower than the target frequency!")
            # if current_time - last_time < 0.9 / frequency:
            #     print("Warning: Control loop is running faster than the target frequency!")
            # last_time = current_time

            lin_vel, ang_vel = get_command_from_controller(spine_observation)
            command = [lin_vel, 0.0, ang_vel]
            # print(f"Command: lin_vel={lin_vel:.2f} m/s, ang_vel={ang_vel:.2f} rad/s")

            print("Spine observation: ", spine_observation)

            for joint in ["left_hip", "left_knee", "right_hip", "right_knee"]:
                print(f"  {joint} position: {spine_observation["servo"][joint]["position"]:.2f}")
            for wheel in ["left_wheel", "right_wheel"]:
                print(f"  {wheel} velocity: {spine_observation["servo"][wheel]["velocity"]:.2f}")

            for wheel in ["left_wheel", "right_wheel"]:
                print(f"  {wheel} q_current: {spine_observation["servo"][wheel]["q_current"]:.2f}")

            observation = get_inputs(last_action, command, spine_observation)
            action = ort_sess.run(None, observation)[0][0]
            last_action = action.tolist()

            action = [0, 0, 0, 0, lin_vel, lin_vel] # for testing purposes
            action_dict = {
                "left_hip": create_servo_target(position=action[0], max_torque=16.0),
                "left_knee": create_servo_target(position=action[1], max_torque=16.0),
                "right_hip": create_servo_target(position=action[2], max_torque=16.0),
                "right_knee": create_servo_target(position=action[3], max_torque=16.0),
                "left_wheel": create_servo_target(velocity=action[4], max_torque=1.7),
                "right_wheel": create_servo_target(velocity=action[5], max_torque=1.7),
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
