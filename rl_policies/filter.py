
from rl_policies.fit import sine_model


class IIRFilter:
    def __init__(self, cutoff_hz, sampling_hz, initial_value=0.0):
        dt = 1.0 / sampling_hz
        tau = 1.0 / (2 * 3.14159 * cutoff_hz)
        self.alpha = dt / (dt + tau)        
        self.last_output = initial_value

    def filter(self, current_input):
        output = self.alpha * current_input + (1.0 - self.alpha) * self.last_output
        self.last_output = output
        return output


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import os
    import argparse


    parser = argparse.ArgumentParser(description="Apply low-pass filters to wheel velocity data from RL agent logs.")
    parser.add_argument("--source", type=str, default="sim", help="Directory containing RL agent log files in JSON format.")
    args = parser.parse_args()

    for log in os.listdir(f"rl_policies/logs/{args.source}"):
        if log.endswith(".json"):
            with open(os.path.join(f"rl_policies/logs/{args.source}", log), "r") as f:
                data = json.load(f)

                if args.source == "sim":
                    timestamps = data["timestamps"]
                    right_wheel_vel = data["right_wheel"]["observation"]    
                    right_wheel_target_vel = None
                elif args.source == "robot":
                    timestamps = data["timestamp"]
                    right_wheel_vel = data["read_vel"]
                    right_wheel_target_vel = data["target_vel"] if data["target_vel"] else None

                filter_10 = IIRFilter(cutoff_hz=10, sampling_hz=50)
                filter_20 = IIRFilter(cutoff_hz=20, sampling_hz=50)
                filter_30 = IIRFilter(cutoff_hz=30, sampling_hz=50)

                right_wheel_vel_filtered_10 = [filter_10.filter(v) for v in right_wheel_vel]
                right_wheel_vel_filtered_20 = [filter_20.filter(v) for v in right_wheel_vel]
                right_wheel_vel_filtered_30 = [filter_30.filter(v) for v in right_wheel_vel]
                
                plt.figure(figsize=(12, 6))
                plt.plot(timestamps, right_wheel_vel, label="Right Wheel Velocity")
                plt.plot(timestamps, right_wheel_vel_filtered_10, label="Right Wheel Velocity (Filter 10 Hz)")
                plt.plot(timestamps, right_wheel_vel_filtered_20, label="Right Wheel Velocity (Filter 20 Hz)")
                plt.plot(timestamps, right_wheel_vel_filtered_30, label="Right Wheel Velocity (Filter 30 Hz)")
                if right_wheel_target_vel is not None:
                    plt.plot(timestamps, right_wheel_target_vel, label="Right Wheel Target Velocity", linestyle="--")
                plt.xlabel("Time [s]")
                plt.ylabel("Velocity [rad/s]")
                plt.title(f"Right Wheel Velocity from {log}")
                plt.legend()
                plt.show()