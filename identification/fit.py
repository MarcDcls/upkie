import json
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from scipy.optimize import curve_fit


def sine_model(t, period, amplitude, phase, offset):
    """Sinusoidal model for curve fitting."""
    return amplitude * np.sin(2 * np.pi * t / period + phase) + offset


parser = argparse.ArgumentParser(description="Apply low-pass filters to wheel velocity data from RL agent logs.")
parser.add_argument("--path", type=str, default="logs/robot/servo_trajectories.json", help="Path to the JSON file containing the recorded trajectories.")
parser.add_argument("--period", type=float, default=4.0, help="Initial guess for the period of the sinusoidal trajectory.")
parser.add_argument("--amplitude", type=float, default=6.0, help="Initial guess for the amplitude of the sinusoidal trajectory.")
args = parser.parse_args()

for log in os.listdir(f"logs/{args.source}"):
    if log.endswith(".json"):
        with open(os.path.join(f"logs/{args.source}", log), "r") as f:
            data = json.load(f)

            timestamps = np.array(data["timestamp"])
            read = np.array(data["read"])
            target = np.array(data["target"])

            # Sinus fitting
            popt, pcov = curve_fit(sine_model, timestamps, read, 
                                p0=[args.period, args.amplitude, 0.0, 0.0],
                                maxfev=5000)

            period, amplitude, phase, offset = popt
            fitted = sine_model(timestamps, *popt)

            print(f"Period: {period:.3f} s")
            print(f"Amplitude: {amplitude:.3f}")
            print(f"Phase: {phase:.3f} rad")
            print(f"Offset: {offset:.3f}")
            
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, read, label="Read Value")
            plt.plot(timestamps, fitted, label="Fitted Sinusoid", linestyle="--")
            plt.plot(timestamps, target, label="Target Value", linestyle=":")
            plt.xlabel("Time [s]")
            plt.ylabel("Value")
            plt.title(f"Data from {args.path}")
            plt.legend()
            plt.show()