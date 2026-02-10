#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Marc Duclusaud


import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.optimize import curve_fit


def sine_model(t, period, amplitude, phase, offset):
    """Sinusoidal model for curve fitting."""
    return amplitude * np.sin(2 * np.pi * t / period + phase) + offset


parser = argparse.ArgumentParser(description="Apply low-pass filters to wheel velocity data from RL agent logs.")
parser.add_argument("--path", type=str, default="logs/robot/position_trajectory.json", help="Path to the JSON file containing the recorded trajectories.")
parser.add_argument("--period", type=float, default=2.0, help="Initial guess for the period of the sinusoidal trajectory.")
parser.add_argument("--amplitude", type=float, default=0.3, help="Initial guess for the amplitude of the sinusoidal trajectory.")
args = parser.parse_args()

with open(args.path, "r") as f:
    data = json.load(f)

    framerate = 50.0
    quarter_period = args.period / 4.0
    quarter_period_ts = int(quarter_period * framerate)

    timestamps = np.array(data["timestamp"][quarter_period_ts:-quarter_period_ts]) - data["timestamp"][quarter_period_ts]
    read = np.array(data["read"][quarter_period_ts:-quarter_period_ts])
    target = np.array(data["target"][quarter_period_ts:-quarter_period_ts])

    # Sinus fitting
    popt, pcov = curve_fit(sine_model, timestamps, read, 
                        p0=[args.period, args.amplitude, np.pi/2, 0.0],
                        maxfev=5000)

    period, amplitude, phase, offset = popt
    fitted = sine_model(timestamps, *popt)

    print(f"Period: {period:.3f} s")
    print(f"Amplitude: {amplitude:.3f}")
    print(f"Phase: {phase:.3f} rad")
    print(f"Offset: {offset:.3f}")

    # Measuring delay at 0 crossing
    fitted_zero_crossings = np.where(np.diff(np.sign(fitted - offset)))[0]
    target_zero_crossings = np.where(np.diff(np.sign(target)))[0]
    delays = fitted_zero_crossings - target_zero_crossings
    average_delay = np.mean(delays) / framerate
    print(f"Average delay: {average_delay*1000:.1f} ms")
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, read, label="Read Value")
    plt.plot(timestamps, fitted - offset, label="Fitted Sinusoid (without offset)", linestyle="--")
    plt.plot(timestamps, target, label="Target Value", linestyle=":")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.title(f"Data from {args.path}")
    plt.legend()
    plt.show()