import numpy as np
import random


def get_position_trajectory(duration: float, 
                            framerate: float, 
                            max_angle: float = 1.2,
                            nb_sinusoids: int = 3,
                            min_amplitude: float = 0.1, 
                            max_amplitude: float = 1.0, 
                            min_frequency: float = 0.1, 
                            max_frequency: float = 1.0, 
                            hold_duration: float = 0.0) -> np.ndarray:
    """
    Generate a random trajectory for the position-controlled joints.
    """
    actions = {
        "left_hip": [0.0],
        "left_knee": [0.0],
        "right_hip": [0.0],
        "right_knee": [0.0],
    }

    # Generate random sinusoids for each joint and sum them up to create the trajectory
    nb_ts = int(duration * framerate)
    for joint in actions:
        params = []
        for _ in range(nb_sinusoids):
            frequency = random.uniform(min_frequency, max_frequency)
            amplitude = random.uniform(min_amplitude, max_amplitude)
            params.append((frequency, amplitude))
        
        for _ in range(nb_ts):
            t = len(actions[joint]) / framerate
            value = 0.0
            for freq, amp in params:
                value += np.sin(2 * np.pi * freq * t) * amp
            value = np.clip(value, -max_angle, max_angle)
            actions[joint].append(value)
        
    # Add hold at the end of the trajectory
    if hold_duration > 0.0:
        hold_steps = int(hold_duration * framerate)
        for joint in actions:
            last_value = actions[joint][-1]
            actions[joint].extend([last_value] * hold_steps)
            
    return actions


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    duration = 6.0
    framerate = 50.0
    trajectory = get_position_trajectory(duration, framerate, hold_duration=1.0)
    t = np.arange(0, duration + 1/framerate + 1.0, 1/framerate)
    for joint, values in trajectory.items():
        plt.plot(t, values, label=joint)
    plt.legend()
    plt.show()