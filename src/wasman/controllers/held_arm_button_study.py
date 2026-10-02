"""Frozen robot-independent reset/trajectory specification for the embodiment study."""
import numpy as np

FIXTURE_POSITION = (3.0, 0.0, 1.1)
START_STANDOFF = 0.12
APPROACH_DISTANCE = 0.10
SETTLE_SECONDS = 5.0
APPROACH_SECONDS = 8.0
HORIZON_SECONDS = 20.0
CONTROL_HZ = 30
PHYSICS_HZ = 240


def reset_offsets(seeds):
    """Independent equal physical displacement per paired seed, in metres."""
    return np.array([np.random.default_rng(seed).uniform(-0.01, 0.01, 3) for seed in seeds])


def approach_fraction(time_s):
    return min(1.0, max(0.0, (time_s - SETTLE_SECONDS) / APPROACH_SECONDS))
