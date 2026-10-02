"""Camera-only staging for a continuous swim-in shot (never moves the robot)."""

import math


def approach_camera(time_s: float):
    """Wide establishing view to three-quarter contact view over 16 seconds."""
    t = min(1.0, max(0.0, time_s / 16.0))
    # Quintic easing: zero velocity and acceleration at either end of the dolly.
    ease = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    target = (0.02 + 0.46 * ease, -0.04, 0.76)
    radius = 2.65 - 0.90 * ease
    angle = math.radians(-110.0 - 15.0 * ease)
    eye = (
        target[0] + radius * math.cos(angle),
        target[1] + radius * math.sin(angle),
        1.65 - 0.20 * ease,
    )
    return eye, target


def valve_camera(time_s: float):
    """Establish the vehicle, then settle on the real finger/wheel contact."""
    t = min(1.0, max(0.0, time_s / 18.0))
    ease = t**3 * (10.0 - 15.0 * t + 6.0 * t**2)
    wide_eye, close_eye = (-1.05, -1.80, 1.65), (0.02, -0.92, 1.18)
    wide_target, close_target = (0.18, -0.04, 0.79), (0.51, -0.05, 0.79)
    eye = tuple(a + (b - a) * ease for a, b in zip(wide_eye, close_eye, strict=True))
    target = tuple(a + (b - a) * ease for a, b in zip(wide_target, close_target, strict=True))
    return eye, target


def hatch_camera(time_s: float):
    """Continuous orbit from folded swim-in to the lid's handle-facing side."""
    t = min(1.0, max(0.0, time_s / 26.0))
    ease = t**3 * (10 - 15 * t + 6 * t**2)
    target = (0.15 + 0.50 * ease, 0.0, 0.30 + 0.05 * ease)
    radius = 2.6 - 0.85 * ease
    angle = math.radians(-125 + 70 * ease)
    return (
        target[0] + radius * math.cos(angle),
        target[1] + radius * math.sin(angle),
        1.65 - 0.50 * ease,
    ), target
