"""Keyboard pose references in physical units, with XYZW quaternions."""

import math
from dataclasses import dataclass, field

import numpy as np

ENVIRONMENTS = {
    "Pool": "Wasman-Underwater-PressButton-Approach-T200-Pool-Direct",
    "PressButton": "Wasman-Underwater-PressButton-T200-Direct",
    "RotateValve": "Wasman-Underwater-RotateValve-T200-Direct",
    "OpenHatch": "Wasman-Underwater-OpenHatch-Direct",
    "CollectShell": "Wasman-Underwater-CollectShell-Direct",
    "PushSlider": "Wasman-Underwater-PushSlider-Marine-v1",
    "PullLever": "Wasman-Underwater-PullLever-Marine-v1",
}

CONTROLS = (
    "W/S forward/back | A/D left/right | R/F up/down\n"
    "Q/E turn left/right | T/G pitch up/down | Z/X roll\n"
    "Shift fast | Ctrl precise | Space hold position\n"
    "Backspace reset | C follow camera | Esc exit\n"
    "Arm locked. Release keys to hold. Use the English keyboard layout."
)


def quaternion_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product in XYZW order."""
    return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - a[:3] @ b[:3]]


def rotate_vector(q: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return vector + 2 * np.cross(q[:3], np.cross(q[:3], vector) + q[3] * vector)


@dataclass
class KeyboardState:
    """Held keys form a set: OS key repeat cannot increase a command."""

    held: set[str] = field(default_factory=set)
    commands: list[str] = field(default_factory=list)

    def event(self, key: str, pressed: bool) -> bool:
        key = key.upper()
        movement = {"W", "S", "A", "D", "R", "F", "Q", "E", "T", "G", "Z", "X"}
        modifiers = {"LEFT_SHIFT", "RIGHT_SHIFT", "LEFT_CONTROL", "RIGHT_CONTROL"}
        one_shot = {"SPACE": "hold", "BACKSPACE": "reset", "C": "camera", "ESCAPE": "quit"}
        if key not in movement | modifiers | one_shot.keys():
            return False
        if pressed:
            if key not in self.held and key in one_shot:
                self.commands.append(one_shot[key])
            self.held.add(key)
        else:
            self.held.discard(key)
        return True

    def clear(self):
        self.held.clear()
        self.commands.clear()

    def velocities(self, speed: float, turn_speed: float) -> tuple[np.ndarray, np.ndarray]:
        def axis(positive, negative):
            return float(positive in self.held) - float(negative in self.held)

        translation = np.array([axis("W", "S"), axis("A", "D"), axis("R", "F")])
        rotation = np.array([axis("Z", "X"), axis("G", "T"), axis("Q", "E")])
        translation /= max(1.0, np.linalg.norm(translation))
        rotation /= max(1.0, np.linalg.norm(rotation))
        scale = 2.0 if self.held & {"LEFT_SHIFT", "RIGHT_SHIFT"} else 1.0
        if self.held & {"LEFT_CONTROL", "RIGHT_CONTROL"}:
            scale = 0.25
        if "SPACE" in self.held:
            scale = 0.0
        return translation * speed * scale, rotation * turn_speed * scale


class PoseReference:
    """Smooth velocity commands with bounded positional and angular lead.

    R/F use world vertical; other axes follow the vehicle. Bounded lead prevents
    keys held at an obstacle from accumulating a distant position target.
    """

    def __init__(self, position, quaternion, max_lead=0.35, max_angle=0.45, smoothing=0.18):
        self.max_lead, self.max_angle, self.smoothing = max_lead, max_angle, smoothing
        self.hold(position, quaternion)

    def hold(self, position, quaternion):
        self.position = np.array(position, dtype=float, copy=True)
        self.quaternion = np.array(quaternion, dtype=float, copy=True)
        self.quaternion /= np.linalg.norm(self.quaternion)
        self.velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def advance(self, linear, angular, measured_position, measured_quaternion, dt):
        if dt <= 0:
            raise ValueError("dt must be positive")
        alpha = -math.expm1(-dt / self.smoothing)
        displacement = linear * dt + (self.velocity - linear) * self.smoothing * alpha
        rotation = angular * dt + (self.angular_velocity - angular) * self.smoothing * alpha
        self.velocity += alpha * (linear - self.velocity)
        self.angular_velocity += alpha * (angular - self.angular_velocity)
        world = rotate_vector(measured_quaternion, np.r_[displacement[:2], 0.0])
        world[2] += displacement[2]
        self.position += world
        error = self.position - measured_position
        self.position = measured_position + error * min(1.0, self.max_lead / max(1e-12, np.linalg.norm(error)))
        angle = np.linalg.norm(rotation)
        delta = np.r_[rotation * (math.sin(angle / 2) / angle if angle > 1e-12 else 0.5), math.cos(angle / 2)]
        self.quaternion = quaternion_product(self.quaternion, delta)
        self.quaternion /= np.linalg.norm(self.quaternion)
        relative = quaternion_product(np.r_[-measured_quaternion[:3], measured_quaternion[3]], self.quaternion)
        if relative[3] < 0:
            relative = -relative
        error_angle = 2 * math.atan2(np.linalg.norm(relative[:3]), relative[3])
        if error_angle > self.max_angle:
            limited = np.r_[
                relative[:3] / np.linalg.norm(relative[:3]) * math.sin(self.max_angle / 2), math.cos(self.max_angle / 2)
            ]
            self.quaternion = quaternion_product(measured_quaternion, limited)
        return self.position, self.quaternion
