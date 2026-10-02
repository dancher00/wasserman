"""Two 90-degree rim turns separated by physical release/regrasp commands."""

import numpy as np
from scipy.spatial.transform import Rotation

RADIUS = 0.059 * 4
OPEN_JAW = 0.5
RELEASE_JAW = 0.25
ACQUISITION_ROLL = -np.pi / 8  # Halfway between the native eight spokes.
BASE_POSITION = np.array([0.7, 0.0, 1.5])
TURN_END = 18 + np.pi / 2 / 0.07
REGRASP_DURATION = 36.0
REGRASP_RETRACTION = 0.10
TURN_FINISH = TURN_END + 2 * REGRASP_DURATION + np.pi / 2 / 0.07
FINISH_OPEN = TURN_FINISH + 2
FINISH_RETREAT = FINISH_OPEN + 4
FINISH_END = FINISH_RETREAT + 7


def two_hand_plan(t):
    # Settle 5 s, approach 7 s, close 6 s, turn at <=0.07 rad/s.
    # During each regrasp the opposite hand remains closed at its last pose.
    angles = {"left": 0.0, "right": 0.0}
    back = {"left": 0.0, "right": 0.0}
    grip = {"left": OPEN_JAW, "right": OPEN_JAW}
    stage = "approach"
    approach = np.clip((t - 5) / 7, 0, 1)
    for s in back:
        back[s] = 0.10 * (1 - approach)
    if t >= 12:
        grip = {s: 0.0 for s in grip}
        stage = "close"
    if t >= 18:
        a = min(np.pi / 2, (t - 18) * 0.07)
        angles = {s: a for s in angles}
        stage = "turn1"
    end = TURN_END
    for side, other, start in [("left", "right", end), ("right", "left", end + REGRASP_DURATION)]:
        if start <= t < start + REGRASP_DURATION:
            u = t - start
            angles[other] = np.pi / 2 if side == "left" else 0.0
            angles[side] = np.pi / 2 * (1 - np.clip((u - 8) / 20, 0, 1))
            back[side] = REGRASP_RETRACTION * min(1.0, max(0.0, (u - 4) / 4), max(0.0, (32 - u) / 4))
            grip[side] = RELEASE_JAW if u < 32 else 0.0
            stage = "regrasp-" + side
    if t >= end + 2 * REGRASP_DURATION:
        a = min(np.pi / 2, (t - end - 2 * REGRASP_DURATION) * 0.07)
        angles = {s: a for s in angles}
        stage = "turn2"
    if t >= TURN_FINISH:
        stage = "settle"
    if t >= FINISH_OPEN:
        grip = {s: RELEASE_JAW for s in grip}
        stage = "release"
    if t >= FINISH_RETREAT:
        back = {s: 0.1 * np.clip((t - FINISH_RETREAT) / 7, 0, 1) for s in back}
        stage = "retreat" if t < FINISH_END else "finished"
    positions = {}
    rotations = {}
    for side, sign in [("left", 1), ("right", -1)]:
        rot = Rotation.from_rotvec([angles[side] + ACQUISITION_ROLL, 0, 0]).as_matrix()
        positions[side] = np.array([3.0 - back[side], 0.0, 1.1]) + rot @ np.array([0.0, sign * RADIUS, 0.0])
        rotations[side] = rot
    return positions, rotations, grip, stage


def mode_plan(t, mode):
    positions, rotations, grip, stage = two_hand_plan(t)
    if mode != "two-hands":
        positions["left"] = np.array(
            [3.0 - 0.1 * (1 - np.clip((t - 5) / 7, 0, 1)) if mode == "support" else 2.9, 0.5, 1.1]
        )
        rotations["left"] = np.eye(3)
        grip["left"] = OPEN_JAW if mode == "free" or t < 12 else 0.0
        if t >= FINISH_OPEN:
            grip["left"] = RELEASE_JAW
        if t >= FINISH_RETREAT and mode == "support":
            positions["left"][0] = 3 - 0.1 * np.clip((t - FINISH_RETREAT) / 7, 0, 1)
    return positions, rotations, grip, stage


class ContactPlanClock:
    """Per-environment pauses at physical acquisition and regrasp boundaries.

    `ready` checks measured pose, jaw opening or opposing contact. The motor-off
    control advances without these guards; it never receives arm motion commands.
    """

    def __init__(self, num_envs, mode):
        # Free/support use the identical right-arm feedback law. The left rail
        # contact is measured as an outcome, never used to delay the right arm.
        active = ["left", "right"] if mode == "two-hands" else ["right"]
        self.events = [(12.0, "pose", active), (18.0, "closed", active), (TURN_END, "pose", active)]
        for side, start in [("left", TURN_END), ("right", TURN_END + REGRASP_DURATION)]:
            if side == "left" and mode != "two-hands":
                continue
            self.events.extend(
                [
                    (start + 4, "open", [side]),
                    (start + 8, "pose", [side]),
                    (start + 28, "pose", [side]),
                    (start + 32, "pose", [side]),
                    (start + 36, "closed", [side]),
                ]
            )
        self.events.extend([(FINISH_OPEN, "pose", active), (FINISH_RETREAT, "open", active)])
        self.time = np.zeros(num_envs)
        self.event = np.zeros(num_envs, dtype=int)
        self.stable = np.zeros(num_envs, dtype=int)
        self.waiting = np.zeros(num_envs, dtype=bool)

    def advance(self, ready, dt=1 / 30):
        for n in range(len(self.time)):
            self.waiting[n] = False
            candidate = self.time[n] + dt
            if self.event[n] < len(self.events):
                boundary, kind, sides = self.events[self.event[n]]
                if candidate >= boundary:
                    ok = n == len(self.time) - 1 or ready(n, boundary, kind, sides)
                    self.stable[n] = self.stable[n] + 1 if ok else 0
                    if self.stable[n] < 3:
                        candidate = boundary - 1e-6
                        self.waiting[n] = True
                    else:
                        candidate = boundary
                        self.event[n] += 1
                        self.stable[n] = 0
            self.time[n] = candidate
        return self.time.copy()
