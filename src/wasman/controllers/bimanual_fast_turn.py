"""One rapid 90-degree command after the v1 contact acquisition sequence.

This separate demonstration changes the requested motion, not physical limits.
It is not a scored completion of the original two-turn 170-degree benchmark.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from wasman.controllers.bimanual_turn_plan import (
    BASE_POSITION, OPEN_JAW, RELEASE_JAW, RADIUS, ACQUISITION_ROLL,
    ContactPlanClock as BaseClock, mode_plan as acquisition_plan,
)

TURN_SPEED = 1.4  # rad/s requested; 20 times the original 0.07 rad/s.
TURN_END = 18 + np.pi / 2 / TURN_SPEED
FINISH_OPEN = TURN_END + 8
FINISH_RETREAT = FINISH_OPEN + 4
FINISH_END = FINISH_RETREAT + 7


def mode_plan(t, mode):
    positions, rotations, grip, stage = acquisition_plan(min(t, 18), mode)
    if t >= 18:
        angle = min(np.pi / 2, (t - 18) * TURN_SPEED)
        rotation = Rotation.from_rotvec([angle + ACQUISITION_ROLL, 0, 0]).as_matrix()
        for side, sign in [('left', 1), ('right', -1)]:
            if side == 'left' and mode != 'two-hands':
                continue
            positions[side] = np.array([3., 0., 1.1]) + rotation @ np.array([0., sign * RADIUS, 0.])
            rotations[side] = rotation
        stage = 'turn1' if t < TURN_END else 'settle'
    if t >= FINISH_OPEN:
        grip = {s: RELEASE_JAW for s in grip}
        stage = 'release'
    if t >= FINISH_RETREAT:
        for side in positions:
            if side == 'left' and mode == 'free':
                continue
            positions[side][0] -= .1 * np.clip((t - FINISH_RETREAT) / 7, 0, 1)
        stage = 'retreat' if t < FINISH_END else 'finished'
    return positions, rotations, grip, stage


def two_hand_plan(t):
    return mode_plan(t, 'two-hands')


class ContactPlanClock(BaseClock):
    def __init__(self, num_envs, mode):
        super().__init__(num_envs, mode)
        active = ['left', 'right'] if mode == 'two-hands' else ['right']
        # Preserve physical acquisition guards. The rapid request subsequently
        # runs on a clock, allowing tracking failure and slip to remain visible.
        self.events = [(12., 'pose', active), (18., 'closed', active),
                       (FINISH_RETREAT, 'open', active)]
