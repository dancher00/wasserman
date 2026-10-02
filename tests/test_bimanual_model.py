"""Independent physical/kinematic contracts for the experimental twin-arm asset."""

import json
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

from wasman.controllers.bimanual_kinematics import BimanualKinematics
from wasman.controllers.bimanual_turn_plan import two_hand_plan
from wasman.physics.rexrov2 import load_parameters
from wasman.physics.rexrov2_bimanual import load_bimanual_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_mass_buoyancy_and_actuation_are_not_artificially_compensated():
    original = load_parameters()
    dual = load_bimanual_parameters()
    assert np.isclose(dual.mass.sum(), original.mass[0] + 2 * original.mass[1:].sum())
    assert np.isclose(dual.volume.sum(), original.volume[0] + 2 * original.volume[1:].sum())
    np.testing.assert_array_equal(dual.allocation, original.allocation)
    assert dual.max_thrust == original.max_thrust
    assert np.linalg.eigvalsh(dual.rigid_composite + dual.added_mass).min() > 0


def test_two_hand_trajectory_is_continuous_and_keeps_opposite_grip_closed():
    previous = None
    for t in np.arange(0, 180, 0.02):
        poses, rot, grip, stage = two_hand_plan(t)
        for side in poses:
            if previous:
                assert np.linalg.norm(poses[side] - previous[0][side]) / 0.02 < 0.10
                speed = Rotation.from_matrix(previous[1][side].T @ rot[side]).magnitude() / 0.02
                assert speed < 0.15
        if stage == "regrasp-left":
            assert grip["right"] == 0
        if stage == "regrasp-right":
            assert grip["left"] == 0
        previous = (poses, rot)


def test_both_arm_jacobians_match_finite_differences_without_cross_talk():
    w = BimanualKinematics()
    q = np.zeros(w.model.nq)
    pose = json.loads((ROOT / "configs/studies/rex-centered-button-pose.json").read_text())["targets"]["wall"]["q"]
    for side in w.indices:
        q[w.indices[side]] = pose
    for side, f in w.frames.items():
        analytic = pin.computeFrameJacobian(w.model, w.data, q, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
        pin.framesForwardKinematics(w.model, w.data, q)
        offset = w.data.oMf[f].rotation @ np.array([0.16, 0, 0])
        analytic[:3] += np.cross(analytic[3:].T, offset).T
        p0 = w.poses(q)[side][0]
        for j in range(w.model.nq):
            shifted = q.copy()
            shifted[j] += 1e-6
            finite = (w.poses(shifted)[side][0] - p0) / 1e-6
            np.testing.assert_allclose(analytic[:3, j], finite, atol=1e-6)


def test_regrasp_waits_for_physical_clearance_independently_per_environment():
    from wasman.controllers.bimanual_turn_plan import RELEASE_JAW, TURN_END, ContactPlanClock

    clock = ContactPlanClock(3, "two-hands")

    def readiness(env, boundary, kind, sides):
        return not (env == 0 and abs(boundary - (TURN_END + 8)) < 1e-6)

    for _ in range(70 * 30):
        clock.advance(readiness)
    assert clock.waiting[0]
    assert TURN_END + 7.99 < clock.time[0] < TURN_END + 8
    assert clock.time[1] > clock.time[0] + 15
    positions, rotations, grip, stage = two_hand_plan(clock.time[0])
    assert stage == "regrasp-left" and grip["right"] == 0 and grip["left"] == RELEASE_JAW
    assert np.isclose(positions["left"][0], 2.9)
    # A stalled physical clearance must prevent the return arc, without
    # freezing the independently simulated neighbour's expert.
    for _ in range(30):
        clock.advance(lambda *unused: True)
    assert clock.time[0] > TURN_END + 8.8


def test_ik_accepts_recorded_float32_seeds_without_zero_numerical_gradients():
    w = BimanualKinematics()
    q = np.zeros(w.model.nq, dtype=np.float32)
    pose = json.loads((ROOT / "configs/studies/rex-centered-button-pose.json").read_text())["targets"]["wall"]["q"]
    for side in w.indices:
        q[w.indices[side]] = pose
    position, rotation = w.poses(q)["right"]
    target = position + np.array([0.005, 0, 0])
    solved, error = w.solve_arm("right", target, rotation, q)
    assert solved.dtype == np.float64 and error < 1e-5
    np.testing.assert_allclose(w.poses(solved)["right"][0], target, atol=1e-5)


def test_level_frame_feedback_does_not_turn_base_motion_into_arm_deflection():
    """Rigidly moving the whole assembly must not change level-frame arm commands."""
    from types import SimpleNamespace as NS
    import torch
    from wasman.controllers.bimanual_control import BimanualController

    w = BimanualKinematics()
    q = np.zeros(w.model.nq)
    pose = json.loads((ROOT / "configs/studies/rex-centered-button-pose.json").read_text())["targets"]["wall"]["q"]
    for side in w.indices:
        q[w.indices[side]] = pose
    names = [w.model.names[j] for j in range(1, w.model.njoints)]
    nominal_base = np.array([.7, 0, 1.5])
    goals = {s: torch.tensor(np.tile(nominal_base + p + [0, 0, .005], (2, 1)), dtype=torch.float32)
             for s, (p, _) in w.poses(q).items()}
    rotations = {s: r.copy() for s, (_, r) in w.poses(q).items()}
    controllers = []
    for rotation, translation in [(np.eye(3), np.zeros(3)),
                                  (Rotation.from_euler('xyz', [.1, -.07, .03]).as_matrix(), np.array([.03, -.02, .01]))]:
        bp = nominal_base + translation
        positions = [bp]
        orientations = [Rotation.from_matrix(rotation).as_quat()]
        jacobians = [np.zeros((6, 6+w.model.nv))]
        for side in ['left', 'right']:
            J = pin.computeFrameJacobian(w.model, w.data, q, w.frames[side], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
            pin.framesForwardKinematics(w.model, w.data, q)
            T = w.data.oMf[w.frames[side]]
            positions.append(bp + rotation @ T.translation)
            orientations.append(Rotation.from_matrix(rotation @ T.rotation).as_quat())
            full = np.zeros((6, 6+w.model.nv))
            full[:3, 6:] = rotation @ J[:3]
            full[3:, 6:] = rotation @ J[3:]
            jacobians.append(full)
        def batch(value):
            return NS(torch=torch.tensor(np.stack([value, value]), dtype=torch.float32))
        data = NS(body_link_pos_w=batch(np.array(positions)), body_link_quat_w=batch(np.array(orientations)),
                  body_link_jacobian_w=batch(np.array(jacobians)), joint_pos=batch(q), joint_vel=batch(np.zeros_like(q)),
                  soft_joint_pos_limits=batch(np.stack([w.low, w.high], axis=-1)))
        robot = NS(device='cpu', data=data, joint_names=names)
        controllers.append(BimanualController(robot, 0, {'left':1, 'right':2},
                           torch.tensor(np.tile(nominal_base,(2,1)),dtype=torch.float32),
                           data.joint_pos.torch.clone(), 'two-hands', feedback_frame='level'))
    for _ in range(3):
        outputs = [c.targets(0, goals, rotations, {'left':.5, 'right':.5}) for c in controllers]
        torch.testing.assert_close(outputs[0][1], outputs[1][1], rtol=0, atol=2e-5)


def test_hydrostatic_holding_effort_matches_potential_energy_gradient():
    w = BimanualKinematics()
    p = load_bimanual_parameters()
    q = np.zeros(w.model.nq)
    pose = json.loads((ROOT / 'configs/studies/rex-centered-button-pose.json').read_text())['targets']['wall']['q']
    for side in w.indices:
        q[w.indices[side]] = pose
    def potential(position):
        pin.framesForwardKinematics(w.model, w.data, position)
        energy = 0.
        for i, name in enumerate(p.names):
            T = w.data.oMf[w.model.getFrameId(name)]
            com = T.translation + T.rotation @ p.com[i]
            cob = T.translation + T.rotation @ p.cob[i]
            energy += p.gravity * (p.mass[i] * com[2] - p.water_density * p.volume[i] * cob[2])
        return energy
    effort = w.hydrostatic_holding_torque(q, p)
    eps = 1e-6
    gradient = []
    for j in range(w.model.nq):
        delta = np.zeros(w.model.nq);delta[j] = eps
        gradient.append((potential(q + delta) - potential(q - delta)) / (2 * eps))
    np.testing.assert_allclose(effort, gradient, atol=1e-4)
