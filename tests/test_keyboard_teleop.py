"""Keyboard behavior, reference frame and rate independence."""

import numpy as np

from wasman.teleoperation import KeyboardState, PoseReference


def test_repeat_release_opposing_keys_and_modifiers():
    keys = KeyboardState()
    for _ in range(20):
        keys.event("W", True)
    np.testing.assert_allclose(keys.velocities(0.2, 0.4)[0], [0.2, 0, 0])
    keys.event("S", True)
    np.testing.assert_allclose(keys.velocities(0.2, 0.4)[0], 0)
    keys.event("S", False)
    keys.event("LEFT_CONTROL", True)
    np.testing.assert_allclose(keys.velocities(0.2, 0.4)[0], [0.05, 0, 0])
    keys.event("W", False)
    np.testing.assert_allclose(keys.velocities(0.2, 0.4)[0], 0)
    for _ in range(20):
        keys.event("BACKSPACE", True)
    assert keys.commands == ["reset"]


def test_frame_rate_independence_and_release_hold():
    positions = []
    for hz in [30, 120]:
        reference = PoseReference([0, 0, 1], [0, 0, 0, 1], max_lead=100)
        for i in range(4 * hz):
            command = np.array([0.2 if i < 2 * hz else 0, 0, 0])
            reference.advance(command, np.zeros(3), reference.position.copy(), reference.quaternion.copy(), 1 / hz)
        positions.append(reference.position)
        assert np.linalg.norm(reference.velocity) < 4e-6
    np.testing.assert_allclose(positions[0], positions[1], atol=1e-10)


def test_body_frame_motion_and_world_vertical():
    yaw_90 = np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])
    reference = PoseReference([0, 0, 1], yaw_90, max_lead=10)
    reference.advance(np.array([1.0, 0, 1]), np.zeros(3), np.array([0, 0, 1]), yaw_90, 1)
    assert abs(reference.position[0]) < 1e-10
    assert reference.position[1] > 0.8
    assert reference.position[2] > 1.8


def test_obstacle_lead_is_bounded_and_space_holds_measured_pose():
    reference = PoseReference([0, 0, 1], [0, 0, 0, 1])
    for _ in range(600):
        reference.advance(np.ones(3), np.ones(3), np.array([0, 0, 1]), np.array([0, 0, 0, 1]), 1 / 30)
    assert np.linalg.norm(reference.position - [0, 0, 1]) <= 0.350001
    assert 2 * np.arccos(reference.quaternion[3]) <= 0.450001
    reference.hold([0.1, 0.2, 1.2], [0, 0, 0, 1])
    np.testing.assert_allclose(reference.velocity, 0)
    np.testing.assert_allclose(reference.angular_velocity, 0)
    np.testing.assert_allclose(reference.position, [0.1, 0.2, 1.2])
