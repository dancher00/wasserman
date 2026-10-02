"""A mean-point approximation must not erase a pure contact-force couple."""

import numpy as np
import torch
import warp as wp
from types import SimpleNamespace

from wasman.physics.contact_wrench import sum_contact_wrench
from wasman.physics.contact_wrench import ResolvedContactWrench


def test_contact_couple_and_reference_translation():
    wp.init()
    def array(value, dtype):
        return wp.array(np.asarray(value), dtype=dtype, device="cpu")
    normal = array([1, 1], wp.float32)
    points = array([[0, 1, 0], [0, -1, 0]], wp.vec3f)
    directions = array([[1, 0, 0], [-1, 0, 0]], wp.vec3f)
    count = array([[2]], wp.uint32)
    start = array([[0]], wp.uint32)
    friction = array([[0, 1, 0]], wp.vec3f)
    friction_point = array([[0, 0, 1]], wp.vec3f)
    friction_count = array([[1]], wp.uint32)
    force = wp.zeros((1, 1), dtype=wp.vec3f, device="cpu")
    torque = wp.zeros((1, 1), dtype=wp.vec3f, device="cpu")
    for reference, expected in [([0, 0, 0], [-1, 0, -2]), ([1, 2, 3], [2, 0, -3])]:
        wp.launch(sum_contact_wrench, dim=(1, 1), inputs=[
            normal, points, directions, count, start, friction, friction_point,
            friction_count, start, array([reference], wp.vec3f),
        ], outputs=[force, torque], device="cpu")
        np.testing.assert_allclose(force.numpy()[0, 0], [0, 1, 0], atol=1e-7)
        np.testing.assert_allclose(torque.numpy()[0, 0], expected, atol=1e-7)


def test_physx_shaped_buffers_and_shared_count_storage():
    def array(value, dtype=wp.float32):
        return wp.array(np.asarray(value), dtype=dtype, device="cpu")
    # Public PhysX arrays have scalar dtype and a trailing vector dimension.
    normal = array([[1], [1]])
    points = array([[0, 1, 0], [0, -1, 0]])
    directions = array([[1, 0, 0], [-1, 0, 0]])
    shared_counts = array([[2]], wp.uint32)
    staged_counts = wp.clone(shared_counts)
    starts = array([[0]], wp.uint32)
    friction = array([[0, 1, 0]])
    friction_point = array([[0, 0, 1]])
    def normal_data(**kwargs):
        shared_counts.fill_(2)
        return normal, points, directions, array([[0], [0]]), shared_counts, starts
    def friction_data(**kwargs):
        shared_counts.fill_(1)  # Backend reuses and overwrites this storage.
        return friction, friction_point, shared_counts, starts
    sensor = SimpleNamespace(
        data=None, _sim_physics_dt=1/240,
        contact_view=SimpleNamespace(get_contact_data=normal_data, get_friction_data=friction_data),
        _contact_counts=staged_counts, _contact_start_indices=starts,
        _friction_counts=shared_counts, _friction_start_indices=starts,
    )
    reader = ResolvedContactWrench(sensor)
    force, torque = reader.evaluate(torch.zeros(1, 3))
    torch.testing.assert_close(force[0, 0], torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(torque[0, 0], torch.tensor([-1., 0., -2.]))
    # Existing views must observe the in-place refresh performed by PhysX.
    normal.fill_(2)
    _, torque = reader.evaluate(torch.zeros(1, 3))
    torch.testing.assert_close(torque[0, 0], torch.tensor([-1., 0., -4.]))
