"""Verify physical water changes and local, depth-dependent optical path lengths."""

from types import SimpleNamespace

import pytest
import torch

from wasman.physics.hydrodynamics import BatchedHydrodynamics, LinkHydrodynamics
from wasman.teleop_environment import WaterControls, cloud_path_length


def environment():
    link = LinkHydrodynamics("test", 0.01, (0, 0, 0), (0,) * 6, (2,) * 6, (3,) * 6)
    hydro = BatchedHydrodynamics([link], 1, 0.01, "cpu", water_density=1000)
    return SimpleNamespace(
        _hydrodynamics=hydro,
        cfg=SimpleNamespace(water_density=1000, turbulence_sigma=0.1),
        device="cpu",
        _mean_current_w=torch.tensor([[0.1, 0.2, 0.03]]),
        _turbulent_current_w=torch.ones(1, 3),
    )


def test_water_changes_real_force_and_buoyancy_without_compounding():
    env = environment()
    controls = WaterControls(env)
    velocity = torch.tensor([[[1.0, 0, 0, 0, 0, 0]]])
    identity = torch.tensor([[[0.0, 0, 0, 1]]])
    initial, _, _ = env._hydrodynamics.compute(velocity, identity)
    torch.testing.assert_close(initial, torch.tensor([[[-5.0, 0, 98.1]]]))
    controls.set(density=1100, viscosity=2, resistance=0.5)
    controls.apply()
    changed, _, _ = env._hydrodynamics.compute(velocity, identity)
    torch.testing.assert_close(changed, torch.tensor([[[-3.65, 0, 107.91]]]))
    controls.apply(force=True)
    repeated, _, _ = env._hydrodynamics.compute(velocity, identity)
    torch.testing.assert_close(repeated, changed)


def test_exact_current_actions_and_persistence_after_scene_reset():
    env = environment()
    controls = WaterControls(env)
    controls.set(current=(0.3, -0.2, 0.1), density=1050, viscosity=1.5)
    controls.rotate_current()
    controls.reverse_current()
    controls.apply()
    torch.testing.assert_close(env._mean_current_w, torch.tensor([[-0.2, -0.3, -0.1]]))
    torch.testing.assert_close(env._turbulent_current_w, torch.zeros(1, 3))
    env._mean_current_w.fill_(0.5)  # A native scene reset randomizes this buffer.
    controls.apply(force=True)
    torch.testing.assert_close(env._mean_current_w, torch.tensor([[-0.2, -0.3, -0.1]]))
    assert env.cfg.water_density == 1050 and env.cfg.turbulence_sigma == 0
    controls.calm()
    controls.apply()
    torch.testing.assert_close(env._mean_current_w, torch.zeros(1, 3))
    controls.restore_defaults()
    controls.apply()
    assert controls.settings.density == 1000 and controls.settings.viscosity == 1


def test_cloud_path_respects_mount_offset_depth_and_finite_local_radius():
    rays = torch.tensor([[[1.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0]]])
    depth = torch.tensor([[[float("inf")], [float("inf")], [0.3]]])
    path = cloud_path_length(depth, rays, torch.tensor([0.2, 0, 0]), 1)
    torch.testing.assert_close(path, torch.tensor([[[0.8], [1.2], [0.3]]]))
    outside = cloud_path_length(
        torch.tensor([[[0.5], [2.0], [float("inf")]]]),
        torch.tensor([[[-1.0, 0, 0]]]).expand(1, 3, 3),
        torch.tensor([2.0, 0, 0]),
        1,
    )
    torch.testing.assert_close(outside, torch.tensor([[[0.0], [1.0], [2.0]]]))


def test_invalid_live_setting_does_not_change_existing_water():
    controls = WaterControls(environment())
    old = controls.metadata()
    with pytest.raises(ValueError):
        controls.set(density=float("nan"))
    assert controls.metadata() == old
