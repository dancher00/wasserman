from types import SimpleNamespace

import torch

from wasman.controllers.tool_pose import ToolPoseController, damped_step


def test_cartesian_damped_solver_handles_singularities():
    jacobian = torch.zeros(4, 6, 7)
    error = torch.ones(4, 6)
    assert torch.equal(damped_step(jacobian, error, torch.ones(7)), torch.zeros(4, 7))


def test_cartesian_damped_solver_reduces_reachable_error():
    torch.manual_seed(27)
    jacobian = torch.randn(8, 6, 7)
    error = torch.randn(8, 6)
    delta = damped_step(jacobian, error, torch.tensor([0.35, 0.35, 0.35, 1, 1, 1, 1]))
    residual = error - (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    assert (residual.norm(dim=-1) < error.norm(dim=-1) * 0.05).all()


def test_hatch_posture_is_opt_in_and_preserves_default_controller():
    env = SimpleNamespace(device="cpu")
    default = ToolPoseController(env, posture_gain=0.1)
    hatch = ToolPoseController(env, posture_gain=0.5, posture_joint_target=(3.14, -0.5, 2.64, 0.52))
    assert default.posture_joint_target is None
    assert torch.allclose(hatch.posture_joint_target, torch.tensor([3.14, -0.5, 2.64, 0.52]))
    assert torch.equal(default.max_speed, hatch.max_speed)
