"""Closing-force bias must not erase the parallel gripper's centering response."""
import torch

from wasman.controllers.bimanual_control import parallel_jaw_target


def test_same_closing_bias_and_restoring_differential_effort():
    q = torch.tensor([[.13, .13], [.14, .12], [.12, .14]], dtype=torch.float64)
    target = parallel_jaw_target(q, 4)
    torch.testing.assert_close(target, torch.full((3, 1), .12, dtype=q.dtype))
    effort = 400 * (target - q)
    # No increase in total commanded closing effort under opposite jaw motion.
    torch.testing.assert_close(effort.sum(-1), torch.full((3,), -8., dtype=q.dtype))
    # The differential effort restores equal aperture, in either direction.
    torch.testing.assert_close(effort[:, 0] - effort[:, 1], -400 * (q[:, 0] - q[:, 1]))
    assert torch.all((effort[1:, 0] - effort[1:, 1]) * (q[1:, 0] - q[1:, 1]) < 0)
