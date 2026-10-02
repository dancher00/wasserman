from types import SimpleNamespace

import pytest
import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from wasman.controllers.ee_target import (
    ABSOLUTE_EE,
    RELATIVE_EE,
    EETargetInterface,
    MeasuredToolAnchor,
    decode_ee_target,
    encode_ee_target,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("frame", [ABSOLUTE_EE, RELATIVE_EE])
def test_pose_roundtrip_with_rotations_and_parallel_environment_origins(frame):
    torch.manual_seed(2130)
    origins = torch.randn(8, 3) * 20
    position = origins + torch.randn(8, 3)
    quaternion = quat_from_euler_xyz(*torch.randn(3, 8))
    jaw = torch.linspace(-1, 1, 8)
    anchor = MeasuredToolAnchor.capture(origins + torch.randn(8, 3), quat_from_euler_xyz(*torch.randn(3, 8)))
    encoded = encode_ee_target(position, quaternion, jaw, origins=origins, frame=frame, anchor=anchor)
    p, q, j = decode_ee_target(encoded, origins=origins, frame=frame, anchor=anchor)
    torch.testing.assert_close(p, position, atol=5e-6, rtol=1e-6)
    torch.testing.assert_close((q * quaternion).sum(-1).abs(), torch.ones(8))
    torch.testing.assert_close(j, jaw)


def test_relative_translation_is_tool_local_and_anchor_is_frozen():
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    q = quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor([torch.pi / 2]))
    anchor = MeasuredToolAnchor.capture(pos, q)
    command = torch.tensor([[1.0, 0, 0, 0, 0, 0, 1, -0.4]])
    pos.add_(100)  # Simulate an updated measurement after predicting a chunk.
    target, _, jaw = decode_ee_target(command, origins=torch.zeros(1, 3), frame=RELATIVE_EE, anchor=anchor)
    torch.testing.assert_close(target, torch.tensor([[1.0, 3.0, 3.0]]))
    torch.testing.assert_close(jaw, torch.tensor([-0.4]))


def test_interface_forwards_only_decoded_pose_and_absolute_jaw():
    # No scene object, task angle, contact, expert phase or success is available.
    env = SimpleNamespace(device="cpu", scene=SimpleNamespace(env_origins=torch.tensor([[10.0, 20.0, 0.0]])))
    interface = EETargetInterface(env)
    interface.controller = SimpleNamespace(actions=lambda p, q, jaw: (p, q, jaw))
    p, q, jaw = interface.actions(torch.tensor([[1.0, 2, 3, 0, 0, 0, -2, 1.5]]))
    torch.testing.assert_close(p, torch.tensor([[11.0, 22, 3]]))
    torch.testing.assert_close(quat_apply(q, torch.ones(1, 3)), torch.ones(1, 3))
    assert jaw.item() == 1


def test_invalid_commands_and_missing_anchor_are_rejected():
    origins = torch.zeros(1, 3)
    command = torch.tensor([[0.0, 0, 0, 0, 0, 0, 1, 0]])
    with pytest.raises(ValueError, match="anchor"):
        decode_ee_target(command, origins=origins, frame=RELATIVE_EE)
    with pytest.raises(ValueError, match="nonzero"):
        decode_ee_target(torch.zeros(1, 8), origins=origins, frame=ABSOLUTE_EE)
    with pytest.raises(ValueError, match="finite"):
        decode_ee_target(command * torch.nan, origins=origins, frame=ABSOLUTE_EE)
    with pytest.raises(ValueError, match="Unknown"):
        decode_ee_target(command, origins=origins, frame="typo")
