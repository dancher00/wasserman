"""Task-neutral EE pose + jaw interface over the existing robot IK controller.

Positions are meters, orientations are XYZW unit quaternions, jaw is an absolute
normalized actuator command. Relative targets use a frozen *measured* tool pose
at prediction time, never the previous command. No task phase/contact/success
signal is read here.
"""

from dataclasses import dataclass

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

from .tool_pose import ToolPoseController

ABSOLUTE_EE = "ee-env-local-pose-xyzw-v1"
RELATIVE_EE = "ee-measured-local-pose-xyzw-v1"


@dataclass(frozen=True)
class MeasuredToolAnchor:
    position_w: torch.Tensor
    quaternion_w: torch.Tensor

    @classmethod
    def capture(cls, position_w, quaternion_w):
        """Own a snapshot so later simulator buffer updates cannot move the anchor."""
        return cls(position_w.clone(), quaternion_w.clone())


def unit_quaternion(quaternion):
    norm = quaternion.norm(dim=-1, keepdim=True)
    if not torch.isfinite(quaternion).all() or (norm < 1e-8).any():
        raise ValueError("EE orientation must be finite and nonzero")
    return quaternion / norm


def encode_ee_target(position_w, quaternion_w, jaw, *, origins, frame, anchor=None):
    quaternion_w = unit_quaternion(quaternion_w)
    if frame == ABSOLUTE_EE:
        position, quaternion = position_w - origins, quaternion_w
    elif frame == RELATIVE_EE:
        if anchor is None:
            raise ValueError("Relative EE targets require a measured anchor")
        orientation = unit_quaternion(anchor.quaternion_w)
        position = quat_apply_inverse(orientation, position_w - anchor.position_w)
        quaternion = quat_mul(quat_conjugate(orientation), quaternion_w)
    else:
        raise ValueError(f"Unknown EE frame: {frame}")
    # Canonical sign for labels. q and -q describe the same orientation.
    quaternion = torch.where(quaternion[..., 3:] < 0, -quaternion, quaternion)
    return torch.cat((position, quaternion, jaw.unsqueeze(-1)), -1)


def decode_ee_target(command, *, origins, frame, anchor=None):
    if command.ndim != 2 or command.shape[-1] != 8 or not torch.isfinite(command).all():
        raise ValueError("EE commands must be finite tensors of shape (N,8)")
    position, quaternion = command[:, :3], unit_quaternion(command[:, 3:7])
    if frame == ABSOLUTE_EE:
        position = position + origins
    elif frame == RELATIVE_EE:
        if anchor is None:
            raise ValueError("Relative EE targets require a measured anchor")
        orientation = unit_quaternion(anchor.quaternion_w)
        position = anchor.position_w + quat_apply(orientation, position)
        quaternion = quat_mul(orientation, quaternion)
    else:
        raise ValueError(f"Unknown EE frame: {frame}")
    return position, quaternion, command[:, 7].clamp(-1, 1)


class EETargetInterface:
    """8-D tool commands to the ordinary 11-D base/arm/jaw actuator interface.

    The existing IK supplies rate limits, arm posture preference and level base
    attitude. There is no phase router, valve controller, success gate, attachment
    or direct mechanism force. The caller supplies any relative command's anchor.
    """

    def __init__(self, env, *, frame=ABSOLUTE_EE, posture_gain=0.1):
        if frame not in (ABSOLUTE_EE, RELATIVE_EE):
            raise ValueError(f"Unknown EE frame: {frame}")
        self.env, self.frame = env, frame
        self.controller = ToolPoseController(env, posture_gain=posture_gain)

    def capture_anchor(self):
        return MeasuredToolAnchor.capture(
            self.env.robot.data.body_link_pos_w.torch[:, self.env._tool_body_id],
            self.env.robot.data.body_link_quat_w.torch[:, self.env._tool_body_id],
        )

    def actions(self, command, *, anchor=None):
        position, quaternion, jaw = decode_ee_target(
            command, origins=self.env.scene.env_origins, frame=self.frame, anchor=anchor
        )
        return self.controller.actions(position, quaternion, jaw)
