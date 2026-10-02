"""Illustration-only assembly transforms; never change training task defaults."""

import math

import torch

TILT_DEGREES = (-25, -10, 10, 25)
# Camera faces +X, so +Y appears on the left of the image.
PLACEMENTS = ((0.18, 0.53), (-0.38, 0.53), (0.18, 1.03), (-0.38, 1.03))
PANEL_CENTER = (0.86, -0.10, 0.78)
CAMERA_EYE = (-0.55, -0.495, 1.01)
CAMERA_TARGET = PANEL_CENTER


def tilt_assembly(panel_pose, mechanism_pose, degrees):
    """Pitch both rigid roots about the panel centre, preserving their attachment.

    Poses use world XYZ position and XYZW quaternion, matching Isaac Lab 3.
    No geometry scaling, articulation-joint drive or image transformation.
    """
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    relative = mechanism_pose[:, :3] - panel_pose[:, :3]
    rotated = torch.stack(
        (c * relative[:, 0] + s * relative[:, 2], relative[:, 1], -s * relative[:, 0] + c * relative[:, 2]), dim=-1
    )
    q = torch.zeros_like(panel_pose[:, 3:7])
    q[:, 1], q[:, 3] = math.sin(angle / 2), math.cos(angle / 2)

    def compose(r):
        xyz = q[:, 3:] * r[:, :3] + r[:, 3:] * q[:, :3] + torch.linalg.cross(q[:, :3], r[:, :3])
        w = q[:, 3:] * r[:, 3:] - (q[:, :3] * r[:, :3]).sum(-1, keepdim=True)
        return torch.cat((xyz, w), dim=-1)

    panel, mechanism = panel_pose.clone(), mechanism_pose.clone()
    panel[:, 3:7] = compose(panel[:, 3:7])
    mechanism[:, :3] = panel_pose[:, :3] + rotated
    mechanism[:, 3:7] = compose(mechanism[:, 3:7])
    return panel, mechanism
