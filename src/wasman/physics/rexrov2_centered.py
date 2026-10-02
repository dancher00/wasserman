"""Remount-specific held-arm parameters; never mutate the native Rex model."""

import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from wasman.controllers.rexrov2_workspace import URDF, RexWorkspace
from wasman.physics.rexrov2 import load_parameters, skew

FOLDED_Q = np.array([0, 1.5, -1.5, np.pi / 2, 1.5, 0, 0.25, 0.25])
BASE_HEIGHT = 1.5


def composite_inertia(workspace, q):
    workspace.fk(q)
    matrix = np.zeros((6, 6))
    for link in ET.parse(URDF).getroot().findall("link"):
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        center = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        local_rotation = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
        frame = workspace.data.oMf[workspace.model.getFrameId(link.get("name"))]
        center = frame.translation + frame.rotation @ center
        rotation = frame.rotation @ local_rotation
        values = inertial.find("inertia")
        xx, xy, xz, yy, yz, zz = (float(values.get(key)) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
        inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        matrix[:3, :3] += np.eye(3) * mass
        matrix[:3, 3:] -= mass * skew(center)
        matrix[3:, :3] += mass * skew(center)
        matrix[3:, 3:] += rotation @ inertia @ rotation.T - mass * skew(center) @ skew(center)
    return matrix


def load_centered_parameters(q=FOLDED_Q, workspace=None):
    """Composite is pose/mount specific. Base added mass stays in native base axes.

    Link-local buoyancy/drag properties stay attached to their respective links
    and must be transformed using the new articulation's measured link poses.
    Moving-joint added inertia is still not modeled: this is a held-arm model.
    """
    workspace = RexWorkspace() if workspace is None else workspace
    return replace(load_parameters(), rigid_composite=composite_inertia(workspace, q))
