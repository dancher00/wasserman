"""Native-limit FK/IK and conservative CAD-hull self-collision checks, CPU only."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ASSET = Path(__file__).parents[1] / "assets/data/robots/rexrov2_oberon7_centered"
URDF = ASSET / "rexrov2_oberon7_centered.urdf"
TCP_OFFSET = np.array([0.18, 0.0, 0.0])


class RexWorkspace:
    def __init__(self, urdf=URDF):
        self.model, self.geometry, _ = pin.buildModelsFromUrdf(str(urdf), [str(urdf.parent)])
        for geometry in self.geometry.geometryObjects:
            geometry.geometry.computeLocalAABB()
        self.data = self.model.createData()
        self.tool_frame = self.model.getFrameId("oberon_end_effector")
        root = ET.parse(urdf).getroot()
        self.adjacent = {
            frozenset((joint.find("parent").get("link"), joint.find("child").get("link")))
            for joint in root.findall("joint")
        }
        self.links = [self.model.frames[g.parentFrame].name for g in self.geometry.geometryObjects]
        # Check base vs shoulder across the fixed pedestal too. Only directly
        # connected/same-link CAD is exempt, not all neighboring joint indices.
        for first in range(len(self.links)):
            for second in range(first + 1, len(self.links)):
                adjacent = frozenset((self.links[first], self.links[second])) in self.adjacent
                if self.links[first] != self.links[second] and not adjacent:
                    self.geometry.addCollisionPair(pin.CollisionPair(first, second))
        self.geometry_data = self.geometry.createData()
        for request in self.geometry_data.distanceRequests:
            request.enable_signed_distance = True
        self.low = self.model.lowerPositionLimit.copy()
        self.high = self.model.upperPositionLimit.copy()

    def fk(self, q):
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
        frame = self.data.oMf[self.tool_frame]
        return frame.translation + frame.rotation @ TCP_OFFSET, frame.rotation.copy()

    def collisions(self, q):
        pin.computeCollisions(self.model, self.data, self.geometry, self.geometry_data, np.asarray(q), False)
        result = []
        for pair, contact in zip(self.geometry.collisionPairs, self.geometry_data.collisionResults, strict=True):
            if contact.isCollision():
                result.append((self.links[pair.first], self.links[pair.second]))
        return sorted(set(result))

    def clearance(self, q):
        pin.computeDistances(self.model, self.data, self.geometry, self.geometry_data, np.asarray(q))
        distances = np.array([result.min_distance for result in self.geometry_data.distanceResults])
        index = int(distances.argmin())
        pair = self.geometry.collisionPairs[index]
        return float(distances[index]), (self.links[pair.first], self.links[pair.second])

    def bounds(self, q, include_base=False):
        """Conservative transformed per-CAD-component AABBs, not link origins."""
        pin.updateGeometryPlacements(self.model, self.data, self.geometry, self.geometry_data, np.asarray(q))
        points = []
        for index, geometry in enumerate(self.geometry.geometryObjects):
            if self.links[index] == "base_link" and not include_base:
                continue
            box = geometry.geometry.aabb_local
            corners = np.array(
                [
                    [x, y, z]
                    for x in (box.min_[0], box.max_[0])
                    for y in (box.min_[1], box.max_[1])
                    for z in (box.min_[2], box.max_[2])
                ]
            )
            placement = self.geometry_data.oMg[index]
            points.append(corners @ placement.rotation.T + placement.translation)
        points = np.concatenate(points)
        return np.array([points.min(0), points.max(0)])

    def solve(self, position, rotation, seed, jaw=0.25):
        position, rotation = np.asarray(position), np.asarray(rotation)

        def residual(arm):
            actual_position, actual_rotation = self.fk(np.r_[arm, jaw, jaw])
            orientation = Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec()
            return np.r_[actual_position - position, 0.25 * orientation]

        result = least_squares(
            residual,
            seed,
            bounds=(self.low[:6] + 1e-4, self.high[:6] - 1e-4),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        q = np.r_[result.x, jaw, jaw]
        actual_position, actual_rotation = self.fk(q)
        return {
            "q": q,
            "position_error_m": float(np.linalg.norm(actual_position - position)),
            "orientation_error_rad": float(
                np.linalg.norm(Rotation.from_matrix(rotation.T @ actual_rotation).as_rotvec())
            ),
            "collisions": self.collisions(q),
        }
