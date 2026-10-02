"""Native-limit dual Oberon FK/IK; no simulator dependencies."""

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from wasman.controllers.rexrov2_workspace import RexWorkspace
from wasman.physics.rexrov2_bimanual import URDF

TCP = np.array([0.16, 0, 0])


class BimanualKinematics(RexWorkspace):
    def __init__(self):
        super().__init__(URDF)
        self.frames = {s: self.model.getFrameId(s + "_oberon_end_effector") for s in ["left", "right"]}
        self.indices = {
            s: [
                self.model.joints[self.model.getJointId(s + "_oberon_" + n)].idx_q
                for n in [
                    "azimuth",
                    "shoulder",
                    "elbow",
                    "roll",
                    "pitch",
                    "wrist",
                    "finger_left_joint",
                    "finger_right_joint",
                ]
            ]
            for s in self.frames
        }

    def poses(self, q):
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q))
        return {
            s: (self.data.oMf[f].translation + self.data.oMf[f].rotation @ TCP, self.data.oMf[f].rotation.copy())
            for s, f in self.frames.items()
        }

    def hydrostatic_holding_torque(self, q, parameters):
        """Joint effort balancing gravity/buoyancy at the initial level base.

        This is a controller calculation, never an external simulator wrench.
        Buoyancy uses the same native link volumes and centers as the runtime.
        """
        q = np.asarray(q, dtype=np.float64)
        self.model.gravity.linear[:] = [0, 0, -parameters.gravity]
        torque = pin.computeGeneralizedGravity(self.model, self.data, q).copy()
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        for i, name in enumerate(parameters.names):
            frame = self.model.getFrameId(name)
            jac = pin.getFrameJacobian(self.model, self.data, frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            force = np.array([0, 0, parameters.water_density * parameters.gravity * parameters.volume[i]])
            lever = self.data.oMf[frame].rotation @ parameters.cob[i]
            torque -= jac.T @ np.r_[force, np.cross(lever, force)]
        return torque

    def solve_arm(self, side, position, rotation, q):
        ids = self.indices[side][:6]
        # Numerical Jacobians need double-precision storage even when the seed
        # came from a float32 simulator trace. Otherwise finite differences can
        # vanish when assigned into q, producing false infeasibility reports.
        q = np.array(q, dtype=np.float64, copy=True)

        def error(x):
            q[ids] = x
            p, r = self.poses(q)[side]
            return np.r_[p - position, 0.25 * Rotation.from_matrix(rotation.T @ r).as_rotvec()]

        result = least_squares(
            error,
            np.clip(q[ids], self.low[ids] + 1e-5, self.high[ids] - 1e-5),
            bounds=(self.low[ids] + 1e-5, self.high[ids] - 1e-5),
            max_nfev=150,
            gtol=1e-9,
            ftol=1e-9,
            xtol=1e-9,
        )
        q[ids] = result.x
        return q, float(np.linalg.norm(error(result.x)))
