"""Bounded dual-arm IK with a selectable reference frame for feedback."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from wasman.controllers.bimanual_kinematics import TCP, BimanualKinematics


def parallel_jaw_target(position, closing_bias, stiffness=400.0):
    """Keep total closing bias while restoring the native differential stiffness.

    Both physical joints receive the same aperture target; they remain independent
    dynamic joints. Per-jaw q_i-bias/K would remove restoring stiffness from the
    opposite-motion coordinate and permit different centering equilibria.
    """
    return position.mean(dim=-1, keepdim=True) - closing_bias / stiffness


class BimanualController:
    def __init__(self, robot, base_id, tool_ids, initial_base, initial_arm, mode, grasp_effort=None, feedback_frame="world", balance_initial_load=False):
        self.robot = robot
        self.base_id = base_id
        self.tool_ids = tool_ids
        self.base = initial_base.clone()
        self.reference = initial_arm.clone()
        self.initial = initial_arm.clone()
        self.mode = mode
        # Optional command-level torque request, via the unchanged native jaw
        # position drive (400 Nm/rad, 30 Nms/rad). This is not an effort-limit
        # override: the native 30 Nm drive/physical limits stay in force.
        if grasp_effort is not None and not 0 < grasp_effort <= 30:
            raise ValueError("Grasp effort must be in the native (0, 30] Nm range")
        self.grasp_effort = grasp_effort
        if feedback_frame not in ("world", "level"):
            raise ValueError(feedback_frame)
        self.feedback_frame = feedback_frame
        self.workspace = BimanualKinematics()
        self.pin_ids = [
            self.workspace.model.joints[self.workspace.model.getJointId(n)].idx_q for n in robot.joint_names
        ]
        self.ids = {
            s: [
                robot.joint_names.index(s + "_oberon_" + n)
                for n in ["azimuth", "shoulder", "elbow", "roll", "pitch", "wrist"]
            ]
            for s in ["left", "right"]
        }
        self.grip = {
            s: [robot.joint_names.index(s + "_oberon_finger_" + n + "_joint") for n in ["left", "right"]]
            for s in ["left", "right"]
        }
        self.speeds = torch.tensor([0.17, 0.17, 0.15, 0.25, 0.30, 0.15], device=robot.device)
        self.bias = torch.zeros_like(initial_arm)
        self.contact_latched = torch.zeros(len(initial_base), dtype=torch.bool, device=robot.device)
        self.ik_residual = np.zeros((len(initial_base), 2))
        self.initial_holding_torque = np.zeros((len(initial_base), self.workspace.model.nq))
        if balance_initial_load:
            from wasman.physics.rexrov2_bimanual import load_bimanual_parameters

            parameters = load_bimanual_parameters()
            for n in range(len(initial_base)):
                q = np.zeros(self.workspace.model.nq)
                q[self.pin_ids] = initial_arm[n].detach().cpu().numpy()
                effort = self.workspace.hydrostatic_holding_torque(q, parameters)
                if np.any(abs(effort) > self.workspace.model.effortLimit + 1e-6):
                    raise ValueError("Initial hydrostatic load exceeds a native joint effort limit")
                self.initial_holding_torque[n] = effort
                for side in self.ids:
                    ids = self.ids[side]
                    bias = torch.as_tensor(effort[self.workspace.indices[side][:6]] / 3000,
                                           device=robot.device, dtype=initial_arm.dtype)
                    self.bias[n, ids] = bias
                    # Establish the loaded drive target before the first step;
                    # measured joint state stays at the same reset pose. All
                    # subsequent reference changes retain native rate bounds.
                    self.reference[n, ids] += bias

    def targets(self, now, goals, rotations, grip):
        d = self.robot.data
        bp = d.body_link_pos_w.torch[:, self.base_id]
        bq = d.body_link_quat_w.torch[:, self.base_id]
        for side in ["left", "right"]:
            ids = self.ids[side]
            tid = self.tool_ids[side]
            goal = goals[side].detach().cpu().numpy()
            rot = rotations[side]
            if rot.ndim == 2:
                rot = np.broadcast_to(rot, (len(bp), 3, 3))
            # The base reference stays fixed. The selected arm feedback frame
            # determines whether measured base motion enters the IK correction.
            from wasman.physics.hydrodynamics import quat_apply_xyzw

            offset = quat_apply_xyzw(
                d.body_link_quat_w.torch[:, tid], torch.tensor(TCP, device=bp.device, dtype=bp.dtype).expand(len(bp), 3)
            )
            bases = bp.detach().cpu().numpy()
            base_rotations = Rotation.from_quat(bq.detach().cpu().numpy()).as_matrix()
            if self.feedback_frame == "level":
                # As in the validated single-arm adapter, compensate joint
                # deflection in the commanded level-base frame. Base motion
                # remains the responsibility of the native station keeper.
                bases = self.base.detach().cpu().numpy()
                base_rotations = np.broadcast_to(np.eye(3), base_rotations.shape)
            nominals = []
            for n in range(len(bp)):
                # The last environment is the motors-off/held-arm control.
                # Its arm command is overwritten by the runner; solving IK for
                # a freely drifting unreachable base only wastes CPU work.
                if n == len(bp) - 1:
                    nominals.append(self.initial[n, ids].detach().cpu().numpy())
                    continue
                local_goal = base_rotations[n].T @ (goal[n] - bases[n])
                local_rotation = base_rotations[n].T @ rot[n]
                seed = np.zeros(self.workspace.model.nq)
                seed[self.pin_ids] = self.reference[n].detach().cpu().numpy()
                candidate, err = self.workspace.solve_arm(side, local_goal, local_rotation, seed)
                self.ik_residual[n, 0 if side == "left" else 1] = err
                nominals.append(candidate[self.workspace.indices[side][:6]])
            nominal = torch.as_tensor(np.array(nominals), device=bp.device, dtype=bp.dtype)
            from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul

            relative_q = d.body_link_quat_w.torch[:, tid]
            measured_tcp = d.body_link_pos_w.torch[:, tid] + offset
            if self.feedback_frame == "level":
                from wasman.physics.hydrodynamics import quat_apply_inverse_xyzw

                measured_tcp = self.base + quat_apply_inverse_xyzw(bq, measured_tcp - bp)
                relative_q = quat_mul(quat_conjugate(bq), relative_q)
            target_q = torch.tensor(Rotation.from_matrix(rot).as_quat(), device=bp.device, dtype=bp.dtype)
            rotation_error = axis_angle_from_quat(quat_mul(target_q, quat_conjugate(relative_q)))
            error = torch.cat((goals[side] - measured_tcp, rotation_error), -1)
            jac = d.body_link_jacobian_w.torch[:, tid, :, [6 + j for j in ids]].clone()
            jac[:, :3] += torch.cross(jac[:, 3:].transpose(1, 2), offset[:, None].expand(-1, 6, -1), dim=-1).transpose(
                1, 2
            )
            if self.feedback_frame == "level":
                inverse = torch.tensor(
                    Rotation.from_quat(bq.detach().cpu().numpy()).as_matrix().transpose(0, 2, 1).copy(),
                    device=bp.device, dtype=bp.dtype,
                )
                jac[:, :3] = inverse @ jac[:, :3]
                jac[:, 3:] = inverse @ jac[:, 3:]
            delta = (
                jac.transpose(1, 2)
                @ torch.linalg.solve(
                    jac @ jac.transpose(1, 2) + 0.002**2 * torch.eye(6, device=bp.device), error.unsqueeze(-1)
                )
            ).squeeze(-1)
            self.bias[:, ids] = (self.bias[:, ids] + 0.6 * delta.clamp(-self.speeds, self.speeds) / 30).clamp(
                -0.15, 0.15
            )
            target = nominal + self.bias[:, ids]
            self.reference[:, ids] += (target - self.reference[:, ids]).clamp(-self.speeds / 30, self.speeds / 30)
            gid = self.grip[side]
            opening = torch.as_tensor(grip[side], device=bp.device, dtype=bp.dtype)
            if opening.ndim == 1:
                opening = opening[:, None]
            if self.grasp_effort is not None:
                closing = opening <= 0
                # Request a closing bias while retaining the native drive's
                # dissipative damping. Do not cancel damping with sampled
                # velocity feedback. No physical gain/effort limit is changed.
                torque_target = parallel_jaw_target(d.joint_pos.torch[:, gid], self.grasp_effort)
                opening = torch.where(closing, torque_target.clamp_min(0), opening)
            self.reference[:, gid] += (opening - self.reference[:, gid]).clamp(-0.15 / 30, 0.15 / 30)
        limits = d.soft_joint_pos_limits.torch
        self.reference = self.reference.clamp(limits[..., 0], limits[..., 1])
        return self.base.clone(), self.reference.clone()
