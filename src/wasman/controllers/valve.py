"""State-feedback valve expert. Only ordinary robot actuator commands are sent."""

import math

import torch

from wasman.controllers.tool_pose import ToolPoseController


def adaptive_release_stroke(phase, forces, actual_stroke, release_stroke):
    """Increase opening only after reaching a target that still loads the wheel."""
    needs_clearance = (phase == 5) & (forces >= 0.05).any(-1)
    needs_clearance &= actual_stroke >= release_stroke - 0.00005
    return torch.where(needs_clearance, (release_stroke + 0.000025).clamp(max=0.004), release_stroke)


def contact_release_offset(offset, phase, force_vectors, dt):
    reaction = force_vectors.sum(1)
    velocity = (0.002 * reaction).clamp(-0.003, 0.003)
    return (offset + velocity * dt * (phase == 5)[:, None]).clamp(-0.01, 0.01)


class ValveExpert:
    phase_names = ("standoff", "approach", "grasp", "turn", "hold", "release", "withdraw", "complete")

    def __init__(
        self,
        env,
        *,
        strict_unload=False,
        compliant_unload=False,
        hold_angle_deg=173.8,
        axial_release=False,
        approach_offset_x=0.0,
    ):
        self.env = env
        if not -0.04 <= approach_offset_x <= 0.04:
            raise ValueError("Development approach offset must be within 40 mm")
        self.approach_offset_x = approach_offset_x
        if not 170.6 <= hold_angle_deg <= 174.3:
            raise ValueError("Hold reference and its release window must stay within the unchanged task goal")
        self.hold_angle_deg = hold_angle_deg
        self.axial_release = axial_release
        if compliant_unload and not strict_unload:
            raise ValueError("Compliant unloading requires strict force-free withdrawal")
        self.strict_unload = strict_unload
        self.compliant_unload = compliant_unload
        self.ik = ToolPoseController(env, posture_gain=0.1)
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.settled = torch.zeros_like(self.phase)
        self.reference_angle = torch.zeros(env.num_envs, device=env.device)
        self.gripper_reference = torch.zeros_like(self.reference_angle)
        self.previous_gripper_position = torch.zeros_like(self.reference_angle)
        self.grasp_angle = torch.zeros_like(self.reference_angle)
        self.grasp_offset = torch.zeros(env.num_envs, 3, device=env.device)
        self.grasp_quaternion = torch.zeros(env.num_envs, 4, device=env.device)
        self.grasp_quaternion[:, 3] = 1
        self.target = self.grasp_offset.clone()
        self.hold_base = self.grasp_offset.clone()
        self.hold_base_reference = self.grasp_offset.clone()
        self.hold_arm = torch.zeros(env.num_envs, 4, device=env.device)
        self.hold_offset = self.grasp_offset.clone()
        self.hold_adjustment = self.reference_angle.clone()
        self.release_stroke = self.reference_angle.clone()
        self.release_offset = self.grasp_offset.clone()
        self.reset(torch.arange(env.num_envs, device=env.device))

    def reset(self, ids):
        self.phase[ids] = 0
        self.settled[ids] = 0
        self.reference_angle[ids] = 0
        self.gripper_reference[ids] = 0
        self.previous_gripper_position[ids] = self.env.robot.data.joint_pos.torch[ids, self.env._gripper_joint_ids[0]]
        self.grasp_angle[ids] = 0
        self.hold_adjustment[ids] = 0
        self.release_stroke[ids] = 0
        self.release_offset[ids] = 0

    def actions(self):
        from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

        env = self.env
        pivot = env.button.data.body_link_pos_w.torch[:, env._button_body_id]
        angle = env.button.data.joint_pos.torch[:, env._button_joint_id]
        tool = env.robot.data.body_link_pos_w.torch[:, env._tool_body_id]
        tool_q = env.robot.data.body_link_quat_w.torch[:, env._tool_body_id]
        zeros = torch.zeros_like(angle)
        initial_q = quat_from_euler_xyz(zeros + env.cfg.valve_acquisition_roll, zeros, zeros)
        offset = pivot.new_tensor(env.cfg.mechanism_contact_offset).expand_as(pivot).clone()
        offset[:, 0] += 0.012 + self.approach_offset_x
        target = pivot + offset
        target[self.phase == 0, 0] -= 0.162
        off_axis = (tool[:, 1:] - target[:, 1:]).norm(dim=-1) > 0.008
        wait_alignment = (self.phase == 1) & off_axis
        target[wait_alignment, 0] = torch.minimum(target[wait_alignment, 0], tool[wait_alignment, 0])
        target_q = initial_q.clone()
        turning = self.phase >= 3
        forces = env.finger_forces()
        bilateral = (forces > 0.05).all(-1)
        # Never outrun the measured wheel by more than ~14 degrees. This also
        # pauses the turning reference when the physical grasp is lost.
        advance = (self.phase == 3) & bilateral
        previous_reference = self.reference_angle.clone()
        self.reference_angle[advance] += 0.14 * env.step_dt
        # Permit tracking-error compensation; the transition is governed by
        # measured angle, not arrival of this reference at a nominal endpoint.
        goal = math.radians(178) - self.grasp_angle
        self.reference_angle[:] = torch.minimum(self.reference_angle, goal.clamp_min(0))
        self.reference_angle[:] = torch.minimum(
            self.reference_angle, torch.maximum(previous_reference, angle - self.grasp_angle + 0.25)
        )
        rotation = quat_from_euler_xyz(self.reference_angle, zeros, zeros)
        target[turning] = (pivot + quat_apply(rotation, self.grasp_offset))[turning]
        target_q[turning] = quat_mul(rotation, self.grasp_quaternion)[turning]
        # Unload the pinch with a small measured stroke before withdrawing.
        # Fully spreading inside the wheel sweeps the inner finger into the
        # spokes. Reserve full opening until clear of the wheel plane.
        if self.strict_unload:
            actual_stroke = env.robot.data.joint_pos.torch[:, env._gripper_joint_ids[0]]
            # Adapt only once the previous opening target has been reached.
            # Never withdraw while a fingertip is still loading the wheel.
            self.release_stroke.copy_(adaptive_release_stroke(self.phase, forces, actual_stroke, self.release_stroke))
        release_action = 2 * self.release_stroke / 0.0098 - 1
        gripper = torch.where(self.phase < 2, 0.0, torch.where(self.phase >= 5, release_action, -1.0))
        gripper = torch.where((self.phase >= 6) & (env.wheel_clearance() >= 0.08), 1.0, gripper)
        grip_delta = (gripper - self.gripper_reference).clamp(-0.8 * env.step_dt, 0.8 * env.step_dt)
        if self.axial_release:
            slow_delta = (gripper - self.gripper_reference).clamp(-0.15 * env.step_dt, 0.15 * env.step_dt)
            grip_delta = torch.where(self.phase == 5, slow_delta, grip_delta)
        self.gripper_reference += grip_delta
        target[self.phase >= 6, 0] -= 0.14
        self.target.copy_(target)

        error = (tool - target).norm(dim=-1)
        tool_speed = env.robot.data.body_com_lin_vel_w.torch[:, env._tool_body_id].norm(dim=-1)
        tolerance = torch.where(self.phase == 0, 0.03, 0.012)
        eligible = (error < tolerance) & (tool_speed < 0.035)
        eligible &= (self.phase != 0) | ((tool[:, 1:] - target[:, 1:]).norm(dim=-1) < 0.006)
        eligible = torch.where(
            self.phase == 1, ((tool - pivot - offset).norm(dim=-1) < 0.012) & (tool_speed < 0.035), eligible
        )
        grip_position = env.robot.data.joint_pos.torch[:, env._gripper_joint_ids[0]]
        grip_speed = (grip_position - self.previous_gripper_position).abs() / env.step_dt
        self.previous_gripper_position.copy_(grip_position)
        grip_settled = (forces > 0.5).all(-1) & (grip_speed < 0.0002) & (self.gripper_reference <= -0.999)
        eligible = torch.where(self.phase == 2, grip_settled, eligible)
        eligible = torch.where(self.phase == 3, angle >= math.radians(170), eligible)
        release_ready = env.valve_contract.held & (angle >= math.radians(self.hold_angle_deg - 0.6))
        release_ready &= angle <= math.radians(self.hold_angle_deg + 0.7)
        release_ready &= env.button.data.joint_vel.torch[:, env._button_joint_id].abs() < 0.05
        eligible = torch.where(self.phase == 4, release_ready, eligible)
        grip_position = env.robot.data.joint_pos.torch[:, env._gripper_joint_ids[0]]
        # Legacy mode permits light unilateral contact before withdrawal.
        # The explicit strict variant adapts the opening and waits for the
        # same force-free threshold required by the measured release contract.
        unload_force = 0.05 if self.strict_unload else 1.0
        unloaded = (grip_position >= self.release_stroke - 0.0001) & (forces < unload_force).all(-1)
        unloaded &= ~env.opposing_contacts()
        eligible = torch.where(self.phase == 5, unloaded, eligible)
        eligible = torch.where(self.phase == 6, env.valve_contract.success, eligible)
        self.settled[:] = torch.where(eligible, self.settled + 1, 0)
        required_steps = torch.where(
            self.phase == 2,
            15,
            torch.where((self.phase == 3) | (self.phase == 5), 3, torch.where(self.phase == 4, 10, 9)),
        )
        transition = (self.settled >= required_steps) & (self.phase < 7)
        grasped = transition & (self.phase == 2)
        # Preserve the measured tool-to-wheel transform at acquisition. The
        # nominal rim point is for approach, not an invented rigid attachment.
        self.grasp_offset[grasped] = (tool - pivot)[grasped]
        self.grasp_quaternion[grasped] = tool_q[grasped]
        self.grasp_angle[grasped] = angle[grasped]
        stopping = transition & (self.phase == 3)
        self.hold_base[stopping] = env.robot.data.body_link_pos_w.torch[stopping, env._base_body_id]
        self.hold_base_reference[stopping] = self.hold_base[stopping]
        self.hold_arm[stopping] = env.robot.data.joint_pos.torch[stopping][:, env._arm_joint_ids]
        self.hold_offset[stopping] = (tool - pivot)[stopping]
        self.hold_adjustment[stopping] = 0
        releasing = transition & (self.phase == 4)
        self.release_stroke[releasing] = (grip_position[releasing] + 0.0007).clamp(max=0.004)
        self.phase[transition] += 1
        self.settled[transition] = 0
        actions = self.ik.actions(target, target_q, self.gripper_reference)
        # Stop solving a redundant IK problem once the turn reaches its target:
        # latch the measured actuator pose, brake, then trim at <=0.01 rad/s.
        # A coordinated tiny vehicle translation + wrist rotation retains the
        # grasp transform. No constraint or force is added to the valve.
        holding = self.phase == 4
        trim_rate = (0.3 * (math.radians(self.hold_angle_deg) - angle)).clamp(-0.015, 0.015)
        self.hold_adjustment[holding] += trim_rate[holding] * env.step_dt
        trim_rotation = quat_from_euler_xyz(self.hold_adjustment, zeros, zeros)
        hold_base_target = self.hold_base + quat_apply(trim_rotation, self.hold_offset) - self.hold_offset
        if self.axial_release:
            # Follow a short axial escape while gently unloading the jaws.
            # This tests a contact trajectory, not an object-force override.
            hold_base_target[self.phase == 5, 0] -= 0.025
        if self.compliant_unload:
            # Admittance: move with the measured contact reaction to unload a
            # unilateral fingertip. Ordinary base position commands only;
            # no wheel forces, constraints, or collision changes.
            self.release_offset.copy_(
                contact_release_offset(self.release_offset, self.phase, env.finger_force_vectors(), env.step_dt)
            )
            hold_base_target += self.release_offset
        hold_base_target[self.phase >= 6, 0] -= 0.14
        withdrawal_speed = torch.where((self.phase >= 6)[:, None], 0.08, 0.035)
        if self.axial_release:
            withdrawal_speed = torch.where((self.phase == 5)[:, None], 0.02, withdrawal_speed)
        self.hold_base_reference += (hold_base_target - self.hold_base_reference).clamp(
            -withdrawal_speed * env.step_dt, withdrawal_speed * env.step_dt
        )
        hold_arm_target = self.hold_arm.clone()
        hold_arm_target[:, 3] -= self.hold_adjustment
        latched = self.phase >= 4
        actions[latched, :3] = (
            (self.hold_base_reference - env.scene.env_origins - env._base_target_nominal)
            / env._base_target_position_scale
        )[latched]
        actions[latched, 6:10] = ((hold_arm_target - env._arm_nominal_targets) / env._arm_target_scale)[latched]
        self.target[latched] = (pivot + self.hold_offset + self.hold_base_reference - self.hold_base)[latched]
        return actions.clamp(-1, 1)
