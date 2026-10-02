"""PPO with a measured-state transit/arm interlock, not an end-to-end learned gate."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from wasman.controllers.approach import ApproachGate
from wasman.controllers.press_hold import PressHoldController, normal_press_force

from .smooth import SmoothPressButtonEnv


class ApproachPressButtonEnv(SmoothPressButtonEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._approach_gate = ApproachGate(self.num_envs, self.device)
        self._transit_arm_targets = self._arm_targets.clone()
        self._staging_position = torch.tensor(self.cfg.staging_position, device=self.device)
        self._arm_nominal_targets[:] = torch.tensor(self.cfg.policy_arm_nominal, device=self.device)
        self._arm_deployed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._deployment_settled = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._press_hold = PressHoldController(self)
        self._normal_force_bias = torch.zeros(self.num_envs, device=self.device)
        self._press_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _pre_physics_step(self, actions):
        self._previous_base_position = self._base_target_pos_w.clone()
        self._previous_base_quaternion = self._base_target_quat_w.clone()
        self._previous_arm_target = self._arm_targets.clone()
        # The checkpoint was trained with an extended arm. During folded transit
        # and deployment use a station-keeping waypoint, not out-of-distribution
        # PPO requests. Hand off with the action filter starting from zero.
        super()._pre_physics_step(torch.where(self._arm_deployed.unsqueeze(-1), actions, 0.0))

    def _limit_requested_targets(self):
        position = self.robot.data.root_pos_w.torch - self.scene.env_origins
        linear_speed = self.robot.data.body_com_lin_vel_w.torch[:, self._base_body_id].norm(dim=-1)
        angular_speed = self.robot.data.body_com_ang_vel_w.torch[:, self._base_body_id].norm(dim=-1)
        staging_error = position - self._staging_position
        enabled = self._approach_gate.update(staging_error, linear_speed, angular_speed, self._base_attitude_error)
        joint_error = (self.robot.data.joint_pos.torch[:, self._arm_joint_ids] - self._arm_nominal_targets).abs()
        # Solver-step joint velocity has high-frequency chatter even when the
        # measured pose is within 0.001 rad. Require persistent measured pose
        # agreement AND a finished reference trajectory, not one noisy sample.
        settled = enabled & (joint_error.amax(-1) < 0.02) & (self._arm_reference_velocity.abs().amax(-1) < 0.01)
        self._deployment_settled[:] = torch.where(settled, self._deployment_settled + 1, 0)
        self._arm_deployed |= self._deployment_settled >= 6
        transit = ~enabled
        # Stay at the navigation waypoint until the measured arm deployment is complete.
        blocked = ~self._arm_deployed
        self._base_target_pos_w[blocked] = self.scene.env_origins[blocked] + self._staging_position
        self._base_target_quat_w[blocked] = self._world_identity_quat[blocked]
        self._arm_targets[transit] = self._transit_arm_targets[transit]
        deploying = enabled & ~self._arm_deployed
        self._arm_targets[deploying] = self._arm_nominal_targets[deploying]
        self._press_hold.update(
            self._arm_deployed, self._previous_base_position, self._previous_base_quaternion, self._previous_arm_target
        )

    def _reference_limits(self):
        manipulating = self._arm_deployed.unsqueeze(-1)
        near = (self._distance < 0.18).unsqueeze(-1)
        # Brake well before contact, not only in the last 10 cm when the vehicle
        # has already accumulated forward speed and station-keeper error.
        base = torch.where(manipulating, torch.where(near, 0.015, 0.035), 0.10)
        arm = torch.where(manipulating, torch.where(near, 0.06, 0.20), 0.35)
        arm = torch.where(self._press_hold.active.unsqueeze(-1), 0.12, arm)
        return base, arm

    def _apply_contact_control(self, force_b, torque_b, base_quat_w):
        # Contact-normal depth regulation replaces only the X position loop. The arm
        # keeps its reached pose; Y/Z and attitude remain under station keeping.
        # All force requests still pass through the eight physical T200 motors.
        station_world = quat_apply(base_quat_w, force_b)
        depth = self.button.data.joint_pos.torch[:, self._button_joint_id]
        normal_speed = self.robot.data.body_com_lin_vel_w.torch[:, self._tool_body_id, 0]
        self._normal_force_bias += 150.0 * (0.0055 - depth) * self._press_hold.active * self.physics_dt
        self._normal_force_bias.clamp_(-8.0, 8.0)
        normal = normal_press_force(depth, normal_speed, self._normal_force_bias)
        station_world[self._press_hold.active, 0] = normal[self._press_hold.active]
        force_b[:] = quat_apply_inverse(base_quat_w, station_world)

    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        self._press_started |= self._press_hold.active & (self._button_travel >= 0.004)
        interrupted = self._press_started & (self._button_travel < 0.003) & ~self._episode_succeeded
        self.extras["wasman_manipulation_enabled"] = self._arm_deployed.clone()
        self.extras["wasman_arm_deployment_enabled"] = self._approach_gate.enabled.clone()
        self.extras["wasman_contact_hold_enabled"] = self._press_hold.active.clone()
        self.extras["wasman_press_interrupted"] = interrupted.clone()
        self.extras["wasman_deployment_error"] = (
            self.robot.data.joint_pos.torch[:, self._arm_joint_ids] - self._arm_nominal_targets
        ).clone()
        self.extras["wasman_arm_velocity"] = self.robot.data.joint_vel.torch[:, self._arm_joint_ids].clone()
        return terminated | interrupted, truncated

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        indices = slice(None) if env_ids is None else env_ids
        self._approach_gate.reset(env_ids)
        self._arm_deployed[indices] = False
        self._deployment_settled[indices] = 0
        self._press_hold.reset(env_ids)
        self._normal_force_bias[indices] = 0.0
        self._press_started[indices] = False
        self._transit_arm_targets[indices] = self._arm_targets[indices]
