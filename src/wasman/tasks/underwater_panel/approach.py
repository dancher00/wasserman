"""Optional folded-arm swim-in; sequencing is engineered, not learned."""

import torch
from isaaclab.utils import configclass

from wasman.controllers.approach import ApproachGate
from wasman.controllers.trajectory import advance_target

from .env_cfg import UnderwaterRotateValveT200EnvCfg
from .valve import UnderwaterRotateValveEnv


@configclass
class ApproachRotateValveEnvCfg(UnderwaterRotateValveT200EnvCfg):
    episode_length_s = 95.0
    staging_position = (-0.12, 0.0, 0.82)
    deployed_arm = (3.14, 0.15, 1.62, 3.20)

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.pos = (-0.85, 0.0, 0.82)
        self.scene.robot.init_state.joint_pos["alpha_axis_c"] = 0.30


class ApproachRotateValveEnv(UnderwaterRotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.navigation = ApproachGate(self.num_envs, self.device)
        self.manipulation_ready = torch.zeros_like(self.navigation.enabled)
        self.deployment_steps = torch.zeros_like(self.navigation.stable_steps)
        self.folded_arm = self._arm_targets.clone()
        self.staging = torch.tensor(self.cfg.staging_position, device=self.device)
        self._arm_nominal_targets[:] = torch.tensor(self.cfg.deployed_arm, device=self.device)
        self.base_reference_velocity = torch.zeros_like(self._base_target_pos_w)
        self.arm_reference_velocity = torch.zeros_like(self._arm_targets)

    def _pre_physics_step(self, actions):
        old_base, old_arm = self._base_target_pos_w.clone(), self._arm_targets.clone()
        super()._pre_physics_step(torch.where(self.manipulation_ready[:, None], actions, 0.0))
        position_error = self.robot.data.root_pos_w.torch - self.scene.env_origins - self.staging
        can_deploy = self.navigation.update(
            position_error,
            self.robot.data.body_com_lin_vel_w.torch[:, self._base_body_id].norm(dim=-1),
            self.robot.data.body_com_ang_vel_w.torch[:, self._base_body_id].norm(dim=-1),
            self._base_attitude_error,
        )
        arm_error = (self.robot.data.joint_pos.torch[:, self._arm_joint_ids] - self._arm_nominal_targets).abs().amax(-1)
        settled = can_deploy & (arm_error < 0.02) & (self.arm_reference_velocity.abs().amax(-1) < 0.01)
        self.deployment_steps[:] = torch.where(settled, self.deployment_steps + 1, 0)
        self.manipulation_ready |= self.deployment_steps >= 6
        blocked = ~self.manipulation_ready
        base_goal = self.scene.env_origins + self.staging
        arm_goal = torch.where(can_deploy[:, None], self._arm_nominal_targets, self.folded_arm)
        base, base_velocity = advance_target(
            old_base, self.base_reference_velocity, base_goal, 0.10, 0.18, self.step_dt
        )
        arm, arm_velocity = advance_target(old_arm, self.arm_reference_velocity, arm_goal, 0.35, 0.6, self.step_dt)
        self.base_reference_velocity[blocked] = base_velocity[blocked]
        self.arm_reference_velocity[blocked] = arm_velocity[blocked]
        self._base_target_pos_w[blocked] = base[blocked]
        self._arm_targets[blocked] = arm[blocked]
        self._base_target_quat_w[blocked] = self._world_identity_quat[blocked]

    def _get_dones(self):
        done = super()._get_dones()
        self.extras["wasman_manipulation_enabled"] = self.manipulation_ready.clone()
        self.extras["wasman_arm_deployment_enabled"] = self.navigation.enabled.clone()
        return done

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "navigation"):
            return
        ids = slice(None) if env_ids is None else env_ids
        self.navigation.reset(env_ids)
        self.manipulation_ready[ids] = False
        self.deployment_steps[ids] = 0
        self.base_reference_velocity[ids] = 0
        self.arm_reference_velocity[ids] = 0
        self.folded_arm[ids] = self._arm_targets[ids]
        self._base_target_pos_w[ids] = self.robot.data.root_pos_w.torch[ids]
