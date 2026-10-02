"""Direct RL environment for contact-rich underwater button intervention."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz

from wasman.assets import (
    BLUEROV2_ALPHA_ARM_JOINT_NAMES,
    BLUEROV2_ALPHA_BODY_NAMES,
    BLUEROV2_ALPHA_GRIPPER_JOINT_NAME,
    BLUEROV2_ALPHA_HYDRODYNAMICS,
)
from wasman.assets.panels import centered_panel_pose
from wasman.controllers import BatchedStationKeepingController, StationKeepingGains
from wasman.controllers.station_keeping import quaternion_angle_error_xyzw
from wasman.controllers.thruster_visuals import ThrusterVisuals
from wasman.physics import BatchedHydrodynamics
from wasman.physics.boundary_effects import BlueROVBoundaryEffect, BoundaryEffectCfg, validate_boundary_scene
from wasman.physics.thrusters import BatchedT200

if TYPE_CHECKING:
    from .env_cfg import UnderwaterPressButtonEnvCfg


class UnderwaterPressButtonEnv(DirectRLEnv):
    """Use a free-floating BlueROV2-Alpha UVMS to press a panel button."""

    cfg: UnderwaterPressButtonEnvCfg

    def __init__(self, cfg: UnderwaterPressButtonEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.scene.button is None and not getattr(self, "diagnostic_only", False):
            raise ValueError("A manipulation task requires its mechanism; use the controller diagnostic subclass")
        boundary_pool = validate_boundary_scene(cfg) if cfg.enable_boundary_effects else None
        super().__init__(cfg, render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self.button = None if cfg.scene.button is None else self.scene["button"]

        body_ids, body_names = self.robot.find_bodies(list(BLUEROV2_ALPHA_BODY_NAMES), preserve_order=True)
        if tuple(body_names) != BLUEROV2_ALPHA_BODY_NAMES:
            raise RuntimeError(f"Unexpected BlueROV2-Alpha body order: {body_names}")
        self._hydro_body_ids = body_ids
        self._base_body_id = body_ids[0]
        self._tool_body_id = body_ids[-1]

        self._arm_joint_ids, arm_joint_names = self.robot.find_joints(
            list(BLUEROV2_ALPHA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(arm_joint_names) != BLUEROV2_ALPHA_ARM_JOINT_NAMES:
            raise RuntimeError(f"Unexpected Alpha arm joint order: {arm_joint_names}")
        self._gripper_joint_ids, gripper_joint_names = self.robot.find_joints(BLUEROV2_ALPHA_GRIPPER_JOINT_NAME)
        if gripper_joint_names != [BLUEROV2_ALPHA_GRIPPER_JOINT_NAME]:
            raise RuntimeError(f"Unexpected Alpha gripper joints: {gripper_joint_names}")

        if self.button is not None:
            self._button_body_ids, button_body_names = self.button.find_bodies(self.cfg.mechanism_body_name)
            self._button_joint_ids, button_joint_names = self.button.find_joints(self.cfg.mechanism_joint_name)
            if button_body_names != [self.cfg.mechanism_body_name] or button_joint_names != [
                self.cfg.mechanism_joint_name
            ]:
                raise RuntimeError(
                    f"Unexpected panel mechanism topology: bodies={button_body_names}, joints={button_joint_names}"
                )
            self._button_body_id = self._button_body_ids[0]
            self._button_joint_id = self._button_joint_ids[0]
            if self.cfg.mechanism_joint_limits is not None:
                limits = torch.tensor(self.cfg.mechanism_joint_limits, device=self.device).reshape(1, 1, 2)
                self.button.write_joint_position_limit_to_sim_index(
                    limits=limits.expand(self.num_envs, 1, 2).contiguous(), joint_ids=self._button_joint_ids
                )

        self._hydrodynamics = BatchedHydrodynamics(
            getattr(self.cfg, "link_hydrodynamics", BLUEROV2_ALPHA_HYDRODYNAMICS),
            num_envs=self.num_envs,
            dt=self.physics_dt,
            device=self.device,
            water_density=self.cfg.water_density,
            acceleration_filter=self.cfg.acceleration_filter,
        )

        self.previous_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._arm_targets = self.robot.data.default_joint_pos.torch[:, self._arm_joint_ids].clone()
        self._arm_nominal_targets = self._arm_targets.clone()
        self._arm_target_scale = torch.tensor(self.cfg.arm_target_scale, device=self.device)
        self._button_targets = torch.full((self.num_envs, 1), self.cfg.mechanism_initial_position, device=self.device)

        self._base_target_nominal = torch.tensor(self.cfg.base_target_position, device=self.device)
        self._base_target_position_scale = torch.tensor(self.cfg.base_target_position_scale, device=self.device)
        self._base_target_rpy_scale = torch.tensor(self.cfg.base_target_rpy_scale, device=self.device)
        self._base_target_pos_w = self.scene.env_origins + self._base_target_nominal
        self._base_target_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self._base_target_quat_w[:, 3] = 1.0
        self._world_identity_quat = self._base_target_quat_w.clone()
        self._station_position_error_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._station_orientation_error_b = torch.zeros_like(self._station_position_error_w)
        self._station_keeper = BatchedStationKeepingController(
            num_envs=self.num_envs,
            dt=self.physics_dt,
            device=self.device,
            gains=StationKeepingGains(
                position_kp=self.cfg.station_position_kp,
                position_kd=self.cfg.station_position_kd,
                position_ki=self.cfg.station_position_ki,
                rotation_kp=self.cfg.station_rotation_kp,
                rotation_kd=self.cfg.station_rotation_kd,
                rotation_ki=self.cfg.station_rotation_ki,
                max_force=self.cfg.station_max_force,
                max_torque=self.cfg.station_max_torque,
                position_integral_limit=self.cfg.station_position_integral_limit,
                rotation_integral_limit=self.cfg.station_rotation_integral_limit,
            ),
        )

        self._mean_current_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._thrusters = None
        if self.cfg.use_physical_thrusters:
            self._thrusters = BatchedT200(
                self.num_envs,
                self.device,
                dt=self.physics_dt,
                time_constant=self.cfg.thruster_time_constant,
                command_delay_steps=self.cfg.thruster_command_delay_steps,
                positions=self.cfg.thruster_positions,
            )
            actual_com = self.robot.data.body_com_pos_b.torch[:, self._base_body_id]
            expected_com = actual_com.new_tensor((0.0, 0.0, 0.011)).expand_as(actual_com)
            if not torch.allclose(actual_com, expected_com, atol=1e-5):
                raise RuntimeError("T200 allocation COM differs from the robot's actual COM")
        self._turbulent_current_w = torch.zeros_like(self._mean_current_w)
        self._boundary_effect = None
        if self.cfg.enable_boundary_effects:
            if self._thrusters is None:
                raise ValueError("Experimental boundary effects require physical T200 thrusters")
            self._boundary_effect = BlueROVBoundaryEffect(
                self.device,
                BoundaryEffectCfg(
                    seabed_loss=self.cfg.boundary_seabed_loss,
                    wall_loss=self.cfg.boundary_wall_loss,
                    range_diameters=self.cfg.boundary_range_diameters,
                ),
                pool_geometry=boundary_pool,
                pool_center=cfg.scene.seabed.init_state.pos if boundary_pool is not None else (0, 0, 0),
                env_origins=self.scene.env_origins if boundary_pool is not None else None,
                positions=self.cfg.thruster_positions,
            )
            self._boundary_gain = torch.ones_like(self._thrusters.force)
            self._boundary_applied_wrench = torch.zeros_like(self._thrusters.realized_wrench)
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._approached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pressed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._max_button_travel = torch.zeros(self.num_envs, device=self.device)
        self._base_attitude_error = torch.zeros(self.num_envs, device=self.device)
        self._base_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self._max_base_attitude_error = torch.zeros(self.num_envs, device=self.device)
        self._max_base_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self._just_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._invalid_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._distance = torch.zeros(self.num_envs, device=self.device)
        self._alignment = torch.zeros(self.num_envs, device=self.device)
        self._button_travel = torch.zeros(self.num_envs, device=self.device)
        self._last_distance = torch.zeros(self.num_envs, device=self.device)
        self._last_button_travel = torch.zeros(self.num_envs, device=self.device)
        self._target_delta_b = torch.zeros((self.num_envs, 3), device=self.device)
        self._alignment_error_b = torch.zeros_like(self._target_delta_b)
        self._projected_up_b = torch.zeros_like(self._target_delta_b)

        reward_names = (
            "position",
            "precision",
            "alignment",
            "contact",
            "press_progress",
            "press",
            "success",
            "distance_progress",
            "action",
            "action_rate",
            "base_velocity",
            "joint_velocity",
            "attitude",
        )
        self._episode_reward_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in reward_names}

        self._thruster_visuals = None
        self._button_indicator = None
        if self.sim.is_rendering:
            from isaaclab.sim import get_current_stage

            self._thruster_visuals = ThrusterVisuals(
                get_current_stage(), self.num_envs, self.device, positions=self.cfg.thruster_positions
            )
            if self.button is not None and self.cfg.mechanism_joint_name == "PlungerSlideJoint":
                from wasman.controllers.button_indicator import ButtonIndicator

                self._button_indicator = ButtonIndicator(get_current_stage(), self.num_envs, self.device)

        if self.sim.has_gui:
            self.set_debug_vis(False)

    @property
    def current_w(self) -> torch.Tensor:
        """Per-environment water current in world axes [m/s]."""
        return self._mean_current_w + self._turbulent_current_w

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(torch.clamp(actions, -1.0, 1.0))
        self._base_target_pos_w.copy_(
            self.scene.env_origins + self._base_target_nominal + self._base_target_position_scale * self.actions[:, :3]
        )
        target_rpy = self._base_target_rpy_scale * self.actions[:, 3:6]
        self._base_target_quat_w.copy_(quat_from_euler_xyz(target_rpy[:, 0], target_rpy[:, 1], target_rpy[:, 2]))
        self._arm_targets.copy_(self._arm_nominal_targets + self._arm_target_scale * self.actions[:, 6:10])
        arm_limits = self.robot.data.soft_joint_pos_limits.torch[:, self._arm_joint_ids]
        self._arm_targets.clamp_(arm_limits[..., 0], arm_limits[..., 1])

    def _apply_action(self) -> None:
        dt = self.physics_dt
        self._turbulent_current_w += -self.cfg.turbulence_theta * self._turbulent_current_w * dt
        self._turbulent_current_w += (
            self.cfg.turbulence_sigma * math.sqrt(dt) * torch.randn_like(self._turbulent_current_w)
        )
        self._turbulent_current_w.clamp_(-self.cfg.turbulence_limit, self.cfg.turbulence_limit)

        # Hydrodynamic coefficients and control wrenches are expressed in each
        # URDF link frame.  ``body_com_quat_w`` is the principal-inertia frame
        # in Isaac Lab 3.x and can be rotated relative to that link frame.
        body_quat_w = self.robot.data.body_link_quat_w.torch[:, self._hydro_body_ids]
        valid_quat = torch.isfinite(body_quat_w).all(dim=-1, keepdim=True)
        identity_quat = torch.zeros_like(body_quat_w)
        identity_quat[..., 3] = 1.0
        body_quat_w = torch.where(valid_quat, body_quat_w, identity_quat)
        body_lin_vel_w = self.robot.data.body_com_lin_vel_w.torch[:, self._hydro_body_ids]
        body_ang_vel_w = self.robot.data.body_com_ang_vel_w.torch[:, self._hydro_body_ids]
        body_lin_vel_w = torch.nan_to_num(body_lin_vel_w).clamp(
            -self.cfg.max_relative_linear_speed, self.cfg.max_relative_linear_speed
        )
        body_ang_vel_w = torch.nan_to_num(body_ang_vel_w).clamp(
            -self.cfg.max_relative_angular_speed, self.cfg.max_relative_angular_speed
        )
        current_w = self.current_w[:, None, :].expand(-1, len(self._hydro_body_ids), -1)
        relative_lin_vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w - current_w)
        relative_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
        relative_twist_b = torch.cat((relative_lin_vel_b, relative_ang_vel_b), dim=-1)

        hydro_force_b, hydro_torque_b, _ = self._hydrodynamics.compute(relative_twist_b, body_quat_w)
        hydro_force_b = torch.nan_to_num(hydro_force_b).clamp(
            -self.cfg.max_hydrodynamic_force, self.cfg.max_hydrodynamic_force
        )
        hydro_torque_b = torch.nan_to_num(hydro_torque_b).clamp(
            -self.cfg.max_hydrodynamic_torque, self.cfg.max_hydrodynamic_torque
        )
        composer = self.robot.instantaneous_wrench_composer
        composer.set_forces_and_torques_index(
            forces=hydro_force_b,
            torques=hydro_torque_b,
            body_ids=self._hydro_body_ids,
            is_global=False,
        )
        base_quat_w = body_quat_w[:, 0]
        station_force_b, station_torque_b, position_error_w, orientation_error_b = self._station_keeper.compute(
            position_w=self.robot.data.body_link_pos_w.torch[:, self._base_body_id],
            quaternion_w=base_quat_w,
            linear_velocity_w=body_lin_vel_w[:, 0],
            angular_velocity_w=body_ang_vel_w[:, 0],
            target_position_w=self._base_target_pos_w,
            target_quaternion_w=self._base_target_quat_w,
        )
        self._station_position_error_w.copy_(position_error_w)
        self._station_orientation_error_b.copy_(orientation_error_b)
        self._apply_contact_control(station_force_b, station_torque_b, base_quat_w)
        if self._thrusters is not None:
            forces = self._thrusters.step(station_force_b, station_torque_b)
            if self._boundary_effect is not None:
                base_pose = torch.cat((self.robot.data.body_link_pos_w.torch[:, self._base_body_id], base_quat_w), -1)
                panel_pose = None if self.cfg.scene.panel is None else self.scene["panel"].data.root_link_pose_w.torch
                forces, gain, _ = self._boundary_effect.apply(base_pose, self._thrusters.force, panel_pose)
                self._boundary_gain.copy_(gain)
                self._boundary_applied_wrench.copy_((self._thrusters.force * gain) @ self._thrusters.matrix.T)
            # Apply every motor's axial force at its actual mount relative to
            # COM. PhysX receives the realized forces, not the requested wrench.
            for motor in range(8):
                composer.add_forces_and_torques_index(
                    forces=forces[:, motor : motor + 1].contiguous(),
                    positions=self._thrusters.offsets[motor].view(1, 1, 3).expand(self.num_envs, 1, 3).contiguous(),
                    body_ids=[self._base_body_id],
                    is_global=False,
                )
        else:
            composer.add_forces_and_torques_index(
                forces=station_force_b.unsqueeze(1),
                torques=station_torque_b.unsqueeze(1),
                body_ids=[self._base_body_id],
                is_global=False,
            )
        if self._thruster_visuals is not None:
            self._thruster_visuals.update(
                station_force_b,
                station_torque_b,
                self.physics_dt,
                render=self._sim_step_counter % self.cfg.sim.render_interval == 0,
                realized_force=None if self._thrusters is None else self._thrusters.force,
            )

        self.robot.set_joint_position_target_index(target=self._arm_targets, joint_ids=self._arm_joint_ids)
        gripper_target = torch.zeros((self.num_envs, 1), device=self.device)
        if self.cfg.policy_gripper:
            limits = self.robot.data.soft_joint_pos_limits.torch[:, self._gripper_joint_ids]
            fraction = 0.5 * (self.actions[:, 10:11] + 1.0)
            gripper_target = limits[..., 0] + fraction * (limits[..., 1] - limits[..., 0])
        self.robot.set_joint_position_target_index(target=gripper_target, joint_ids=self._gripper_joint_ids)
        if self.button is None:
            return  # Controller-only scenes contain no mechanism or invisible dummy collider.
        self.button.set_joint_position_target_index(target=self._button_targets, joint_ids=self._button_joint_ids)
        if self._button_indicator is not None:
            self._button_indicator.update(
                self.button.data.joint_pos.torch[:, self._button_joint_id] - self.cfg.mechanism_initial_position,
                render=self._sim_step_counter % self.cfg.sim.render_interval == 0,
            )
        if self.cfg.mechanism_displaced_volume:
            buoyancy = torch.zeros(self.num_envs, 1, 3, device=self.device)
            buoyancy[:, :, 2] = self.cfg.water_density * 9.81 * self.cfg.mechanism_displaced_volume
            self.button.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=buoyancy,
                body_ids=[self._button_body_id],
                is_global=True,
            )

    def _apply_contact_control(self, force_b, torque_b, base_quat_w):
        """Optional variant hook; released baselines use no contact compensation."""

    def _update_task_state(self) -> None:
        base_quat_w = self.robot.data.body_link_quat_w.torch[:, self._base_body_id]
        tool_quat_w = self.robot.data.body_link_quat_w.torch[:, self._tool_body_id]
        tool_pos_w = self.robot.data.body_link_pos_w.torch[:, self._tool_body_id]
        button_pos_w = self.button.data.body_link_pos_w.torch[:, self._button_body_id]
        contact_offset = torch.tensor(self.cfg.mechanism_contact_offset, device=self.device).expand_as(button_pos_w)
        button_pos_w = button_pos_w + quat_apply(
            self.button.data.body_link_quat_w.torch[:, self._button_body_id], contact_offset
        )

        world_x = torch.tensor((1.0, 0.0, 0.0), device=self.device).expand_as(tool_pos_w)
        world_up = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand_as(tool_pos_w)
        tool_axis_w = quat_apply(tool_quat_w, world_x)
        target_delta_w = button_pos_w - tool_pos_w

        self._distance.copy_(torch.linalg.vector_norm(target_delta_w, dim=-1))
        self._alignment.copy_(torch.sum(tool_axis_w * world_x, dim=-1).clamp(-1.0, 1.0))
        self._button_travel.copy_(
            (
                self.cfg.mechanism_direction
                * (self.button.data.joint_pos.torch[:, self._button_joint_id] - self.cfg.mechanism_initial_position)
            ).clamp(0.0, self.cfg.button_max_travel)
        )
        self._max_button_travel.copy_(torch.maximum(self._max_button_travel, self._button_travel))
        self._target_delta_b.copy_(quat_apply_inverse(base_quat_w, target_delta_w))
        tool_axis_b = quat_apply_inverse(base_quat_w, tool_axis_w)
        desired_axis_b = quat_apply_inverse(base_quat_w, world_x)
        self._alignment_error_b.copy_(desired_axis_b - tool_axis_b)
        self._projected_up_b.copy_(quat_apply_inverse(base_quat_w, world_up))
        attitude_error = quaternion_angle_error_xyzw(base_quat_w, self._world_identity_quat)
        angular_speed = torch.linalg.vector_norm(
            self.robot.data.body_com_ang_vel_w.torch[:, self._base_body_id], dim=-1
        )
        self._base_attitude_error.copy_(torch.nan_to_num(attitude_error, nan=math.pi))
        self._base_angular_speed.copy_(torch.nan_to_num(angular_speed, nan=self.cfg.max_valid_body_angular_speed))
        self._max_base_attitude_error.copy_(torch.maximum(self._max_base_attitude_error, self._base_attitude_error))
        self._max_base_angular_speed.copy_(torch.maximum(self._max_base_angular_speed, self._base_angular_speed))

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_task_state()
        base_quat_w = self.robot.data.body_link_quat_w.torch[:, self._base_body_id]
        base_lin_vel_b = quat_apply_inverse(
            base_quat_w, self.robot.data.body_com_lin_vel_w.torch[:, self._base_body_id]
        )
        base_ang_vel_b = quat_apply_inverse(
            base_quat_w, self.robot.data.body_com_ang_vel_w.torch[:, self._base_body_id]
        )
        current_b = quat_apply_inverse(base_quat_w, self.current_w)
        joint_pos = self.robot.data.joint_pos.torch[:, self._arm_joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self._arm_joint_ids]
        button_fraction = (self._button_travel / self.cfg.button_pressed_threshold).clamp(0.0, 1.5).unsqueeze(-1)
        observation = torch.cat(
            (
                self._target_delta_b,
                self._alignment_error_b,
                base_lin_vel_b,
                base_ang_vel_b,
                self._projected_up_b,
                joint_pos,
                joint_vel,
                button_fraction,
                current_b,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_state()
        approached_now = self._distance < self.cfg.approach_threshold
        tool_contact_pose = (self._distance < self.cfg.tool_contact_distance) & (
            self._alignment > self.cfg.tool_contact_alignment
        )
        if self.cfg.require_finger_contact:
            grasped = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for sensor in ("left_contact", "right_contact"):
                forces = self.scene[sensor].data.normal_force_matrix_w.torch
                grasped &= forces.norm(dim=-1).reshape(self.num_envs, -1).sum(-1) > self.cfg.finger_contact_threshold_n
            tool_contact_pose &= grasped
            self.extras["wasman_grasped"] = grasped.clone()
        contacted_now = (self._button_travel > self.cfg.contact_threshold) & tool_contact_pose
        stable_base = (self._base_attitude_error < self.cfg.success_max_attitude_error) & (
            self._base_angular_speed < self.cfg.success_max_angular_speed
        )
        pressed_now = (self._button_travel >= self.cfg.button_pressed_threshold) & tool_contact_pose & stable_base
        pressed_now &= self._button_travel <= self.cfg.success_max_travel
        pressed_now &= (
            self.button.data.joint_vel.torch[:, self._button_joint_id].abs() <= self.cfg.mechanism_success_max_speed
        )
        pressed_now &= (
            self.robot.data.body_com_lin_vel_w.torch[:, self._tool_body_id].norm(dim=-1)
            <= self.cfg.success_max_tool_speed
        )
        self._approached |= approached_now
        self._contacted |= contacted_now
        self._pressed |= pressed_now

        self._success_counter = torch.where(pressed_now, self._success_counter + 1, 0)
        self._episode_succeeded |= self._success_counter >= self.cfg.success_hold_steps

        base_pos_w = self.robot.data.body_com_pos_w.torch[:, self._base_body_id]
        base_offset = base_pos_w - self.scene.env_origins
        escaped = torch.linalg.vector_norm(base_offset, dim=-1) > self.cfg.max_base_distance
        escaped |= base_pos_w[:, 2] < self.cfg.min_base_height
        escaped |= base_pos_w[:, 2] > self.cfg.max_base_height
        escaped |= self._base_attitude_error > self.cfg.max_base_attitude_error
        body_vel = self.robot.data.body_com_vel_w.torch
        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch
        finite_state = torch.isfinite(self.robot.data.body_link_pos_w.torch).all(dim=(1, 2))
        finite_state &= torch.isfinite(body_vel).all(dim=(1, 2))
        finite_state &= torch.isfinite(joint_pos).all(dim=1)
        finite_state &= torch.isfinite(joint_vel).all(dim=1)
        finite_state &= torch.isfinite(self.button.data.joint_pos.torch).all(dim=1)
        valid_speed = body_vel[..., :3].abs().amax(dim=(1, 2)) < self.cfg.max_valid_body_linear_speed
        valid_speed &= body_vel[..., 3:].abs().amax(dim=(1, 2)) < self.cfg.max_valid_body_angular_speed
        valid_speed &= joint_vel.abs().amax(dim=1) < self.cfg.max_valid_joint_speed
        self._invalid_state.copy_(~(finite_state & valid_speed))
        time_out = self.episode_length_buf >= self.max_episode_length

        # Clone terminal metrics because DirectRLEnv performs same-step resets
        # before returning ``extras`` to evaluation clients.
        self.extras["wasman_success"] = self._episode_succeeded.clone()
        if self._thrusters is not None:
            self.extras["wasman_thruster_saturation_scale"] = self._thrusters.saturation_scale.clone()
            self.extras["wasman_thruster_force"] = self._thrusters.force.clone()
            self.extras["wasman_thruster_rpm"] = self._thrusters.rpm.clone()
            if self._boundary_effect is not None:
                self.extras["wasman_boundary_gain"] = self._boundary_gain.clone()
                self.extras["wasman_boundary_applied_wrench"] = self._boundary_applied_wrench.clone()
                self.extras["wasman_boundary_actuation_error"] = (
                    self._thrusters.requested_wrench - self._boundary_applied_wrench
                ).clone()
            # Legacy metric is actuator-only (before optional boundary losses).
            self.extras["wasman_actuation_error"] = (
                self._thrusters.requested_wrench - self._thrusters.realized_wrench
            ).clone()
        self.extras["wasman_distance"] = self._distance.clone()
        self.extras["wasman_tool_speed"] = (
            self.robot.data.body_com_lin_vel_w.torch[:, self._tool_body_id].norm(dim=-1).clone()
        )
        self.extras["wasman_alignment"] = self._alignment.clone()
        self.extras["wasman_button_travel"] = self._button_travel.clone()
        self.extras["wasman_max_button_travel"] = self._max_button_travel.clone()
        self.extras["wasman_mechanism_progress"] = self._button_travel.clone()
        self.extras["wasman_max_mechanism_progress"] = self._max_button_travel.clone()
        self.extras["wasman_mechanism_progress_unit"] = self.cfg.mechanism_progress_unit
        self.extras["wasman_approached"] = self._approached.clone()
        self.extras["wasman_contacted"] = self._contacted.clone()
        self.extras["wasman_pressed"] = self._pressed.clone()
        self.extras["wasman_base_attitude_error"] = self._base_attitude_error.clone()
        self.extras["wasman_base_angular_speed"] = self._base_angular_speed.clone()
        self.extras["wasman_max_base_attitude_error"] = self._max_base_attitude_error.clone()
        self.extras["wasman_max_base_angular_speed"] = self._max_base_angular_speed.clone()
        return escaped | self._invalid_state, time_out

    def _get_rewards(self) -> torch.Tensor:
        distance_progress = torch.where(self._just_reset, 0.0, self._last_distance - self._distance).clamp(-0.1, 0.1)
        press_progress = torch.where(
            self._just_reset,
            0.0,
            (self._button_travel - self._last_button_travel).clamp_min(0.0),
        )
        self._just_reset.fill_(False)
        self._last_distance.copy_(self._distance)
        self._last_button_travel.copy_(self._button_travel)

        precision = torch.exp(-70.0 * self._distance.square())
        alignment_score = 0.5 * (self._alignment + 1.0)
        tool_contact_pose = (self._distance < self.cfg.tool_contact_distance) & (
            self._alignment > self.cfg.tool_contact_alignment
        )
        contacted = (self._button_travel > self.cfg.contact_threshold) & tool_contact_pose
        stable_base = (self._base_attitude_error < self.cfg.success_max_attitude_error) & (
            self._base_angular_speed < self.cfg.success_max_angular_speed
        )
        pressed = (self._button_travel >= self.cfg.button_pressed_threshold) & tool_contact_pose & stable_base
        base_lin_vel = self.robot.data.body_com_lin_vel_w.torch[:, self._base_body_id].clamp(-5.0, 5.0)
        base_ang_vel = self.robot.data.body_com_ang_vel_w.torch[:, self._base_body_id].clamp(-10.0, 10.0)
        arm_joint_vel = self.robot.data.joint_vel.torch[:, self._arm_joint_ids].clamp(-3.0, 3.0)

        reward_terms = {
            "position": self.cfg.reward_position * torch.exp(-4.0 * self._distance),
            "precision": self.cfg.reward_precision * precision,
            "alignment": self.cfg.reward_alignment * alignment_score * torch.exp(-2.0 * self._distance),
            "contact": self.cfg.reward_contact * contacted.float(),
            "press_progress": self.cfg.reward_press_progress * press_progress / self.cfg.button_pressed_threshold,
            "press": self.cfg.reward_press * pressed.float(),
            "success": self.cfg.reward_success * self._episode_succeeded.float(),
            "distance_progress": self.cfg.reward_distance_progress * distance_progress,
            "action": self.cfg.penalty_action * torch.sum(self.actions.square(), dim=-1),
            "action_rate": self.cfg.penalty_action_rate
            * torch.sum((self.actions - self.previous_actions).square(), dim=-1),
            "base_velocity": self.cfg.penalty_base_velocity
            * (torch.sum(base_lin_vel.square(), dim=-1) + 0.5 * torch.sum(base_ang_vel.square(), dim=-1)),
            "joint_velocity": self.cfg.penalty_joint_velocity * torch.sum(arm_joint_vel.square(), dim=-1),
            "attitude": self.cfg.penalty_attitude * self._base_attitude_error.square(),
        }
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, value in reward_terms.items():
            scaled_value = torch.nan_to_num(value * self.step_dt)
            reward += scaled_value
            self._episode_reward_sums[name] += scaled_value
        reward = torch.where(
            self._invalid_state,
            torch.full_like(reward, self.cfg.invalid_state_penalty),
            reward,
        )
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if self._thruster_visuals is not None:
            self._thruster_visuals.reset(env_ids)
        if env_ids is None:
            indices = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if indices.numel() == 0:
            return
        if self._button_indicator is not None:
            self._button_indicator.reset(indices)
        if self._thrusters is not None:
            self._thrusters.reset(indices)
        if self._boundary_effect is not None:
            self._boundary_gain[indices] = 1.0
            self._boundary_applied_wrench[indices] = 0.0

        log = self.extras.setdefault("log", {})
        log["Metrics/success_rate"] = self._episode_succeeded[indices].float().mean().item()
        log["Metrics/approach_rate"] = self._approached[indices].float().mean().item()
        log["Metrics/contact_rate"] = self._contacted[indices].float().mean().item()
        log["Metrics/press_rate"] = self._pressed[indices].float().mean().item()
        log["Metrics/terminal_distance"] = (
            torch.nan_to_num(self._distance[indices], nan=self.cfg.max_base_distance)
            .clamp_max(self.cfg.max_base_distance)
            .mean()
            .item()
        )
        log["Metrics/max_button_travel"] = (
            torch.nan_to_num(self._max_button_travel[indices]).clamp(0.0, self.cfg.button_max_travel).mean().item()
        )
        log["Metrics/max_base_attitude_error"] = self._max_base_attitude_error[indices].mean().item()
        log["Metrics/max_base_angular_speed"] = self._max_base_angular_speed[indices].mean().item()
        for name, episode_sum in self._episode_reward_sums.items():
            log[f"Episode_Reward/{name}"] = episode_sum[indices].mean().item() / self.max_episode_length_s
            episode_sum[indices] = 0.0

        super()._reset_idx(indices)
        count = indices.numel()

        root_pose = self.robot.data.default_root_pose.torch[indices].clone()
        root_pose[:, :3] += self.scene.env_origins[indices]
        root_pose[:, 0] += self._uniform(-0.035, 0.035, (count,))
        root_pose[:, 1] += self._uniform(-0.045, 0.045, (count,))
        root_pose[:, 2] += self._uniform(-0.035, 0.035, (count,))
        yaw = self._uniform(-0.08, 0.08, (count,))
        root_pose[:, 3:6] = 0.0
        root_pose[:, 5] = torch.sin(0.5 * yaw)
        root_pose[:, 6] = torch.cos(0.5 * yaw)
        root_velocity = self.robot.data.default_root_vel.torch[indices].clone()

        joint_pos = self.robot.data.default_joint_pos.torch[indices].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[indices].clone()
        joint_pos[:, self._arm_joint_ids] += self._uniform(-0.025, 0.025, (count, len(self._arm_joint_ids)))
        joint_pos[:, self._gripper_joint_ids] = 0.0
        arm_limits = self.robot.data.soft_joint_pos_limits.torch[indices][:, self._arm_joint_ids]
        joint_pos[:, self._arm_joint_ids] = joint_pos[:, self._arm_joint_ids].clamp(
            arm_limits[..., 0], arm_limits[..., 1]
        )

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=indices)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=indices)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=indices)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=indices)
        self._arm_targets[indices] = joint_pos[:, self._arm_joint_ids]
        self.robot.set_joint_position_target_index(
            target=self._arm_targets[indices], joint_ids=self._arm_joint_ids, env_ids=indices
        )
        self.robot.set_joint_position_target_index(
            target=torch.zeros((count, 1), device=self.device),
            joint_ids=self._gripper_joint_ids,
            env_ids=indices,
        )

        if self.button is not None:
            self._reset_mechanism(indices, count)

        speed = self._uniform(*self.cfg.current_speed_range, (count,))
        heading = self._uniform(-math.pi, math.pi, (count,))
        self._mean_current_w[indices, 0] = speed * torch.cos(heading)
        self._mean_current_w[indices, 1] = speed * torch.sin(heading)
        self._mean_current_w[indices, 2] = self._uniform(*self.cfg.current_vertical_range, (count,))
        self._turbulent_current_w[indices] = 0.0
        self._hydrodynamics.set_parameter_scales(
            indices,
            volume=self._uniform(*self.cfg.volume_scale_range, (count,)),
            damping=self._uniform(*self.cfg.damping_scale_range, (count,)),
            added_mass=self._uniform(*self.cfg.added_mass_scale_range, (count,)),
        )
        self._hydrodynamics.reset(indices)
        self._station_keeper.reset(indices)

        self.actions[indices] = 0.0
        self.previous_actions[indices] = 0.0
        self._base_target_pos_w[indices] = self.scene.env_origins[indices] + self._base_target_nominal
        self._base_target_quat_w[indices] = self._world_identity_quat[indices]
        self._success_counter[indices] = 0
        self._episode_succeeded[indices] = False
        self._approached[indices] = False
        self._contacted[indices] = False
        self._pressed[indices] = False
        self._max_button_travel[indices] = 0.0
        self._base_attitude_error[indices] = 0.0
        self._base_angular_speed[indices] = 0.0
        self._max_base_attitude_error[indices] = 0.0
        self._max_base_angular_speed[indices] = 0.0
        self._invalid_state[indices] = False
        self._just_reset[indices] = True
        self._last_distance[indices] = 0.0
        self._last_button_travel[indices] = 0.0

    def _reset_mechanism(self, indices, count):
        """Original reset sequence, separated so object-free diagnostics need no dummy asset."""
        button_pose = self.button.data.default_root_pose.torch[indices].clone()
        button_pose[:, :3] += self.scene.env_origins[indices]
        button_pose[:, 1] = self.scene.env_origins[indices, 1] + self._uniform(*self.cfg.button_y_range, (count,))
        button_pose[:, 2] = self._uniform(*self.cfg.button_z_range, (count,))
        button_velocity = self.button.data.default_root_vel.torch[indices].clone()
        button_joint_pos = self.button.data.default_joint_pos.torch[indices].clone()
        button_joint_vel = self.button.data.default_joint_vel.torch[indices].clone()
        button_joint_pos[:, self._button_joint_id] = self.cfg.mechanism_initial_position
        button_joint_vel[:, self._button_joint_id] = 0.0
        self.button.write_root_pose_to_sim_index(root_pose=button_pose, env_ids=indices)
        # Randomize the assembly, not the fixture within its backing panel.
        # Keep the learned mechanism/robot reference unchanged and move both
        # wall geometry and collision together at reset (never during contact).
        if self.cfg.scene.panel is not None:
            panel = self.scene["panel"]
            panel_pose = panel.data.default_root_pose.torch[indices].clone()
            panel_pose[:, :3] += self.scene.env_origins[indices]
            panel_pose = centered_panel_pose(panel_pose, button_pose)
            panel.write_root_pose_to_sim_index(root_pose=panel_pose, env_ids=indices)
            panel.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((count, 6), device=self.device), env_ids=indices
            )
        self.button.write_root_velocity_to_sim_index(root_velocity=button_velocity, env_ids=indices)
        self.button.write_joint_position_to_sim_index(position=button_joint_pos, env_ids=indices)
        self.button.write_joint_velocity_to_sim_index(velocity=button_joint_vel, env_ids=indices)
        self.button.set_joint_position_target_index(
            target=torch.zeros((count, 1), device=self.device), joint_ids=self._button_joint_ids, env_ids=indices
        )

    def _uniform(self, low: float, high: float, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=self.device).uniform_(low, high)
