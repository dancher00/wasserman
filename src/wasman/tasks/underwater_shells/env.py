"""Free dynamic shells: ordinary contact/gravity/drag, no object pose writes during actions."""

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from pxr import Usd, UsdGeom

from wasman.controller_diagnostic import ControllerDiagnosticEnv
from wasman.controllers.hatch_actuator import measured_state
from wasman.controllers.shell_collection_contract import ShellCollectionContract
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.env import UnderwaterPressButtonEnv

from .env_cfg import ASSETS


class CollectShellEnv(UnderwaterPressButtonEnv):
    # Legacy vehicle-plant flag skips the panel-mechanism guard. This subclass
    # instead requires explicit independent manipulated bodies and its own score.
    diagnostic_only = True

    def __init__(self, cfg, render_mode=None, **kwargs):
        if cfg.scene.button is not None or any(not hasattr(cfg.scene, f"shell_{i}") for i in range(cfg.shell_count)):
            raise ValueError("Shell task requires its independent dynamic objects, without a dummy panel")
        super().__init__(cfg, render_mode, **kwargs)
        self.shells = [self.scene[f"shell_{i}"] for i in range(cfg.shell_count)]
        stage = Usd.Stage.Open(str(ASSETS / "Shell.usda"))
        points = UsdGeom.Mesh(stage.GetPrimAtPath("/Shell/Surface")).GetPointsAttr().Get()
        # Mesh vertices bound its convex collision hull exactly. Rotated box
        # corners can lie below the floor while the rounded shell rests on it.
        self.local_bounds = torch.tensor([tuple(p) for p in points], device=self.device)
        self.hoop_center = self.scene.env_origins + torch.tensor(cfg.hoop_center, device=self.device)
        self.contract = ShellCollectionContract(self.num_envs, cfg.shell_count, self.device, self.step_dt)
        self.collection_state = None

    def shell_positions(self):
        return torch.stack([x.data.root_link_pos_w.torch for x in self.shells], 1)

    def finger_force_vectors(self):
        # One sensed finger body, K filter bodies; preserve shell identity.
        return torch.stack(
            [
                self.scene[name].data.normal_force_matrix_w.torch.reshape(self.num_envs, self.cfg.shell_count, 3)
                for name in ("left_contact", "right_contact")
            ],
            dim=2,
        )

    def _apply_action(self):
        super()._apply_action()
        for obj in self.shells:
            velocity = obj.data.root_com_lin_vel_w.torch - self.current_w
            force = -self.cfg.shell_linear_drag * velocity - self.cfg.shell_quadratic_drag * velocity.abs() * velocity
            force[:, 2] += self.cfg.water_density * 9.81 * self.cfg.shell_displaced_volume
            torque = -self.cfg.shell_angular_drag * obj.data.root_com_ang_vel_w.torch
            obj.instantaneous_wrench_composer.set_forces_and_torques_index(
                forces=force[:, None, :],
                torques=torque[:, None, :],
                is_global=True,
            )

    def contract_inputs(self):
        positions = self.shell_positions()
        quaternion = torch.stack([x.data.root_link_quat_w.torch for x in self.shells], 1)
        corners = self.local_bounds[None, None].expand(self.num_envs, self.cfg.shell_count, -1, -1)
        rotation = quaternion[:, :, None, :].expand(*corners.shape[:-1], 4)
        bounds = quat_apply(rotation.reshape(-1, 4), corners.reshape(-1, 3)).reshape_as(corners)
        tool = self.robot.data.body_link_pos_w.torch[:, self._tool_body_id]
        return dict(
            positions=positions,
            bounds=bounds + positions[:, :, None, :],
            linear_velocity=torch.stack([x.data.root_com_lin_vel_w.torch for x in self.shells], 1),
            angular_velocity=torch.stack([x.data.root_com_ang_vel_w.torch for x in self.shells], 1),
            finger_forces=self.finger_force_vectors(),
            tool_distance=(positions - tool[:, None, :]).norm(dim=-1),
            hoop_center_xy=self.hoop_center[:, :2],
            hoop_inner_radius=self.cfg.hoop_inner_radius,
            floor_z=0.0,
        )

    def _get_dones(self):
        failed, timeout = ControllerDiagnosticEnv._get_dones(self)
        inputs = self.contract_inputs()
        finite = torch.isfinite(inputs["positions"]).all(dim=(1, 2))
        finite &= torch.isfinite(inputs["linear_velocity"]).all(dim=(1, 2))
        escaped = (inputs["positions"] - self.scene.env_origins[:, None]).norm(dim=-1).amax(-1) > 3
        self.collection_state = self.contract.update(**inputs)
        self._episode_succeeded.copy_(self.collection_state["success"])
        self.extras["wasman_success"] = self._episode_succeeded.clone()
        for key, value in self.collection_state.items():
            self.extras["wasman_shell_" + key] = value.clone()
        return failed | ~finite | escaped, timeout

    def _get_rewards(self):
        return self._episode_succeeded.float()

    def _get_observations(self):
        positions = self.shell_positions() - self.scene.env_origins[:, None]
        velocity = torch.stack([x.data.root_com_lin_vel_w.torch for x in self.shells], 1)
        lifted = self.contract.lifted.float()
        transported = self.contract.transported.float()
        placed = torch.zeros_like(lifted) if self.collection_state is None else self.collection_state["placed"].float()
        state = torch.cat(
            (
                measured_state(self),
                positions.flatten(1),
                velocity.flatten(1),
                lifted,
                transported,
                placed,
                self.actions,
            ),
            -1,
        )
        return {"policy": state}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, "shells"):
            return
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device)
        )
        n = len(ids)
        for obj in self.shells:
            pose = obj.data.default_root_pose.torch[ids].clone()
            pose[:, :3] += self.scene.env_origins[ids]
            pose[:, :2] += self._uniform(-self.cfg.shell_position_jitter, self.cfg.shell_position_jitter, (n, 2))
            yaw = self._uniform(*self.cfg.shell_yaw_range, (n,))
            pose[:, 3:] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
            obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=ids)
        self.contract.reset(ids)
        if self.collection_state is not None:
            for value in self.collection_state.values():
                value[ids] = 0
