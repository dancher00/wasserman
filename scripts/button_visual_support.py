"""Unchanged smooth T200 task: robot-only inputs and frozen PPO demonstration teacher."""

import numpy as np
import torch


def measured_state(e):
    b = e._base_body_id
    return torch.cat(
        (
            e.robot.data.body_link_pos_w.torch[:, b] - e.scene.env_origins,
            e.robot.data.body_link_quat_w.torch[:, b],
            e.robot.data.body_com_lin_vel_w.torch[:, b],
            e.robot.data.body_com_ang_vel_w.torch[:, b],
            e.robot.data.joint_pos.torch[:, e._arm_joint_ids],
            e.robot.data.joint_vel.torch[:, e._arm_joint_ids],
        ),
        -1,
    ).clone()


def pack_action(action):
    if action.shape[-1] != 10 or not torch.isfinite(action).all():
        raise ValueError("Expected ten finite original actuator channels")
    return action.clone()


unpack_action = pack_action


class ButtonTeacher:
    """Training data only. Uses privileged state; students never receive it."""

    def __init__(self, env):
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        self.env = env
        observation = TensorDict(env._get_observations(), batch_size=[env.num_envs])
        self.model = MLPModel(
            observation,
            {"actor": ["policy"]},
            "actor",
            10,
            hidden_dims=[256, 256, 128],
            activation="elu",
            obs_normalization=True,
        )
        state = torch.load("checkpoints/wasman_press_button_smooth_seed42.pt", map_location="cpu", weights_only=True)[
            "actor_state_dict"
        ]
        self.model.load_state_dict({k: v for k, v in state.items() if not k.startswith("distribution.")}, strict=True)
        self.model.to(env.device).eval()

    def actions(self):
        from tensordict import TensorDict

        return self.model(TensorDict(self.env._get_observations(), batch_size=[self.env.num_envs])).clamp(-1, 1)


def contract(c):
    return dict(
        threshold=c.button_pressed_threshold,
        max_travel=c.success_max_travel,
        distance=c.tool_contact_distance,
        alignment=c.tool_contact_alignment,
        attitude=c.success_max_attitude_error,
        angular_speed=c.success_max_angular_speed,
        tool_speed=c.success_max_tool_speed,
        mechanism_speed=c.mechanism_success_max_speed,
        hold_steps=c.success_hold_steps,
        initial=c.mechanism_initial_position,
        direction=c.mechanism_direction,
        unit="m",
    )


def capture(e):
    b = e._base_body_id
    return dict(
        q=e._button_travel.clone(),
        mechanism_speed=e.button.data.joint_vel.torch[:, e._button_joint_id].clone(),
        tool_speed=e.robot.data.body_com_lin_vel_w.torch[:, e._tool_body_id].norm(dim=-1),
        distance=e._distance.clone(),
        alignment=e._alignment.clone(),
        attitude=e._base_attitude_error.clone(),
        angular_speed=e._base_angular_speed.clone(),
        tool=e.robot.data.body_link_pos_w.torch[:, e._tool_body_id].clone() - e.scene.env_origins,
        mechanism=e.button.data.body_link_pos_w.torch[:, e._button_body_id].clone() - e.scene.env_origins,
        base=e.robot.data.body_link_pos_w.torch[:, b].clone() - e.scene.env_origins,
        motor_force=e._thrusters.force.clone(),
    )


def verify(d, c):
    for key, value in d.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {key}")

    def col(key):
        return np.atleast_2d(d[key].T).T

    eligible = (col("q") >= c["threshold"]) & (col("q") <= c["max_travel"])
    eligible &= col("alignment") > c["alignment"]
    for key in ["distance", "attitude", "angular_speed"]:
        eligible &= col(key) < c[key]
    eligible &= (col("tool_speed") <= c["tool_speed"]) & (abs(col("mechanism_speed")) <= c["mechanism_speed"])
    counts = np.zeros(eligible.shape[1], dtype=int)
    seen = np.zeros_like(counts, dtype=bool)
    first = np.full_like(counts, -1)
    for i, row in enumerate(eligible):
        counts = np.where(row, counts + 1, 0)
        seen |= counts >= c["hold_steps"]
        valid = col("active")[i] & ~col("terminal")[i]
        if np.any(seen[valid] != col("success")[i, valid]):
            raise ValueError(f"Contract mismatch row{i}")
        first[(first < 0) & seen & valid] = i + 1
    return dict(contract_replayed=True, first_success_step=first.tolist(), success_per_seed=(first >= 0).tolist())
