"""Raw physical evidence and independent NumPy replay of marine success criteria."""

import numpy as np


def contract(cfg):
    return dict(
        initial=cfg.mechanism_initial_position,
        direction=cfg.mechanism_direction,
        threshold=cfg.button_pressed_threshold,
        distance=cfg.tool_contact_distance,
        alignment=cfg.tool_contact_alignment,
        attitude=cfg.success_max_attitude_error,
        angular_speed=cfg.success_max_angular_speed,
        hold_steps=cfg.success_hold_steps,
        unit=cfg.mechanism_progress_unit,
    )


def capture(env):
    import torch

    e = env
    return dict(
        q=e.button.data.joint_pos.torch[:, e._button_joint_id].clone(),
        mechanism_speed=e.button.data.joint_vel.torch[:, e._button_joint_id].clone(),
        distance=e._distance.clone(),
        alignment=e._alignment.clone(),
        attitude=e._base_attitude_error.clone(),
        angular_speed=e._base_angular_speed.clone(),
        forces=torch.stack(
            [
                e.scene[n].data.normal_force_matrix_w.torch.reshape(e.num_envs, -1, 3).sum(1)
                for n in ["left_contact", "right_contact"]
            ],
            1,
        ),
        tool=e.robot.data.body_link_pos_w.torch[:, e._tool_body_id].clone() - e.scene.env_origins,
        mechanism=e.button.data.body_link_pos_w.torch[:, e._button_body_id].clone() - e.scene.env_origins,
    )


def verify(data, criteria):
    """Replay consecutive-step success without using the recorded success flag."""
    d, c = data, criteria
    for k in ["q", "distance", "alignment", "attitude", "angular_speed", "forces"]:
        if not np.isfinite(d[k]).all():
            raise ValueError(f"Nonfinite {k}")
    q = np.atleast_2d(d["q"].T).T
    qualified = (q - c["initial"]) * c["direction"] >= c["threshold"]
    for key, threshold, greater in [
        ("distance", c["distance"], False),
        ("alignment", c["alignment"], True),
        ("attitude", c["attitude"], False),
        ("angular_speed", c["angular_speed"], False),
    ]:
        values = np.atleast_2d(d[key].T).T
        qualified &= values > threshold if greater else values < threshold
    active = np.atleast_2d(d["active"].T).T
    terminal = np.atleast_2d(d["terminal"].T).T
    observed = np.atleast_2d(d["success"].T).T
    counts = np.zeros(q.shape[1], dtype=int)
    seen = np.zeros(q.shape[1], dtype=bool)
    first = np.full(q.shape[1], -1, dtype=int)
    for i in range(len(q)):
        counts = np.where(qualified[i], counts + 1, 0)
        seen |= counts >= c["hold_steps"]
        valid = active[i] & ~terminal[i]
        if np.any(seen[valid] != observed[i, valid]):
            raise ValueError(f"Success contract mismatch at row{i}")
        new = (first < 0) & seen & valid
        first[new] = i + 1
    return dict(contract_replayed=True, first_success_step=first.tolist(), success_per_seed=(first >= 0).tolist())
