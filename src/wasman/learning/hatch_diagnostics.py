"""Read-only NumPy diagnostics; observed proxies never replace recorded task success."""

import numpy as np

THRESHOLDS = {
    "bilateral_each_finger_n": 0.15,
    "opposing_grasp_each_finger_n": 0.10,
    "opposing_force_unit_dot_less_than": -0.25,
    "near_handle_m": 0.06,
    "arm_deployment_max_abs_command": 0.02,
    "arm_c_deployed_rad": 2.5,
    "arm_c_refolded_rad": 1.8,
    "angular_hold_min_deg": 80.0,
    "angular_hold_max_deg": 105.0,
    "grasp_motion_min_deg": 75.0,
    "ungrasped_motion_max_deg": 5.0,
    "lid_hold_speed_max_rad_s": 0.05,
    "tool_hold_speed_max_m_s": 0.04,
    "base_attitude_max_rad": 0.25,
    "base_angular_speed_max_rad_s": 0.35,
}


def _first(mask, times):
    indices = np.flatnonzero(mask)
    return float(times[indices[0]]) if len(indices) else None


def _longest(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def _trailing(mask):
    missing = np.flatnonzero(~mask[::-1])
    return int(missing[0]) if len(missing) else len(mask)


def analyze_hatch_rollout(metadata, diagnostics, proprio, sample_steps, terminal=None):
    """Analyze one no-reset batch without loading images, querying a policy or GPU.

    Contact rows are POST-action at ``(step+1)*dt``. Proprioception is PRE-action
    at ``sample_step*dt``; joint/velocity timing is therefore sampled/proxy only.
    A terminal row can contain auto-reset state, so it and all following rows for
    that environment are excluded from physical metrics, not counted as motion.
    """
    count, dt = metadata["num_envs"], metadata["dt_s"]
    steps = len(diagnostics)
    if steps != metadata["steps"] or steps < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive dt and nonempty trace matching metadata required")
    times = np.array([row["t"] for row in diagnostics], dtype=float)
    if not np.allclose(times, (np.arange(steps) + 1) * dt, rtol=0, atol=1e-5):
        raise ValueError("Discontinuous post-action trace timestamps")
    fields = {}
    for name, shape in {
        "angle_rad": (steps, count),
        "speed_rad_s": (steps, count),
        "force_vectors": (steps, count, 2, 3),
        "distance": (steps, count),
        "tool_speed": (steps, count),
        "attitude": (steps, count),
        "base_angular_speed": (steps, count),
        "action": (steps, count, 11),
    }.items():
        values = np.array([row[name] for row in diagnostics], dtype=float)
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid diagnostic field {name}")
        fields[name] = values
    sample_steps = np.asarray(sample_steps)
    proprio = np.asarray(proprio)
    slices, offset = {}, 0
    for name, width in metadata["proprio_fields"]:
        slices[name] = slice(offset, offset + width)
        offset += width
    if proprio.shape != (len(sample_steps), count, offset) or not np.isfinite(proprio).all():
        raise ValueError("Invalid proprioception shape or values")
    if (
        len(sample_steps) == 0
        or not np.issubdtype(sample_steps.dtype, np.integer)
        or sample_steps[0] != 0
        or np.any(np.diff(sample_steps) != metadata["sample_every"])
        or sample_steps[-1] >= steps
    ):
        raise ValueError("Invalid pre-action sample steps")
    if slices["arm_joint_positions"].stop - slices["arm_joint_positions"].start != 4:
        raise ValueError("Expected Alpha four-joint arm schema")
    statuses = {}
    for name in ("success_per_env", "terminal_per_env", "censored_per_env"):
        status = np.asarray(metadata[name])
        if status.shape != (count,) or status.dtype != np.bool_:
            raise ValueError(f"Invalid recorded status {name}")
        statuses[name] = status
    if terminal is None:
        terminal = np.zeros((steps, count), dtype=bool)
        terminal[-1] = statuses["terminal_per_env"]
        terminal_source = "metadata final flags; final terminal row excluded"
    else:
        terminal = np.asarray(terminal)
        if terminal.shape != (steps, count) or terminal.dtype != np.bool_:
            raise ValueError("Invalid terminal buffer")
        if not np.array_equal(terminal.any(0), statuses["terminal_per_env"]):
            raise ValueError("Terminal buffer disagrees with recorded metadata")
        terminal_source = "terminal.npy"
    magnitudes = np.linalg.norm(fields["force_vectors"], axis=-1)
    units = fields["force_vectors"] / np.maximum(magnitudes[..., None], 1e-6)
    dot = (units[:, :, 0] * units[:, :, 1]).sum(-1)
    bilateral = (magnitudes > THRESHOLDS["bilateral_each_finger_n"]).all(-1)
    opposing = (magnitudes > THRESHOLDS["opposing_grasp_each_finger_n"]).all(-1)
    opposing &= (dot < THRESHOLDS["opposing_force_unit_dot_less_than"]) & (
        fields["distance"] < THRESHOLDS["near_handle_m"]
    )
    environments = []
    for env_id in range(count):
        ended = np.flatnonzero(terminal[:, env_id])
        limit = int(ended[0]) if len(ended) else steps
        observed_times = times[:limit]
        samples = sample_steps < limit
        sample_times = sample_steps[samples] * dt
        arm = proprio[samples, env_id, slices["arm_joint_positions"]]
        deployed, folds = False, []
        for time, joint_c in zip(sample_times, arm[:, 2], strict=True):
            if joint_c >= THRESHOLDS["arm_c_deployed_rad"]:
                deployed = True
            elif deployed and joint_c < THRESHOLDS["arm_c_refolded_rad"]:
                folds.append(float(time))
                deployed = False
        actions = fields["action"][sample_steps[samples], env_id]
        deploy = np.max(np.abs(actions[:, 6:10]), axis=-1) > THRESHOLDS["arm_deployment_max_abs_command"]
        start = np.flatnonzero(deploy)
        start_speed = (
            float(np.linalg.norm(proprio[samples, env_id, slices["base_linear_velocity_world"]][start[0]]))
            if len(start)
            else None
        )
        angle = fields["angle_rad"][:limit, env_id]
        grasp = opposing[:limit, env_id]
        delta = np.diff(np.r_[0.0, angle])
        engaged = grasp & np.r_[False, grasp[:-1]] if limit else grasp
        grasp_motion = np.cumsum(np.where(engaged, delta, 0))
        ungrasped_motion = np.cumsum(np.where(~engaged, np.abs(delta), 0))
        hold = (angle >= np.deg2rad(THRESHOLDS["angular_hold_min_deg"])) & (
            angle <= np.deg2rad(THRESHOLDS["angular_hold_max_deg"])
        )
        hold &= (grasp_motion >= np.deg2rad(THRESHOLDS["grasp_motion_min_deg"])) & (
            ungrasped_motion <= np.deg2rad(THRESHOLDS["ungrasped_motion_max_deg"])
        )
        hold &= grasp & (np.abs(fields["speed_rad_s"][:limit, env_id]) <= THRESHOLDS["lid_hold_speed_max_rad_s"])
        hold &= fields["tool_speed"][:limit, env_id] <= THRESHOLDS["tool_hold_speed_max_m_s"]
        hold &= fields["attitude"][:limit, env_id] < THRESHOLDS["base_attitude_max_rad"]
        hold &= fields["base_angular_speed"][:limit, env_id] < THRESHOLDS["base_angular_speed_max_rad_s"]
        progress_mask = (angle > np.deg2rad(5)) & (angle < np.deg2rad(30))
        progress = {
            "angle_band_deg": [5, 30],
            "observed_seconds_in_band": float(progress_mask.sum() * dt),
            "median_lid_speed_rad_s": float(np.median(fields["speed_rad_s"][:limit, env_id][progress_mask]))
            if progress_mask.any()
            else None,
            "median_handle_distance_m": float(np.median(fields["distance"][:limit, env_id][progress_mask]))
            if progress_mask.any()
            else None,
            "median_grip_command": float(np.median(fields["action"][:limit, env_id, 10][progress_mask]))
            if progress_mask.any()
            else None,
        }
        environments.append(
            {
                "env_id": env_id,
                "recorded_success": bool(statuses["success_per_env"][env_id]),
                "terminal": bool(statuses["terminal_per_env"][env_id]),
                "censored": bool(statuses["censored_per_env"][env_id]),
                "usable_post_action_rows": limit,
                "excluded_terminal_and_later_rows": steps - limit,
                "contact_post_action": {
                    "first_bilateral_s": _first(bilateral[:limit, env_id], observed_times),
                    "bilateral_total_s": float(bilateral[:limit, env_id].sum() * dt),
                    "first_opposing_near_grasp_s": _first(grasp, observed_times),
                    "opposing_near_grasp_total_s": float(grasp.sum() * dt),
                    "handle_distance_min_m": float(fields["distance"][:limit, env_id].min()) if limit else None,
                    "handle_distance_max_m": float(fields["distance"][:limit, env_id].max()) if limit else None,
                    "max_angle_deg": float(np.rad2deg(angle.max())) if limit else None,
                    "final_angle_deg": float(np.rad2deg(angle[-1])) if limit else None,
                    "first_80deg_s": _first(angle >= np.deg2rad(80), observed_times),
                    "final_lid_speed_rad_s": float(fields["speed_rad_s"][limit - 1, env_id]) if limit else None,
                    "observed_grasp_motion_deg": float(np.rad2deg(grasp_motion[-1])) if limit else None,
                    "observed_ungrasped_motion_deg": float(np.rad2deg(ungrasped_motion[-1])) if limit else None,
                    "hold_conditions_longest_s": float(_longest(hold) * dt),
                    "hold_conditions_final_run_s": float(_trailing(hold) * dt),
                    "max_base_attitude_rad": float(fields["attitude"][:limit, env_id].max()) if limit else None,
                    "max_base_angular_speed_rad_s": float(fields["base_angular_speed"][:limit, env_id].max())
                    if limit
                    else None,
                    "early_lift": progress,
                },
                "robot_state_sampled_proxies": {
                    "first_deployment_command_s": _first(deploy, sample_times),
                    "base_speed_at_first_deployment_m_s": start_speed,
                    "arm_c_refold_events_s": folds,
                    "arm_c_refold_count": len(folds),
                    "note": (
                        "Pre-action sampled robot state; event times are sample-resolution approximations, "
                        "not expert phases."
                    ),
                },
            }
        )
    return {
        "schema": "wasman-hatch-rollout-diagnostics-v1",
        "seed": metadata["seed"],
        "mode": metadata["mode"],
        "num_envs": count,
        "recorded_successes": int(statuses["success_per_env"].sum()),
        "recorded_terminals": int(statuses["terminal_per_env"].sum()),
        "recorded_censored": int(statuses["censored_per_env"].sum()),
        "horizon_s": float(steps * dt),
        "timing": {
            "contact_hz": 1 / dt,
            "robot_state_hz": 1 / (dt * metadata["sample_every"]),
            "contact_time": "post-action (step+1)*dt",
            "robot_state_time": "pre-action sample_step*dt",
        },
        "terminal_exclusion_source": terminal_source,
        "thresholds": dict(THRESHOLDS),
        "success_authority": (
            "Recorded metadata only; NumPy hold/contact/fold diagnostics are not new task success labels."
        ),
        "environments": environments,
    }
