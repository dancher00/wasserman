"""Independent replay gate for the single-press, staged button film."""

import math


def verified_button_recording(record):
    trace, protocol = record["trace"], record["success_protocol"]
    if not trace or record["automatic_resets"] != 0:
        raise ValueError("Expected an uninterrupted button recording")
    count, succeeded = 0, False
    for i, frame in enumerate(trace):
        measured = [
            frame[name]
            for name in (
                "t",
                "depth_m",
                "tool_distance_m",
                "tool_axis_alignment",
                "base_attitude_error_rad",
                "base_angular_speed_rad_s",
                "tool_speed_m_s",
                "button_speed_m_s",
            )
        ]
        measured.extend(frame["arm_joint_position_rad"])
        measured.extend(frame["arm_joint_target_rad"])
        if not all(math.isfinite(value) for value in measured):
            raise ValueError("Non-finite measured telemetry, including after latched success")
        if not math.isclose(frame["t"], (i + 1) / record["fps"], abs_tol=1e-5):
            raise ValueError("Discontinuous telemetry")
        pressed = (
            protocol["min_depth_m"] <= frame["depth_m"]
            and (protocol["max_depth_m"] is None or frame["depth_m"] <= protocol["max_depth_m"])
            and frame["tool_distance_m"] < protocol["max_tool_distance_m"]
            and frame["tool_axis_alignment"] > protocol["min_tool_alignment"]
            and frame["base_attitude_error_rad"] < protocol["max_attitude_rad"]
            and frame["base_angular_speed_rad_s"] < protocol["max_angular_speed_rad_s"]
            and (protocol["max_tool_speed_m_s"] is None or frame["tool_speed_m_s"] <= protocol["max_tool_speed_m_s"])
            and (
                protocol["max_mechanism_speed_m_s"] is None
                or abs(frame["button_speed_m_s"]) <= protocol["max_mechanism_speed_m_s"]
            )
        )
        count = count + 1 if pressed else 0
        succeeded |= count >= protocol["hold_steps"]
        if succeeded != frame["success"]:
            raise ValueError("Reported success disagrees with measured gate replay")
    if not succeeded:
        raise ValueError("No completed sustained button press")
    first_press = next((i for i, frame in enumerate(trace) if frame["button_indicator_on"]), None)
    if first_press is None or not all(frame["button_indicator_on"] for frame in trace[first_press:]):
        raise ValueError("Button indication must be one uninterrupted press")
    expected_phases = ["transit", "deployment", "manipulation", "contact_hold"]
    if list(dict.fromkeys(frame["phase"] for frame in trace)) != expected_phases:
        raise ValueError("Incorrect approach/deployment/contact sequence")
    transit = [frame for frame in trace if frame["phase"] == "transit"]
    if any(frame["arm_joint_position_rad"][2] >= 0.4 for frame in transit):
        raise ValueError("Arm was not folded during transit")
    if any(frame["arm_joint_target_rad"] != transit[0]["arm_joint_target_rad"] for frame in transit):
        raise ValueError("Arm target changed during transit")
    return {
        "task": record["task"],
        "seed": record["seed"],
        "controller": record["control"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "source_sha256": record["source_sha256"],
        "moviepy_version": record.get("moviepy_version"),
        "automatic_resets": 0,
        "telemetry_contract_replayed": True,
        "episodes": 1,
        "successes": 1,
        "evaluation_role": "Selected hybrid-controller demonstration, not an end-to-end learned approach",
        "duration_s": len(trace) / record["fps"],
        "first_success_time_s": next(f["t"] for f in trace if f["success"]),
        "chapters": [
            {"time": next(f["t"] for f in trace if f["phase"] == phase) - 1 / record["fps"], "label": label}
            for phase, label in zip(expected_phases, ["Swim in", "Deploy", "Align", "Hold"], strict=True)
        ],
    }
