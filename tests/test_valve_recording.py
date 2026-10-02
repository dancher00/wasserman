import math

import pytest

from wasman.controllers.valve_recording import verified_recording
from wasman.controllers.valve_success import AMB_VALVE_SUCCESS, valve_success_metadata


def recording():
    frames = []
    angles = [0, *range(1, 174), *([173] * 50)]
    for i, angle in enumerate(angles):
        released = i >= 204
        frames.append(
            {
                "t": (i + 1) / 30,
                "navigation": None,
                "angle_rad": [math.radians(angle)],
                "wheel_speed_rad_s": [0.0],
                "finger_forces_n": [[0.0, 0.0] if released else [0.5, 0.5]],
                "wheel_clearance_m": [0.10 if released else 0.01],
                "tool_speed_m_s": [0.0],
                "base_attitude_rad": [0.0],
                "base_angular_speed_rad_s": [0.0],
                "tool_axis_alignment": [1.0],
                "opposing_contacts": [not released],
                "success": [i >= 218],
            }
        )
    return {
        "num_envs": 1,
        "failed_resets": 0,
        "steps": len(frames),
        "trace": frames,
        "protocol": {"policy_dt_s": 1 / 30},
        "successes": 1,
        "task": "test",
        "controller": "test expert",
        "checkpoint_sha256": None,
        "seed": 42,
        "source_sha256_at_start": {},
    }


def test_verified_recording_requires_measured_completion():
    result = verified_recording(recording())
    assert result["telemetry_contract_replayed"]
    assert [c["label"] for c in result["chapters"]] == ["Approach", "Turn", "Grasp", "Hold", "Release", "Complete"]
    assert result["final_angle_deg"] == pytest.approx(173)


@pytest.mark.parametrize("fault", ["reset", "false_success", "one_finger", "gap", "final_drift", "unstable_end"])
def test_rejects_invalid_recordings(fault):
    trial = recording()
    if fault == "reset":
        trial["failed_resets"] = 1
    elif fault == "false_success":
        trial["trace"][0]["success"] = [True]
    elif fault == "one_finger":
        for frame in trial["trace"]:
            frame["finger_forces_n"][0][1] = 0
    elif fault == "gap":
        trial["trace"][3]["t"] += 1
    elif fault == "final_drift":
        trial["trace"][-1]["angle_rad"] = [math.radians(176)]
    else:
        trial["trace"][-1]["base_attitude_rad"] = [0.30]
    with pytest.raises(ValueError):
        verified_recording(trial)


def test_angle_only_recording_accepts_turn_without_grasp_hold_or_release():
    trial = recording()
    trial["success_contract"] = valve_success_metadata(AMB_VALVE_SUCCESS)
    for frame in trial["trace"]:
        frame["finger_forces_n"] = [[0.0, 0.0]]
        frame["wheel_clearance_m"] = [0.01]
        frame["success"] = [frame["angle_rad"][0] >= math.radians(170)]
        frame["strict_valve_success"] = [False]
    trial["trace"][-1]["angle_rad"] = [math.radians(185)]
    result = verified_recording(trial)
    assert result["success_contract"]["version"] == AMB_VALVE_SUCCESS
    assert [c["label"] for c in result["chapters"]] == ["Approach", "Complete"]
    trial["trace"][0]["success"] = [True]
    with pytest.raises(ValueError, match="Reported success"):
        verified_recording(trial)
