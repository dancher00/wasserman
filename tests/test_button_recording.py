"""The hero publisher never accepts a success flag without its physical gate."""

from copy import deepcopy

import pytest

from wasman.controllers.button_recording import verified_button_recording


def recording():
    record = {
        "task": "button",
        "seed": 42,
        "fps": 30,
        "automatic_resets": 0,
        "control": "hybrid",
        "checkpoint_sha256": "test",
        "source_sha256": {},
        "success_protocol": {
            "hold_steps": 3,
            "min_depth_m": 0.004,
            "max_depth_m": 0.007,
            "max_tool_speed_m_s": 0.035,
            "max_attitude_rad": 0.25,
            "max_angular_speed_rad_s": 0.35,
            "max_tool_distance_m": 0.13,
            "min_tool_alignment": 0.7,
            "max_mechanism_speed_m_s": None,
        },
        "trace": [],
    }
    for index, phase in enumerate(
        ["transit", "deployment", "manipulation", "contact_hold", "contact_hold", "contact_hold"]
    ):
        record["trace"].append(
            {
                "t": (index + 1) / 30,
                "phase": phase,
                "depth_m": 0.005 if index >= 3 else 0,
                "tool_distance_m": 0.02,
                "tool_axis_alignment": 0.99,
                "base_attitude_error_rad": 0.02,
                "base_angular_speed_rad_s": 0.01,
                "tool_speed_m_s": 0.01,
                "button_speed_m_s": 0.001,
                "button_indicator_on": index >= 3,
                "success": index == 5,
                "arm_joint_position_rad": [0, 0, 0.2, 0],
                "arm_joint_target_rad": [0, 0, 0.2, 0],
            }
        )
    return record


def test_button_film_requires_actual_continuous_hold():
    record = recording()
    assert verified_button_recording(record)["telemetry_contract_replayed"]
    for field, value in [("depth_m", 0.001), ("base_angular_speed_rad_s", 2), ("tool_axis_alignment", 0.5)]:
        invalid = deepcopy(record)
        invalid["trace"][4][field] = value
        with pytest.raises(ValueError, match="replay"):
            verified_button_recording(invalid)


def test_button_film_rejects_second_press_and_unfolded_transit():
    record = recording()
    record["trace"][4]["button_indicator_on"] = False
    with pytest.raises(ValueError, match="uninterrupted press"):
        verified_button_recording(record)
    record = recording()
    record["trace"][0]["arm_joint_position_rad"][2] = 1
    with pytest.raises(ValueError, match="folded"):
        verified_button_recording(record)


@pytest.mark.parametrize("field", ["tool_speed_m_s", "arm_joint_position_rad", "arm_joint_target_rad"])
def test_button_film_rejects_nonfinite_telemetry_after_latched_success(field):
    record = recording()
    frame = deepcopy(record["trace"][-1])
    frame["t"] += 1 / record["fps"]
    if isinstance(frame[field], list):
        frame[field][0] = float("nan")
    else:
        frame[field] = float("inf")
    record["trace"].append(frame)
    with pytest.raises(ValueError, match="Non-finite"):
        verified_button_recording(record)
