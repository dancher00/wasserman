"""Audit the published recording's simulation trace, independently of its video."""

import json
from pathlib import Path

import pytest

TRACE = Path(__file__).parent / "fixtures/recorded_approach/final_approach_cinema/trace.json"


@pytest.mark.unit
def test_published_approach_is_continuous_and_holds_arm_during_transit():
    recording = json.loads(TRACE.read_text())
    trace = recording["trace"]
    assert len(trace) == 1080
    assert recording["playback_speed"] == 1
    assert recording["initial_base_position_w_m"][0] == pytest.approx(-0.85, abs=0.035)
    assert recording["initial_tool_distance_m"] > 1.0
    for i, frame in enumerate(trace):
        assert frame["t"] == pytest.approx((i + 1) / 30)
    assert list(dict.fromkeys(frame["phase"] for frame in trace)) == [
        "transit",
        "deployment",
        "manipulation",
        "contact_hold",
    ]
    release = next(i for i, frame in enumerate(trace) if frame["phase"] == "deployment")
    assert 0 < release < len(trace) - 30
    transit = trace[:release]
    assert all(frame["arm_joint_position_rad"][2] < 0.4 for frame in transit)
    assert all(frame["arm_joint_target_rad"] == transit[0]["arm_joint_target_rad"] for frame in transit)
    assert all(not frame["button_indicator_on"] and not frame["success"] for frame in transit)
    # Physical servos can deflect; this verifies the arm is not being animated.
    assert (
        max(
            abs(q - initial)
            for frame in transit
            for q, initial in zip(frame["arm_joint_position_rad"], transit[0]["arm_joint_position_rad"], strict=True)
        )
        < 0.01
    )
    assert max(frame["base_speed_m_s"] for frame in transit[-12:]) < 0.04
    assert trace[release]["tool_distance_m"] > 0.20
    assert trace[release]["base_position_w_m"][0] == pytest.approx(-0.08, abs=0.05)
    assert trace[-1]["success"]
    first_press = next(i for i, frame in enumerate(trace) if frame["button_indicator_on"])
    assert all(frame["button_indicator_on"] for frame in trace[first_press:])
    assert trace[-1]["t"] - trace[first_press]["t"] > 5.0
    assert max(frame["depth_m"] for frame in trace) < 0.007
    assert trace[-1]["base_position_w_m"][0] - recording["initial_base_position_w_m"][0] > 0.9


@pytest.mark.unit
def test_recorded_arm_release_has_no_target_jump():
    trace = json.loads(TRACE.read_text())["trace"]
    for previous, current in zip(trace[:-1], trace[1:], strict=True):
        assert (
            max(
                abs(a - b)
                for a, b in zip(previous["arm_joint_target_rad"], current["arm_joint_target_rad"], strict=True)
            )
            <= 0.35 / 30 + 1e-6
        )


@pytest.mark.unit
def test_evaluation_counts_exactly_two_trials_per_environment():
    result = json.loads((TRACE.parents[1] / "press_button_approach_evaluation.json").read_text())
    assert result["completed_per_env"] == [2] * 64
    assert result["episodes"] == 128
    assert result["episode_length_s"] == 40
    assert result["success_hold_steps"] == 90
