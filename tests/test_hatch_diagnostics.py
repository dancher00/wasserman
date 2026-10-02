import copy

import numpy as np
import pytest

from wasman.learning.hatch_diagnostics import analyze_hatch_rollout


def fixture():
    steps, count, dt = 9, 2, 1 / 30
    metadata = {
        "num_envs": count,
        "steps": steps,
        "dt_s": dt,
        "sample_every": 3,
        "seed": 3,
        "mode": "evaluate",
        "success_per_env": [False, True],
        "terminal_per_env": [False, False],
        "censored_per_env": [False, False],
        "proprio_fields": [["base_linear_velocity_world", 3], ["arm_joint_positions", 4]],
    }
    trace = []
    for step in range(steps):
        trace.append(
            {
                "t": (step + 1) * dt,
                "angle_rad": [0.0, np.deg2rad(min(85, step * 20))],
                "speed_rad_s": [0.0, 0.0],
                "force_vectors": [[[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [-1, 0, 0]]],
                "distance": [0.2, 0.02],
                "tool_speed": [0.0, 0.0],
                "attitude": [0.0, 0.0],
                "base_angular_speed": [0.0, 0.0],
                "action": [[0.0] * 11, [0.0] * 11],
            }
        )
    trace[3]["action"][0][8] = 0.5
    state = np.zeros((3, count, 7))
    state[1, 0, 0] = 0.08
    state[:, 0, 5] = [2.6, 1.7, 1.6]
    return metadata, trace, state, np.arange(0, steps, 3), np.zeros((steps, count), dtype=bool)


def test_contact_and_pre_action_timestamps_remain_distinct():
    report = analyze_hatch_rollout(*fixture())
    e0, e1 = report["environments"]
    assert e0["robot_state_sampled_proxies"]["first_deployment_command_s"] == pytest.approx(0.1)
    assert e0["robot_state_sampled_proxies"]["base_speed_at_first_deployment_m_s"] == pytest.approx(0.08)
    assert e0["robot_state_sampled_proxies"]["arm_c_refold_events_s"] == pytest.approx([0.1])
    assert e1["contact_post_action"]["first_bilateral_s"] == pytest.approx(1 / 30)
    assert e1["contact_post_action"]["first_opposing_near_grasp_s"] == pytest.approx(1 / 30)
    assert e1["contact_post_action"]["hold_conditions_longest_s"] == pytest.approx(5 / 30)
    assert e1["contact_post_action"]["hold_conditions_final_run_s"] == pytest.approx(5 / 30)
    assert report["timing"]["contact_hz"] == 30
    assert report["timing"]["robot_state_hz"] == 10


def test_bilateral_contact_is_not_an_opposing_grasp_or_task_success():
    meta, trace, state, steps, terminal = fixture()
    meta["success_per_env"] = [False, False]
    for row in trace:
        row["force_vectors"][1][1] = [1, 0, 0]
    report = analyze_hatch_rollout(meta, trace, state, steps, terminal)
    contact = report["environments"][1]["contact_post_action"]
    assert contact["first_bilateral_s"] is not None
    assert contact["first_opposing_near_grasp_s"] is None
    assert contact["hold_conditions_longest_s"] == 0
    assert report["recorded_successes"] == 0


def test_proxies_do_not_override_recorded_success():
    data = fixture()
    data[0]["success_per_env"][1] = False
    report = analyze_hatch_rollout(*data)
    assert report["environments"][1]["contact_post_action"]["hold_conditions_longest_s"] > 0
    assert report["environments"][1]["recorded_success"] is False


def test_terminal_reset_row_excluded_and_peer_censor_preserved():
    meta, trace, state, steps, terminal = fixture()
    terminal[3, 0] = True
    meta["terminal_per_env"][0] = True
    meta["censored_per_env"][1] = True
    trace[3]["distance"][0] = 900
    report = analyze_hatch_rollout(meta, trace, state, steps, terminal)
    e0, e1 = report["environments"]
    assert e0["usable_post_action_rows"] == 3
    assert e0["excluded_terminal_and_later_rows"] == 6
    assert e0["contact_post_action"]["handle_distance_max_m"] == 0.2
    assert e1["censored"] and not e1["terminal"]


def test_immediate_terminal_empty_metrics_are_not_invented():
    data = fixture()
    data[0]["terminal_per_env"][0] = True
    data[4][0, 0] = True
    report = analyze_hatch_rollout(*data)
    env = report["environments"][0]
    assert env["contact_post_action"]["max_angle_deg"] is None
    assert env["robot_state_sampled_proxies"]["first_deployment_command_s"] is None


@pytest.mark.parametrize("kind", ["timestamp", "sample", "nonfinite", "terminal"])
def test_bad_inputs_are_rejected(kind):
    meta, trace, state, steps, terminal = fixture()
    if kind == "timestamp":
        trace[2]["t"] = 0
    elif kind == "sample":
        steps[1] = 0
    elif kind == "nonfinite":
        trace[2]["angle_rad"][0] = float("nan")
    else:
        terminal[0, 0] = True
    with pytest.raises(ValueError):
        analyze_hatch_rollout(meta, trace, state, steps, terminal)


def test_metrics_do_not_mutate_input():
    data = fixture()
    before = copy.deepcopy(data)
    analyze_hatch_rollout(*data)
    assert data[0:2] == before[0:2]
    for actual, expected in zip(data[2:], before[2:], strict=True):
        np.testing.assert_array_equal(actual, expected)
