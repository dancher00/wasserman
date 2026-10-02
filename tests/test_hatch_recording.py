import copy
import math

import pytest

from wasman.controllers.hatch_recording import replay_hatch_trace


def trace():
    rows = []
    for i, deg in enumerate(list(range(83)) + [82] * 30):
        rows.append(
            {
                "t": (i + 1) / 30,
                "angle_rad": [math.radians(deg)],
                "speed_rad_s": [0.2 if i < 83 else 0.0],
                "force_vectors": [[[1.0, 0, 0], [-1.0, 0, 0]]],
                "distance": [0.005],
                "tool_speed": [0.0],
                "attitude": [0.0],
                "base_angular_speed": [0.0],
                "action": [[0.0] * 11],
                "navigation_ready": [True],
                "success": [i == 112],
            }
        )
    return rows


def test_replay_measured_opening():
    assert replay_hatch_trace(trace()) == {"first_success_step": [113], "successes": 1}


@pytest.mark.parametrize("change", ["fake_success", "missing_frame", "no_contact", "early_arm", "nan"])
def test_reject_invented_success_and_corrupted_recording(change):
    rows = copy.deepcopy(trace())
    if change == "fake_success":
        rows[0]["success"] = [True]
    elif change == "missing_frame":
        rows.pop(10)
    elif change == "no_contact":
        for row in rows:
            row["force_vectors"] = [[[0.0, 0, 0], [0.0, 0, 0]]]
    elif change == "early_arm":
        rows[0]["navigation_ready"] = [False]
        rows[0]["action"][0][6] = 0.1
    else:
        rows[0]["angle_rad"] = [float("nan")]
    with pytest.raises(ValueError):
        replay_hatch_trace(rows)
