import math

import pytest
import torch

from scripts.rescore_valve_angle import score

pytestmark = pytest.mark.unit


def test_rescore_excludes_success_on_and_after_reset_but_retains_prior_success():
    angle = torch.tensor([[0, 170, 0], [175, 0, 0], [180, 180, 170]]) * (math.pi / 180)
    done = torch.tensor([[False, False, False], [True, False, False], [False, True, False]])
    result = score(angle, done.int().cumsum(0) == 0, 0.1)
    assert result["success_per_env"] == [False, True, True]
    assert result["successes"] == 2
    assert result["first_success_time_s"] == [None, 0.1, pytest.approx(0.3)]
    assert result["max_angle_before_reset_deg"] == pytest.approx([0, 170, 170])
