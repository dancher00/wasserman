"""The benchmark angle gate and the legacy physical gate are independent."""

import math

import pytest
import torch

from wasman.controllers.valve_success import (
    AMB_VALVE_SUCCESS,
    STRICT_VALVE_SUCCESS,
    select_valve_success,
    valve_angle_criteria,
    valve_success_metadata,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_angle_threshold_includes_boundary_and_has_no_upper_bound(dtype):
    angles = torch.tensor([-190, -170, 0, 8.499, 8.5, 169.999, 170, 175, 185, 360], dtype=dtype)
    criteria = valve_angle_criteria(angles * (math.pi / 180))
    assert criteria["valve_rotated"].tolist() == [False] * 6 + [True] * 4
    assert criteria["valve_engaged"].tolist() == [False] * 4 + [True] * 6


def test_angle_success_does_not_require_or_inherit_the_physical_contract():
    angles = torch.tensor([math.radians(170), 0.0])
    strict = torch.tensor([False, True])
    assert select_valve_success(angles, strict, AMB_VALVE_SUCCESS).tolist() == [True, False]
    assert select_valve_success(angles, strict, STRICT_VALVE_SUCCESS).tolist() == [False, True]


def test_contract_metadata_distinguishes_new_and_historical_scores():
    current = valve_success_metadata(AMB_VALVE_SUCCESS)
    legacy = valve_success_metadata(STRICT_VALVE_SUCCESS)
    assert current["goal_angle_deg"] == [170, None]
    assert current["engagement_angle_deg"] == 8.5
    assert not any(current[key] for key in ("requires_bilateral_contact", "requires_hold", "requires_release"))
    assert legacy["goal_angle_deg"] == [170, 175]
    assert legacy["hold_s"] == 1.0
    assert legacy["release_s"] == 0.5
    assert not current["success_ends_episode"]


def test_unknown_contract_is_rejected_instead_of_silently_changing_scores():
    with pytest.raises(ValueError, match="Unknown valve success contract"):
        valve_success_metadata("typo")
    with pytest.raises(ValueError, match="Unknown valve success contract"):
        select_valve_success(torch.zeros(1), torch.zeros(1, dtype=torch.bool), "typo")
