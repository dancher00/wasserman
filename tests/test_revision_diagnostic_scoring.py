import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from score_revision_diagnostics import (  # noqa: E402
    initialization_comparison,
    paired_metrics,
    success_effect_sensitivity,
)

pytestmark = pytest.mark.unit


def test_reset_pairing_is_distinct_from_condition_active_warmup():
    reference = dict(seeds=[1, 2], initial_reset=dict(base_position=np.zeros((2, 3)).tolist()))
    variant = dict(seeds=[1, 2], initial_reset=dict(base_position=np.zeros((2, 3)).tolist()))
    left = [dict(measured=np.zeros((2, 7)))]
    right = [dict(measured=np.zeros((2, 7)))]
    right[0]["measured"][0, 0] = 2e-7
    result = initialization_comparison("RotateValve", reference, variant, left, right)
    assert result["reset_bitwise_equal_by_field"] == {"base_position": True}
    assert not result["post_warmup_measured_bitwise_equal"]
    assert result["post_warmup_max_abs_difference_by_channel"]["tool_x_m"] == 2e-7
    variant["initial_reset"]["base_position"][0][0] = 0.001
    with pytest.raises(ValueError, match="paired reset mismatch"):
        initialization_comparison("RotateValve", reference, variant, left, right)


def test_initialization_comparison_rejects_broadcast_reset_and_nonfinite_observation():
    reference = dict(seeds=[1, 2], initial_reset=dict(base_position=np.zeros((2, 3)).tolist()))
    variant = dict(seeds=[1, 2], initial_reset=dict(base_position=np.zeros((1, 3)).tolist()))
    trace = [dict(measured=np.zeros((2, 8)))]
    with pytest.raises(ValueError, match="paired reset mismatch"):
        initialization_comparison("OpenHatch", reference, variant, trace, trace)
    variant["initial_reset"] = reference["initial_reset"]
    trace[0]["measured"][0, 0] = np.nan
    with pytest.raises(ValueError, match="post-warmup"):
        initialization_comparison("OpenHatch", reference, variant, trace, trace)


def row(value, *, active=True, reset=False):
    return dict(
        active=np.array([active]),
        reset=np.array([reset]),
        grasped=np.array([True]),
        angle=np.array([0.1]),
        station_position_error=np.array([[value, 0, 0]]),
        base_attitude=np.array([value]),
        motor_force=np.array([[value, value]]),
        motor_saturation_scale=np.array([[1.0]]),
    )


def test_joint_censoring_excludes_post_success_outliers():
    left = [row(1), row(3), row(1000, active=False)]
    right = [row(2), row(4), row(2000)]
    x, y, seconds = paired_metrics(left, right)
    assert seconds.tolist() == pytest.approx([2 / 30])
    assert x["station_position_rms_m"].tolist() == pytest.approx([np.sqrt(5)])
    assert y["motor_force_rms_N"].tolist() == pytest.approx([np.sqrt(10)])


def test_zero_exposure_is_undefined_not_zero_error():
    x, y, seconds = paired_metrics([row(1, reset=True)], [row(1)])
    assert seconds.tolist() == [0.0]
    assert all(np.isnan(value).all() for value in x.values())
    assert all(np.isnan(value).all() for value in y.values())


def test_integral_effect_sensitivity_subtracts_within_model_contrasts():
    def result(left, right):
        return dict(
            task="RotateValve",
            evidence=[dict(training_seed=seed) for seed in (17, 43, 101)],
            success=dict(
                variant_per_reset=np.full((3, 30), left).tolist(), reference_per_reset=np.full((3, 30), right).tolist()
            ),
        )

    # Nominal: Ki removal costs all successes; alternate: both gains succeed.
    nominal, alternate = result(False, True), result(True, True)
    nominal["contrast"] = "ki0-nominal minus ki1-nominal"
    alternate["contrast"] = "ki0-published-both minus ki1-published-both"
    sensitivity = success_effect_sensitivity(nominal, alternate)
    assert sensitivity["difference"] == 1
    assert sensitivity["descriptive_paired_crossed_95"] == [1, 1]
    alternate["evidence"].reverse()
    with pytest.raises(ValueError, match="alignment"):
        success_effect_sensitivity(nominal, alternate)
    alternate["evidence"].reverse()
    alternate["contrast"] = "ki0-published-both minus ki1-nominal"
    with pytest.raises(ValueError, match="within-coefficient"):
        success_effect_sensitivity(nominal, alternate)
    alternate["contrast"] = "ki0-published-both minus ki1-published-both"
    alternate["success"]["variant_per_reset"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Boolean"):
        success_effect_sensitivity(nominal, alternate)
