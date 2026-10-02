import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from revision_scoring_evidence import (  # noqa: E402
    check_embedded_report,
    checked_tensor_outcomes,
    initialization_fingerprint,
    require_matching_initialization,
)

pytestmark = pytest.mark.unit


def test_paired_initialization_requires_actual_states_and_reset_fields():
    report = dict(seeds=[10, 11], initial_reset=dict(damping_scale=[1.0, 1.1]))
    state = np.zeros((2, 8))
    first = initialization_fingerprint(state, report)
    require_matching_initialization([first, initialization_fingerprint(state.copy(), report)])
    state[0, 2] = 1e-9
    with pytest.raises(ValueError, match="initialization mismatch"):
        require_matching_initialization([first, initialization_fingerprint(state, report)])
    state[0, 2] = 0
    report["initial_reset"]["damping_scale"][0] = 1.01
    with pytest.raises(ValueError, match="initialization mismatch"):
        require_matching_initialization([first, initialization_fingerprint(state, report)])
    state[0, 1] = np.nan
    with pytest.raises(ValueError, match="Invalid initial"):
        initialization_fingerprint(state, report)


def test_torch_tuple_and_json_list_reports_match_without_hiding_numeric_changes():
    embedded = dict(diagnostic_condition=dict(added_mass=(5.5, 12.7, 14.6)))
    report = dict(diagnostic_condition=dict(added_mass=[5.5, 12.7, 14.6]))
    check_embedded_report(embedded, report)
    report["diagnostic_condition"]["added_mass"][0] += 1e-9
    with pytest.raises(ValueError, match="Trace/report mismatch"):
        check_embedded_report(embedded, report)
    with pytest.raises(ValueError):
        check_embedded_report(dict(coefficient=float("nan")), dict(coefficient=None))


def test_eight_reset_validation_requires_explicit_scope_and_cannot_be_a_primary_cohort():
    report, trace = valve_fixture(True)
    for key in ("seeds", "success_per_seed", "first_success_step", "first_reset_step"):
        report[key] = report[key][:8]
    report["successes"] = 8
    trace = [
        {key: value[:8] if isinstance(value, torch.Tensor) else value for key, value in row.items()} for row in trace
    ]
    with pytest.raises(ValueError, match="30 declared episodes"):
        checked_tensor_outcomes("RotateValve", report, trace)
    assert checked_tensor_outcomes("RotateValve", report, trace, expected_episodes=8).all()


def valve_fixture(success):
    report = dict(
        seeds=list(range(30)),
        evaluated_steps=2240,
        step_dt=1 / 30,
        success_contract=dict(version="ambench-angle-170-v1"),
        success_per_seed=[success] * 30,
        successes=30 if success else 0,
        first_success_step=[1 if success else -1] * 30,
        first_reset_step=[-1] * 30,
    )
    row = dict(
        active=torch.ones(30, dtype=torch.bool),
        reset=torch.zeros(30, dtype=torch.bool),
        success=torch.full((30,), success),
        angle=torch.full((30,), 3.0 if success else 0.0),
    )
    return report, [row]


def test_physical_success_can_finish_early_but_short_failure_cannot():
    report, trace = valve_fixture(True)
    assert checked_tensor_outcomes("RotateValve", report, trace).all()
    report, trace = valve_fixture(False)
    with pytest.raises(ValueError, match="Incomplete evaluation horizon"):
        checked_tensor_outcomes("RotateValve", report, trace)


def test_report_and_trace_claims_do_not_replace_measured_evidence():
    report, trace = valve_fixture(True)
    trace[0]["angle"].zero_()
    with pytest.raises(ValueError, match="Physical trace/report"):
        checked_tensor_outcomes("RotateValve", report, trace)


def test_reset_censoring_and_active_mask_are_verified():
    report, trace = valve_fixture(False)
    trace[0]["reset"][:] = True
    report["first_reset_step"] = [1] * 30
    assert not checked_tensor_outcomes("RotateValve", report, trace).any()
    report["first_reset_step"] = [-1] * 30
    with pytest.raises(ValueError, match="censoring"):
        checked_tensor_outcomes("RotateValve", report, trace)
    trace[0]["active"][0] = False
    with pytest.raises(ValueError, match="active mask"):
        checked_tensor_outcomes("RotateValve", report, trace)


def test_nonfinite_evidence_cannot_be_scored_as_a_physical_failure():
    report, trace = valve_fixture(False)
    trace[0]["reset"][:] = True
    report["first_reset_step"] = [1] * 30
    trace[0]["inputs"] = {"force": torch.tensor([float("nan")])}
    with pytest.raises(ValueError, match="Nonfinite physical evidence"):
        checked_tensor_outcomes("RotateValve", report, trace)
