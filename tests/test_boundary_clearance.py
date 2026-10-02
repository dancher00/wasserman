"""Replay matched clearance experiments; no simulator needed for the audit."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_boundary_clearance.py"


def pair_comparison():
    # Extract the pure numerical function without launching Kit at import time.
    definition = next(
        n for n in ast.parse(SCRIPT.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "compare_pair"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["compare_pair"]


def test_pair_rejects_nonmatching_initial_state_and_failed_runs():
    compare = pair_comparison()
    baseline = {"id": "off", "initial_state": [0, 0, 1], "passed": True}
    assert not compare(baseline, {**baseline, "initial_state": [1, 0, 1]}, [], [])["valid"]
    assert not compare(baseline, {**baseline, "passed": False}, [], [])["valid"]


def test_pair_numerics_quaternion_double_cover_and_target_mismatch():
    compare = pair_comparison()
    case = {"id": "case", "initial_state": [], "passed": True}
    off = [{"position_w_m": [0, 0, 1], "target_w_m": [0, 0, 1], "quaternion_xyzw": [0, 0, 0, 1]}]
    on = [{"position_w_m": [0.003, 0.004, 1], "target_w_m": [0, 0, 1], "quaternion_xyzw": [0, 0, 0, -1]}]
    result = compare(case, case, off, on)
    assert result["max_on_off_position_delta_mm"] == pytest.approx(5)
    assert result["max_on_off_attitude_delta_deg"] == 0
    assert result["tracking_rmse_off_mm"] == 0
    assert result["tracking_rmse_on_mm"] == pytest.approx(5)
    on[0]["target_w_m"] = [0, 0, 2]
    with pytest.raises(RuntimeError, match="target mismatch"):
        compare(case, case, off, on)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_recorded_clearance_sweep_has_matched_states_no_contacts_and_replayable_metrics(version):
    directory = ROOT / f"artifacts/boundary_clearance_{version}"
    report = json.loads((directory / "report.json").read_text())
    assert report["completed"]
    assert report["loss_coefficient"] == 0.2 and report["range_diameters"] == 10
    assert report["water_current_m_s"] == [0, 0, 0]
    assert report["physics_hz"] == 120 and report["control_hz"] == 30
    assert len(report["usd_collision_bounds"]) == 11
    assert {b["body"] for b in report["usd_collision_bounds"]} <= set(report["contact_sensor_bodies"])
    assert (
        hashlib.sha256((directory / "run_script.py").read_bytes()).hexdigest()
        == report["source_sha256"]["scripts/sweep_boundary_clearance.py"]
    )
    cases = {c["id"]: c for c in report["cases"]}
    frames = {}
    for key, case in cases.items():
        trace = directory / case["trace"]
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == case["trace_sha256"]
        frames[key] = json.loads(trace.read_text())
        assert len(frames[key]) == case["frames"]
        if case["passed"]:
            assert len(frames[key]) == report["steps"]
            assert case["rejection_reason"] is None and not case["reset"]
            assert case["peak_normal_contact_n"] == 0
            assert min(case["min_conservative_gaps_m"]) >= report["minimum_safe_collider_gap_m"]
            gains = np.array([f["boundary_gain"] for f in frames[key]])
            assert np.isfinite(gains).all() and gains.min() >= 0.8 and gains.max() <= 1
        if not case["enabled"]:
            assert case["max_boundary_force_delta_n"] == 0
            assert case["max_boundary_torque_delta_nm"] == 0
            assert case["max_per_motor_loss_percent"] == 0
    compare = pair_comparison()
    for pair in report["pairs"]:
        off, on = cases[pair["off"]], cases[pair["on"]]
        assert off["initial_state"] == on["initial_state"]
        assert off["mode"] == on["mode"] and off["base_to_surface_m"] == on["base_to_surface_m"]
        assert off["protocol"] == on["protocol"] and off["seed"] == on["seed"]
        assert compare(off, on, frames[pair["off"]], frames[pair["on"]]) == pair
    # Unsafe initial placements must remain in the evidence, not become passed
    # tests after a silent reset or after disabling contacts.
    if version == "v1":
        assert any(c["rejection_reason"] == "initial_collider_clearance" for c in cases.values())
