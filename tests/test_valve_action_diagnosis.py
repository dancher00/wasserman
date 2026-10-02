"""Evidence gates for the failed rollout diagnosis, not policy acceptance tests."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "artifacts/valve_action_trace_2065"


def test_instrumentation_reproduces_original_failure_and_actual_actor():
    report = json.loads((SOURCE / "analysis_v2.json").read_text())
    assert not report["diagnostic_freeze_base_x"]
    assert report["steps"] == 1350
    assert report["successes"] == 0
    assert max(report["alignment"].values()) < 1e-5
    assert max(report["instrumentation_comparison"]["maximum_absolute_differences"].values()) == 0
    assert report["trace_sha256"] == hashlib.sha256((SOURCE / "trace.json").read_bytes()).hexdigest()
    summary = json.loads((SOURCE / "summary.json").read_text())
    assert summary["source_sha256_at_start"]["scripts/check_valve_expert.py"] == hashlib.sha256(
        (SOURCE / "recorder_snapshot.py").read_bytes()
    ).hexdigest()


def test_first_sustained_loss_is_usually_not_an_open_gripper_command():
    report = json.loads((SOURCE / "analysis_v2.json").read_text())
    events = [e["sustained_loss_events"][0] for e in report["episodes"]]
    assert len(events) == 8
    assert sum(e["loss"]["gripper_action"] == -1 for e in events) == 7
    for event in events:
        assert event["loss"]["wheel_relative_tool_m"][0] < -0.010
        assert event["loss"]["base_action"][0] < event["before_1s"]["base_action"][0]
        assert event["loss"]["valid_grasp_turn_deg"] < 165


def test_matched_intervention_delays_loss_but_is_not_a_recovered_policy():
    baseline = json.loads((SOURCE / "analysis_v2.json").read_text())
    folder = ROOT / "artifacts/valve_freeze_base_x_2065"
    probe = json.loads((folder / "analysis.json").read_text())
    assert probe["diagnostic_freeze_base_x"]
    assert "NOT standalone" in probe["kind"]
    assert probe["successes"] == 0
    assert probe["checkpoint_sha256"] == baseline["checkpoint_sha256"]
    assert probe["seed"] == baseline["seed"]
    assert max(probe["alignment"].values()) < 1e-5
    pairs = zip(baseline["episodes"], probe["episodes"], strict=True)
    assert sum(
        b["sustained_loss_events"][0]["loss"]["time_s"] > a["sustained_loss_events"][0]["loss"]["time_s"]
        for a, b in pairs
    ) == 7
    assert probe["trace_sha256"] == hashlib.sha256((folder / "trace.json").read_bytes()).hexdigest()
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["failed_resets"] == 0
    assert summary["controller"].startswith("DIAGNOSTIC INTERVENTION")
    assert summary["source_sha256_at_start"]["scripts/check_valve_expert.py"] == hashlib.sha256(
        (folder / "recorder_snapshot.py").read_bytes()
    ).hexdigest()
