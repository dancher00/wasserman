import pytest
import torch

from wasman.motor_replay import MotorCommandTape, verify_matched_tapes
from wasman.physics.thrusters import BatchedT200


def test_replay_ignores_feedback_but_retains_actual_motor_dynamics():
    motor = BatchedT200(1, "cpu", dt=1 / 120)
    original = motor.step
    reference = MotorCommandTape(motor)
    with reference.installed():
        for i in range(20):
            motor.step(torch.tensor([[float(i), 0, 20]]), torch.zeros(1, 3))
    assert motor.step == original
    commands, forces = reference.tensors()
    motor.reset(slice(None))
    replay = MotorCommandTape(motor, commands)
    with replay.installed():
        for _ in range(20):
            motor.step(torch.full((1, 3), 999.0), torch.full((1, 3), -999.0))
        with pytest.raises(RuntimeError, match="exhausted"):
            motor.step(torch.zeros(1, 3), torch.zeros(1, 3))
    matched = verify_matched_tapes(reference, replay)
    assert matched["input_wrenches_exact_match"] and matched["pre_boundary_motor_forces_exact_match"]
    assert forces[0].abs().max() == 0  # Native two-step delay is not bypassed.
    assert forces[-1].abs().max() > 0
    assert motor.step == original


def test_tape_restores_step_after_exception_and_rejects_nonfinite():
    motor = BatchedT200(1, "cpu", dt=1 / 120)
    original = motor.step
    with pytest.raises(ValueError, match="Nonfinite"):
        MotorCommandTape(motor, torch.full((3, 1, 6), float("nan")))
    tape = MotorCommandTape(motor)
    with pytest.raises(RuntimeError, match="Nonfinite"), tape.installed():
        motor.step(torch.full((1, 3), float("nan")), torch.zeros(1, 3))
    assert motor.step == original


def test_recorded_motor_replay_evidence():
    from pathlib import Path

    from wasman.motor_replay_audit import audit_directory

    report = audit_directory(Path(__file__).parents[1] / "artifacts/boundary_motor_replay_v1", decode_video=False)
    assert report["passed"]
    assert {m["id"] for m in report["modes"]} == {"floor", "wall"}
    for mode in report["modes"]:
        assert mode["reference_off_pose_exact_match"]
        assert all(take["physics_frames"] == 960 and take["frames"] == 240 for take in mode["takes"])
        assert 0 < mode["maximum_position_difference_m"] < 0.1


@pytest.mark.parametrize("field,value", [("contact_history_max_n", 0.1), ("collider_gaps_m", [0.01] * 6)])
def test_independent_audit_rejects_semantically_unsafe_trace(tmp_path, field, value):
    import hashlib
    import json
    import shutil
    from pathlib import Path

    from wasman.motor_replay_audit import audit_directory

    source = Path(__file__).parents[1] / "artifacts/boundary_motor_replay_v1"
    destination = tmp_path / "suite"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.mp4", "*.jpg"))
    path = destination / "floor/reference_off/trace.json"
    trace = json.loads(path.read_text())
    trace["physics_frames"][0][field] = value
    path.write_text(json.dumps(trace))
    report_path = path.with_name("report.json")
    report = json.loads(report_path.read_text())
    report["trace_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="guard failed"):
        audit_directory(destination, decode_video=False)
