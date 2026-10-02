"""Publication contract tests; actual media decoding is checked by the recorder/browser."""

import copy
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from wasman.camera_projection import pinhole_view_projection
from wasman.effect_annotations import build_annotations

publisher = runpy.run_path(str(Path(__file__).parents[1] / "scripts/publish_physical_effect_demos.py"))
validate_take, validate_group = publisher["validate_take"], publisher["validate_group"]


@pytest.mark.parametrize("epsilon,accepted", [(1e-13, True), (1e-5, False)])
def test_annotation_comparator_allows_only_subpixel_numeric_roundoff(epsilon, accepted):
    expected = {"points_px": [[225.1128820243773, 180.0]], "valid": [True], "label": "Tool"}
    recorded = copy.deepcopy(expected)
    recorded["points_px"][0][0] += epsilon
    assert publisher["annotations_match"](recorded, expected) is accepted


@pytest.mark.parametrize("fault", ["nan", "inf", "bool_as_number", "string", "missing_key", "length"])
def test_annotation_roundoff_tolerance_never_relaxes_schema_or_finiteness(fault):
    expected = {"points_px": [[225.0, 180.0]], "valid": [True], "label": "Tool"}
    recorded = copy.deepcopy(expected)
    if fault == "nan":
        recorded["points_px"][0][0] = float("nan")
    elif fault == "inf":
        recorded["points_px"][0][0] = float("inf")
    elif fault == "bool_as_number":
        recorded["valid"][0] = 1
    elif fault == "string":
        recorded["label"] = "Other"
    elif fault == "missing_key":
        recorded.pop("label")
    else:
        recorded["points_px"].append([225.0, 180.0])
    assert not publisher["annotations_match"](recorded, expected)


@pytest.fixture
def take(tmp_path):
    frames = [
        {
            "frame": i,
            "t_s": (i + 1) / 30,
            "base_position_w_m": [0, 0, 0.85],
            "target_position_w_m": [0, 0, 0.85],
            "base_quaternion_xyzw": [0, 0, 0, 1],
            "motor_forces_n": [0] * 8,
            "motor_rpm_signed_thrust": [0] * 8,
            "boundary_gain": [1] * 8,
            "current_w_m_s": [0, 0, 0],
            "applied_wrench_body": [0] * 6,
            "robot_upper_envelope_m": 1.35,
            "rotor_axes_w": [[1, 0, 0]] * 8,
            "gripper_position_w_m": [0.5, 0, 0.7],
        }
        for i in range(240)
    ]
    history = []
    camera_position = [0.8, -0.45, 3.6]
    camera_quaternion = [1, 0, 0, 0]  # ROS optical +Z points downward in the world.
    camera_intrinsic = [[400, 0, 320], [0, 400, 240], [0, 0, 1]]
    inset_matrix = pinhole_view_projection(camera_position, camera_quaternion, camera_intrinsic, [640, 480])
    for frame in frames:
        history.append(frame["gripper_position_w_m"])
        frame["camera_view_projection_row_major"] = np.eye(4).tolist()
        frame["annotations"] = build_annotations(frame, np.eye(4), [1280, 720], "current", trajectory_w=history)
        frame["inset_camera_view_projection_row_major"] = inset_matrix.tolist()
        frame["inset_viewport_px"] = [640, 480]
        frame["inset_camera_position_w_m"] = camera_position
        frame["inset_camera_quaternion_xyzw_ros"] = camera_quaternion
        frame["inset_camera_intrinsic_matrix"] = camera_intrinsic
        frame["inset_camera_clipping_range_m"] = [0.01, 100.0]
        frame["inset_projection_source"] = "CameraData Fabric pose and calibrated pinhole intrinsics"
        frame["inset_annotations"] = build_annotations(frame, inset_matrix, [640, 480], "current", trajectory_w=history)
    (tmp_path / "trace.json").write_text(json.dumps({"frames": frames}))
    for name in ("poster.jpg", "physical_effect_0000.mp4", "current_inset_0000.mp4", "inset_poster.jpg"):
        (tmp_path / name).write_bytes(b"unit fixture, not encoded media")
    report = {
        "completed": True,
        "failure": None,
        "policy": None,
        "mode": "current",
        "runtime_pose_writes": 0,
        "boundary_calibrated": False,
        "actuator_timing_identified": False,
        "boundary_enabled": False,
        "scene": {"kind": "sand", "manipulation_objects": []},
        "manipulation_prims_absent": True,
        "frames": 240,
        "fps": 30,
        "telemetry_offset_s": 1 / 30,
        "decoded_video": {"decoded_frames": 240, "fps": 30, "size": [1280, 720]},
        "decoded_inset_video": {"decoded_frames": 240, "fps": 30, "size": [640, 480]},
        "files": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in tmp_path.iterdir()},
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    return tmp_path


def test_controller_recording_contract(take):
    report, frames = validate_take(take, "current")
    assert len(frames) == 240
    assert report["publication_audit"]["position_error_rms_m"] == 0


@pytest.mark.parametrize(
    "fault", ["interrupted", "pose_override", "policy", "time_origin", "frame_count", "hash", "scene", "objects"]
)
def test_invalid_recording_not_published(take, fault):
    path = take / "report.json"
    report = json.loads(path.read_text())
    if fault == "interrupted":
        report["completed"] = False
    elif fault == "pose_override":
        report["runtime_pose_writes"] = 1
    elif fault == "policy":
        report["policy"] = "unrelated actor"
    elif fault == "time_origin":
        report["telemetry_offset_s"] = 0
    elif fault == "frame_count":
        report["decoded_video"]["decoded_frames"] = 239
    elif fault == "scene":
        report["scene"]["kind"] = "pool"
    elif fault == "objects":
        report["scene"]["manipulation_objects"] = ["Button"]
    else:
        (take / "trace.json").write_text((take / "trace.json").read_text() + " ")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_take(take, "current")


def test_fabricated_screen_arrow_is_rejected_even_with_matching_file_hash(take):
    trace_path = take / "trace.json"
    trace = json.loads(trace_path.read_text())
    trace["frames"][0]["annotations"]["current_arrows"] = [
        {"start_px": [100, 100], "end_px": [400, 100], "valid": True}
    ]
    trace_path.write_text(json.dumps(trace))
    report_path = take / "report.json"
    report = json.loads(report_path.read_text())
    report["files"]["trace.json"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Projected annotation"):
        validate_take(take, "current")


@pytest.mark.parametrize("fault", ["frames", "fps", "size", "missing_video", "hash", "missing_metadata"])
def test_current_inset_media_must_match_primary_clock_and_its_hash(take, fault):
    path = take / "report.json"
    report = json.loads(path.read_text())
    if fault == "frames":
        report["decoded_inset_video"]["decoded_frames"] = 239
    elif fault == "fps":
        report["decoded_inset_video"]["fps"] = 24
    elif fault == "size":
        report["decoded_inset_video"]["size"] = [1280, 720]
    elif fault == "missing_video":
        (take / "current_inset_0000.mp4").unlink()
    elif fault == "hash":
        (take / "inset_poster.jpg").write_bytes(b"changed")
    else:
        report.pop("decoded_inset_video")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_take(take, "current")


@pytest.mark.parametrize("fault", ["same_camera", "viewport", "annotation", "axes", "gripper_history"])
def test_secondary_projection_and_measured_geometry_cannot_be_fabricated(take, fault):
    trace_path, report_path = take / "trace.json", take / "report.json"
    trace, report = json.loads(trace_path.read_text()), json.loads(report_path.read_text())
    frame = trace["frames"][3]
    if fault == "same_camera":
        frame["inset_camera_view_projection_row_major"] = frame["camera_view_projection_row_major"]
    elif fault == "viewport":
        frame["inset_viewport_px"] = [1280, 720]
    elif fault == "annotation":
        frame["inset_annotations"] = frame["annotations"]
    elif fault == "axes":
        frame["rotor_axes_w"][0] = [0, 0, 0]
    else:
        frame["gripper_position_w_m"][0] += 0.2
    trace_path.write_text(json.dumps(trace))
    report["files"]["trace.json"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_take(take, "current")


@pytest.mark.parametrize("fault", ["stale_usd", "missing_pose", "bad_intrinsics", "source"])
def test_inset_matrix_is_audited_against_live_sensor_pose_not_only_self_consistent_pixels(take, fault):
    trace_path, report_path = take / "trace.json", take / "report.json"
    trace, report = json.loads(trace_path.read_text()), json.loads(report_path.read_text())
    frame = trace["frames"][3]
    if fault == "stale_usd":
        # Regression: USD camera is identity while Fabric camera really sits at
        # 3.6m. Even self-consistent pixels from that WRONG pose must be rejected.
        bad_matrix = pinhole_view_projection(
            [0, 0, 0], [0, 0, 0, 1], frame["inset_camera_intrinsic_matrix"], [640, 480]
        )
        frame["inset_camera_view_projection_row_major"] = bad_matrix.tolist()
        frame["inset_annotations"] = build_annotations(
            frame,
            bad_matrix,
            [640, 480],
            "current",
            trajectory_w=[f["gripper_position_w_m"] for f in trace["frames"][:4]],
        )
    elif fault == "missing_pose":
        del frame["inset_camera_position_w_m"]
    elif fault == "bad_intrinsics":
        frame["inset_camera_intrinsic_matrix"][0][0] = 0
    else:
        frame["inset_projection_source"] = "UsdGeom.Camera.GetCamera()"
    trace_path.write_text(json.dumps(trace))
    report["files"]["trace.json"]["sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        validate_take(take, "current")


def comparison_triplet():
    report = {
        "scene": {"kind": "sand", "manipulation_objects": []},
        "source_sha256": {"controller.py": "same"},
        "conditions": {"current": [0, 0, 0], "seed": 2058, "motor_tau": 0.08},
        "initial_pose_xyzw": [0, 0, 0.85, 0, 0, 0, 1],
    }
    frame = {
        "camera_eye_w_m": [-1, -2, 1],
        "camera_target_w_m": [0, 0, 0.7],
        "inset_camera_view_projection_row_major": np.eye(4).tolist(),
    }
    names = publisher["GROUPS"]["current"]
    takes = {name: (copy.deepcopy(report), [copy.deepcopy(frame)]) for name in names}
    takes[names[1]][0]["conditions"]["current"] = [0.22, 0, 0]
    takes[names[2]][0]["conditions"]["current"] = [0.8, 0, 0]  # unit fixture, not a chosen release speed
    return names, takes


def test_current_group_supports_three_conditions():
    names, takes = comparison_triplet()
    assert len(names) == 3
    validate_group(takes, "current", names)


@pytest.mark.parametrize("fault", ["source", "seed", "initial", "camera", "inset", "condition_keys"])
def test_third_current_condition_is_checked_against_same_reference(fault):
    names, takes = comparison_triplet()
    report, frames = takes[names[2]]
    if fault == "source":
        report["source_sha256"]["controller.py"] = "other"
    elif fault == "seed":
        report["conditions"]["seed"] = 99
    elif fault == "initial":
        report["initial_pose_xyzw"][0] = 0.1
    elif fault == "camera":
        frames[0]["camera_eye_w_m"][0] = -3
    elif fault == "inset":
        frames[0]["inset_camera_view_projection_row_major"][3][0] = 0.1
    else:
        report["conditions"]["extra"] = 1
    with pytest.raises(ValueError):
        validate_group(takes, "current", names)


def test_publication_copies_three_real_inset_assets_and_writes_manifest_hashes(take, tmp_path, monkeypatch):
    """Exercise manifest/media wiring; media validity is tested separately above."""
    root = tmp_path / "publication"
    source = root / "effects_v3"
    (root / "artifacts").mkdir(parents=True)
    fixture_files = [path for path in take.iterdir() if path.is_file()]
    for names in publisher["GROUPS"].values():
        for name in names:
            (source / name).mkdir(parents=True)
            for path in fixture_files:
                shutil.copyfile(path, source / name / path.name)

    def audited_fixture(folder, mode):
        report = {
            "scene": {"kind": "sand" if mode == "current" else "pool", "center_m": [0, 0, 0]},
            "source_sha256": {"fixture": "same"},
            "conditions": {
                "current": [0, 0, 0],
                "current_onset": 2,
                "current_ramp": 0.5,
                "base_height": 0.85,
                "wall_x": 1,
                "motor_tau": 0.08,
                "motor_delay": 2,
                "step_height": 0.12,
                "step_time": 2,
            },
            "initial_pose_xyzw": [0, 0, 0.85, 0, 0, 0, 1],
            "duration_s": 8,
            "fps": 30,
            "telemetry_offset_s": 1 / 30,
            "publication_audit": {"fixture_only": True},
        }
        frame = {
            "camera_eye_w_m": [0, 0, 1],
            "camera_target_w_m": [0, 0, 0],
            "inset_camera_view_projection_row_major": np.eye(4).tolist(),
            "annotations": {"schema_version": 2},
        }
        return report, [frame]

    scope = publisher["main"].__globals__
    monkeypatch.setitem(scope, "__file__", str(root / "scripts/publisher.py"))
    monkeypatch.setitem(scope, "validate_take", audited_fixture)
    monkeypatch.setattr(sys, "argv", ["publisher.py", "--input-root", str(source), "--tag", "v3"])
    publisher["main"]()
    manifest = json.loads((root / "website/public/static/effects/manifest.json").read_text())
    assert manifest["version"] == "v3"
    assert sum(len(mode["variants"]) for mode in manifest["modes"]) == 9
    for item in manifest["modes"][0]["variants"]:
        assert item["annotations_schema_version"] == 2
        for field, digest_field, suffix in (
            ("inset_video", "inset_sha256", "mp4"),
            ("inset_poster", "inset_poster_sha256", "jpg"),
        ):
            assert item[field].endswith(f"{item['id']}-inset.{suffix}")
            path = root / "website/public" / item[field].removeprefix("./")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item[digest_field]


def test_drift_caption_requires_measured_error_speed_and_allocator_evidence():
    report = {"conditions": {"current": [1.7, 0, 0], "current_onset": 2, "current_ramp": 0.5}}
    assert "Drift observed" not in publisher["label_and_caption"]("current", report)[1]
    report["metrics"] = {
        "last_second_xy_error_m": 1.8,
        "last_second_horizontal_speed_m_s": 0.32,
        "min_allocator_saturation_scale": 0.65,
    }
    label, caption = publisher["label_and_caption"]("current", report)
    assert label == "1.70 m/s"
    assert "Drift observed" in caption and "1.80 m" in caption and "0.32 m/s" in caption
    assert "Not a calibrated critical-current threshold" in caption
    report["metrics"]["min_allocator_saturation_scale"] = 1
    assert "Drift observed" not in publisher["label_and_caption"]("current", report)[1]
