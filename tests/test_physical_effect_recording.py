"""Replay actual controller-demo telemetry and media/provenance manifests."""

import hashlib
import json
from pathlib import Path

import numpy as np

from wasman.camera_projection import pinhole_view_projection

ROOT = Path(__file__).resolve().parents[1]
TAKES = ROOT / "artifacts/effects_v3"
NAMES = (
    "current-still",
    "current-flow",
    "current-overload",
    "seabed-near",
    "seabed-far",
    "wall-near",
    "wall-far",
    "actuator-instant",
    "actuator-lag",
)


def load(name):
    folder = TAKES / name
    return json.loads((folder / "report.json").read_text()), json.loads((folder / "trace.json").read_text())["frames"]


def test_all_nine_recordings_have_matching_frames_hashes_and_finite_data():
    source = json.loads((ROOT / "src/wasman/physics/data/t200_16v.json").read_text())["samples"]
    force_limits = np.array(source)[:, 2]
    for name in NAMES:
        report, frames = load(name)
        assert report["completed"] and report["failure"] is None
        assert report["policy"] is None and report["runtime_pose_writes"] == 0
        assert report["decoded_video"]["decoded_frames"] == len(frames) == 240
        assert report["decoded_video"]["size"] == [1280, 720]
        assert report["decoded_video"]["fps"] == report["fps"] == 30
        assert report["telemetry_offset_s"] == 1 / 30
        assert report["scene"]["kind"] == ("sand" if name.startswith("current") else "pool")
        assert report["scene"]["manipulation_objects"] == []
        assert report["manipulation_prims_absent"]
        if not name.startswith("current"):
            assert report["scene"]["dimensions"]["length_m"] == 25
            assert report["scene"]["dimensions"]["width_m"] == 25
            assert report["scene"]["dimensions"]["water_depth_m"] == 2.5
        assert np.allclose([row["t_s"] for row in frames], np.arange(1, 241) / 30)
        for filename, expected in report["files"].items():
            path = TAKES / name / filename
            if filename == "physical_effect_0000.mp4":
                path = ROOT / "website/public/static/effects/v3" / f"{name}.mp4"
            elif filename == "current_inset_0000.mp4":
                path = ROOT / "website/public/static/effects/v3" / f"{name}-inset.mp4"
            data = path.read_bytes()
            assert len(data) == expected["bytes"]
            assert hashlib.sha256(data).hexdigest() == expected["sha256"]
        for row in frames:
            assert row["robot_upper_envelope_m"] < 2.5
            for key in (
                "base_position_w_m",
                "motor_forces_n",
                "boundary_gain",
                "current_w_m_s",
                "camera_view_projection_row_major",
                "rotor_axes_w",
                "gripper_position_w_m",
            ):
                assert np.isfinite(row[key]).all()
            assert row["annotations"]["schema_version"] == 2
            if name.startswith("current"):
                assert np.isfinite(row["inset_camera_view_projection_row_major"]).all()
                assert row["inset_viewport_px"] == [640, 480]
                assert row["inset_annotations"]["schema_version"] == 2
                expected = pinhole_view_projection(
                    row["inset_camera_position_w_m"],
                    row["inset_camera_quaternion_xyzw_ros"],
                    row["inset_camera_intrinsic_matrix"],
                    row["inset_viewport_px"],
                    row["inset_camera_clipping_range_m"],
                )
                assert np.allclose(row["inset_camera_view_projection_row_major"], expected, rtol=1e-6, atol=1e-7)
            if not name.startswith("current"):
                assert np.allclose(row["current_w_m_s"], 0)
        forces = np.array([row["motor_forces_n"] for row in frames])
        assert forces.min() >= force_limits.min() - 1e-5
        assert forces.max() <= force_limits.max() + 1e-5
        if name.startswith("current"):
            assert report["decoded_inset_video"]["decoded_frames"] == 240
            assert report["decoded_inset_video"]["fps"] == 30
            assert report["decoded_inset_video"]["size"] == [640, 480]


def test_paired_diagnostics_change_only_the_declared_condition():
    for first, second, changed in (
        ("current-still", "current-flow", {"current"}),
        ("current-still", "current-overload", {"current"}),
        ("seabed-near", "seabed-far", {"base_height"}),
        ("wall-near", "wall-far", {"wall_x"}),
        ("actuator-instant", "actuator-lag", {"motor_tau", "motor_delay"}),
    ):
        a, af = load(first)
        b, bf = load(second)
        assert a["source_sha256"] == b["source_sha256"]
        assert {key for key in a["conditions"] if a["conditions"][key] != b["conditions"][key]} == changed
        assert af[0]["camera_eye_w_m"] == bf[0]["camera_eye_w_m"]
        assert af[0]["camera_target_w_m"] == bf[0]["camera_target_w_m"]
    _, calm = load("current-still")
    _, flow = load("current-flow")
    assert calm[:60] == flow[:60]  # matched simulation before current onset
    assert np.allclose(flow[-1]["current_w_m_s"], [0.9, 0, 0])
    overload_report, overload = load("current-overload")
    assert calm[:60] == overload[:60]
    assert np.allclose(overload[-1]["current_w_m_s"], overload_report["conditions"]["current"])
    assert np.allclose(overload_report["conditions"]["current"], [1.7, 0, 0])
    assert np.linalg.norm(overload[-1]["current_w_m_s"]) > np.linalg.norm(flow[-1]["current_w_m_s"])
    for near, far in (("seabed-near", "seabed-far"), ("wall-near", "wall-far")):
        assert load(near)[0]["metrics"]["minimum_boundary_gain"] < 1
        assert load(far)[0]["metrics"]["minimum_boundary_gain"] == 1
    _, instant = load("actuator-instant")
    _, lag = load("actuator-lag")
    assert [row["target_position_w_m"] for row in instant] == [row["target_position_w_m"] for row in lag]
    assert np.allclose(np.array(instant[60]["target_position_w_m"]) - instant[59]["target_position_w_m"], [0, 0, 0.12])
    assert not np.allclose([row["base_position_w_m"] for row in instant], [row["base_position_w_m"] for row in lag])
