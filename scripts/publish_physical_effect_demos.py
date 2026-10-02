"""Publish verified, uncut controller diagnostics and their measured telemetry."""

import argparse
import hashlib
import json
import math
import shutil
from numbers import Real
from pathlib import Path

import numpy as np

from wasman.camera_projection import pinhole_view_projection
from wasman.effect_annotations import build_annotations
from wasman.learning.hatch_reporting import publication_version

GROUPS = {
    "current": ("current-still", "current-flow", "current-overload"),
    "seabed": ("seabed-far", "seabed-near"),
    "wall": ("wall-far", "wall-near"),
    "actuator": ("actuator-instant", "actuator-lag"),
}
AXES = {"current": {"current"}, "seabed": {"base_height"}, "wall": {"wall_x"}, "actuator": {"motor_tau", "motor_delay"}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annotations_match(recorded, expected):
    """Strict schema equality with 1e-9 absolute numeric roundoff allowance.

    JSON reconstructs matrices in C order while a live camera may supply an
    F-contiguous matrix. GEMV accumulation order can differ by ~1e-13 pixels.
    This tolerance is far below a pixel or geometric measurement precision;
    booleans, strings, keys and sequence lengths still match exactly.
    """
    if isinstance(recorded, bool) or isinstance(expected, bool):
        return type(recorded) is type(expected) and recorded == expected
    if isinstance(recorded, Real) and isinstance(expected, Real):
        return (
            math.isfinite(recorded)
            and math.isfinite(expected)
            and math.isclose(recorded, expected, rel_tol=0.0, abs_tol=1e-9)
        )
    if isinstance(recorded, dict) and isinstance(expected, dict):
        return recorded.keys() == expected.keys() and all(
            annotations_match(recorded[key], expected[key]) for key in recorded
        )
    if isinstance(recorded, list) and isinstance(expected, list):
        return len(recorded) == len(expected) and all(
            annotations_match(a, b) for a, b in zip(recorded, expected, strict=True)
        )
    return type(recorded) is type(expected) and recorded == expected


def validate_take(folder, mode):
    report = json.loads((folder / "report.json").read_text())
    frames = json.loads((folder / "trace.json").read_text())["frames"]
    if not report["completed"] or report["failure"] is not None or report["policy"] is not None:
        raise ValueError(f"Not a completed controller diagnostic: {folder}")
    if report["mode"] != mode or report["runtime_pose_writes"] != 0:
        raise ValueError("Unexpected controller mode or pose overrides")
    if report["boundary_calibrated"] or report["actuator_timing_identified"]:
        raise ValueError("Do not claim calibration from simulator recordings")
    if report["boundary_enabled"] != (mode in ("seabed", "wall")):
        raise ValueError("Unexpected boundary-model condition")
    scene = report.get("scene", {})
    dimensions = scene.get("dimensions", {})
    expected_scene = "sand" if mode == "current" else "pool"
    if scene.get("kind") != expected_scene:
        raise ValueError("Current requires sand; floor/wall/actuator require the pool")
    if scene.get("manipulation_objects") != [] or not report.get("manipulation_prims_absent"):
        raise ValueError("No manipulation objects may occur in controller diagnostics")
    pool_bounds = None
    if expected_scene == "pool":
        if any(
            dimensions.get(key) != value
            for key, value in (("length_m", 25.0), ("width_m", 25.0), ("water_depth_m", 2.5))
        ):
            raise ValueError("Publish only standard-size finite pools")
        if any(not 0 < f.get("robot_upper_envelope_m", float("inf")) < 2.5 for f in frames):
            raise ValueError("Submerged robot envelope must remain below the waterline")
        cx, cy, _ = scene["center_m"]
        pool_bounds = [cx - 12.5, cx + 12.5, cy - 12.5, cy + 12.5, 2.5]
        if any(np.linalg.norm(f["current_w_m_s"]) > 1e-8 for f in frames):
            raise ValueError("Pool demos must use still water")
    if (
        mode in ("seabed", "wall")
        and scene.get("boundary_surface") != f"pool_{'floor' if mode == 'seabed' else 'wall'}"
    ):
        raise ValueError("Boundary diagnostics require the real pool surface")
    if report["frames"] != 240 or report["fps"] != 30 or len(frames) != 240:
        raise ValueError("Expected an uncut 240-frame, 8-second take")
    decoded = report["decoded_video"]
    if decoded["decoded_frames"] != len(frames) or decoded["fps"] != 30 or decoded["size"] != [1280, 720]:
        raise ValueError("Decoded media does not match telemetry")
    if mode == "current":
        inset = report.get("decoded_inset_video")
        if not isinstance(inset, dict) or (
            inset.get("decoded_frames") != len(frames) or inset.get("fps") != 30 or inset.get("size") != [640, 480]
        ):
            raise ValueError("Current inset must be an actual synchronized 640x480, 240-frame, 30fps recording")
    if not np.isclose(report["telemetry_offset_s"], 1 / 30):
        raise ValueError("Unexpected observer/telemetry time origin")
    if [frame["frame"] for frame in frames] != list(range(240)):
        raise ValueError("Missing, duplicated or reordered frame")
    if not np.allclose([frame["t_s"] for frame in frames], np.arange(1, 241) / 30, atol=1e-7):
        raise ValueError("Physics times do not match observer frames")
    for key in (
        "base_position_w_m",
        "base_quaternion_xyzw",
        "target_position_w_m",
        "motor_forces_n",
        "motor_rpm_signed_thrust",
        "boundary_gain",
        "current_w_m_s",
        "applied_wrench_body",
        "rotor_axes_w",
        "gripper_position_w_m",
        "camera_view_projection_row_major",
    ):
        if not np.isfinite([frame[key] for frame in frames]).all():
            raise ValueError(f"Non-finite recorded field: {key}")
    gains = np.array([frame["boundary_gain"] for frame in frames])
    if gains.min() < 0.8 - 1e-6 or gains.max() > 1 + 1e-6:
        raise ValueError("Boundary gain outside its declared envelope")
    axes = np.asarray([frame["rotor_axes_w"] for frame in frames])
    if axes.shape != (240, 8, 3) or not np.allclose(np.linalg.norm(axes, axis=-1), 1, atol=1e-5):
        raise ValueError("Native rotor axes must be eight unit world vectors per frame")
    media = ["physical_effect_0000.mp4", "trace.json", "poster.jpg"]
    if mode == "current":
        media.extend(("current_inset_0000.mp4", "inset_poster.jpg"))
    for name in media:
        if not (folder / name).is_file() or name not in report["files"]:
            raise ValueError(f"Missing required recording: {name}")
        if sha(folder / name) != report["files"][name]["sha256"]:
            raise ValueError(f"Recording changed after its report: {folder / name}")
    trajectory = []
    for frame in frames:
        if mode == "current":
            trajectory.append(frame["gripper_position_w_m"])
        if frame.get("annotations", {}).get("schema_version") != 2:
            raise ValueError("New publications require measured-axis annotation schema 2")
        expected = build_annotations(
            frame,
            frame["camera_view_projection_row_major"],
            [1280, 720],
            mode,
            pool_bounds=pool_bounds,
            trajectory_w=trajectory,
        )
        if not annotations_match(frame.get("annotations"), expected):
            raise ValueError("Projected annotation differs from recorded world geometry/camera")
        if mode == "current":
            matrix = np.asarray(frame.get("inset_camera_view_projection_row_major"))
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or frame.get("inset_viewport_px") != [640, 480]:
                raise ValueError("Current inset must carry its own finite camera matrix and viewport")
            if frame.get("inset_projection_source") != "CameraData Fabric pose and calibrated pinhole intrinsics":
                raise ValueError("Current inset projection must use live CameraData, not stale USD transforms")
            live_matrix = pinhole_view_projection(
                frame.get("inset_camera_position_w_m"),
                frame.get("inset_camera_quaternion_xyzw_ros"),
                frame.get("inset_camera_intrinsic_matrix"),
                frame["inset_viewport_px"],
                frame.get("inset_camera_clipping_range_m"),
            )
            if not np.allclose(matrix, live_matrix, rtol=1e-6, atol=1e-7):
                raise ValueError("Inset projection disagrees with live Fabric pose and pinhole intrinsics")
            if np.allclose(matrix, frame["camera_view_projection_row_major"], atol=1e-10):
                raise ValueError("Current inset must use a distinct actual camera, not a resized observer view")
            expected_inset = build_annotations(frame, matrix, [640, 480], mode, trajectory_w=trajectory)
            if not annotations_match(frame.get("inset_annotations"), expected_inset):
                raise ValueError("Inset annotation differs from the recorded secondary camera/trajectory")
    error = np.array([f["base_position_w_m"] for f in frames]) - np.array([f["target_position_w_m"] for f in frames])
    report["publication_audit"] = {
        "position_error_rms_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "last_second_position_error_rms_m": float(np.sqrt(np.mean(np.sum(error[-30:] ** 2, axis=1)))),
        "minimum_boundary_gain": float(gains.min()),
        "source_report_sha256": sha(folder / "report.json"),
        "all_frames_checked": len(frames),
    }
    return report, frames


def validate_group(takes, mode, names):
    """Compare every condition to the same reference; supports current triplets."""
    first, first_frames = takes[names[0]]
    for name in names[1:]:
        other, other_frames = takes[name]
        a_scene, b_scene = dict(first["scene"]), dict(other["scene"])
        if mode == "wall":
            a_scene.pop("center_m")
            b_scene.pop("center_m")
        if a_scene != b_scene or first["source_sha256"] != other["source_sha256"]:
            raise ValueError("Compared conditions must share the physical scene and source revision")
        if first["conditions"].keys() != other["conditions"].keys():
            raise ValueError("Compared conditions must declare the same parameter keys")
        for key, value in first["conditions"].items():
            if key not in AXES[mode] and other["conditions"][key] != value:
                raise ValueError(f"Comparison changes an uncontrolled condition: {mode}/{key}")
        for key in ("camera_eye_w_m", "camera_target_w_m"):
            if first_frames[0][key] != other_frames[0][key]:
                raise ValueError("Compared conditions must share a camera")
        if (
            mode == "current"
            and first_frames[0]["inset_camera_view_projection_row_major"]
            != other_frames[0]["inset_camera_view_projection_row_major"]
        ):
            raise ValueError("Current conditions must share the secondary camera")
        if mode != "seabed" and not np.allclose(first["initial_pose_xyzw"], other["initial_pose_xyzw"], atol=1e-7):
            raise ValueError("Compared conditions must share their initial state")


def label_and_caption(mode, report):
    c = report["conditions"]
    if mode == "current":
        speed = float(np.linalg.norm(c["current"]))
        caption = (
            "Still-water controller reference."
            if speed == 0
            else (
                f"Current ramps from zero at {c['current_onset']:.1f} s over {c['current_ramp']:.1f} s. "
                "Same controller and initial state."
            )
        )
        metrics = report.get("metrics", {})
        error = metrics.get("last_second_xy_error_m", 0)
        velocity = metrics.get("last_second_horizontal_speed_m_s", 0)
        allocation = metrics.get("min_allocator_saturation_scale", 1)
        if error > 1 and velocity > 0.1 and allocation < 1:
            caption += (
                f" Drift observed: final-second XY error {error:.2f} m, horizontal speed {velocity:.2f} m/s; "
                f"minimum allocation scale {allocation:.2f}. Not a calibrated critical-current threshold."
            )
        return f"{speed:.2f} m/s", caption
    if mode == "seabed":
        return f"{c['base_height']:.2f} m", (
            "Commanded base height, not rotor clearance. Optional pool-floor loss enabled; "
            "same motor timing and controller."
        )
    if mode == "wall":
        return f"Wall X {c['wall_x']:.2f} m", (
            "Actual inner pool wall, without manipulation fixtures. "
            "Projected guides use measured rotor/base clearances to that wall."
        )
    tau, delay = c["motor_tau"], c["motor_delay"]
    label = "Instantaneous response" if tau == 0 and delay == 0 else f"{tau * 1000:.0f} ms + {delay}-step delay"
    return label, (
        f"Same +{c['step_height']:.2f} m target step at {c['step_time']:.1f} s; "
        "native T200 static limits/deadband retained. No ideal-wrench bypass."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/effects_v3"))
    parser.add_argument("--tag", type=publication_version, default="v3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dest = root / "website/public/static/effects" / args.tag
    artifact = root / f"artifacts/physical_effect_demos_{args.tag}.json"
    if dest.exists() or artifact.exists():
        raise FileExistsError("Preserve the published version; select a new tag")
    takes = {}
    for mode, names in GROUPS.items():
        for name in names:
            takes[name] = validate_take(args.input_root / name, mode)
        validate_group(takes, mode, names)
    dest.mkdir(parents=True)
    manifest = {
        "version": args.tag,
        "kind": "Recorded controller diagnostics, not policy evaluation or hardware calibration",
        "source": str(args.input_root),
        "telemetry_clock": "Video frame k at k/30 s depicts post-step telemetry (k+1)/30 s",
        "scene_mapping": {"current": "sand", "seabed": "pool", "wall": "pool", "actuator": "pool"},
        "modes": [],
    }
    for mode, names in GROUPS.items():
        variants = []
        for name in names:
            report, frames = takes[name]
            folder = args.input_root / name
            label, caption = label_and_caption(mode, report)
            paths = {}
            media = [
                ("video", "physical_effect_0000.mp4", ".mp4"),
                ("poster", "poster.jpg", ".jpg"),
                ("trace", "trace.json", ".json"),
            ]
            if mode == "current":
                media.extend(
                    (
                        ("inset_video", "current_inset_0000.mp4", "-inset.mp4"),
                        ("inset_poster", "inset_poster.jpg", "-inset.jpg"),
                    )
                )
                paths.update(
                    inset_sha256=sha(folder / "current_inset_0000.mp4"),
                    inset_poster_sha256=sha(folder / "inset_poster.jpg"),
                )
            for field, source, suffix in media:
                target = dest / f"{name}{suffix}"
                shutil.copyfile(folder / source, target)
                paths[field] = f"./static/effects/{args.tag}/{target.name}"
            variants.append(
                {
                    "id": name,
                    "label": label,
                    "caption": caption,
                    "annotations_schema_version": frames[0]["annotations"]["schema_version"],
                    "scene": report["scene"],
                    **paths,
                    "duration_s": report["duration_s"],
                    "fps": report["fps"],
                    "sha256": sha(folder / "physical_effect_0000.mp4"),
                    "trace_sha256": sha(folder / "trace.json"),
                    "poster_sha256": sha(folder / "poster.jpg"),
                    "telemetry_offset_s": report["telemetry_offset_s"],
                    "source_report": str(folder / "report.json"),
                    "audit": report["publication_audit"],
                }
            )
        manifest["modes"].append({"id": mode, "variants": variants})
    data = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    artifact.write_text(data)
    (dest.parent / "manifest.json").write_text(data)
    print(artifact)


if __name__ == "__main__":
    main()
