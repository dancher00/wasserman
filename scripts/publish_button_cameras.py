"""Publish the hero's same-episode cameras only after replaying the sustained-press gate."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

from wasman.controllers.button_recording import verified_button_recording
from wasman.recording_publication import geometry_metadata, verify_decoded_film


def main(argv=None, *, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--version", help="Publish to a fresh versioned directory; preserve legacy films")
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    destination = root / "website/public/static"
    prefix = "./static"
    if args.version:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args.version):
            raise ValueError("Version must be a simple lowercase identifier")
        destination /= f"recordings/press-button/{args.version}"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Preserve published versions")
        prefix += f"/recordings/press-button/{args.version}"
    artifact_name = (
        f"press_button_{args.version.replace('-', '_')}_demo.json" if args.version else "press_button_camera_demo.json"
    )
    artifact = root / "artifacts" / artifact_name
    if args.version and (artifact.exists() or artifact.is_symlink()):
        raise FileExistsError("Preserve published version artifacts")
    trace_path = args.source / "trace.json"
    record = json.loads(trace_path.read_text())
    report = verified_button_recording(record)
    if args.version:
        if record.get("converted_geometry_audit", {}).get("passed") is not True:
            raise ValueError("Versioned geometry film requires a converter-output audit")
        report.update(
            {
                key: record[key]
                for key in (
                    "robot_geometry_version",
                    "robot_urdf_sha256",
                    "thruster_positions_m",
                    "robot_camera_profile",
                    "robot_camera_parameters",
                    "converted_geometry_audit",
                )
            }
        )
        provenance = dict(record, registered_geometry_audit=record["converted_geometry_audit"])
        report.update(geometry_metadata(provenance, args.version))
    cameras = record["camera_streams"]
    if not cameras["enabled"] or cameras["frames"] != len(record["trace"]):
        raise ValueError("Camera and telemetry counts differ")
    trim = cameras["observer_initial_frames_to_trim"]
    sources = [("observer", args.source / "policy_orbit_0000.mp4")]
    sources += [(name, args.source / f"button_{name}.mp4") for name in ("gripper", "base")]
    for name, source in sources:
        reader = imageio_ffmpeg.read_frames(str(source))
        metadata = next(reader)
        reader.close()
        expected_duration = report["duration_s"] + (trim / 30 if name == "observer" else 0)
        if abs(metadata["duration"] - expected_duration) > 0.05 or abs(metadata["fps"] - 30) > 0.01:
            raise ValueError(f"{name} film does not align with telemetry")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    destination.mkdir(parents=True, exist_ok=not bool(args.version))
    output = destination / "press-button.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(sources[0][1]),
            "-vf",
            f"trim=start_frame={trim},setpts=PTS-STARTPTS",
            "-an",
            "-r",
            "30",
            "-fps_mode",
            "cfr",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    report["observer_encoding"] = verify_decoded_film(output, len(record["trace"]), imageio_ffmpeg.read_frames)
    report["video_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    streams = []
    for name, source in sources[1:]:
        verify_decoded_film(source, len(record["trace"]), imageio_ffmpeg.read_frames)
        filename = f"press-button-{name}-sync.mp4"
        shutil.copyfile(source, destination / filename)
        streams.append(
            {
                "id": name,
                "label": name.capitalize(),
                "src": f"{prefix}/{filename}",
                "fps": 30,
                "width": 256,
                "height": 256,
                "frames": cameras["frames"],
                "duration_s": report["duration_s"],
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    report["camera_streams"] = streams
    report["camera_alignment"] = {
        "same_episode": True,
        "capture": "post-physics, same step as observer",
        "initial_hold_s": cameras["initial_hold_s"],
        "observer_trimmed_frames": trim,
        "max_pair_timestamp_difference_s": max(abs(t[0] - t[1]) for t in cameras["timestamps_s"]),
    }
    trace_name = "trace.json" if args.version else "press_button_camera_trace.json"
    report_name = "report.json" if args.version else "press_button_camera_demo.json"
    report["video_url"] = f"{prefix}/press-button.mp4"
    report["report_url"] = f"{prefix}/{report_name}"
    report["trace_url"] = f"{prefix}/{trace_name}"
    shutil.copyfile(trace_path, destination / trace_name)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    for path in (artifact, destination / report_name):
        with path.open("x" if args.version else "w") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
