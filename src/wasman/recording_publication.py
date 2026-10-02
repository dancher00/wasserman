"""Safe, versioned routing and provenance gates for expert film publication."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from wasman.robot_camera_profiles import robot_camera_profile

GEOMETRY_FIELDS = (
    "robot_geometry_version",
    "robot_camera_profile",
    "robot_camera_parameters",
    "registered_geometry_audit",
)


@dataclass(frozen=True)
class PublicationPaths:
    destination: Path
    artifact: Path
    url_prefix: str
    report_name: str
    trace_name: str
    version: str | None

    def prepare(self):
        if self.version:
            if self.artifact.exists() or self.artifact.is_symlink():
                raise FileExistsError("Preserve the existing versioned artifact")
            self.destination.mkdir(parents=True, exist_ok=False)
        else:
            self.destination.mkdir(parents=True, exist_ok=True)

    def write_report(self, report):
        data = json.dumps(report, indent=2, allow_nan=False) + "\n"
        self.artifact.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.artifact, self.destination / self.report_name):
            with path.open("x" if self.version else "w") as stream:
                stream.write(data)


def publication_paths(root, task, version=None):
    names = {
        "rotate-valve": ("rotate_valve", "rotate_valve_site_demo.json", "rotate_valve_trace.json"),
        "open-hatch": ("hatch", "hatch_site_demo.json", "hatch_expert_trace.json"),
    }
    prefix, report_name, trace_name = names[task]
    destination = root / "website/public/static"
    artifact = root / "artifacts" / report_name
    url_prefix = "./static"
    if version is not None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", version):
            raise ValueError("Version must be a safe lowercase identifier of 1–64 characters")
        destination /= f"recordings/{task}/{version}"
        artifact = root / "artifacts" / f"{prefix}_{version}_demo.json"
        url_prefix += f"/recordings/{task}/{version}"
        report_name, trace_name = "report.json", "trace.json"
        if any(path.exists() or path.is_symlink() for path in (destination, artifact)):
            raise FileExistsError("Preserve published versions and their artifacts")
    return PublicationPaths(destination, artifact, url_prefix, report_name, trace_name, version)


def geometry_metadata(record, version):
    """Copy actual recorded provenance; never relabel an old film from live config."""
    result = {key: record[key] for key in GEOMETRY_FIELDS if key in record}
    if version is not None and version.startswith("registered-"):
        if set(result) != set(GEOMETRY_FIELDS):
            raise ValueError("Registered film is missing geometry/camera provenance")
        if result["robot_geometry_version"] != "registered-v1":
            raise ValueError("Registered film must use registered-v1 geometry")
        audit = result["registered_geometry_audit"]
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise ValueError("Registered geometry audit did not pass")
        if result["robot_camera_profile"] != "geometry-v2":
            raise ValueError("Registered film requires geometry-v2 robot cameras")
        expected = json.loads(json.dumps(robot_camera_profile("geometry-v2").metadata()))
        if result["robot_camera_parameters"] != expected:
            raise ValueError("Recorded camera parameters disagree with geometry-v2")
        if record.get("camera_streams", {}).get("enabled") is not True:
            raise ValueError("Registered film requires paired real camera streams")
    if version is not None:
        result["recording_version"] = version
    return result


def verify_decoded_film(path, expected_frames, read_frames):
    """Decode the entire encoded file, not just its possibly misleading header."""
    reader = read_frames(str(path))
    try:
        metadata = next(reader)
        fps, duration = float(metadata["fps"]), float(metadata["duration"])
        if not math.isfinite(fps) or not math.isfinite(duration):
            raise ValueError("Encoded film has non-finite timing")
        if abs(fps - 30) > 0.01 or abs(duration - expected_frames / 30) > 0.05:
            raise ValueError("Encoded film timing differs from 30 Hz telemetry")
        frames = sum(1 for _ in reader)
        if frames != expected_frames:
            raise ValueError(f"Encoded film contains {frames} frames, expected {expected_frames}")
        return {"frames": frames, "fps": fps, "duration_s": duration, "full_decode_verified": True}
    finally:
        reader.close()


def verified_moviepy_tail_loss(record, source_video, log_path, decoded_frames, expected_frames):
    """Recognize only the reproduced 2.2.1 cumulative-duration final-frame loss."""
    if record.get("moviepy_version") != "2.2.1" or decoded_frames != expected_frames - 1:
        raise ValueError("Not the supported one-frame MoviePy 2.2.1 tail-loss case")
    if int(sum([1 / 30] * expected_frames) * 30) != decoded_frames:
        raise ValueError("MoviePy cumulative-duration truncation does not explain this count")
    log = log_path.read_bytes()
    matches = []
    for line in log.decode().splitlines():
        match = re.search(r"\[VideoRecorder\] Wrote (\d+) frames to (.+)$", line)
        if match and Path(match[2]).resolve() == source_video.resolve() and int(match[1]) == expected_frames:
            matches.append(line)
    if len(matches) != 1:
        raise ValueError("Recorder log must establish the exact pre-encoder source frame count")
    import hashlib

    return {
        "reason": "MoviePy 2.2.1 cumulative duration rounds down; iter_frames omits only final index",
        "moviepy_version": "2.2.1",
        "recorder_log_sha256": hashlib.sha256(log).hexdigest(),
        "recorder_log_line": matches[0],
        "observer_frames_before_encoder": expected_frames,
        "observer_source_decoded_frames": decoded_frames,
        "observer_missing_tail_frames": 1,
        "telemetry_dropped_tail_frames": 1,
        "base_dropped_tail_frames": 1,
        "gripper_dropped_tail_frames": 1,
        "padding_or_frame_repetition": False,
    }
