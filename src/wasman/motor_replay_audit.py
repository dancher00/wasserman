"""Read-only independent validation of paired 120 Hz motor-replay recordings."""

import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_directory(directory, *, decode_video=True):
    directory = Path(directory)
    suite = json.loads((directory / "report.json").read_text())
    if not suite["passed"] or suite["calibrated"] or suite["feedback_compensation"]:
        raise ValueError("Incomplete or incorrectly labeled replay suite")
    if suite["loss_coefficient"] != 0.2 or suite["range_diameters"] != 10:
        raise ValueError("Boundary assumptions changed")
    result = {"passed": True, "report_sha256": sha(directory / "report.json"), "modes": []}
    for mode in suite["modes"]:
        command_file = directory / mode["command_file"]
        if sha(command_file) != mode["command_sha256"]:
            raise ValueError("Command tape hash mismatch")
        command = np.load(command_file)
        reports, traces, details = {}, {}, []
        for take in ("reference_off", "off", "on"):
            folder = directory / mode["id"] / take
            report = json.loads((folder / "report.json").read_text())
            trace = json.loads((folder / "trace.json").read_text())
            if not report["passed"] or sha(folder / "trace.json") != report["trace_sha256"]:
                raise ValueError("Take failed or trace modified")
            if report["feedback_compensation"] != (take == "reference_off"):
                raise ValueError("Replay must ignore feedback")
            if report["automatic_reset"] or report["runtime_pose_or_velocity_overwrites"]:
                raise ValueError("Reset or forced motion invalidates replay")
            physics = trace["physics_frames"]
            if len(physics) != len(command) or len(physics) != 4 * len(trace["frames"]):
                raise ValueError("Physics/control frame mismatch")
            for i, frame in enumerate(physics):
                if frame["t_s"] != (i + 1) / 120:
                    raise ValueError("Physics clock mismatch")
                numbers = np.concatenate([np.asarray(v).reshape(-1) for v in frame.values()])
                if not np.isfinite(numbers).all():
                    raise ValueError("Nonfinite recorded telemetry")
                if min(frame["collider_gaps_m"]) < 0.02 or frame["contact_history_max_n"] != 0:
                    raise ValueError("Collision-clearance/contact guard failed")
                low, high = suite["motor_force_limits_n"]
                if (
                    min(frame["motor_force_before_boundary_n"]) < low
                    or max(frame["motor_force_before_boundary_n"]) > high
                ):
                    raise ValueError("Actuator force limit violation")
                gains = np.asarray(frame["boundary_gain"])
                if not ((gains >= 0.8 - 1e-7) & (gains <= 1)).all():
                    raise ValueError("Boundary gain outside original coefficient limits")
                if take != "on" and not (gains == 1).all():
                    raise ValueError("OFF take contains boundary correction")
            if trace["frames"] != physics[3::4]:
                raise ValueError("Video telemetry not corresponding post-step 30 Hz samples")
            if not np.array_equal(command[:, 0], [p["input_wrench_b"] for p in physics]):
                raise ValueError("Motor input sequence is not exact")
            if min(report["initial_conservative_gaps_m"]) < 0.02:
                raise ValueError("Initial collider gap unsafe")
            detail = {"id": take, "physics_frames": len(physics), "frames": len(trace["frames"])}
            if take != "reference_off" and decode_video:
                import imageio_ffmpeg

                video = folder / "observer.mp4"
                if sha(video) != report["video_sha256"]:
                    raise ValueError("Video hash mismatch")
                reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
                try:
                    metadata = next(reader)
                    count = sum(1 for _ in reader)
                finally:
                    reader.close()
                if count != len(trace["frames"]) or metadata["fps"] != 30:
                    raise ValueError("Decoded video timing/frame-count mismatch")
                detail["decoded_video"] = {"frames": count, "fps": metadata["fps"], "size": metadata["size"]}
            reports[take], traces[take] = report, trace
            details.append(detail)
        initial = reports["reference_off"]["initial_state"]
        if any(report["initial_state"] != initial for report in reports.values()):
            raise ValueError("Initial states are not exact matches")
        reference = traces["reference_off"]["physics_frames"]
        reference_forces = [p["motor_force_before_boundary_n"] for p in reference]
        for trace in traces.values():
            if reference_forces != [p["motor_force_before_boundary_n"] for p in trace["physics_frames"]]:
                raise ValueError("Actual pre-boundary actuator forces differ")
        if [p["position_w_m"] for p in reference] != [p["position_w_m"] for p in traces["off"]["physics_frames"]]:
            raise ValueError("OFF replay did not reproduce the reference trajectory exactly")
        if reports["off"]["camera"] != reports["on"]["camera"]:
            raise ValueError("Paired observer cameras differ")
        a, b = [np.asarray([p["position_w_m"] for p in traces[t]["frames"]]) for t in ("off", "on")]
        maximum = float(np.linalg.norm(b - a, axis=1).max())
        if abs(maximum - mode["maximum_position_difference_m"]) > 1e-12:
            raise ValueError("Reported motion difference is not reproduced by telemetry")
        result["modes"].append(
            {
                "id": mode["id"],
                "maximum_position_difference_m": maximum,
                "reference_off_pose_exact_match": True,
                "takes": details,
            }
        )
    return result
