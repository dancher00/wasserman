"""Publish only a reset-free hatch recording that passes measured-contract replay."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

from wasman.controllers.hatch_recording import replay_hatch_trace
from wasman.recording_publication import (
    geometry_metadata,
    publication_paths,
    verified_moviepy_tail_loss,
    verify_decoded_film,
)


def main(argv=None, *, root=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--version", help="Publish a fresh version without replacing legacy recordings")
    parser.add_argument("--allow-moviepy-tail-loss", action="store_true")
    parser.add_argument("--recorder-log", type=Path)
    args = parser.parse_args(argv)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    paths = publication_paths(root, "open-hatch", args.version)
    dest = paths.destination
    summary = json.loads((args.source / "summary.json").read_text())
    recorded_geometry = geometry_metadata(summary, args.version)
    trace_path = args.source / "trace.json"
    trace = json.loads(trace_path.read_text())
    if summary["num_envs"] != 1 or any(summary["failed"]):
        raise ValueError("Publish one reset-free trial only")
    replay = replay_hatch_trace(trace)
    if replay["successes"] != 1 or not trace[-1]["success"][0]:
        raise ValueError("No measured hatch completion")
    video = args.source / "hatch_expert_0000.mp4"
    reader = imageio_ffmpeg.read_frames(str(video))
    metadata = next(reader)
    source_frames = sum(1 for _ in reader)
    reader.close()
    duration = len(trace) / 30
    cameras = summary.get("camera_streams", {})
    trim_frames = cameras.get("observer_initial_frames_to_trim", 0)
    if abs(metadata["duration"] - duration - trim_frames / 30) > 0.05 or abs(metadata["fps"] - 30) > 0.01:
        raise ValueError("Video and telemetry timing differ")
    if cameras.get("enabled"):
        if cameras["frames"] != len(trace):
            raise ValueError("Camera and observer frame counts differ")
        for name in ("gripper", "base"):
            reader = imageio_ffmpeg.read_frames(str(args.source / f"hatch_{name}.mp4"))
            sensor_metadata = next(reader)
            reader.close()
            if abs(sensor_metadata["duration"] - duration) > 0.05 or abs(sensor_metadata["fps"] - 30) > 0.01:
                raise ValueError(f"{name} stream timing differs from observer")
            verify_decoded_film(args.source / f"hatch_{name}.mp4", len(trace), imageio_ffmpeg.read_frames)
    prefix_evidence = None
    source_trace_frames = len(trace)
    if source_frames != len(trace) + trim_frames:
        if not (args.allow_moviepy_tail_loss and args.recorder_log and args.version and trim_frames == 1):
            raise ValueError("Observer source count differs; explicit evidenced tail-loss opt-in required")
        prefix_evidence = verified_moviepy_tail_loss(
            summary, video, args.recorder_log, source_frames, len(trace) + trim_frames
        )
        if not cameras.get("enabled"):
            raise ValueError("Tail-loss recovery requires complete paired sensor streams")
        trace = trace[:-1]
        replay = replay_hatch_trace(trace)
        if replay["successes"] != 1 or not trace[-1]["success"][0]:
            raise ValueError("Common recorded prefix does not complete the measured contract")
        duration = len(trace) / 30
        cameras = dict(cameras, frames=len(trace), timestamps_s=cameras["timestamps_s"][:-1])
        prefix_evidence.update(
            source_trace_frames=source_trace_frames,
            published_frames=len(trace),
            source_telemetry_duration_s=source_trace_frames / 30,
            published_duration_s=duration,
            source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            source_observer_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        )
    paths.prepare()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output = dest / "open-hatch-demo.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"trim=start_frame={trim_frames},setpts=PTS-STARTPTS",
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
    observer_encoding = verify_decoded_film(output, len(trace), imageio_ffmpeg.read_frames)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(min(48, duration - 2)),
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(dest / "open-hatch-demo.png"),
        ],
        check=True,
    )
    chapters = [{"time": 0.0, "label": "Swim in"}]
    for label, predicate in (
        ("Deploy", lambda r: r["navigation_ready"][0]),
        ("Descend", lambda r: r["phase"][0] >= 1),
        ("Grasp", lambda r: r["phase"][0] >= 2),
        ("Lift", lambda r: r["phase"][0] >= 3),
        ("Hold", lambda r: r["phase"][0] >= 4),
    ):
        row = next((r for r in trace if predicate(r)), None)
        if row:
            chapters.append({"time": row["t"], "label": label})
    first_success = replay["first_success_step"][0] / 30
    chapters.append({"time": first_success, "label": "Complete"})
    chapters.sort(key=lambda c: c["time"])
    report = {
        **recorded_geometry,
        "observer_encoding": observer_encoding,
        "task": "Wasman-Underwater-OpenHatch-Direct",
        "seed": summary["seed"],
        "controller": "Physical state-feedback expert, not a learned policy",
        "moviepy_version": summary.get("moviepy_version"),
        "evaluation_role": "Single development demonstration, not a benchmark success rate",
        "episodes": 1,
        "successes": 1,
        "automatic_resets": 0,
        "telemetry_contract_replayed": True,
        "duration_s": duration,
        "frame_alignment_tolerance_s": 1 / 30,
        "first_success_time_s": first_success,
        "final_angle_deg": math.degrees(trace[-1]["angle_rad"][0]),
        "max_base_attitude_deg": math.degrees(max(r["attitude"][0] for r in trace)),
        "chapters": chapters,
        "conditions": summary["conditions"],
        "source_sha256": summary["source_sha256"],
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "protocol": "hatch-grasp-open-hold-v1; flooded, unlocked, pressure equalized; no release required",
    }
    if prefix_evidence:
        report["continuous_prefix_recovery"] = prefix_evidence
        evidence_path = dest / "encoder-tail-loss-evidence.txt"
        evidence_path.write_text(
            "MoviePy 2.2.1: ImageSequenceClip.duration=sum([1/fps]*N); "
            "Clip.iter_frames uses range(int(duration*fps)).\n"
            + prefix_evidence["recorder_log_line"] + "\n"
        )
        prefix_evidence["evidence_url"] = f"{paths.url_prefix}/{evidence_path.name}"
        prefix_evidence["evidence_sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if cameras.get("enabled"):
        camera_streams = []
        for name in ("gripper", "base"):
            source = args.source / f"hatch_{name}.mp4"
            filename = f"open-hatch-{name}-sync.mp4"
            if prefix_evidence:
                subprocess.run([
                    ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
                    "-vf", f"trim=end_frame={len(trace)},setpts=PTS-STARTPTS",
                    "-an", "-r", "30", "-fps_mode", "cfr", "-c:v", "libx264",
                    "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(dest / filename),
                ], check=True)
            else:
                shutil.copyfile(source, dest / filename)
            verify_decoded_film(dest / filename, len(trace), imageio_ffmpeg.read_frames)
            camera_streams.append(
                {
                    "id": name,
                    "label": "Gripper" if name == "gripper" else "Base",
                    "src": f"{paths.url_prefix}/{filename}",
                    "fps": 30,
                    "width": 256,
                    "height": 256,
                    "frames": cameras["frames"],
                    "duration_s": duration,
                    "sha256": hashlib.sha256((dest / filename).read_bytes()).hexdigest(),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
        report["camera_streams"] = camera_streams
        report["camera_alignment"] = {
            "same_episode": True,
            "capture": "post-physics, same step as observer",
            "initial_hold_s": cameras["initial_hold_s"],
            "observer_trimmed_frames": trim_frames,
            "max_pair_timestamp_difference_s": max(abs(t[0] - t[1]) for t in cameras["timestamps_s"]),
        }
    report.update(
        video_url=f"{paths.url_prefix}/{output.name}",
        poster_url=f"{paths.url_prefix}/open-hatch-demo.png",
        report_url=f"{paths.url_prefix}/{paths.report_name}",
        trace_url=f"{paths.url_prefix}/{paths.trace_name}",
    )
    if prefix_evidence:
        (dest / paths.trace_name).write_text(json.dumps(trace, allow_nan=False) + "\n")
    else:
        shutil.copyfile(trace_path, dest / paths.trace_name)
    report["trace_sha256"] = hashlib.sha256((dest / paths.trace_name).read_bytes()).hexdigest()
    paths.write_report(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
