"""Compare paired generated episodes and three declared RGB frames per episode."""

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from wasman.learning.revision_protocol import TASKS, sha256


def frames(path, indices, shape):
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-threads",
            "2",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(indices), *shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("left", "right"):
        parser.add_argument(f"--{side}-data", type=Path, required=True)
        parser.add_argument(f"--{side}-archives", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for task in TASKS:
        audits = [
            json.loads((getattr(args, f"{side}_data") / task / "revision_audit.json").read_text())
            for side in ("left", "right")
        ]
        manifests = [
            json.loads((getattr(args, f"{side}_archives") / task / "rgb-manifest.json").read_text())
            for side in ("left", "right")
        ]
        if not all(audit["passed"] for audit in audits):
            raise ValueError("Generation audit did not pass")
        for left, right in zip(audits[0]["episodes"], audits[1]["episodes"], strict=True):
            if left["seed"] != right["seed"] or left["length"] != right["length"]:
                raise ValueError("Paired frame times require matching seed and length")
            indices = sorted({0, left["length"] // 2, left["length"] - 1})
            images, hashes = [], []
            for side, row, manifest in zip(("left", "right"), (left, right), manifests, strict=True):
                entry = manifest["files"][row["path"] + "/wrist.rgb"]
                path = getattr(args, f"{side}_archives") / task / entry["archive_path"]
                digest = sha256(path)
                if digest != entry["archive_sha256"] or entry["raw_sha256"] != row["files"]["wrist.rgb"]:
                    raise ValueError("Archive/audit identity mismatch")
                images.append(frames(path, indices, entry["image_shape"]))
                hashes.append(digest)
            difference = images[0].astype(np.float64) - images[1].astype(np.float64)
            per_frame = [
                dict(
                    frame=index,
                    mean_absolute_channel_difference=float(np.abs(delta).mean()),
                    channel_rmse=float(np.sqrt(np.square(delta).mean())),
                    max_absolute_channel_difference=float(np.abs(delta).max()),
                    changed_pixel_fraction=float(np.any(delta != 0, axis=-1).mean()),
                )
                for index, delta in zip(indices, difference, strict=True)
            ]
            row = dict(
                task=task,
                seed=left["seed"],
                frames=left["length"],
                trajectory_bytes_identical=left["files"]["trajectory.npz"] == right["files"]["trajectory.npz"],
                rgb_bytes_identical=left["files"]["wrist.rgb"] == right["files"]["wrist.rgb"],
                archive_sha256=hashes,
                sampled_frames=per_frame,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = dict(
        scope="Development expert generation, not learned-policy robustness or full-cohort score replication",
        rgb_units="8-bit channel values, range 0..255",
        sampling="First, floor(length/2), and final recorded pre-action frame of every paired episode",
        episodes=rows,
        script_sha256=sha256(__file__),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
