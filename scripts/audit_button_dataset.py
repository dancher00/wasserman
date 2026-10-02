"""Audit native30Hz episode alignment, lossless targets and raw physical completion."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from button_visual_support import verify

from wasman.learning.marine_visual_dataset import MarineVisualDataset


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while b := stream.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    metas = sorted(args.data.glob("seed_*/metadata.json" if args.pilot else "train_batch*/seed_*/metadata.json"))
    successes = [p.parent for p in metas if json.loads(p.read_text())["outcome"] == "success"][: args.episodes]
    if len(successes) != args.episodes:
        raise ValueError("Insufficient successful whole episodes")
    ds = MarineVisualDataset(successes)
    reports = []
    seeds = []
    tasks = []
    for path in successes:
        meta = json.loads((path / "metadata.json").read_text())
        d = dict(np.load(path / "trajectory.npz"))
        n = meta["length"]
        seeds.append(meta["seed"])
        tasks.append(meta["task"])
        if d["terminal"].any() or d["success_after"][:-1].any() or not d["success_after"][-1]:
            raise ValueError("Censoring violation")
        if len(d["measured"]) != n or not all(
            np.isfinite(d[k]).all() for k in ["measured", "command", "actuator", "time", "camera_time"]
        ):
            raise ValueError("Invalid data")
        expected = d["actuator"]
        if not np.array_equal(expected, d["command"]):
            raise ValueError("Actuator packing loses information")
        if np.abs(d["time"] - np.arange(n) / 30).max() > 1e-6:
            raise ValueError("Sample timing")
        clock = np.cumsum(np.full(n * 4, meta["physics_dt"], dtype=np.float32), dtype=np.float32)[3::4]
        if np.max(np.abs(clock - d["camera_time"])) > 1e-5:
            raise ValueError("Camera clock mismatch")
        verified = verify(d, meta["success_contract"])
        if verified["first_success_step"] != [n]:
            raise ValueError("First success censoring mismatch")
        rgb = np.memmap(path / "wrist.rgb", mode="r", dtype=np.uint8, shape=(n, 384, 384, 3))
        sample = np.asarray(rgb[np.linspace(0, n - 1, 9).astype(int)])
        if np.any(sample.reshape(9, -1).std(-1) < 1):
            raise ValueError("Blank RGB")
        change = np.abs(np.diff(sample.astype(float), axis=0)).mean()
        if change < 0.1:
            raise ValueError("Stale RGB")
        reports.append(
            dict(
                seed=meta["seed"],
                length=n,
                physical_replay=verified,
                mean_sample_image_change=float(change),
                hashes={f: sha(path / f) for f in ["metadata.json", "trajectory.npz", "wrist.rgb"]},
            )
        )
        print("verified", meta["seed"], n, flush=True)
    if len(set(seeds)) != len(seeds) or len(set(tasks)) != 1:
        raise ValueError("Duplicate seeds/mixed tasks")
    output = args.data / "dataset_audit.json"
    if output.exists():
        raise ValueError("Preserve previous audit")
    output.write_text(
        json.dumps(
            dict(
                passed=True,
                task=tasks[0],
                episodes=reports,
                selected_seeds=seeds,
                native_hz=30,
                samples=len(ds),
                reset_splicing=False,
                pilot=args.pilot,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
