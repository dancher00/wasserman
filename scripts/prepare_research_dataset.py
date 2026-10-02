"""Verified nested dataset views; reuse RGB files without duplicating disk storage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wasman.controllers.marine_evidence import verify


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, choices=[10, 20, 40, 80], required=True)
    p.add_argument("--interface", choices=["actuator", "ee"], default="actuator")
    a = p.parse_args()
    if a.output_dir.exists():
        p.error("Fresh output required")
    audit_path = a.source / "dataset_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["pilot"] or len(audit["selected_seeds"]) != 80:
        raise ValueError("Complete original80-episode audit required")
    seeds = audit["selected_seeds"][: a.episodes]
    records = []
    a.output_dir.mkdir(parents=True)
    for seed in seeds:
        source = next(a.source.glob(f"train_batch*/seed_{seed}"))
        item = next(x for x in audit["episodes"] if x["seed"] == seed)
        for name in ["metadata.json", "trajectory.npz"]:
            if sha(source / name) != item["hashes"][name]:
                raise ValueError(f"Source changed: {source / name}")
        meta = json.loads((source / "metadata.json").read_text())
        data = dict(np.load(source / "trajectory.npz"))
        outcome = verify(data, meta["success_contract"])
        if outcome["first_success_step"] != [meta["length"]]:
            raise ValueError("Physical episode mismatch")
        target = a.output_dir / "train_batch00" / f"seed_{seed}"
        target.mkdir(parents=True)
        if a.interface == "ee":
            if meta["task"] not in ["PullLever", "PushSlider"]:
                raise ValueError("Unsupported EE teacher")
            data["original_actuator_command"] = data["command"].copy()
            quaternion = np.tile([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], (meta["length"], 1)).astype(np.float32)
            data["command"] = np.concatenate([data["expert_target"], quaternion, data["actuator"][:, -1:]], -1).astype(
                np.float32
            )
            np.savez_compressed(target / "trajectory.npz", **data)
        else:
            (target / "trajectory.npz").symlink_to((source / "trajectory.npz").resolve())
        (target / "wrist.rgb").symlink_to((source / "wrist.rgb").resolve())
        meta.update(interface=a.interface, source_episode=str(source.resolve()))
        (target / "metadata.json").write_text(json.dumps(meta, indent=2))
        records.append(
            dict(
                seed=seed,
                length=meta["length"],
                physical_replay=outcome,
                source_hashes=item["hashes"],
                hashes={n: sha(target / n) for n in ["metadata.json", "trajectory.npz"]},
            )
        )
    result = dict(
        passed=True,
        pilot=False,
        task=audit["task"],
        selected_seeds=seeds,
        episodes=records,
        interface=a.interface,
        source_audit_sha256=sha(audit_path),
        source=str(a.source.resolve()),
        nested_prefix=True,
        ee_closed_loop_replay_required=a.interface == "ee",
    )
    (a.output_dir / "dataset_audit.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
