"""Create a checksummed EE view of the very same v2 marine episodes and RGB."""

import argparse
import json
from pathlib import Path

import numpy as np

from wasman.learning.revision_protocol import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    audit_path = a.data / "revision_audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit["task"] not in ("PushSlider", "PullLever") or a.output_dir.exists():
        raise ValueError("Require audited marine episodes and a fresh view directory")
    for row in audit["episodes"]:
        source = a.data / row["path"]
        for name, digest in row["files"].items():
            if sha256(source / name) != digest:
                raise ValueError("Source changed after independent audit")
        data = dict(np.load(source / "trajectory.npz"))
        meta = json.loads((source / "metadata.json").read_text())
        target = a.output_dir / "train_batch00" / f"seed_{row['seed']}"
        target.mkdir(parents=True)
        data["original_actuator_command"] = data["command"].copy()
        quaternion = np.tile([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], (meta["length"], 1)).astype(np.float32)
        data["command"] = np.concatenate((data["expert_target"], quaternion, data["actuator"][:, -1:]), -1).astype(
            np.float32
        )
        np.savez_compressed(target / "trajectory.npz", **data)
        (target / "wrist.rgb").symlink_to((source / "wrist.rgb").resolve())
        meta.update(interface="ee", source_audit_sha256=sha256(audit_path), source_episode=row["path"])
        (target / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Preserve failed and unused attempts as well as the selected successes.
    selected = {row["seed"] for row in audit["episodes"]}
    for source in sorted(a.data.glob("train_batch*/seed_*/metadata.json")):
        meta = json.loads(source.read_text())
        if meta["seed"] in selected:
            continue
        target = a.output_dir / "train_batch00" / f"seed_{meta['seed']}"
        target.mkdir(parents=True)
        meta.update(
            interface="ee",
            source_audit_sha256=sha256(audit_path),
            source_episode=str(source.parent.relative_to(a.data)),
        )
        (target / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (a.output_dir / "view.json").write_text(
        json.dumps(
            dict(
                task=audit["task"],
                interface="ee",
                source_audit_sha256=sha256(audit_path),
                paired_native_and_ee_replay_required=True,
                pilot=audit["pilot"],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
