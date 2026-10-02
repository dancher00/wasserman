"""Retrospective angle-only scoring; preserve inputs and censor at the first reset."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wasman.controllers.valve_success import AMB_VALVE_SUCCESS, valve_angle_criteria, valve_success_metadata


def score(angle, valid, dt):
    passed = valve_angle_criteria(angle)["valve_rotated"] & valid
    return {
        "episodes": angle.shape[1],
        "successes": int(passed.any(0).sum()),
        "success_per_env": passed.any(0).tolist(),
        "first_success_time_s": [
            (int(ids[0]) + 1) * dt if len(ids := passed[:, i].nonzero().flatten()) else None
            for i in range(angle.shape[1])
        ],
        "max_angle_before_reset_deg": torch.rad2deg(torch.where(valid, angle, -torch.inf).amax(0)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--trace", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    data = torch.load(args.paired, map_location="cpu", weights_only=True)
    summary, h = data["summary"], data["history"]
    n = len(summary["seeds"])
    report = {
        "role": "Retrospective rescore, not new rollouts or independent acceptance",
        "success_contract": valve_success_metadata(AMB_VALVE_SUCCESS),
        "censoring": "Exclude first reset frame and all subsequent frames; JSON traces must have zero resets",
        "paired_seeds": summary["seeds"],
        "expert_repeat_is_not_independent_sample": True,
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.paired, *args.trace]},
        "paired": [],
        "traces": [],
    }
    for group, variant in enumerate(summary["variants"]):
        sl = slice(group * n, (group + 1) * n)
        valid = h["done"][:, sl].int().cumsum(0) == 0
        report["paired"].append(
            {
                "name": variant["name"],
                "legacy_successes": sum(variant["success"]),
                **score(h["angle"][:, sl], valid, summary["dt"]),
            }
        )
    for path in args.trace:
        r = json.loads(path.read_text())
        if r["failed_resets"]:
            raise ValueError(f"Cannot safely censor JSON trace without per-frame reset flags: {path}")
        angle = torch.tensor([f["angle_rad"] for f in r["trace"]])
        report["traces"].append(
            {
                "path": str(path),
                "seed": r["seed"],
                "legacy_successes": r["successes"],
                **score(angle, torch.ones_like(angle, dtype=torch.bool), r["protocol"]["policy_dt_s"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["paired"] + report["traces"]:
        print(row.get("name", row.get("path")), row["legacy_successes"], "->", row["successes"], "/", row["episodes"])


if __name__ == "__main__":
    main()
