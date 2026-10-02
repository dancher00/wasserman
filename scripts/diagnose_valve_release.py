"""Summarize measured unloading/withdrawal from a completed recorder trace.

This is a diagnostic, not a replacement for physical contract replay.
"""

import argparse
import json
import math
from pathlib import Path


def summarize(data):
    frames = data["trace"]
    if not frames:
        raise ValueError("Empty trace")
    rows = []
    for env in range(len(frames[0]["angle_rad"])):
        held = next((i for i, f in enumerate(frames) if f["held"][env]), None)
        # Use measured displacement after hold, not a policy's latent mode.
        withdrawal = None
        if held is not None:
            start_x = frames[held]["tool_position_m"][env][0]
            withdrawal = next(
                (i for i in range(held, len(frames)) if start_x - frames[i]["tool_position_m"][env][0] >= 0.01),
                None,
            )
        end = frames[-1]
        row = {
            "environment": env,
            "success": any(f["success"][env] for f in frames),
            "held": held is not None,
            "final_angle_deg": math.degrees(end["angle_rad"][env]),
            "withdrawal_definition": "TCP retreats 10 mm along world X from first measured hold",
            "withdrawal_time_s": None,
            "withdrawal_force_n": None,
            "withdrawal_angle_deg": None,
            "post_withdrawal_angle_change_deg": None,
        }
        if withdrawal is not None:
            f = frames[withdrawal]
            row.update(
                withdrawal_time_s=f["t"],
                withdrawal_force_n=max(f["finger_forces_n"][env]),
                withdrawal_angle_deg=math.degrees(f["angle_rad"][env]),
                post_withdrawal_angle_change_deg=math.degrees(end["angle_rad"][env] - f["angle_rad"][env]),
            )
        rows.append(row)
    return {
        "role": "Measured release diagnostic; not a success-contract validator or causal proof",
        "environments": rows,
        "successes": sum(r["success"] for r in rows),
        "held": sum(r["held"] for r in rows),
        "withdrawing_under_contact": sum(
            r["withdrawal_force_n"] is not None and r["withdrawal_force_n"] >= 0.05 for r in rows
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(json.loads(args.trace.read_text())), indent=2))


if __name__ == "__main__":
    main()
