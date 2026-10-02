"""Localize failed physical milestones without treating diagnosis as a new test score."""

import argparse
import json
from pathlib import Path

import numpy as np

from wasman.controllers.marine_evidence import verify


def summarize(folder):
    r = json.loads((folder / "report.json").read_text())
    c = json.loads((folder / "contract.json").read_text())
    d = dict(np.load(folder / "trace.npz"))
    v = verify(d, c)
    rows = []
    for i, seed in enumerate(r["seeds"]):
        valid = d["active"][:, i] & ~d["terminal"][:, i]
        indices = np.flatnonzero(valid)
        forces = np.linalg.norm(d["forces"][:, i], axis=-1)
        bilateral = (forces > 0.12).all(-1) & valid
        progress = (d["q"][:, i] - c["initial"]) * c["direction"]
        first = np.flatnonzero(bilateral)
        category = (
            "success"
            if v["success_per_seed"][i]
            else "no_bilateral_contact"
            if not len(first)
            else "incomplete_constrained_motion"
            if progress[valid].max() < c["threshold"]
            else "terminal_alignment_or_stability"
        )
        rows.append(
            dict(
                seed=seed,
                success=v["success_per_seed"][i],
                first_missing_stage=category,
                min_tool_distance_m=float(d["distance"][valid, i].min()),
                max_progress=float(progress[valid].max()),
                progress_unit=c["unit"],
                first_bilateral_contact_s=float(first[0] * r["dt"]) if len(first) else None,
                bilateral_fraction=float(bilateral[valid].mean()),
                peak_attitude_rad=float(d["attitude"][valid, i].max()),
                time_min_distance_s=float(indices[np.argmin(d["distance"][valid, i])] * r["dt"]),
            )
        )
    return dict(run=str(folder), seeds=r["seeds"], model_has_expert=r["expert_at_inference"], episodes=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["PullLever", "PushSlider"], required=True)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/research_core_20260924"))
    a = p.parse_args()
    root = Path("artifacts/marine_mechanisms_20260924") / a.task
    runs = [root / "train_batch00"] + sorted(root.glob("validation_*")) + sorted(root.glob("test_*"))
    report = dict(
        task=a.task,
        kind="Physical failure localization, not a causal architecture attribution",
        definition=(
            "Diagnostic bilateral contact means both finger force norms exceed 0.12 N; "
            "original success criteria unchanged."
        ),
        caveat=(
            "Training teacher episodes and validation/test states differ. "
            "A shared-state shadow rollout is required to attribute prediction error causally."
        ),
        runs=[summarize(x) for x in runs if (x / "report.json").exists()],
    )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    target = a.output_dir / f"{a.task}_failure_analysis.json"
    target.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
