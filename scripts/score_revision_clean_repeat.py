"""Physically rescore the six same-checkpoint DP repeats from a separate checkout."""

import argparse
import json
from pathlib import Path

from revision_scoring_evidence import audit_saved_trace
from score_revision_campaign import load_run

from wasman.learning.revision_protocol import CONTROL_STEPS, PROTOCOL, TASKS, sha256


def normalized_sources(values):
    """Normalize only relocated entry-script paths, retaining identity and hashes."""
    if isinstance(values, str):
        if len(values) != 64 or any(c not in "0123456789abcdef" for c in values):
            raise ValueError("Invalid entry-script digest")
        return values
    frozen = json.loads((Path(__file__).resolve().parents[1] / "research/revision-v2-runtime.json").read_text())[
        "source_sha256"
    ]
    result = {}
    for name, digest in values.items():
        path = Path(name)
        if path.is_absolute():
            normalized = "scripts/" + path.name
            if path.parent.name != "scripts" or frozen.get(normalized) != digest:
                raise ValueError("Unrecognized relocated source path/hash")
            name = normalized
        if name in result:
            raise ValueError("Ambiguous normalized source paths")
        result[name] = digest
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing score output")
    rows = []
    for task in TASKS:
        label = f"{task}-DP-17"
        folder = args.repeat / "evaluation" / label
        name = "report.json" if task in ("PressButton", "PushSlider", "PullLever") else "summary.json"
        primary = json.loads((args.core / "primary-evaluation" / label / name).read_text())
        report = json.loads((folder / name).read_text())
        original, original_evidence = load_run(args.core, task, "DP", 17)
        for key in (
            "checkpoint_sha256",
            "seeds",
            "purpose",
            "asset_profile",
            "diagnostic_condition",
            "initial_reset",
            "policy_random_seed",
            "inference_precision",
            "replan_every_control_steps",
            "policy_hz",
            "target_interpolation",
            "inference_inputs",
        ):
            if report.get(key) != primary.get(key):
                raise ValueError(f"{task}: mismatched {key}")
        if normalized_sources(report["source_sha256"]) != normalized_sources(primary["source_sha256"]):
            raise ValueError("Source bytes/relative identities differ")
        if report["purpose"] != "test" or report.get("expert_at_inference"):
            raise ValueError("Unexpected inference role")
        if report.get("evaluated_steps", report.get("steps")) != CONTROL_STEPS[task]:
            raise ValueError("Incomplete repeat horizon")
        repeated, evidence = audit_saved_trace(folder, task, report)
        if evidence["initialization_sha256"] != original_evidence["initialization_sha256"]:
            raise ValueError("Recorded initialization mismatch")
        seeds = report["seeds"]
        rows.append(
            dict(
                task=task,
                model="DP",
                training_seed=17,
                episodes=30,
                source_successes=int(original.sum()),
                repeat_successes=int(repeated.sum()),
                matching_outcomes=int((original == repeated).sum()),
                lost_success_seeds=[s for s, x, y in zip(seeds, original, repeated, strict=True) if x and not y],
                gained_success_seeds=[s for s, x, y in zip(seeds, original, repeated, strict=True) if y and not x],
                source_evidence=original_evidence,
                repeat_evidence=evidence,
                repeat_report_sha256=sha256(folder / name),
            )
        )
    result = dict(
        protocol=PROTOCOL,
        complete=True,
        rows=rows,
        matching_outcomes=sum(r["matching_outcomes"] for r in rows),
        episodes=180,
        scorer_sha256=sha256(__file__),
        scope=(
            "Six full-horizon seed-17 DP checkpoints repeated in a separate checkout on the same GPU. "
            "Not additional training seeds or a new primary score table; "
            "not causal attribution of execution differences."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
