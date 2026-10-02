"""Audit fixed reference controllers on all primary states without feeding outcomes into training."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from audit_revision_dataset import replay
from compare_revision_reproduction import initial_state
from revision_scoring_evidence import audit_saved_trace
from run_revision_campaign import PYTHON, run

from wasman.learning.revision_protocol import CONTROL_STEPS, TASKS, sha256, validate_runtime_snapshot


def collected_outcomes(folder, task, seeds):
    rows, initial = [], []
    for seed in seeds:
        episode = folder / f"seed_{seed}"
        metadata = json.loads((episode / "metadata.json").read_text())
        with np.load(episode / "trajectory.npz", allow_pickle=False) as archive:
            trace = dict(archive)
        n = metadata["length"]
        if metadata["seed"] != seed or any(len(value) != n or not np.isfinite(value).all() for value in trace.values()):
            raise ValueError("Invalid reference trajectory")
        first = replay(task, trace, metadata)
        success = first > 0
        if success != (metadata["outcome"] == "success"):
            raise ValueError("Reference success disagrees with physical replay")
        if not 0 < n <= CONTROL_STEPS[task] or (not success and n < CONTROL_STEPS[task] and not trace["terminal"][-1]):
            raise ValueError("Incomplete reference horizon")
        rows.append(
            dict(
                seed=seed,
                success=success,
                first_success_step=first,
                metadata_sha256=sha256(episode / "metadata.json"),
                trajectory_sha256=sha256(episode / "trajectory.npz"),
            )
        )
        initial.append(trace["measured"][0])
    return rows, np.stack(initial)


def main():
    import subprocess

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--clean-completion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_runtime_snapshot()
    deadline = time.monotonic() + 36 * 3600
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, task in enumerate(TASKS):
        if index:
            while not (args.core / "completed-all.json").exists() or not args.clean_completion.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Primary/clean evaluation gates did not finish")
                time.sleep(30)
        seeds = list(range(60000 + index * 1000, 60030 + index * 1000))
        native = task in ("PressButton", "PushSlider", "PullLever")
        folder = args.output / "data" / task / "train_batch00"
        if native:
            script = (
                "scripts/rollout_button_visual.py" if task == "PressButton" else "scripts/rollout_research_visual.py"
            )
            command = [script, "--task", task, "--mode", "expert", "--purpose", "test"]
        else:
            command = ["scripts/benchmark.py", "collect", "--task", task]
        run(
            args.output,
            task,
            [*command, "--seeds", *seeds, "--steps", CONTROL_STEPS[task], "--output-dir", folder],
            folder,
        )
        if native:
            report = json.loads((folder / "report.json").read_text())
            if report["seeds"] != seeds or report["mode"] != "expert" or not report["expert_at_inference"]:
                raise ValueError("Wrong reference identity")
            outcomes, evidence = audit_saved_trace(folder, task, report)
            rows = [dict(seed=seed, success=bool(success)) for seed, success in zip(seeds, outcomes, strict=True)]
            measured = initial_state(folder)
        else:
            rows, measured = collected_outcomes(folder, task, seeds)
            evidence = dict(summary_sha256=sha256(folder / "summary.json"), physical_contract_replayed=True)
        reference = initial_state(args.core / "primary-evaluation" / f"{task}-ACT-17")
        pairing = dict(schema_matches=reference.shape == measured.shape)
        if pairing["schema_matches"]:
            pairing.update(
                bitwise_equal=bool(np.array_equal(reference, measured)),
                max_absolute_difference=float(np.abs(reference - measured).max()),
            )
        else:
            pairing["note"] = "Stored initial-state schemas differ; no full-state equality claim"
        row = dict(
            task=task,
            episodes=30,
            successes=sum(x["success"] for x in rows),
            outcomes=rows,
            evidence=evidence,
            initial_measured_state_comparison=pairing,
            scope="Supplemental fixed privileged reference; not another learned-policy training replicate",
        )
        results.append(row)
        (args.output / "scores.json").write_text(
            json.dumps(dict(complete=len(results) == 6, rows=results), indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
        if not native:
            subprocess.run(
                [
                    PYTHON,
                    "scripts/archive_revision_rgb.py",
                    "--data",
                    str(folder.parent),
                    "--archive",
                    str(args.output / "archives" / task),
                    "--codec",
                    "libx264rgb",
                    "--prune-rgb",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
