"""Replay unchanged hatch success contract from measured rollout signals and audit provenance."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wasman.controllers.hatch_contract import HatchContract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.root / "experiment.json").read_text())
    for name, digest in protocol["protected_source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Protected source changed: {name}")
    records = []
    for run in args.runs:
        path = args.root / run
        rollout = torch.load(path / "rollout.pt", weights_only=True, map_location="cpu")
        report, trace = rollout["summary"], rollout["trace"]
        if report != json.loads((path / "summary.json").read_text()):
            raise ValueError("Standalone summary differs from trace")
        n = len(report["seeds"])
        contract = HatchContract(n, "cpu", report["step_dt"])
        first = torch.full((n,), -1, dtype=torch.long)
        valid = torch.ones(n, dtype=torch.bool)
        for step, row in enumerate(trace):
            valid &= ~row["reset"]
            success = contract.update(
                row["angle"], row["speed"], row["forces"], row["near_handle"], row["tool_speed"], row["stable"]
            )
            new = success & valid & (first < 0)
            first[new] = step + 1
            expected_new = row["success"]
            if not torch.equal(new, expected_new):
                raise ValueError(f"Success contract replay differs: {run}, step{step}")
        if first.tolist() != report["first_success_step"]:
            raise ValueError(f"First successes disagree: {run}")
        if "checkpoint" in report:
            checkpoint = Path(report["checkpoint"])
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash differs")
            config = json.loads((checkpoint.parent / "config.json").read_text())
            if set(report["seeds"]) & set(config["train_seeds"] + config.get("validation_seeds", [])):
                raise ValueError("Training/test seed leakage")
            if report["expert_at_inference"]:
                raise ValueError("Learned score contains expert assistance")
        records.append(
            dict(
                run=run,
                successes=int((first >= 0).sum()),
                episodes=n,
                contract_replayed=True,
                trace_sha256=hashlib.sha256((path / "rollout.pt").read_bytes()).hexdigest(),
                summary_sha256=hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
            )
        )
    if args.output.exists():
        raise ValueError("Fresh verification path required")
    args.output.write_text(json.dumps(dict(runs=records, protected_sources_unchanged=True), indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
