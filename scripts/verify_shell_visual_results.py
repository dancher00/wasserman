"""Recompute final model success from raw contact, geometry and velocity evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from wasman.controllers.shell_collection_contract import ShellCollectionContract


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--scene-protocol", type=Path, required=True)
    p.add_argument("--asset-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads(args.scene_protocol.read_text())
    assets = json.loads(args.asset_manifest.read_text())
    for name, digest in protocol["source_sha256"].items():
        if sha(name) != digest:
            raise ValueError(f"Frozen scene/controller changed: {name}")
    for name, digest in assets["files"].items():
        if sha(Path("src/wasman/assets/data/objects/shell_collection") / name) != digest:
            raise ValueError(f"Frozen asset changed: {name}")
    records = []
    initial_reference = None
    for model in ("act", "dp"):
        for suffix in ("test31", "single42"):
            name = f"{model}_{suffix}"
            run = args.root / name
            data = torch.load(run / "rollout.pt", weights_only=True, map_location="cpu")
            r, trace = data["summary"], data["trace"]
            if r != json.loads((run / "summary.json").read_text()):
                raise ValueError("Summary mismatch")
            expected = [*range(5500, 5530), 42] if suffix == "test31" else [42]
            if (
                r["seeds"] != expected
                or r["expert_at_inference"]
                or r["task"] != "Wasman-Underwater-CollectShell-Direct"
            ):
                raise ValueError("Wrong seeds, policy or task")
            if r["policy_hz"] != 30:
                raise ValueError("Different policy command frequency")
            if r["evaluated_steps"] != 7190 or r["task_horizon_s"] != 240 or r["step_dt"] != 1 / 30:
                raise ValueError("Different evaluation window")
            if sha(r["checkpoint"]) != r["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed")
            source = json.loads((run / "source_manifest.json").read_text())
            for key, digest in protocol["source_sha256"].items():
                if "shell_collection.py" not in key and source.get(key) != digest:
                    raise ValueError(f"Different rollout source: {key}")
            if json.loads((run / "asset_manifest.json").read_text()) != assets:
                raise ValueError("Different rollout assets")
            if suffix == "test31":
                initial = trace[0]["measured"]
                if initial_reference is not None and (initial - initial_reference).abs().max() > 2e-5:
                    raise ValueError("Model initial states are not paired")
                initial_reference = initial
            n = len(expected)
            contract = ShellCollectionContract(n, 1, "cpu", r["step_dt"])
            active = torch.ones(n, dtype=torch.bool)
            first = torch.full((n,), -1)
            resets = first.clone()
            for step, row in enumerate(trace):
                state = contract.update(**row["inputs"])
                success = state["success"] & active & ~row["reset"]
                if not torch.equal(active, row["active"]) or not torch.equal(success, row["success"]):
                    raise ValueError(f"Physical contract replay mismatch: {name}/{step}")
                first[success] = step + 1
                resets[active & row["reset"]] = step + 1
                active &= ~(success | row["reset"])
            if active.any() and len(trace) != r["evaluated_steps"]:
                raise ValueError("Unfinished short evaluation")
            if first.tolist() != r["first_success_step"] or resets.tolist() != r["first_reset_step"]:
                raise ValueError("Episode censoring mismatch")
            if (first > 0).tolist() != r["success_per_seed"] or int((first > 0).sum()) != r["successes"]:
                raise ValueError("Aggregate success mismatch")
            records.append(
                dict(
                    run=name,
                    successes=r["successes"],
                    episodes=n,
                    contract_replayed=True,
                    trace_sha256=sha(run / "rollout.pt"),
                    summary_sha256=sha(run / "summary.json"),
                )
            )
    args.output.write_text(
        json.dumps(
            dict(runs=records, scene_protocol_sha256=sha(args.scene_protocol), assets_sha256=sha(args.asset_manifest)),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
