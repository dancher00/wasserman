"""Independently recompute angle-only scores from saved physical rollout traces."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    report = json.loads((directory / "summary.json").read_text())
    data = torch.load(directory / "rollout.pt", weights_only=True, map_location="cpu")
    assert data["summary"] == report
    assert report["success_contract"]["version"] == "ambench-angle-170-v1"
    n = len(report["seeds"])
    active = torch.ones(n, dtype=torch.bool)
    first_success = torch.full((n,), -1, dtype=torch.long)
    first_reset = first_success.clone()
    maxima = torch.full((n,), -torch.inf)
    contact = torch.zeros(n, dtype=torch.bool)
    for step, row in enumerate(data["trace"], 1):
        assert torch.equal(row["active"], active), (directory, step, "active mask")
        assert torch.isfinite(row["angle"][active]).all()
        assert torch.isfinite(row["action"][active]).all()
        reset = row["reset"] & active
        valid = active & ~reset
        success = valid & (row["angle"] >= math.radians(170))
        first_success[success] = step
        first_reset[reset] = step
        maxima = torch.maximum(maxima, torch.where(valid, row["angle"], -torch.inf))
        contact |= valid & row["grasped"]
        active &= ~(success | reset)
    flags = first_success >= 0
    assert first_success.tolist() == report["first_success_step"]
    assert first_reset.tolist() == report["first_reset_step"]
    assert flags.tolist() == report["success_per_seed"]
    assert int(flags.sum()) == report["successes"]
    assert n == report["episodes"]
    assert torch.allclose(torch.rad2deg(maxima), torch.tensor(report["max_angle_deg"]), atol=1e-5)
    if "ever_grasped" in report:
        assert contact.tolist() == report["ever_grasped"]
    if "checkpoint" in report:
        checkpoint_path = Path(report["checkpoint"])
        assert sha256(checkpoint_path) == report["checkpoint_sha256"]
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        config = checkpoint["config"]
        assert not set(report["seeds"]) & set(config["train_seeds"] + config.get("validation_seeds", []))
        for source, expected in config["source_sha256"].items():
            assert sha256(Path(source)) == expected, source
        assert not report["expert_at_inference"]
    return dict(
        directory=str(directory),
        episodes=n,
        successes=int(flags.sum()),
        resets=int((first_reset >= 0).sum()),
        bilateral_contact_episodes=int(contact.sum()),
        rollout_sha256=sha256(directory / "rollout.pt"),
        summary_sha256=sha256(directory / "summary.json"),
        score_recomputed_from_signed_angle=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Fresh output required")
    protected = json.loads((args.root / "experiment.json").read_text())["protected_source_sha256"]
    for source, expected in protected.items():
        assert sha256(Path(source)) == expected, source
    directories = ["expert_test31", "act_test31", "act_single42", "dp_test31", "dp_single42"]
    results = [verify(args.root / d) for d in directories if (args.root / d / "summary.json").exists()]
    result = dict(
        protected_sources_unchanged=True,
        reports=results,
        pending=[d for d in directories if not (args.root / d / "summary.json").exists()],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
