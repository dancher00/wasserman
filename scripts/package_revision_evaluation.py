"""Package complete physically scored primary and intervention evidence separately from training data."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from package_revision_task import pack
from revision_package_assets import split_archive
from score_revision_diagnostics import CONDITIONS

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "report.json",
    "summary.json",
    "trace.npz",
    "rollout.pt",
    "contract.json",
    "source_manifest.json",
    "asset_manifest.json",
    "physical_asset.json",
    "initial_rgb.npz",
    "predictions.pt",
    "wrist_0.png",
}


def groups():
    for task in TASKS:
        yield (
            f"primary-{task}",
            [f"primary-evaluation/{task}-{model}-{seed}" for model in ("ACT", "DP", "BC") for seed in (17, 43, 101)],
        )
    for task in ("OpenHatch", "RotateValve"):
        yield (
            f"controller-{task}",
            [
                f"controller-evaluation/{condition}/{task}-DP-{seed}"
                for condition in ("ki1-nominal", *CONDITIONS)
                for seed in (17, 43, 101)
            ],
        )
    for task in ("PushSlider", "PullLever"):
        yield (
            f"interface-{task}",
            [
                f"interface-evaluation/{interface}/{task}-DP-{seed}" + ("-ee" if interface == "ee" else "")
                for interface in ("actuator", "ee")
                for seed in (17, 43, 101)
            ],
        )


def group_files(campaign, folders):
    files = {}
    for name in folders:
        folder = campaign / name
        if not folder.is_dir():
            raise ValueError(f"Incomplete evaluation: {name}")
        required = (
            {"summary.json", "rollout.pt"}
            if (folder / "summary.json").exists()
            else {"report.json", "trace.npz", "contract.json"}
        )
        if any(not (folder / item).is_file() for item in required):
            raise ValueError(f"Incomplete physical evidence: {name}")
        for path in sorted(folder.iterdir()):
            if path.name in ALLOWED:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("Evaluation evidence must be a regular file")
                files[str(path.relative_to(campaign))] = path
        receipt = campaign / "jobs" / ("eval-" + name.replace("/", "-") + ".json")
        if json.loads(receipt.read_text()).get("returncode") != 0:
            raise ValueError(f"Unfinished evaluation job: {name}")
        files[str(receipt.relative_to(campaign))] = receipt
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh output; preserve partial packages for investigation")
    runtime_hash = validate_runtime_snapshot()
    environment_path = args.campaign / "environment-observed.json"
    environment = json.loads(environment_path.read_text())
    if (
        environment.get("schema") != "wm-environment-observation-v1"
        or environment.get("protocol") != PROTOCOL
        or environment.get("runtime_manifest_sha256") != runtime_hash
        or environment.get("capture_source_sha256") != sha256(ROOT / "scripts/capture_revision_environment.py")
    ):
        raise ValueError("Missing or incompatible environment observation")
    plan = [(name, folders, group_files(args.campaign, folders)) for name, folders in groups()]
    if sum(len(folders) for _, folders, _ in plan) != 108:
        raise ValueError("Expected 54 primary, 42 controller and 12 interface cohorts")
    args.output.mkdir(parents=True)
    analyses = {
        "primary-scores": "score_revision_campaign.py",
        "controller-scores": "score_revision_diagnostics.py",
        "interface-scores": "score_revision_interfaces.py",
        "training-resources": "summarize_revision_training.py",
    }
    analysis_files = {"evaluation-provenance/environment-observed.json": environment_path}
    for name, script in analyses.items():
        target = args.output / f"{name}.json"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--root", str(args.campaign), "--output", str(target)],
            check=True,
        )
        if not json.loads(target.read_text())["complete"]:
            raise ValueError("Cannot publish an incomplete analysis as the complete evaluation")
        analysis_files[f"analysis/{target.name}"] = target
    for name in (
        "research/revision-v2-runtime.json",
        "research/policy-sources.json",
        "research/training-backbones.json",
        "docs/revision-v2-protocol.md",
        "docs/revision-v2-analysis.md",
        *[f"scripts/{script}" for script in analyses.values()],
        "scripts/revision_scoring_evidence.py",
        "scripts/revision_package_assets.py",
        "scripts/restore_revision_task.py",
        "scripts/package_revision_evaluation.py",
        "scripts/capture_revision_environment.py",
        "scripts/restore_revision_evaluation.py",
        "scripts/check_revision_evaluation_restore.py",
    ):
        analysis_files[f"evaluation-provenance/{name}"] = ROOT / name
    assets = [pack(args.output / "evaluation-analysis.tar.zst", analysis_files)]
    for name, _, files in plan:
        assets.append(split_archive(args.output, pack(args.output / f"evaluation-{name}.tar.zst", files)))
    index = dict(
        schema="wm-open-evaluation-package-v1",
        protocol=PROTOCOL,
        complete=True,
        primary_cohorts=54,
        controller_cohorts=42,
        interface_cohorts=12,
        runtime_manifest_sha256=runtime_hash,
        assets=assets,
        packager_sha256=sha256(__file__),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        scope=(
            "Primary and predeclared intervention reports, physical trajectories and complete analyses. "
            "Training data/models are separate task packages. Cross-workstation and post hoc checks are separate. "
            "Packaging does not establish public access or author submission approval."
        ),
    )
    index_path = args.output / "evaluation-index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(assets=len(assets), index_sha256=sha256(index_path))))


if __name__ == "__main__":
    main()
