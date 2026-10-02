"""Package complete second-workstation test and supplemental validation evidence."""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

from package_revision_evaluation import ALLOWED
from package_revision_task import pack
from revision_package_assets import split_archive

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = {
    "cross-machine.json": "compare_revision_reproduction.py",
    "validation.json": "score_revision_validation.py",
}


def cohort_files(external, label):
    folder = external / label
    receipt = json.loads((folder / "local-retrieval.json").read_text())
    manifest = json.loads((folder / "retrieval-manifest.json").read_text())
    if not receipt.get("members_verified") or receipt["files"] != len(manifest):
        raise ValueError("Incomplete retrieved cohort")
    if json.loads((folder / "job.json").read_text()).get("returncode") != 0:
        raise ValueError("Unsuccessful external job")
    required = {"job.json", "gpu.jsonl", "job.log"}
    required |= (
        {"rollout/summary.json", "rollout/rollout.pt"}
        if "rollout/summary.json" in manifest
        else {"rollout/report.json", "rollout/trace.npz", "rollout/contract.json"}
    )
    if not required <= manifest.keys():
        raise ValueError("Missing physical evidence in retrieval manifest")
    files = {}
    for name, digest in manifest.items():
        relative = Path(name)
        allowed = name in {"job.json", "gpu.jsonl", "job.log"} or (
            relative.parent == Path("rollout")
            and (relative.name in ALLOWED or (relative.name.startswith("wrist_") and relative.suffix == ".png"))
        )
        if not allowed or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected retrieval member")
        path = folder / relative
        if any(p.is_symlink() for p in [folder, path, path.parent]) or not path.is_file() or sha256(path) != digest:
            raise ValueError("Changed or nonregular retrieval member")
        files[f"external/{label}/{name}"] = path
    for name in ("local-retrieval.json", "retrieval-manifest.json"):
        path = folder / name
        if path.is_symlink():
            raise ValueError("Symlink receipt")
        files[f"external/{label}/{name}"] = path
    return files


def analyze(core, external, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, script in ANALYSES.items():
        target = output / name
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--core",
                str(core),
                "--external",
                str(external),
                "--output",
                str(target),
            ],
            check=True,
        )
        if not json.loads(target.read_text())["complete"]:
            raise ValueError("Incomplete portability analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required; preserve partial packages")
    runtime = validate_runtime_snapshot()
    plans = []
    for task in TASKS:
        files = {}
        for model, seed, suffix in itertools.product(("ACT", "DP", "BC"), (17, 43, 101), ("", "-validation")):
            files.update(cohort_files(args.external, f"{task}-{model}-{seed}{suffix}"))
        plans.append((task, files))
    provenance = {}
    for name, folder in (("primary", args.core), ("secondary", args.external)):
        path = folder / "environment-observed.json"
        env = json.loads(path.read_text())
        if (
            env.get("schema") != "wm-environment-observation-v1"
            or env.get("protocol") != PROTOCOL
            or env.get("runtime_manifest_sha256") != runtime
        ):
            raise ValueError("Incompatible environment observation")
        provenance[f"portability-provenance/{name}-environment.json"] = path
    args.output.mkdir(parents=True)
    analyze(args.core, args.external, args.output / "analysis")
    provenance.update({f"portability-analysis/{name}": args.output / "analysis" / name for name in ANALYSES})
    sources = [
        "research/revision-v2-runtime.json",
        "docs/revision-v2-protocol.md",
        "docs/revision-v2-analysis.md",
        "scripts/revision_scoring_evidence.py",
        "scripts/revision_package_assets.py",
        "scripts/restore_revision_task.py",
        "scripts/score_revision_campaign.py",
        "scripts/package_revision_portability.py",
        "scripts/restore_revision_portability.py",
        *[f"scripts/{script}" for script in ANALYSES.values()],
    ]
    for name in sources:
        provenance[f"portability-provenance/{name}"] = ROOT / name
    assets = [pack(args.output / "portability-analysis.tar.zst", provenance)]
    for task, files in plans:
        assets.append(split_archive(args.output, pack(args.output / f"portability-{task}.tar.zst", files)))
    index = dict(
        schema="wm-open-portability-package-v1",
        protocol=PROTOCOL,
        complete=True,
        test_cohorts=54,
        validation_cohorts=54,
        runtime_manifest_sha256=runtime,
        assets=assets,
        source_hashes={name: sha256(ROOT / name) for name in sources},
        scope=(
            "Same primary checkpoints repeated on a second workstation; supplemental validation declared "
            "after initial test exposure. No extra independent training seeds. Primary evidence and models "
            "supplied separately. Excludes separate clean-training and post hoc image-intervention evidence."
        ),
    )
    path = args.output / "portability-index.json"
    path.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(assets=len(assets), index_sha256=sha256(path))))


if __name__ == "__main__":
    main()
