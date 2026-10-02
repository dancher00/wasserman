"""Recompute every complete analysis from restored, digest-bound evaluation evidence."""

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot

ROOT = Path(__file__).resolve().parents[1]
ANALYSES = {
    "primary-scores": "score_revision_campaign.py",
    "controller-scores": "score_revision_diagnostics.py",
    "interface-scores": "score_revision_interfaces.py",
    "training-resources": "summarize_revision_training.py",
}


def verify_restored_members(restored, assets):
    restored = restored.resolve()
    found = {}
    for asset in assets:
        for name, expected in asset["members"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Unsafe restored member path")
            target = restored.joinpath(*path.parts)
            if any(p.is_symlink() for p in (target, *target.parents) if p != restored and restored in p.parents):
                raise ValueError("Restored evaluation evidence cannot traverse symlinks")
            if name in found or not target.is_file() or sha256(target) != expected:
                raise ValueError(f"Missing, duplicate or changed restored member: {name}")
            found[name] = expected
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.package / "evaluation-index.json"
    if sha256(index_path) != args.index_sha256:
        raise ValueError("Evaluation index differs from the independently supplied digest")
    index = json.loads(index_path.read_text())
    if (
        index.get("schema") != "wm-open-evaluation-package-v1"
        or index.get("protocol") != PROTOCOL
        or not index.get("complete")
        or [index.get(k) for k in ("primary_cohorts", "controller_cohorts", "interface_cohorts")] != [54, 42, 12]
        or index.get("runtime_manifest_sha256") != validate_runtime_snapshot()
        or len(index["assets"]) != 11
    ):
        raise ValueError("Require the complete, compatible 108-cohort evaluation package")
    output = args.output.resolve()
    inputs = [args.package.resolve(), args.restored.resolve()]
    if args.output.exists() or any(output == p or output in p.parents or p in output.parents for p in inputs):
        raise ValueError("Use a fresh output directory disjoint from package and restored evidence")
    members = verify_restored_members(args.restored, index["assets"])
    # A different implementation could change tables without changing the saved traces.
    for name, expected in members.items():
        relative = name.removeprefix("evaluation-provenance/")
        if (
            name != relative
            and relative.startswith(("scripts/", "research/", "docs/"))
            and sha256(ROOT / relative) != expected
        ):
            raise ValueError(f"Analysis checkout differs from packaged provenance: {relative}")
    for name, script in ANALYSES.items():
        if f"analysis/{name}.json" not in members or f"evaluation-provenance/scripts/{script}" not in members:
            raise ValueError(f"Missing complete analysis or its source: {name}")
    output.mkdir(parents=True)
    checks = {}
    for name, script in ANALYSES.items():
        target = output / f"{name}.json"
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--root", str(args.restored), "--output", str(target)],
            check=True,
        )
        if not json.loads(target.read_text())["complete"] or sha256(target) != members[f"analysis/{name}.json"]:
            raise ValueError(f"Regenerated analysis differs from the complete released table: {name}")
        checks[name] = sha256(target)
    receipt = dict(
        index_sha256=args.index_sha256,
        restored_members_verified=len(members),
        all_four_analyses_byte_identical=True,
        analysis_sha256=checks,
        checker_sha256=sha256(__file__),
        scope=(
            "Physical outcomes and complete analyses reconstructed from restored evaluation evidence. "
            "The 60 models and training records are supplied separately and checked by the scorers; "
            "this check alone does not establish their download provenance, public access, "
            "new simulator execution or independent training reproduction."
        ),
    )
    (output / "verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
