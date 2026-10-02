"""Restore full-retraining evidence and regenerate both comparisons exactly."""

import argparse
import json
from pathlib import Path

from package_revision_retraining import ROOT, run_audit
from restore_revision_task import extract_verified

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "core", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args()
    path = args.package / "retraining-index.json"
    if sha256(path) != args.index_sha256:
        raise ValueError("Retraining index differs from supplied digest")
    index = json.loads(path.read_text())
    if (
        index.get("schema") != "wm-open-retraining-package-v1"
        or index.get("protocol") != PROTOCOL
        or not index.get("complete")
        or index.get("cases") != ["clean", "external"]
        or index.get("runtime_manifest_sha256") != validate_runtime_snapshot()
    ):
        raise ValueError("Incompatible retraining package")
    for name, digest in index["source_hashes"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(ROOT / name) != digest:
            raise ValueError("Packaged analysis source differs")
    if args.output.exists():
        raise ValueError("Fresh restore destination required")
    args.output.mkdir(parents=True)
    seen = set()
    for record in index["assets"]:
        if seen.intersection(record["members"]):
            raise ValueError("Duplicate archive members")
        seen.update(record["members"])
        extract_verified(args.package, record, args.output)
    for name in index["cases"]:
        folder = args.output / name
        target = args.output / f"{name}-regenerated.json"
        run_audit(args.core, folder / "model", folder / "dataset-audit.json", folder / "rollout", target)
        if target.read_bytes() != (args.output / "retraining-analysis" / f"{name}-retraining.json").read_bytes():
            raise ValueError("Regenerated retraining comparison differs")
    (args.output / "retraining-verified.json").write_text(
        json.dumps(
            dict(
                index_sha256=args.index_sha256,
                all_members_verified=True,
                both_analyses_byte_identical=True,
                scope=(
                    "Restored recorded retraining evidence; primary model/audit provided separately. "
                    "No new training or public-access claim."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
