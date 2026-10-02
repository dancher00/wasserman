"""Restore portability evidence and regenerate both analyses against separate core evidence."""

import argparse
import json
from pathlib import Path

from package_revision_portability import ANALYSES, ROOT, analyze
from restore_revision_task import extract_verified

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = args.package / "portability-index.json"
    if sha256(path) != args.index_sha256:
        raise ValueError("Changed portability index")
    index = json.loads(path.read_text())
    if (
        index.get("schema") != "wm-open-portability-package-v1"
        or index.get("protocol") != PROTOCOL
        or not index.get("complete")
        or [index.get("test_cohorts"), index.get("validation_cohorts")] != [54, 54]
        or index.get("runtime_manifest_sha256") != validate_runtime_snapshot()
    ):
        raise ValueError("Incompatible or incomplete portability package")
    for name, digest in index["source_hashes"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(ROOT / name) != digest:
            raise ValueError("Analysis source differs from packaged version")
    if args.output.exists():
        raise ValueError("Fresh restoration destination required")
    args.output.mkdir(parents=True)
    members = set()
    for record in index["assets"]:
        if members.intersection(record["members"]):
            raise ValueError("Duplicate archive members")
        members.update(record["members"])
        extract_verified(args.package, record, args.output)
    analyze(args.core, args.output / "external", args.output / "regenerated")
    for name in ANALYSES:
        if (args.output / "regenerated" / name).read_bytes() != (
            args.output / "portability-analysis" / name
        ).read_bytes():
            raise ValueError("Regenerated portability analysis differs")
    (args.output / "portability-verified.json").write_text(
        json.dumps(
            dict(
                index_sha256=args.index_sha256,
                all_members_verified=True,
                both_analyses_byte_identical=True,
                scope=(
                    "Restored second-workstation evidence; core models and primary traces provided separately. "
                    "No new training or public-access verification."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
