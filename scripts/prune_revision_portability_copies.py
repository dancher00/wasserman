"""Remove verified portability restoration/download duplicates; retain originals."""

import argparse
import json
from pathlib import Path, PurePosixPath

from prune_revision_evaluation_copies import regular
from revision_package_assets import delivery_records, verified_asset_paths

from wasman.learning.revision_protocol import PROTOCOL, sha256


def plan_duplicates(external, restored, assets):
    a, b = external.resolve(), restored.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("Original and restored trees must be disjoint")
    rows = []
    seen = set()
    for asset in assets:
        for name, digest in asset["members"].items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Unsafe member path")
            if path.parts[0] != "external":
                continue
            relative = str(PurePosixPath(*path.parts[1:]))
            if name in seen:
                raise ValueError("Duplicate member")
            seen.add(name)
            original, copy = regular(external, relative), regular(restored, name)
            if sha256(original) != digest or sha256(copy) != digest:
                raise ValueError("Original or duplicate differs")
            rows.append(dict(path=name, original=relative, sha256=digest, bytes=copy.stat().st_size))
    if not rows:
        raise ValueError("No duplicate external evidence")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["external", "restored", "package", "download", "completion", "output"]:
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing cleanup receipt")
    if (
        args.package.resolve() == args.download.resolve()
        or args.package.resolve() in args.download.resolve().parents
        or args.download.resolve() in args.package.resolve().parents
    ):
        raise ValueError("Canonical and downloaded archives must be disjoint")
    completion = json.loads(args.completion.read_text())
    index_path = args.package / "portability-index.json"
    index = json.loads(index_path.read_text())
    verification_path = args.restored / "portability-verified.json"
    verification = json.loads(verification_path.read_text())
    if (
        sha256(index_path) != completion["index_sha256"]
        or sha256(verification_path) != completion["verification_sha256"]
        or verification["index_sha256"] != completion["index_sha256"]
        or verification.get("both_analyses_byte_identical") is not True
        or verification.get("all_members_verified") is not True
        or index.get("schema") != "wm-open-portability-package-v1"
        or index.get("protocol") != PROTOCOL
        or index.get("complete") is not True
        or [index.get("test_cohorts"), index.get("validation_cohorts")] != [54, 54]
    ):
        raise ValueError("Require complete verified portability roundtrip")
    for asset in index["assets"]:
        verified_asset_paths(args.package, asset)
    files = plan_duplicates(args.external, args.restored, index["assets"])
    downloads = delivery_records(index["assets"])
    for row in downloads:
        copy = regular(args.download, row["path"])
        if sha256(copy) != row["sha256"] or copy.stat().st_size != row["bytes"]:
            raise ValueError("Downloaded copy differs")
    receipt = dict(
        status="verified-plan",
        apply=args.apply,
        files=files,
        downloaded_archive_copies=downloads,
        duplicate_bytes=sum(x["bytes"] for x in files + downloads),
        index_sha256=sha256(index_path),
        scope=(
            "Only indexed restored external evidence and downloaded archive copies; canonical external evidence, "
            "canonical archives, both regenerated analyses and verification receipts retained."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.apply:
        for row in files:
            original = regular(args.external, row["original"])
            copy = regular(args.restored, row["path"])
            if sha256(original) != row["sha256"] or sha256(copy) != row["sha256"]:
                raise ValueError("Evidence changed after preflight")
            copy.unlink()
        for row in downloads:
            original = regular(args.package, row["path"])
            copy = regular(args.download, row["path"])
            if sha256(original) != row["sha256"] or sha256(copy) != row["sha256"]:
                raise ValueError("Archive changed after preflight")
            copy.unlink()
        receipt["status"] = "completed"
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ["files", "downloaded_archive_copies"]}))


if __name__ == "__main__":
    main()
