"""Remove only restored evaluation duplicates after complete release roundtrip checks."""

import argparse
import json
from pathlib import Path, PurePosixPath

from wasman.learning.revision_protocol import PROTOCOL, sha256
from revision_package_assets import delivery_records, verified_asset_paths

PREFIXES = {"primary-evaluation", "controller-evaluation", "interface-evaluation"}


def regular(root, name):
    part = PurePosixPath(name)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError("Unsafe duplicate path")
    path = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("Duplicate or original is missing or traverses a symlink")
    return path


def plan_duplicates(campaign, restored, assets):
    if campaign.resolve() == restored.resolve() or campaign.resolve() in restored.resolve().parents or restored.resolve() in campaign.resolve().parents:
        raise ValueError("Canonical and restored trees must be disjoint")
    plan = []
    seen = set()
    for asset in assets:
        for name, digest in asset["members"].items():
            parts = PurePosixPath(name).parts
            if not parts or PurePosixPath(name).is_absolute() or ".." in parts:
                raise ValueError("Unsafe manifest path")
            if parts[0] not in PREFIXES:
                continue
            if name in seen:
                raise ValueError("Duplicate manifest member")
            seen.add(name)
            original, duplicate = regular(campaign, name), regular(restored, name)
            if sha256(original) != digest or sha256(duplicate) != digest:
                raise ValueError("Canonical or restored bytes differ from verified package")
            plan.append(dict(path=name, sha256=digest, bytes=duplicate.stat().st_size))
    if not plan:
        raise ValueError("No evaluation duplicates found")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("campaign", "package", "reproduction", "completion", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Otherwise record a dry-run plan")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing cleanup receipt")
    completion = json.loads(args.completion.read_text())
    index_path = args.package / "evaluation-index.json"
    index = json.loads(index_path.read_text())
    verification_path = args.reproduction / "analysis-check/verified.json"
    verification = json.loads(verification_path.read_text())
    if (sha256(index_path) != completion["index_sha256"]
        or sha256(verification_path) != completion["verification_sha256"]
        or verification["index_sha256"] != completion["index_sha256"]
        or verification.get("all_four_analyses_byte_identical") is not True
        or index.get("schema") != "wm-open-evaluation-package-v1"
        or index.get("protocol") != PROTOCOL
        or index.get("complete") is not True
        or [index.get(k) for k in ("primary_cohorts", "controller_cohorts", "interface_cohorts")] != [54, 42, 12]):
        raise ValueError("Complete evaluation roundtrip verification is required")
    # Retain and rehash all canonical delivery archives before removing any copies.
    for asset in index["assets"]:
        try:
            verified_asset_paths(args.package, asset)
        except (ValueError, OSError) as error:
            raise ValueError("Canonical delivery archive changed") from error
    downloads = []
    download_root = args.reproduction / "download"
    if download_root.exists():
        for row in delivery_records(index["assets"]):
            duplicate = regular(download_root, row["path"])
            if sha256(duplicate) != row["sha256"] or duplicate.stat().st_size != row["bytes"]:
                raise ValueError("Downloaded delivery copy changed")
            downloads.append({k: row[k] for k in ("path", "sha256", "bytes")})
    restored = args.reproduction / "restored"
    plan = plan_duplicates(args.campaign, restored, index["assets"])
    receipt = dict(status="verified-plan", apply=args.apply, files=plan, downloaded_archive_copies=downloads,
                   duplicate_bytes=sum(row["bytes"] for row in plan + downloads),
                   index_sha256=sha256(index_path), verification_sha256=sha256(verification_path),
                   scope="Only restored evaluation and downloaded archive copies removed. Canonical traces, archives, models, analyses and reproduction receipts retained.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    if args.apply:
        for row in plan:
            duplicate = regular(restored, row["path"])
            if sha256(regular(args.campaign, row["path"])) != row["sha256"] or sha256(duplicate) != row["sha256"]:
                raise ValueError("File changed after cleanup preflight")
            duplicate.unlink()
        for row in downloads:
            duplicate = regular(download_root, row["path"])
            if sha256(regular(args.package, row["path"])) != row["sha256"] or sha256(duplicate) != row["sha256"]:
                raise ValueError("Delivery archive changed after cleanup preflight")
            duplicate.unlink()
        receipt["status"] = "completed"
        args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"files", "downloaded_archive_copies"}}, indent=2))


if __name__ == "__main__":
    main()
