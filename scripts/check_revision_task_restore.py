"""Reaudit a downloaded, relocated task package and optionally prune verified duplicates."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from restore_revision_rgb import safe_relative

from wasman.learning.revision_protocol import sha256


def save_receipt(path, receipt):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prune_duplicates(candidates, receipt_path, receipt):
    # Check all paths before deleting any, and persist the complete deletion record first.
    for path, expected in candidates:
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Unsafe or changed staged duplicate: {path}")
        receipt["pruned_files"].append(dict(path=str(path), bytes=path.stat().st_size, sha256=expected))
    receipt["cleanup_planned_bytes"] = sum(row["bytes"] for row in receipt["pruned_files"])
    save_receipt(receipt_path, receipt)
    for path, expected in candidates:
        if sha256(path) != expected:
            raise ValueError("Staged duplicate changed after planning")
        path.unlink()
    receipt["cleanup_complete"] = True
    save_receipt(receipt_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--restored", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--prune-verified-duplicates", action="store_true")
    args = parser.parse_args()
    package, restored, canonical = (p.resolve() for p in (args.package, args.restored, args.canonical))
    for left, right in ((package, restored), (package, canonical), (restored, canonical)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError("Package, restored and canonical roots must be disjoint")
    index = json.loads((package / "index.json").read_text())
    task = index["task"]
    download = json.loads((package / "download-receipt.json").read_text())
    restoration = json.loads((restored / f"{task}-package-restored.json").read_text())
    index_hash = sha256(package / "index.json")
    if download["index_sha256"] != index_hash or restoration["package_index_sha256"] != index_hash:
        raise ValueError("Download/restore receipts do not bind this package")
    audits = []
    for name in (task, f"{task}-ee"):
        data = restored / "data" / name
        if not data.exists():
            continue
        audit = data / "revision_audit.json"
        expected = sha256(canonical / "data" / name / "revision_audit.json")
        if sha256(audit) != expected:
            raise ValueError("Packaged audit differs from canonical audit")
        supplied = data / "revision_audit.supplied.json"
        if supplied.exists():
            raise ValueError("Preserve previous reaudit attempt")
        audit.rename(supplied)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("audit_revision_dataset.py")),
                "--task",
                task,
                "--data",
                str(data),
            ],
            check=True,
        )
        if sha256(audit) != expected:
            raise ValueError("Regenerated audit is not byte-identical")
        result = json.loads(audit.read_text())
        audits.append(
            dict(
                dataset=name,
                audit_sha256=expected,
                selected=len(result["episodes"]),
                attempts=len(result["attempts"]),
                regenerated_byte_identical=True,
            )
        )
    models = []
    for path in sorted((restored / "models").glob("*/final.pt")):
        original = canonical / path.relative_to(restored)
        expected = json.loads((original.parent / "completed.json").read_text())["final_sha256"]
        if sha256(path) != expected or sha256(original) != expected:
            raise ValueError("Restored model differs from completed canonical training")
        models.append(dict(label=path.parent.name, final_sha256=expected))
    if len(models) != (12 if task in ("PushSlider", "PullLever") else 9):
        raise ValueError("Incomplete model set")
    receipt = dict(
        task=task,
        index_sha256=index_hash,
        audits=audits,
        models=models,
        scope=(
            "Authenticated private-release download and relocated full dataset/model restoration; "
            "not anonymous accessibility or a new training replicate"
        ),
        cleanup_complete=False,
        pruned_files=[],
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    save_receipt(args.receipt, receipt)
    if not args.prune_verified_duplicates:
        return
    manifest = json.loads((canonical / "archives" / task / "rgb-manifest.json").read_text())
    candidates = []
    for relative, row in manifest["files"].items():
        raw = safe_relative(restored / "data" / task, relative)
        archive_relative = safe_relative(Path("archives") / task, row["archive_path"])
        source = canonical / archive_relative
        if sha256(source) != row["archive_sha256"] or sha256(raw) != row["raw_sha256"]:
            raise ValueError("Canonical lossless archive or restored RGB changed")
        candidates.append((raw, row["raw_sha256"]))
        copied_archive = restored / archive_relative
        if sha256(copied_archive) != row["archive_sha256"]:
            raise ValueError("Copied archive changed")
        candidates.append((copied_archive, row["archive_sha256"]))
    for row in models:
        candidates.append((restored / "models" / row["label"] / "final.pt", row["final_sha256"]))
    for row in index["assets"]:
        candidates.append((safe_relative(package, row["path"]), row["sha256"]))
    prune_duplicates(candidates, args.receipt, receipt)
    print(json.dumps(dict(task=task, verified=True, freed_bytes=receipt["cleanup_planned_bytes"])), flush=True)


if __name__ == "__main__":
    main()
