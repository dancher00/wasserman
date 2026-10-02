"""Verify and restore a complete task package without workstation-specific paths."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from restore_revision_rgb import safe_relative
from revision_package_assets import verified_asset_paths

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256


def extract_verified(package, record, output):
    sources = verified_asset_paths(package, record)
    # Concatenate checked byte parts into the decompressor without a second
    # large on-disk archive. Single-file packages follow the same path.
    producer = subprocess.Popen(["cat", "--", *map(str, sources)], stdout=subprocess.PIPE)
    process = subprocess.Popen(["zstd", "-q", "-d", "-c"], stdin=producer.stdout, stdout=subprocess.PIPE)
    producer.stdout.close()
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name
                if not member.isfile() or name not in record["members"] or name in seen:
                    raise ValueError("Unexpected, duplicate or non-regular archive member")
                target = safe_relative(output, name)
                # Refuse symlink traversal, including an existing final path.
                if any(path.is_symlink() for path in (target, *target.parents)):
                    raise ValueError("Package destination traverses a symlink")
                target.parent.mkdir(parents=True, exist_ok=True)
                expected = record["members"][name]
                if target.exists():
                    if sha256(target) != expected:
                        raise ValueError(f"Existing file differs: {target}")
                    if hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest() != expected:
                        raise ValueError("Member checksum mismatch")
                else:
                    if shutil.disk_usage(target.parent).free < member.size + 40 * 2**30:
                        raise RuntimeError("Insufficient space including 40 GiB reserve")
                    temporary = target.with_name(target.name + ".unpack-partial")
                    if temporary.exists() or temporary.is_symlink():
                        raise ValueError("Preserve/investigate an existing partial extraction")
                    with temporary.open("xb") as stream:
                        shutil.copyfileobj(archive.extractfile(member), stream, length=8 * 1024**2)
                    if sha256(temporary) != expected:
                        raise ValueError("Member checksum mismatch")
                    temporary.replace(target)
                seen.add(name)
        while process.stdout.read(1024**2):
            pass
        if process.wait() or producer.wait() or seen != set(record["members"]):
            raise ValueError("Incomplete package extraction")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()
        if producer.poll() is None:
            producer.kill()
        producer.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index_path = args.package / "index.json"
    index = json.loads(index_path.read_text())
    task = index["task"]
    if index["schema"] != "wm-open-task-package-v1" or index["protocol"] != PROTOCOL or task not in TASKS:
        raise ValueError("Unsupported package")
    args.output.mkdir(parents=True, exist_ok=True)
    for record in index["assets"]:
        extract_verified(args.package, record, args.output)
        print(record["path"], flush=True)
    data = args.output / "data" / task
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("restore_revision_rgb.py")),
            "--archive",
            str(args.output / "archives" / task),
            "--output",
            str(data),
        ],
        check=True,
    )
    if sha256(data / "revision_audit.json") != index["dataset_audit_sha256"]:
        raise ValueError("Restored audit mismatch")
    ee = args.output / "data" / f"{task}-ee"
    if ee.exists():
        audit = json.loads((ee / "revision_audit.json").read_text())
        for row in audit["episodes"]:
            folder = safe_relative(ee, row["path"])
            metadata = json.loads((folder / "metadata.json").read_text())
            source = safe_relative(data, metadata["source_episode"]) / "wrist.rgb"
            if sha256(source) != row["files"]["wrist.rgb"]:
                raise ValueError("EE view RGB differs from audited native episode")
            target = folder / "wrist.rgb"
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve() != source.resolve():
                    raise ValueError("Existing EE view points elsewhere")
            else:
                target.symlink_to(os.path.relpath(source, folder))
    receipt = dict(package_index_sha256=sha256(index_path), task=task, all_members_verified=True)
    (args.output / f"{task}-package-restored.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
