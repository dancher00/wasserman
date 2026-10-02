"""Restore verified raw RGB and episode records from the public lossless shards."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from archive_revision_rgb import decode_command, digest_file


def safe_relative(root, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Archive manifest path must stay within its root")
    return root / relative


def restore_one(archive, output, name, record, *, reserve_bytes=40 * 2**30):
    codec = record.get("codec", "zstd")
    compressed = safe_relative(archive, record.get("archive_path", str(Path(name).with_suffix(".rgb.zst"))))
    target = safe_relative(output, name)
    if compressed.stat().st_size != record["archive_bytes"] or digest_file(compressed) != record["archive_sha256"]:
        raise ValueError("Compressed shard failed checksum")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != record["raw_bytes"] or digest_file(target) != record["raw_sha256"]:
            raise ValueError("Existing destination differs; refusing to overwrite")
    else:
        if shutil.disk_usage(target.parent).free < record["raw_bytes"] + reserve_bytes:
            raise RuntimeError("Insufficient space including the declared reserve")
        temporary = target.with_suffix(".rgb.partial")
        with temporary.open("wb") as destination:
            subprocess.run(decode_command(compressed, codec), stdout=destination, check=True)
        if temporary.stat().st_size != record["raw_bytes"] or digest_file(temporary) != record["raw_sha256"]:
            raise ValueError("Decoded bytes failed checksum")
        temporary.replace(target)
    for sidecar, expected in record.get("sidecar_sha256", {}).items():
        source = safe_relative(compressed.parent, sidecar)
        destination = safe_relative(target.parent, sidecar)
        if digest_file(source) != expected:
            raise ValueError("Episode record failed checksum")
        if destination.exists() and digest_file(destination) != expected:
            raise ValueError("Existing episode record differs")
        if not destination.exists():
            shutil.copyfile(source, destination)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode", action="append", help="Restore only these seed_NNN directories; repeatable")
    a = p.parse_args()
    manifest_path = a.archive / "rgb-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    restored = []
    for name, record in manifest["files"].items():
        if a.episode and Path(name).parent.name not in a.episode:
            continue
        restore_one(a.archive, a.output, name, record)
        restored.append(name)
        print(name, flush=True)
    if not restored:
        raise ValueError("No matching episodes")
    audit = a.archive / "revision_audit.json"
    if audit.exists() and not a.episode:
        target = a.output / audit.name
        if target.exists() and digest_file(target) != digest_file(audit):
            raise ValueError("Existing audit differs")
        if not target.exists():
            shutil.copyfile(audit, target)
    receipt = dict(
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        files=restored,
        decoded_bytes_verified=True,
    )
    (a.output / "restore-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
