"""Checksummed transport parts for large logical archives; no reassembly file needed."""

import hashlib
import shutil
from pathlib import Path

from wasman.learning.revision_protocol import sha256

MAX_PART_BYTES = 1024**3


def delivery_records(assets):
    records = []
    seen = set()
    for asset in assets:
        parts = asset.get("parts")
        if parts is not None and (not parts or sum(p["bytes"] for p in parts) != asset["bytes"]):
            raise ValueError("Invalid multipart byte total")
        for row in parts if parts is not None else [asset]:
            name = row["path"]
            if Path(name).name != name or name in {"", ".", ".."} or name in seen or row["bytes"] <= 0:
                raise ValueError("Unsafe, empty or duplicate delivery asset")
            seen.add(name)
            records.append(row)
    return records


def verified_asset_paths(package, asset):
    rows = delivery_records([asset])
    paths = []
    whole = hashlib.sha256()
    for row in rows:
        path = package / row["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError("Missing or unsafe delivery asset")
        if path.stat().st_size != row["bytes"]:
            raise ValueError("Delivery asset checksum/size mismatch")
        part_hash = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                whole.update(block)
                part_hash.update(block)
        if part_hash.hexdigest() != row["sha256"]:
            raise ValueError("Delivery asset checksum mismatch")
        paths.append(path)
    if whole.hexdigest() != asset["sha256"]:
        raise ValueError("Logical archive checksum mismatch or reordered parts")
    return paths


def split_archive(package, record, *, max_part_bytes=MAX_PART_BYTES):
    if max_part_bytes <= 0:
        raise ValueError("Positive part size required")
    source = package / record["path"]
    verified_asset_paths(package, record)
    if record["bytes"] <= max_part_bytes:
        return record
    if shutil.disk_usage(package).free < record["bytes"] + 40 * 2**30:
        raise RuntimeError("Insufficient multipart staging space including 40 GiB reserve")
    parts = []
    with source.open("rb") as stream:
        index = 0
        while stream.tell() < record["bytes"]:
            name = f"{record['path']}.part-{index:04d}"
            target = package / name
            if target.exists():
                raise ValueError("Preserve existing transport part")
            remaining = min(max_part_bytes, record["bytes"] - stream.tell())
            with target.open("xb") as output:
                while remaining:
                    block = stream.read(min(8 * 1024**2, remaining))
                    if not block:
                        raise ValueError("Archive changed while splitting")
                    output.write(block)
                    remaining -= len(block)
            parts.append(dict(path=name, bytes=target.stat().st_size, sha256=sha256(target)))
            index += 1
    result = dict(record, parts=parts)
    verified_asset_paths(package, result)
    # Only the duplicate compressed aggregate is removed. All original input
    # evidence remains, and ordered parts reproduce its exact compressed bytes.
    if sha256(source) != record["sha256"]:
        raise ValueError("Archive changed before aggregate cleanup")
    source.unlink()
    return result
