"""Losslessly archive RGB one episode at a time; optional prune only after byte verification."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_command(path, codec="zstd"):
    if codec == "zstd":
        return ["zstd", "-q", "-d", "-c", str(path)]
    if codec != "libx264rgb":
        raise ValueError(codec)
    import imageio_ffmpeg

    return [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-threads",
        "2",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def decoded_digest(path, codec="zstd"):
    process = subprocess.Popen(decode_command(path, codec), stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: process.stdout.read(8 * 1024**2), b""):
        digest.update(chunk)
        size += len(chunk)
    process.stdout.close()
    if process.wait() != 0:
        raise ValueError(f"Decompression failed: {path}")
    return digest.hexdigest(), size


def archive_one(source, destination, *, prune=False, codec="zstd", image_shape=None):
    """Existing compressed files must match before the source can be unlinked."""
    expected, size = digest_file(source), source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".partial")
        if codec == "zstd":
            with temporary.open("wb") as output:
                subprocess.run(["zstd", "-q", "-T4", "-3", "-c", str(source)], stdout=output, check=True)
        elif codec == "libx264rgb":
            import imageio_ffmpeg

            if image_shape is None or len(image_shape) != 3 or image_shape[2] != 3:
                raise ValueError("Lossless RGB video requires height,width,3")
            height, width, _ = image_shape
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    "30",
                    "-i",
                    str(source),
                    "-an",
                    "-c:v",
                    "libx264rgb",
                    "-crf",
                    "0",
                    "-preset",
                    "fast",
                    "-threads",
                    "4",
                    "-f",
                    "matroska",
                    str(temporary),
                ],
                check=True,
            )
        else:
            raise ValueError(codec)
        if decoded_digest(temporary, codec) != (expected, size):
            raise ValueError("Lossless roundtrip mismatch; source retained")
        temporary.replace(destination)
    elif decoded_digest(destination, codec) != (expected, size):
        raise ValueError("Existing archive mismatch; source retained")
    entry = dict(
        codec=codec,
        image_shape=image_shape,
        raw_sha256=expected,
        raw_bytes=size,
        archive_sha256=digest_file(destination),
        archive_bytes=destination.stat().st_size,
        decoded_bytes_verified=True,
    )
    if prune:
        # Recheck raw bytes in case a producer was incorrectly left running.
        if source.stat().st_size != size or digest_file(source) != expected:
            raise ValueError("Source changed during compression; source retained")
        source.unlink()
    entry["raw_pruned"] = prune
    return entry


def store_manifest(manifest, state):
    temporary = manifest.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(state, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(manifest)
    descriptor = os.open(manifest.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_before_prune(manifest, state, source, entry, *, prune):
    # A stopped process must never leave a removed RGB file absent from its
    # recovery index. Commit the verified archive and sidecars first.
    destination = manifest.parent / entry["archive_path"]
    for path in (destination, *(destination.parent / name for name in entry.get("sidecar_sha256", {}))):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    store_manifest(manifest, state)
    if prune:
        if source.stat().st_size != entry["raw_bytes"] or digest_file(source) != entry["raw_sha256"]:
            raise ValueError("Source changed after manifest publication; source retained")
        source.unlink()
        entry["raw_pruned"] = True
        store_manifest(manifest, state)


def reconcile_pruned(manifest, state, data):
    changed = False
    for name, entry in state["files"].items():
        if not entry.get("raw_pruned") and not (data / name).exists():
            destination = manifest.parent / entry["archive_path"]
            if digest_file(destination) != entry["archive_sha256"] or decoded_digest(destination, entry["codec"]) != (
                entry["raw_sha256"],
                entry["raw_bytes"],
            ):
                raise ValueError("Cannot recover absent RGB from a changed archive")
            entry.update(raw_pruned=True, source_absent_verified_on_resume=True)
            changed = True
    if changed:
        store_manifest(manifest, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--codec", choices=["zstd", "libx264rgb"], default="zstd")
    p.add_argument("--prune-rgb", action="store_true", help="Remove raw RGB only after verified lossless decoding")
    a = p.parse_args()
    manifest = a.archive / "rgb-manifest.json"
    if manifest.exists():
        state = json.loads(manifest.read_text())
        if state["data_root"] != str(a.data.resolve()):
            raise ValueError("Archive belongs to another dataset")
    else:
        state = dict(schema="wm-lossless-rgb-v1", data_root=str(a.data.resolve()), files={})
    reconcile_pruned(manifest, state, a.data)
    for path in sorted(a.data.rglob("wrist.rgb")):
        relative = path.relative_to(a.data)
        destination = a.archive / relative.with_suffix(".rgb.zst" if a.codec == "zstd" else ".rgb.mkv")
        meta = json.loads((path.parent / "metadata.json").read_text())
        entry = archive_one(path, destination, prune=False, codec=a.codec, image_shape=meta["image_shape"])
        entry["archive_path"] = str(destination.relative_to(a.archive))
        state["files"][str(relative)] = entry
        # Keep trajectory and metadata available even when the working RGB is removed.
        for name in ("metadata.json", "trajectory.npz"):
            original = path.parent / name
            if original.exists():
                (destination.parent / name).write_bytes(original.read_bytes())
                entry.setdefault("sidecar_sha256", {})[name] = digest_file(original)
        publish_before_prune(manifest, state, path, entry, prune=a.prune_rgb)
        print(json.dumps(dict(path=str(relative), **entry)), flush=True)
    audit = a.data / "revision_audit.json"
    if audit.exists():
        (a.archive / "revision_audit.json").write_bytes(audit.read_bytes())


if __name__ == "__main__":
    main()
