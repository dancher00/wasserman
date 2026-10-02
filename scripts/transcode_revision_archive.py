"""Replace raw-Zstandard RGB shards with byte-verified lossless RGB video, without restoring raw files."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg
from archive_revision_rgb import decoded_digest, digest_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prune-superseded", action="store_true")
    args = parser.parse_args()
    manifest_path = args.archive / "rgb-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ledger_path = args.archive / "transcode-receipts.jsonl"
    for name, previous in list(manifest["files"].items()):
        if previous.get("codec", "zstd") != "zstd":
            continue
        relative = Path(previous.get("archive_path", str(Path(name).with_suffix(".rgb.zst"))))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Archive member must remain within the archive")
        source = args.archive / relative
        if digest_file(source) != previous["archive_sha256"]:
            raise ValueError("Source shard changed")
        target = source.with_suffix(".mkv")
        shape = previous.get("image_shape")
        if shape is None:
            shape = json.loads((source.parent / "metadata.json").read_text())["image_shape"]
        height, width, channels = shape
        if channels != 3:
            raise ValueError("Require RGB24")
        if shutil.disk_usage(args.archive).free < previous["raw_bytes"] + 40 * 2**30:
            raise RuntimeError("Insufficient reserve for verified transcoding")
        if not target.exists():
            temporary = target.with_suffix(".mkv.partial")
            if temporary.exists():
                raise ValueError("Preserve/investigate existing partial conversion")
            decoder = subprocess.Popen(["zstd", "-q", "-d", "-c", str(source)], stdout=subprocess.PIPE)
            try:
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-v",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        "30",
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264rgb",
                        "-crf",
                        "0",
                        "-preset",
                        "fast",
                        "-threads",
                        "2",
                        "-f",
                        "matroska",
                        str(temporary),
                    ],
                    stdin=decoder.stdout,
                    check=True,
                )
                decoder.stdout.close()
                if decoder.wait():
                    raise ValueError("Source decompression failed")
            finally:
                if decoder.poll() is None:
                    decoder.kill()
                    decoder.wait()
            if decoded_digest(temporary, "libx264rgb") != (previous["raw_sha256"], previous["raw_bytes"]):
                raise ValueError("Transcoding changed raw RGB bytes; source retained")
            temporary.replace(target)
        elif decoded_digest(target, "libx264rgb") != (previous["raw_sha256"], previous["raw_bytes"]):
            raise ValueError("Existing video differs; source retained")
        current = {
            **previous,
            "codec": "libx264rgb",
            "image_shape": shape,
            "archive_path": str(target.relative_to(args.archive)),
            "archive_sha256": digest_file(target),
            "archive_bytes": target.stat().st_size,
            "decoded_bytes_verified": True,
        }
        receipt = dict(
            path=name,
            previous=previous,
            current=current,
            previous_manifest_sha256=digest_file(manifest_path),
            prune_requested=args.prune_superseded,
        )
        with ledger_path.open("a") as ledger:
            ledger.write(json.dumps(receipt) + "\n")
        manifest["files"][name] = current
        temporary_manifest = manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary_manifest.replace(manifest_path)
        if args.prune_superseded:
            if digest_file(source) != previous["archive_sha256"]:
                raise ValueError("Source changed during conversion; retaining it")
            source.unlink()
        print(
            json.dumps(
                dict(
                    path=name,
                    saved_bytes=previous["archive_bytes"] - current["archive_bytes"],
                    raw_bytes_verified=True,
                    superseded_pruned=args.prune_superseded,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
