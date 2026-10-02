"""Download one indexed task package from a public URL or an authenticated GitHub release."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from upload_revision_task import api_pages

from wasman.learning.revision_protocol import TASKS, sha256


@contextmanager
def incoming(url=None, *, repository=None, asset=None):
    if url is not None:
        with urllib.request.urlopen(url, timeout=120) as response:
            yield response
        return
    process = subprocess.Popen(
        [
            "gh",
            "api",
            "-H",
            "Accept: application/octet-stream",
            f"repos/{repository}/releases/assets/{asset}",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        yield process.stdout
        if process.wait():
            raise RuntimeError("GitHub download failed")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def fetch(target, expected, size, opener):
    if target.exists():
        if sha256(target) != expected or (size is not None and target.stat().st_size != size):
            raise ValueError("Existing download differs; refusing to overwrite")
        return
    if shutil.disk_usage(target.parent).free < (size or 16 * 1024**2) + 40 * 2**30:
        raise RuntimeError("Insufficient space including 40 GiB reserve")
    temporary = target.with_name(target.name + ".download-partial")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("Existing partial download requires inspection")
    digest, count = hashlib.sha256(), 0
    try:
        with temporary.open("xb") as output, opener() as source:
            for block in iter(lambda: source.read(8 * 1024**2), b""):
                count += len(block)
                if count > (size if size is not None else 16 * 1024**2):
                    raise ValueError("Download exceeds indexed size")
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != expected or (size is not None and count != size):
            raise ValueError("Downloaded bytes fail the supplied checksum/size")
        temporary.replace(target)
    finally:
        # This temporary file was created exclusively by this invocation.
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="Public directory URL containing TASK-index.json and its shards")
    source.add_argument("--repository", help="owner/name; uses gh authentication, including private drafts")
    parser.add_argument("--release", help="Required with --repository")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--index-sha256", required=True, help="Expected index hash from the release record")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.repository and not args.release:
        parser.error("--repository requires --release")
    args.destination.mkdir(parents=True, exist_ok=True)
    assets = {}
    if args.repository:
        releases = api_pages(f"repos/{args.repository}/releases?per_page=100")
        found = [release for release in releases if release["tag_name"] == args.release]
        if len(found) != 1:
            raise ValueError("Release not found or ambiguous")
        assets = {
            row["name"]: row
            for row in api_pages(f"repos/{args.repository}/releases/{found[0]['id']}/assets?per_page=100")
        }

    def opener(name):
        if args.base_url:
            return lambda: incoming(args.base_url.rstrip("/") + "/" + name)
        return lambda: incoming(repository=args.repository, asset=assets[name]["id"])

    index_name = f"{args.task}-index.json"
    index_path = args.destination / "index.json"
    fetch(index_path, args.index_sha256, assets[index_name]["size"] if assets else None, opener(index_name))
    index = json.loads(index_path.read_text())
    if index["schema"] != "wm-open-task-package-v1" or index["task"] != args.task:
        raise ValueError("Wrong task package")
    for row in index["assets"]:
        if Path(row["path"]).name != row["path"]:
            raise ValueError("Shard path must be a plain filename")
        fetch(args.destination / row["path"], row["sha256"], row["bytes"], opener(row["path"]))
        print(row["path"], flush=True)
    receipt = dict(
        task=args.task,
        index_sha256=args.index_sha256,
        assets=len(index["assets"]),
        source=args.base_url or f"{args.repository}/{args.release}",
        all_downloaded_bytes_verified=True,
    )
    (args.destination / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
