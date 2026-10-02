"""Upload a completed task package to a private draft release and verify downloaded bytes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from wasman.learning.revision_protocol import sha256


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint]))


def api_pages(endpoint):
    pages = json.loads(subprocess.check_output(["gh", "api", "--paginate", "--slurp", endpoint]))
    return [row for page in pages for row in page]


def verify_download(repository, asset_id, expected, size):
    process = subprocess.Popen(
        ["gh", "api", "-H", "Accept: application/octet-stream", f"repos/{repository}/releases/assets/{asset_id}"],
        stdout=subprocess.PIPE,
    )
    digest, count = hashlib.sha256(), 0
    try:
        for block in iter(lambda: process.stdout.read(8 * 1024**2), b""):
            digest.update(block)
            count += len(block)
        if process.wait() or digest.hexdigest() != expected or count != size:
            raise ValueError("Downloaded release asset differs from local package")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--prune-verified-packages", action="store_true")
    args = parser.parse_args()
    if not api(f"repos/{args.repository}")["private"]:
        raise ValueError("This staging command requires the existing private repository")
    # GitHub's by-tag endpoint omits drafts; authenticated release listing includes them.
    matches = [
        row for row in api_pages(f"repos/{args.repository}/releases?per_page=100") if row["tag_name"] == args.release
    ]
    if len(matches) != 1:
        raise ValueError("Expected one existing release with the requested tag")
    release = matches[0]
    if not release["draft"]:
        raise ValueError("Use an explicit draft release while evaluations remain incomplete")
    index_path = args.package / "index.json"
    index = json.loads(index_path.read_text())
    if index["schema"] != "wm-open-task-package-v1":
        raise ValueError("Unsupported package schema")
    # Unique name allows six task indices in one release without overwriting.
    published_index = args.package / f"{index['task']}-index.json"
    published_index.write_bytes(index_path.read_bytes())
    records = [
        *index["assets"],
        dict(path=published_index.name, sha256=sha256(published_index), bytes=published_index.stat().st_size),
    ]
    receipt_path = args.package / "upload-receipt.json"
    receipt = dict(
        repository=args.repository,
        release_id=release["id"],
        release_url=release["html_url"],
        visibility="private draft",
        package_index_sha256=sha256(index_path),
        assets={},
    )
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        for key in ("repository", "release_id", "package_index_sha256"):
            if previous[key] != receipt[key]:
                raise ValueError("Previous upload belongs to different evidence/release")
        receipt = previous
    for record in records:
        path = args.package / record["path"]
        if path.parent.resolve() != args.package.resolve():
            raise ValueError("Package asset must be directly within its directory")
        previous = receipt["assets"].get(record["path"])
        if not path.exists() and not (previous and previous.get("download_verified")):
            raise ValueError("Missing asset without a verified upload receipt")
        if path.exists() and (sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]):
            raise ValueError("Local package changed after indexing")
        assets = api(f"repos/{args.repository}/releases/{release['id']}/assets?per_page=100")
        # Releases may exceed 100 shards; follow explicit page numbers.
        page = 2
        while len(assets) % 100 == 0 and assets:
            more = api(f"repos/{args.repository}/releases/{release['id']}/assets?per_page=100&page={page}")
            assets.extend(more)
            page += 1
            if len(more) < 100:
                break
        found = [asset for asset in assets if asset["name"] == record["path"]]
        if not found:
            subprocess.run(["gh", "release", "upload", args.release, str(path), "--repo", args.repository], check=True)
            # The newly uploaded asset can be on a later page.
            assets = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        "--paginate",
                        "--slurp",
                        f"repos/{args.repository}/releases/{release['id']}/assets?per_page=100",
                    ]
                )
            )
            found = [asset for page_assets in assets for asset in page_assets if asset["name"] == record["path"]]
        if len(found) != 1 or found[0]["size"] != record["bytes"]:
            raise ValueError("Release asset identity/size mismatch")
        asset = found[0]
        verify_download(args.repository, asset["id"], record["sha256"], record["bytes"])
        receipt["assets"][record["path"]] = dict(
            asset_id=asset["id"],
            sha256=record["sha256"],
            bytes=record["bytes"],
            download_verified=True,
            local_package_pruned=False,
        )
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        if args.prune_verified_packages and path.suffixes[-2:] == [".tar", ".zst"]:
            if path.exists():
                path.unlink()
            receipt["assets"][record["path"]]["local_package_pruned"] = True
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(dict(task=index["task"], asset=record["path"], verified=True)), flush=True)
    receipt["complete"] = True
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
