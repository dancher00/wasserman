"""Download a source/evaluation supplement with verified ordered multipart archives."""

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from fetch_revision_task import fetch, incoming
from revision_package_assets import delivery_records, verified_asset_paths
from upload_revision_task import api_pages

from wasman.learning.revision_protocol import PROTOCOL, sha256

SCHEMAS = {
    "source-index.json": "wm-open-source-package-v1",
    "evaluation-index.json": "wm-open-evaluation-package-v1",
    "portability-index.json": "wm-open-portability-package-v1",
    "retraining-index.json": "wm-open-retraining-package-v1",
}
RECEIPT = "evidence-download-verified.json"


def download_package(destination, index_name, digest, opener):
    """opener(name) returns a context-managed binary stream; no archive extraction."""
    destination = Path(destination)
    if any(p.is_symlink() for p in (destination, *destination.parents)):
        raise ValueError("Destination traverses a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / index_name
    if index_name not in SCHEMAS or index_path.is_symlink():
        raise ValueError("Unsupported index or symlink")
    fetch(index_path, digest, None, lambda: opener(index_name))
    index = json.loads(index_path.read_text())
    if index.get("schema") != SCHEMAS[index_name] or index.get("protocol") != PROTOCOL:
        raise ValueError("Wrong evidence package schema/protocol")
    if index_name != "source-index.json" and not index.get("complete"):
        raise ValueError("Incomplete evidence package")
    if not index.get("assets"):
        raise ValueError("Empty evidence package")
    for asset in index["assets"]:
        if Path(asset["path"]).name != asset["path"] or asset["path"] in {"", ".", ".."}:
            raise ValueError("Unsafe logical archive path")
    records = delivery_records(index["assets"])
    # Validate the complete namespace before downloading any archive.
    for row in records:
        if row["path"] in {*SCHEMAS, RECEIPT} or (destination / row["path"]).is_symlink():
            raise ValueError("Reserved or symlink archive path")
    receipt_path = destination / RECEIPT
    if receipt_path.exists() or receipt_path.is_symlink():
        raise ValueError("Preserve existing verification receipt; use a fresh destination")
    for row in records:
        fetch(destination / row["path"], row["sha256"], row["bytes"], lambda row=row: opener(row["path"]))
    for asset in index["assets"]:
        verified_asset_paths(destination, asset)
    receipt = dict(
        index=index_name,
        index_sha256=sha256(index_path),
        logical_archives=len(index["assets"]),
        delivery_files=len(records),
        all_downloaded_bytes_verified=True,
        scope="Transport and ordered archive hashes verified; restoration and scientific rescoring are separate checks",
    )
    with receipt_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-url", help="Directory URL containing the index and all listed delivery files")
    source.add_argument("--repository", help="owner/name; uses gh authentication")
    p.add_argument("--release")
    p.add_argument("--index", choices=SCHEMAS, required=True)
    p.add_argument("--index-sha256", required=True)
    p.add_argument("--destination", type=Path, required=True)
    a = p.parse_args()
    if a.repository and not a.release:
        p.error("--repository requires --release")
    if a.base_url:

        def opener(name):
            return incoming(a.base_url.rstrip("/") + "/" + quote(name, safe=""))
    else:
        releases = [r for r in api_pages(f"repos/{a.repository}/releases?per_page=100") if r["tag_name"] == a.release]
        if len(releases) != 1:
            raise ValueError("Release not found or ambiguous")
        rows = api_pages(f"repos/{a.repository}/releases/{releases[0]['id']}/assets?per_page=100")
        assets = {r["name"]: r["id"] for r in rows}
        if len(assets) != len(rows):
            raise ValueError("Ambiguous release asset names")

        def opener(name):
            return incoming(repository=a.repository, asset=assets[name])

    print(json.dumps(download_package(a.destination, a.index, a.index_sha256, opener)))


if __name__ == "__main__":
    main()
