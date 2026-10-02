"""Stage source or portability packages in the private draft and verify retrieved bytes."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from upload_revision_task import api, api_pages
from revision_package_assets import delivery_records, verified_asset_paths

from wasman.learning.revision_protocol import PROTOCOL, sha256

SCHEMAS = {
    "retraining-index.json": "wm-open-retraining-package-v1",
    "source-index.json": "wm-open-source-package-v1",
    "portability-index.json": "wm-open-portability-package-v1",
}


def package_records(package, index_name):
    index_path = package / index_name
    index = json.loads(index_path.read_text())
    if index.get("schema") != SCHEMAS[index_name] or index.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible supplemental package")
    if index_name == "portability-index.json" and (
        not index.get("complete") or [index.get("test_cohorts"), index.get("validation_cohorts")] != [54, 54]
    ):
        raise ValueError("Incomplete portability package")
    if index_name == "retraining-index.json" and (
        not index.get("complete") or index.get("cases") != ["clean", "external"]
    ):
        raise ValueError("Incomplete retraining package")
    try:
        transport = delivery_records(index["assets"])
    except ValueError as error:
        raise ValueError("Duplicate or unsafe supplemental asset name/size") from error
    records = [*transport, dict(path=index_name, bytes=index_path.stat().st_size, sha256=sha256(index_path))]
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate supplemental asset name")
    for row in records:
        if row["bytes"] >= 2 * 2**30:
            raise ValueError("Split delivery assets below the 2 GiB upload bound")
        path = package / row["path"]
        if Path(row["path"]).name != row["path"] or path.is_symlink() or not path.is_file():
            raise ValueError("Supplemental asset must be a regular direct child")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("Changed supplemental package")
    for asset in index["assets"]:
        verified_asset_paths(package, asset)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--index", choices=SCHEMAS, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--download", type=Path, required=True)
    args = parser.parse_args()
    records = package_records(args.package, args.index)
    if not api(f"repos/{args.repository}")["private"]:
        raise ValueError("This staging command requires a private repository")
    matches = [r for r in api_pages(f"repos/{args.repository}/releases?per_page=100") if r["tag_name"] == args.release]
    if len(matches) != 1 or not matches[0]["draft"]:
        raise ValueError("Expected one existing private draft release")
    release = matches[0]
    args.download.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.download).free < sum(r["bytes"] for r in records) + 40 * 2**30:
        raise RuntimeError("Insufficient space including40GiB reserve")
    verified = []
    for row in records:
        name = row["path"]
        assets = api_pages(f"repos/{args.repository}/releases/{release['id']}/assets?per_page=100")
        found = [a for a in assets if a["name"] == name]
        if not found:
            subprocess.run(
                ["gh", "release", "upload", args.release, str(args.package / name), "--repo", args.repository],
                check=True,
            )
            found = [
                a
                for a in api_pages(f"repos/{args.repository}/releases/{release['id']}/assets?per_page=100")
                if a["name"] == name
            ]
        if len(found) != 1 or found[0]["size"] != row["bytes"]:
            raise ValueError("Release asset identity/size differs; do not overwrite")
        target = args.download / name
        if target.is_symlink() or (
            target.exists() and (target.stat().st_size != row["bytes"] or sha256(target) != row["sha256"])
        ):
            raise ValueError("Preserve changed existing download")
        if not target.exists():
            temporary = target.with_name(target.name + ".partial")
            with temporary.open("xb") as stream:
                subprocess.run(
                    [
                        "gh",
                        "api",
                        "-H",
                        "Accept: application/octet-stream",
                        f"repos/{args.repository}/releases/assets/{found[0]['id']}",
                    ],
                    stdout=stream,
                    check=True,
                )
            if temporary.stat().st_size != row["bytes"] or sha256(temporary) != row["sha256"]:
                raise ValueError("Downloaded supplemental asset differs")
            temporary.replace(target)
        verified.append(dict(path=name, asset_id=found[0]["id"], sha256=row["sha256"], bytes=row["bytes"]))
        (args.download / "download-progress.json").write_text(json.dumps(verified, indent=2) + "\n")
        print(json.dumps(dict(download_verified=name)), flush=True)
    (args.download / "private-stage-verified.json").write_text(
        json.dumps(
            dict(
                repository=args.repository,
                release_id=release["id"],
                index_sha256=sha256(args.package / args.index),
                assets=verified,
                public_access_verified=False,
                scope=(
                    "Authenticated private-draft upload and verified local downloads; "
                    "archive restoration is a separate check."
                ),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
