"""Install the exact ACT/DP source snapshots and isolated Python overlay used by the benchmark."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify", action="store_true", help="Only check the installed source bytes and package versions"
    )
    parser.add_argument(
        "--archive-dir", type=Path, help="Optional directory of pinned AM-Bench/LeRobot source tarballs"
    )
    args = parser.parse_args()
    sources = json.loads((ROOT / "research/policy-sources.json").read_text())["sources"]
    for source in sources:
        destination = ROOT / ".deps" / source["directory"]
        missing = []
        for name, expected in source["files"].items():
            path = destination / name
            if path.exists():
                if sha(path) != expected:
                    raise ValueError(f"Changed dependency; refusing to overwrite: {path}")
            else:
                missing.append(name)
        if missing and args.verify:
            raise FileNotFoundError(f"{source['name']}: {len(missing)} missing source files; run this installer first")
        if missing:
            with tempfile.TemporaryDirectory(prefix="wm-policy-") as temporary:
                archive = (
                    args.archive_dir / f"{source['name']}.tar.gz"
                    if args.archive_dir
                    else Path(temporary) / "source.tar.gz"
                )
                if not args.archive_dir:
                    url = f"https://codeload.github.com/{source['repository']}/tar.gz/{source['commit']}"
                    with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                if sha(archive) != source["archive_sha256"]:
                    raise ValueError(f"Archive checksum mismatch: {source['name']}")
                with tarfile.open(archive) as bundle:
                    members = {m.name.split("/", 1)[1]: m for m in bundle if "/" in m.name and m.isfile()}
                    # Validate every requested member before writing; never extract arbitrary archive paths.
                    contents = {name: bundle.extractfile(members[name]).read() for name in missing}
                    for name, data in contents.items():
                        if hashlib.sha256(data).hexdigest() != source["files"][name]:
                            raise ValueError(f"Source checksum mismatch: {name}")
                    for name, data in contents.items():
                        path = destination / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
        print(f"{source['name']}: {len(source['files'])} source hashes verified", flush=True)
    overlay = ROOT / ".deps/valve-policy-deps"
    requirements = ROOT / "research/policy-requirements.txt"
    if not args.verify:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(overlay),
                "--no-deps",
                "--requirements",
                str(requirements),
            ],
            check=True,
        )
    # Use distribution metadata at the isolated target, not a same-named global package.
    from importlib.metadata import distributions

    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in distributions(path=[str(overlay)])}
    for line in requirements.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        if installed.get(name.lower().replace("_", "-")) != version:
            raise ValueError(f"Overlay package mismatch: {line}")
    print("Policy overlay versions verified. Training backbones and selected checkpoints are separate data products.")


if __name__ == "__main__":
    main()
