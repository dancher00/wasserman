"""Restore the pinned reproduction archive without replacing existing evidence.

The checkout includes its verified scientific runtime/evidence package.
No release download or GitHub credentials are needed. Original file paths and
bytes are retained; manuscript drafts, peer reviews and handoffs are excluded.
"""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

from fetch_paper_artifacts import safe_path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore(archive, destination, manifest):
    if archive.stat().st_size != manifest["bytes"] or sha(archive) != manifest["sha256"]:
        raise ValueError("Reproduction archive differs from its pinned digest")
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or set(names) != set(manifest["files"]):
            raise ValueError("Reproduction archive membership differs from its manifest")
        # Check every incoming member and existing destination before writing.
        for member in members:
            target = safe_path(destination, member.name)
            expected = manifest["files"][member.name]
            if not member.isfile() or member.size != expected["bytes"]:
                raise ValueError("Unexpected archive member: " + member.name)
            with stream.extractfile(member) as incoming:
                if hashlib.file_digest(incoming, "sha256").hexdigest() != expected["sha256"]:
                    raise ValueError("Archive member digest differs: " + member.name)
            if target.exists() and (not target.is_file() or sha(target) != expected["sha256"]):
                raise ValueError("Existing evidence differs; preserved: " + member.name)
        for member in members:
            target = safe_path(destination, member.name)
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".reproduction-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as output, stream.extractfile(member) as incoming:
                    shutil.copyfileobj(incoming, output)
                os.link(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
    print(f"Verified and restored {len(manifest['files'])} original reproduction files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive")
    parser.add_argument("--destination", type=Path, default=ROOT)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/reproduction-package.json").read_text())
    archive = args.archive or ROOT / manifest["bundled_path"]
    restore(archive, args.destination.resolve(), manifest)


if __name__ == "__main__":
    main()
