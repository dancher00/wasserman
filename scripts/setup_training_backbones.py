"""Fetch or verify the exact publicly available training initializations."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_backbones(*, download=False):
    import torch

    manifest = ROOT / "research/training-backbones.json"
    records = []
    for row in json.loads(manifest.read_text())["backbones"]:
        location = row["destination"]
        path = (
            Path(torch.hub.get_dir()) / location.removeprefix("torch-hub:")
            if location.startswith("torch-hub:")
            else ROOT / location
        )
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing training backbone: {path}; run scripts/setup_training_backbones.py")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(row["url"], timeout=60) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            if temporary.stat().st_size != row["bytes"] or digest(temporary) != row["sha256"]:
                raise ValueError("Downloaded initialization failed checksum")
            temporary.replace(path)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Existing initialization differs; refusing to replace: {path}")
        records.append(dict(name=row["name"], sha256=row["sha256"], bytes=row["bytes"]))
    return dict(manifest_sha256=digest(manifest), files=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    print(json.dumps(verify_backbones(download=not a.verify), indent=2))


if __name__ == "__main__":
    main()
