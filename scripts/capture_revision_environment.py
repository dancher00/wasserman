"""Record a bounded software/GPU observation without hostnames, paths or credentials."""

import argparse
import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from wasman.learning.revision_protocol import PROTOCOL, sha256, validate_runtime_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Experimental role, without a personal machine name")
    args = parser.parse_args()
    runtime_hash = validate_runtime_snapshot()
    root = Path(__file__).resolve().parents[1]
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "isaacsim", "isaaclab", "lerobot", "diffusers", "timm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    # Rollout/training backends put this pinned overlay ahead of base site-packages.
    for distribution in importlib.metadata.distributions(path=[str(root / ".deps/valve-policy-deps")]):
        name = distribution.metadata["Name"].lower().replace("_", "-")
        if name in packages:
            packages[name] = distribution.version
    query = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"]
    gpu = subprocess.check_output(query, text=True, timeout=15).strip().splitlines()
    keys = ("model", "driver", "memory_mib", "compute_capability")
    devices = [dict(zip(keys, [value.strip() for value in line.split(",")], strict=True)) for line in gpu]
    os_release = platform.freedesktop_os_release()
    record = dict(
        schema="wm-environment-observation-v1",
        protocol=PROTOCOL,
        label=args.label,
        observed_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        runtime_manifest_sha256=runtime_hash,
        python=sys.version.split()[0],
        operating_system={key: os_release.get(key) for key in ("ID", "VERSION_ID")},
        kernel=platform.release(),
        architecture=platform.machine(),
        packages=packages,
        package_version_scope=(
            "Policy overlay takes precedence; null means no distribution metadata (source pins remain authoritative)"
        ),
        policy_sources_sha256=sha256(root / "research/policy-sources.json"),
        training_backbones_sha256=sha256(root / "research/training-backbones.json"),
        torch_cuda_build=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        gpu=devices,
        capture_source_sha256=sha256(__file__),
        scope=(
            "Environment observed during the running campaign; not a retroactive per-job attestation. "
            "Training/inference settings remain in model configurations and pinned runtime sources. "
            "No GPU serial/UUID, hostname, username, environment variables or credentials are collected."
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(label=args.label, gpu=devices, runtime_manifest_sha256=runtime_hash)))


if __name__ == "__main__":
    main()
