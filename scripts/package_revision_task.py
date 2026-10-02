"""Package a completed open task as independently checksummed, restorable shards."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("ACT", "DP", "BC")
SEEDS = (17, 43, 101)


def pack(target, files):
    """No opaque model loading; caller verifies all input hashes before packaging."""
    if target.exists():
        raise ValueError("Use a fresh package destination")
    if shutil.disk_usage(target.parent).free < sum(path.stat().st_size for path in files.values()) + 40 * 2**30:
        raise RuntimeError("Insufficient space including 40 GiB reserve")
    expected = {name: sha256(path) for name, path in files.items()}
    temporary = target.with_suffix(target.suffix + ".partial")
    process = subprocess.Popen(["zstd", "-q", "-T2", "-3", "-o", str(temporary)], stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
            for name, path in sorted(files.items()):
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as source:
                    archive.addfile(info, source)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Package compression failed")
        verify_members(temporary, expected)
        temporary.replace(target)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return dict(path=target.name, bytes=target.stat().st_size, sha256=sha256(target), members=expected)


def verify_members(path, expected):
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE)
    found = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in found:
                    raise ValueError("Unexpected archive member")
                found[member.name] = hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest()
        # Consume tar padding before waiting for the decompressor.
        while process.stdout.read(1024**2):
            pass
        if process.wait() or found != expected:
            raise ValueError("Package member roundtrip mismatch")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Fresh output required; preserve partial packages for investigation")
    data = a.campaign / "data" / a.task
    archive = a.campaign / "archives" / a.task
    complete = json.loads((data / "completed.json").read_text())
    audit_path = data / "revision_audit.json"
    audit = json.loads(audit_path.read_text())
    if (
        complete["protocol"] != PROTOCOL
        or complete["task"] != a.task
        or complete["dataset_audit_sha256"] != sha256(audit_path)
        or not audit["passed"]
        or audit["pilot"]
        or len(audit["episodes"]) != 80
    ):
        raise ValueError("A complete audited primary dataset is required")
    manifest_path = archive / "rgb-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attempt_hashes = {}
    for path in data.glob("train_batch*/seed_*/metadata.json"):
        seed = json.loads(path.read_text())["seed"]
        if seed in attempt_hashes:
            raise ValueError("Duplicate collection attempt")
        attempt_hashes[seed] = sha256(path)
    if attempt_hashes != {row["seed"]: row["metadata_sha256"] for row in audit["attempts"]}:
        raise ValueError("Collection attempts changed after audit")
    for row in audit["episodes"]:
        entry = manifest["files"][f"{row['path']}/wrist.rgb"]
        if entry["raw_sha256"] != row["files"]["wrist.rgb"]:
            raise ValueError("Archive RGB differs from selected audited episode")
        for name in ("metadata.json", "trajectory.npz"):
            if entry["sidecar_sha256"][name] != row["files"][name]:
                raise ValueError("Archive records differ from selected audited episode")
    # Relocatable manifest removes the original workstation path, retains all file hashes.
    portable = {**manifest, "data_root": a.task}
    a.output.mkdir(parents=True)
    portable_path = a.output / "rgb-manifest.json"
    portable_path.write_text(json.dumps(portable, indent=2) + "\n")
    assets = []
    metadata = {
        f"archives/{a.task}/rgb-manifest.json": portable_path,
        f"archives/{a.task}/revision_audit.json": audit_path,
        f"data/{a.task}/revision_audit.json": audit_path,
        f"data/{a.task}/completed.json": data / "completed.json",
        "LICENSE": ROOT / "LICENSE",
        "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
    }
    # Include unsuccessful/unused attempts too: they establish first-success selection.
    for path in sorted(data.glob("train_batch*/seed_*/metadata.json")):
        metadata[f"data/{a.task}/{path.relative_to(data)}"] = path
    assets.append(
        pack(
            a.output / f"{a.task}-metadata.tar.zst",
            metadata,
        )
    )
    for name, record in sorted(manifest["files"].items()):
        compressed = archive / record["archive_path"]
        if sha256(compressed) != record["archive_sha256"]:
            raise ValueError("RGB shard changed")
        files = {record["archive_path"]: compressed}
        for sidecar, expected in record["sidecar_sha256"].items():
            path = compressed.parent / sidecar
            if sha256(path) != expected:
                raise ValueError("Episode record changed")
            files[str(path.relative_to(archive))] = path
        seed = Path(name).parent.name
        assets.append(
            pack(
                a.output / f"{a.task}-{seed}.tar.zst", {f"archives/{a.task}/{key}": path for key, path in files.items()}
            )
        )
    configurations = [(model, seed, "") for model in MODELS for seed in SEEDS]
    if a.task in ("PushSlider", "PullLever"):
        configurations += [("DP", seed, "-ee") for seed in SEEDS]
        ee = a.campaign / "data" / f"{a.task}-ee"
        evidence = {}
        for name in ("view.json", "revision_audit.json", "ee_replay_gate.json"):
            evidence[f"data/{a.task}-ee/{name}"] = ee / name
        for pattern in ("train_batch*/seed_*/metadata.json", "train_batch*/seed_*/trajectory.npz"):
            for path in sorted(ee.glob(pattern)):
                evidence[f"data/{a.task}-ee/{path.relative_to(ee)}"] = path
        for interface in ("actuator", "ee"):
            folder = a.campaign / "interface-gates" / f"{a.task}-{interface}"
            for name in ("report.json", "trace.npz", "contract.json", "physical_asset.json", "source_manifest.json"):
                evidence[f"interface-gates/{a.task}-{interface}/{name}"] = folder / name
        assets.append(pack(a.output / f"{a.task}-interface-evidence.tar.zst", evidence))
    for model, seed, suffix in configurations:
        label = f"{a.task}-{model}-{seed}{suffix}"
        folder = a.campaign / "models" / label
        completed = json.loads((folder / "completed.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        if (
            completed["pilot"]
            or config["pilot"]
            or config["task"] != a.task
            or config["seed"] != seed
            or config["model"] != model
            or config["protocol"] != PROTOCOL
            or config["runtime_manifest_sha256"] != sha256(ROOT / "research/revision-v2-runtime.json")
            or config["sample_budget"] != 320000
            or completed["samples"] != 320000
            or completed["final_sha256"] != sha256(folder / "final.pt")
        ):
            raise ValueError("Final model identity/hash mismatch")
        names = ["final.pt", "config.json", "completed.json", "metrics.jsonl"]
        files = {f"models/{label}/{name}": folder / name for name in names}
        for path in sorted((a.campaign / "jobs").glob(f"{label}.*")):
            files[f"jobs/{path.name}"] = path
        assets.append(pack(a.output / f"{label}.tar.zst", files))
    index = dict(
        schema="wm-open-task-package-v1",
        protocol=PROTOCOL,
        task=a.task,
        asset_profile="open-procedural-v1",
        primary_training_runs=9,
        interface_training_runs=len(configurations) - 9,
        selected_demonstrations=80,
        attempted_demonstrations=audit["attempted_episodes"],
        dataset_audit_sha256=sha256(audit_path),
        runtime_manifest_sha256=sha256(ROOT / "research/revision-v2-runtime.json"),
        assets=assets,
        packager_sha256=sha256(__file__),
        evaluation_status="Packaged training evidence; scored reports are separate",
        restoration=(
            "Run scripts/restore_revision_task.py --package PACKAGE_DIRECTORY --output NEW_CAMPAIGN_DIRECTORY. "
            "All shards extract relative to the campaign directory. RGB is decoded with byte verification; "
            "EE views reuse native RGB through relative symlinks after metadata and audit verification."
        ),
    )
    (a.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(task=a.task, assets=len(assets), bytes=sum(row["bytes"] for row in assets))))


if __name__ == "__main__":
    main()
