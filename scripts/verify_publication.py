"""Check the publication checkout without importing simulator dependencies."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    errors = []
    checked = 0
    optional = []
    manifest = ROOT / "research/frozen-hotstab.json"
    if not manifest.exists():
        manifest = ROOT / "research/frozen-core.json"
    frozen = json.loads(manifest.read_text())
    revision_path = ROOT / "research/revision-v2-runtime.json"
    overrides = {}
    if revision_path.exists() and manifest.name == "frozen-core.json":
        revision = json.loads(revision_path.read_text())
        if sha(manifest) != revision["historical_manifest_sha256"]:
            errors.append("Historical manifest was altered by the new revision")
        pinned = json.loads((ROOT / "research/paper-release.json").read_text())["runtimes"]["core"]["commit"]
        if revision["historical_runtime_commit"] != pinned:
            errors.append("Historical runtime pin was altered by the new revision")
        overrides = revision["historical_source_overrides"]
        for name, row in overrides.items():
            if frozen["source_sha256"].get(name) != row["historical_sha256"]:
                errors.append("Revision override has wrong historical parent: " + name)
        for name, expected in revision["source_sha256"].items():
            path = ROOT / name
            if not path.is_file() or sha(path) != expected:
                errors.append("New runtime snapshot mismatch: " + name)
    allow_missing = set(frozen.get("optional_weights", []) + frozen.get("external_licensed_assets", []))
    for name, expected in frozen["source_sha256"].items():
        if name in overrides:
            expected = overrides[name]["revision_sha256"]
        path = ROOT / name
        if not path.is_file() and name in allow_missing:
            optional.append(name)
            continue
        if not path.is_file() or sha(path) != expected:
            errors.append("Frozen source mismatch: " + name)
        checked += 1
    for name in ["LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"]:
        if not (ROOT / name).is_file():
            errors.append("Missing attribution: " + name)
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "ls-files", "-z"], text=True).split("\0")
    for name in filter(None, tracked):
        if "/meshes/alpha/" in name or name.endswith(".rgb") or name.startswith((".venv/", ".deps/", "logs/")):
            errors.append("Excluded payload tracked: " + name)
    artifacts = json.loads((ROOT / "research/paper-artifacts.json").read_text())["artifacts"]
    by_hash = {row["sha256"]: row for row in artifacts if row["kind"] == "checkpoint"}
    for row in json.loads((ROOT / "research/paper-results-index.json").read_text())["records"]:
        model = by_hash.get(row["checkpoint_sha256"])
        if model is None or model["path"] != row["checkpoint"]:
            errors.append("Unknown selected model: " + row["id"])
        pairs = list(zip(row["recorded_seeds_including_separate42"], row["recorded_success_per_seed"], strict=True))
        paired = [(seed, success) for seed, success in pairs if seed != 42]
        if len(paired) != row["episodes"] or sum(success for _, success in paired) != row["successes"]:
            errors.append("Recorded result count differs: " + row["id"])
    for row in artifacts:
        if sum(part["bytes"] for part in row["parts"]) != row["bytes"]:
            errors.append("Multipart size mismatch: " + row["id"])
        for part in row["parts"]:
            if not 0 < part["bytes"] < 2 * 1024**3:
                errors.append("Invalid release part size: " + part["name"])
    result = {
        "frozen_manifest": str(manifest.relative_to(ROOT)),
        "checked_sources": checked,
        "optional_local_assets_absent": optional,
        "errors": errors,
        "scope": "Source/packaging integrity, not simulator success-rate replication",
    }
    print(json.dumps(result, indent=2))
    return not errors


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
