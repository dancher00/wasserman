"""Explicitly snapshot the new runtime without rewriting historical source hashes."""

import argparse
import datetime
import json
import subprocess
from pathlib import Path

from wasman.learning.revision_protocol import PROTOCOL, sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "research/revision-v2-runtime.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", action="store_true", help="Seal after engineering gates; subsequent updates are refused")
    a = p.parse_args()
    if MANIFEST.exists() and json.loads(MANIFEST.read_text())["status"] == "frozen":
        raise ValueError("Frozen runtime is immutable; define a new version for changes")
    historical_path = ROOT / "research/frozen-core.json"
    historical = json.loads(historical_path.read_text())
    changes = {}
    for name, digest in historical["source_sha256"].items():
        path = ROOT / name
        if path.is_file() and sha256(path) != digest:
            changes[name] = dict(historical_sha256=digest, revision_sha256=sha256(path))
    sources = set((ROOT / "src/wasman").rglob("*.py"))
    for pattern in ("collect_*visual.py", "evaluate_*visual.py", "rollout_*visual.py", "train_*act.py", "train_*dp.py"):
        sources.update((ROOT / "scripts").glob(pattern))
    sources.update(
        ROOT / "scripts" / name
        for name in (
            "benchmark.py",
            "train_revision_policy.py",
            "button_dp.py",
            "button_visual_support.py",
            "audit_revision_dataset.py",
            "prepare_revision_interface.py",
        )
    )
    sources.update(ROOT / name for name in changes)
    sources.add(ROOT / "docs/revision-v2-protocol.md")
    sources.add(ROOT / "docs/studies/hydrodynamics-anchor-v2.md")
    record = dict(
        protocol=PROTOCOL,
        status="frozen" if a.freeze else "development",
        snapshot_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        historical_runtime_commit=json.loads((ROOT / "research/paper-release.json").read_text())["runtimes"]["core"][
            "commit"
        ],
        historical_manifest_sha256=sha256(historical_path),
        scope="Separate open-asset learning runtime; historical checkpoints and result hashes are unchanged",
        historical_source_overrides=changes,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sorted(sources)},
    )
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(status=record["status"], sources=len(sources), explicit_overrides=len(changes))))


if __name__ == "__main__":
    main()
