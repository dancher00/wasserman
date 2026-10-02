"""Run one frozen bimanual condition; final states cannot precede acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--mode", choices=["free", "support", "two-hands"], required=True)
p.add_argument("--split", choices=["final", "development", "demo"], default="final")
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--check-only", action="store_true")
p.add_argument("--physics-only", action="store_true", help="Record the separate demo trace without live camera contention")
a = p.parse_args()
manifest = json.loads(a.manifest.read_text())
assert manifest["study"] == "bimanual-valve-v1"
assert set(manifest["gates"]) == {"free", "support", "two-hands"}
assert all(g["result"]["passed"] for g in manifest["gates"].values())
assert manifest["fixtures_spawned_at_reset_pose"]
errors = [name for name, expected in manifest["source_sha256"].items()
          if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
if errors:
    raise SystemExit("Frozen runtime differs: " + ", ".join(errors[:20]))
seeds = manifest["final_seeds"] if a.split == "final" else manifest["development_seeds"] if a.split == "development" else [manifest["demo_seed"]]
script = "record_bimanual_observer.py" if a.split == "demo" and not a.physics_only else "probe_bimanual_valve.py"
cmd = [sys.executable, str(ROOT / "scripts" / script), "--fixture", "large", "--mode", a.mode,
       "--seeds", *map(str, seeds), "--seconds", str(manifest["seconds"]), "--dt", str(manifest["dt"]),
       "--solver-type", str(manifest["solver_type"]), "--feedback-frame", manifest["feedback_frame"],
       "--spawn-fixture-at-reset-pose", "--output-dir", str(a.output_dir.resolve())]
if manifest["grasp_effort_Nm"] is not None:
    cmd += ["--grasp-effort", str(manifest["grasp_effort_Nm"])]
print(json.dumps({"split": a.split, "mode": a.mode, "seeds": seeds, "command": cmd}, indent=2), flush=True)
if not a.check_only:
    if a.output_dir.exists():
        raise SystemExit("Choose a new output directory to preserve evidence")
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))
