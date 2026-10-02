"""One full expert or DP episode from the frozen core; dry-run unless --execute.

This is an installation/replication diagnostic, not a new benchmark cohort.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from paper_release import ROOT, check_runtime


def plan(runtime, metadata, mode, output):
    pin = json.loads((ROOT / "research/paper-release.json").read_text())["runtimes"]["core"]["commit"]
    check_runtime(runtime, pin)
    if output.exists():
        raise ValueError("Use a fresh output directory; retain earlier failed runs")
    row = next(r for r in json.loads((ROOT / "research/paper-results-index.json").read_text())["records"]
               if r["id"] == "baseline-RotateValve-DP")
    if mode == "dp":
        if metadata is None:
            raise ValueError("DP requires --metadata pointing to extracted core-metadata")
        backend = metadata / row["backend"]
        if not backend.is_file() or hashlib.sha256(backend.read_bytes()).hexdigest() != row["backend_sha256"]:
            raise ValueError("Missing or modified original DP evaluator")
        checkpoint = runtime / row["checkpoint"]
        if not checkpoint.is_file():
            raise ValueError("Download valve_dp_v1 into the prepared runtime first")
        with checkpoint.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["checkpoint_sha256"]:
                raise ValueError("Checkpoint SHA-256 mismatch")
        options = ["--checkpoint", row["checkpoint"]]
    else:
        backend = runtime / "scripts/collect_valve_visual.py"
        options = []
    command = [str(runtime / ".venv/bin/python"), str(backend), *options,
               "--seeds", "2500", "--steps", "2240", "--output-dir", str(output)]
    return dict(runtime_commit=pin, mode=mode, seed=2500, steps=2240, command=command,
                backend_sha256=hashlib.sha256(backend.read_bytes()).hexdigest(),
                scope="Single full episode; no estimate of success rate or training reproducibility")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--mode", choices=["expert", "dp"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    output = args.output_dir.resolve()
    metadata = args.metadata.resolve() if args.metadata else None
    try:
        record = plan(runtime, metadata, args.mode, output)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2), flush=True)
    if not args.execute:
        return
    if not Path(record["command"][0]).is_file():
        parser.error("Install the runtime environment before --execute")
    receipt = output.with_name(output.name + "-launch.json")
    if receipt.exists():
        parser.error("Launch receipt exists; choose a fresh output name")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(runtime / "scripts"), str(runtime / ".deps/valve-policy-deps")])
    environment["WARP_CACHE_PATH"] = str(runtime / ".deps/second-machine-warp-cache")
    environment.setdefault("OMP_NUM_THREADS", "4")
    record["platform"] = platform.platform()
    record["publication_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
    record["gpu"] = gpu.stdout.strip()
    record["status"] = "started"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    code = subprocess.call(record["command"], cwd=runtime, env=environment)
    record.update(exit_code=code, status="process_finished" if code == 0 else "process_failed")
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print("Retain the launch receipt and all outputs; inspect measured success separately from process exit.")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
