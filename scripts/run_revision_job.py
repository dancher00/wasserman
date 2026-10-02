"""Run one revision job with bounded disk use, provenance, timeout and GPU telemetry."""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timeout", type=int, default=14400)
    p.add_argument("--resume", action="store_true", help="Fresh receipt for a explicitly resumed training command")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or a.record.exists() or a.output_dir.exists() != a.resume:
        p.error("Command and fresh receipt/output paths required")
    if a.resume and "--resume" not in command:
        p.error("Resumed job must explicitly request backend recovery")
    if shutil.disk_usage(ROOT).free < 40 * 2**30:
        raise RuntimeError("Less than 40 GiB free: stage verified redundant files before launching")
    sources = sorted(list((ROOT / "src/wasman").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")))
    hashes = {str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    environment = dict(os.environ)
    environment.update(WASMAN_ASSET_PROFILE="open-procedural-v1", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    environment["PYTHONPATH"] = str(ROOT / ".deps/valve-policy-deps")
    a.record.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(
        command=command,
        cwd=str(ROOT),
        output_dir=str(a.output_dir),
        asset_profile=environment["WASMAN_ASSET_PROFILE"],
        source_sha256=hashes,
        started_unix=time.time(),
        timeout_s=a.timeout,
        free_bytes_before=shutil.disk_usage(ROOT).free,
    )
    if any(Path(value).name == "train_revision_policy.py" for value in command):
        from setup_training_backbones import verify_backbones

        receipt["training_backbones"] = verify_backbones()
    a.record.write_text(json.dumps(receipt, indent=2) + "\n")
    log_path = a.record.with_suffix(".log")
    start = time.monotonic()
    with log_path.open("w") as log, a.record.with_suffix(".gpu.jsonl").open("w", buffering=1) as gpu:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        receipt["pid"] = process.pid
        a.record.write_text(json.dumps(receipt, indent=2) + "\n")
        while process.poll() is None:
            sample = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu.write(json.dumps(dict(elapsed_s=time.monotonic() - start, sample=sample.stdout.strip())) + "\n")
            low_disk = shutil.disk_usage(ROOT).free < 40 * 2**30
            if time.monotonic() - start > a.timeout or low_disk:
                receipt["timed_out"] = time.monotonic() - start > a.timeout
                receipt["disk_reserve_reached"] = low_disk
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(10)
        receipt["returncode"] = process.wait()
    receipt.update(
        elapsed_s=time.monotonic() - start, finished_unix=time.time(), free_bytes_after=shutil.disk_usage(ROOT).free
    )
    if receipt["returncode"] == 0:
        for path in a.output_dir.glob("seed_*/metadata.json"):
            meta = json.loads(path.read_text())
            meta["asset_profile"] = receipt["asset_profile"]
            meta["launch_receipt"] = a.record.name
            meta["launch_source_sha256"] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            path.write_text(json.dumps(meta, indent=2) + "\n")
    a.record.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "source_sha256"}), flush=True)
    raise SystemExit(receipt["returncode"])


if __name__ == "__main__":
    main()
