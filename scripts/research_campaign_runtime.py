"""Finite stage journal shared by the authorized research campaign."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/research_core_20260924"
ENV = dict(
    os.environ,
    PYTHONPATH=str(ROOT / ".deps/valve-policy-deps"),
    WARP_CACHE_PATH=str(ROOT / ".deps/research-core-warp-cache"),
    OMP_NUM_THREADS="4",
    MKL_NUM_THREADS="4",
)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(p)


def frozen_sources():
    own = BASE / "frozen_sources.json"
    if own.exists():
        for rel, digest in read(own).items():
            if sha(ROOT / rel) != digest:
                raise ValueError("Research source changed: " + rel)
    for rel, h in read(ROOT / "artifacts/marine_mechanisms_20260924/acceptance.json")["source_sha256"].items():
        if sha(ROOT / rel) != h:
            raise ValueError("Original frozen source changed: " + rel)


def run(label, command, expected):
    expected = Path(expected)
    folder = BASE / "stages" / label
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "stage.json"
    if receipt.exists() and read(receipt).get("status") == "complete":
        if not expected.is_file() or sha(expected) != read(receipt)["expected_sha256"]:
            raise ValueError("Completed artifact changed")
        return
    if receipt.exists():
        raise ValueError(f"Preserve failed/running stage and diagnose: {label}")
    if expected.exists():
        raise ValueError(f"Unowned completion artifact: {expected}")
    frozen_sources()
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError("Less than25GiB disk headroom; no stage started")
    cmd = [str(ROOT / ".venv/bin/python"), *map(str, command)]
    state = dict(status="running", label=label, command=cmd, expected=str(expected), started_unix=time.time())
    write(receipt, state)
    write(BASE / "queue.json", state)
    with (folder / "stdout.log").open("x") as stream:
        code = subprocess.call(cmd, cwd=ROOT, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
    if code or not expected.is_file() or (expected.parent / "error.json").exists():
        state.update(status="failed", exit_code=code)
        write(receipt, state)
        write(BASE / "queue.json", state)
        raise RuntimeError(f"{label} failed; evidence preserved")
    state.update(status="complete", completed_unix=time.time(), expected_sha256=sha(expected))
    write(receipt, state)
    write(BASE / "queue.json", state)


def select(task, model, data, logs, *, button=False, interface="actuator", label=""):
    budgets = [5000, 10000, 20000] if model == "act" else [10, 20, 40]
    val = 9100 if button else 8300 if task == "PushSlider" else 8500
    directory = Path(data)
    result = []
    for budget in budgets:
        checkpoint = logs / (f"model_{budget}.pt" if model == "act" else f"model_epoch_{budget}.pt")
        out = directory / f"validation_{model}_{budget}"
        cmd = [
            "scripts/rollout_button_visual.py" if button else "scripts/rollout_research_visual.py",
            "--task",
            task,
            "--mode",
            "policy",
            "--purpose",
            "validation",
            "--checkpoint",
            checkpoint,
            "--seeds",
            *range(val, val + 8),
            "--output-dir",
            out,
        ]
        if not button:
            cmd += ["--interface", interface]
        run(f"{label}_validation_{model}_{budget}", cmd, out / "report.json")
        r = read(out / "report.json")
        if not r["contract_replayed"] or r["expert_at_inference"]:
            raise ValueError("Validation provenance")
        result.append(
            dict(
                budget=budget,
                successes=sum(r["success_per_seed"]),
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),
            )
        )
    chosen = max(result, key=lambda x: (x["successes"], -x["budget"]))
    path = directory / f"selection_{model}.json"
    value = dict(selected=chosen, validation=result, decision_unix=time.time(), test_opened=False)
    if path.exists():
        if read(path)["selected"] != chosen:
            raise ValueError("Selection changed")
    else:
        write(path, value)
    return chosen
