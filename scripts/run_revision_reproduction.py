"""Finite clean-checkout reproduction from the completed source campaign."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source-campaign", type=Path, required=True)
parser.add_argument("--checkout", type=Path, required=True)
parser.add_argument("--timeout-hours", type=float, default=24)
parser.add_argument(
    "--wait-for-campaign-completion",
    action="store_true",
    help="Defer repeated scored rollouts until the primary campaign releases the GPU",
)
args = parser.parse_args()
ROOT = args.checkout.resolve()
P = str(ROOT / ".venv/bin/python")
CORE = args.source_campaign.resolve()
OUT = ROOT / "artifacts/repro-full"
DEADLINE = time.monotonic() + args.timeout_hours * 3600
OUT.mkdir(parents=True, exist_ok=True)


def wait_for(path):
    while not path.exists():
        if time.monotonic() > DEADLINE:
            raise TimeoutError(str(path))
        time.sleep(30)


def run(label, command, out):
    receipt = OUT / "jobs" / f"{label}.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("returncode") == 0:
        return
    subprocess.run(
        [
            P,
            "scripts/run_revision_job.py",
            "--record",
            str(receipt),
            "--output-dir",
            str(out),
            "--",
            P,
            *map(str, command),
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )


wait_for(CORE / "data/PressButton/completed.json")
data = OUT / "data/PressButton"
subprocess.run(
    [P, "scripts/restore_revision_rgb.py", "--archive", str(CORE / "archives/PressButton"), "--output", str(data)],
    cwd=ROOT,
    check=True,
)
model = OUT / "models/PressButton-ACT-17"
run(
    "PressButton-ACT-17-retrain",
    ["scripts/train_revision_policy.py", "--task", "PressButton", "--model", "ACT", "--seed", "17", "--data", data],
    model,
)
original = torch.load(CORE / "models/PressButton-ACT-17/final.pt", map_location="cpu", weights_only=True)
repeated = torch.load(model / "final.pt", map_location="cpu", weights_only=True)
assert original["config"] == repeated["config"]
differences = {
    k: float((v - repeated["state_dict"][k]).abs().max())
    for k, v in original["state_dict"].items()
    if not torch.equal(v, repeated["state_dict"][k])
}
record = dict(
    task="PressButton",
    model="ACT",
    training_seed=17,
    sample_budget=320000,
    configs_equal=True,
    bitwise_equal=not differences,
    max_absolute_parameter_difference=max(differences.values(), default=0),
    differing_tensors=len(differences),
    scope=(
        "One complete independent checkout training on the same GPU; shared verified dependency caches, no CAD assets"
    ),
)
(OUT / "training-reproduction.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
del original, repeated
if differences:
    raise RuntimeError("Complete training differed; inspect the recorded discrepancy before inference comparison")
# Full fixed-cohort inference reproduction, not additional independent models.
if args.wait_for_campaign_completion:
    wait_for(CORE / "completed-all.json")
for task, model_name in [
    ("PressButton", "ACT"),
    *[(t, "DP") for t in ["PressButton", "RotateValve", "OpenHatch", "CollectShell", "PushSlider", "PullLever"]],
]:
    i = ["PressButton", "RotateValve", "OpenHatch", "CollectShell", "PushSlider", "PullLever"].index(task)
    label = f"{task}-{model_name}-17"
    filename = "report.json" if task in ["PressButton", "PushSlider", "PullLever"] else "summary.json"
    source_report = CORE / "primary-evaluation" / label / filename
    wait_for(source_report)
    checkpoint = model / "final.pt" if model_name == "ACT" else CORE / "models" / label / "final.pt"
    out = OUT / "evaluation" / label
    run(
        label + "-evaluate",
        [
            "scripts/benchmark.py",
            "evaluate",
            "--task",
            task,
            "--checkpoint",
            checkpoint,
            "--purpose",
            "test",
            "--seeds",
            *range(60000 + 1000 * i, 60030 + 1000 * i),
        ],
        out,
    )
    ref = json.loads(source_report.read_text())
    rep = json.loads((out / filename).read_text())
    assert ref["seeds"] == rep["seeds"]
    comparison = dict(
        task=task,
        model=model_name,
        training_seed=17,
        source_successes=ref["success_per_seed"],
        repeat_successes=rep["success_per_seed"],
        outcomes_identical=ref["success_per_seed"] == rep["success_per_seed"],
        source_report_sha256=hashlib.sha256(source_report.read_bytes()).hexdigest(),
        repeat_report_sha256=hashlib.sha256((out / filename).read_bytes()).hexdigest(),
        scope="Same fixed checkpoint and reset cohort; not extra training replicates or a new success-rate estimate",
    )
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison), flush=True)
(OUT / "completed.json").write_text(
    json.dumps(dict(completed_unix=time.time(), training_runs=1, evaluation_cohorts=7)) + "\n"
)
