"""One command interface to the versioned visual benchmark implementations.

The backends remain byte-identical to the implementations recorded in existing
checkpoint provenance. This module only selects the correct task and interface;
it never changes learning defaults, success criteria or observation semantics.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TASKS = ("PressButton", "RotateValve", "OpenHatch", "CollectShell", "PushSlider", "PullLever")
STEMS = {"RotateValve": "valve", "OpenHatch": "hatch", "CollectShell": "shell"}


def command(task, operation, model=None, interface=None):
    """Return the existing backend and fixed arguments, before user run arguments."""
    if task not in TASKS or operation not in {"collect", "train", "evaluate"}:
        raise ValueError("Unknown task or operation")
    interface = interface or ("ee" if task == "RotateValve" else "actuator")
    allowed = (
        {"ee"} if task == "RotateValve" else {"actuator", "ee"} if task in {"PushSlider", "PullLever"} else {"actuator"}
    )
    if interface not in allowed:
        raise ValueError(f"{task} has no benchmarked {interface} interface")
    if operation == "collect" and interface == "ee" and task in {"PushSlider", "PullLever"}:
        raise ValueError(
            "Collect native trajectories, then use prepare_research_dataset.py to construct an audited EE view"
        )
    if operation == "train":
        if model not in {"ACT", "DP"}:
            raise ValueError("Training requires --model ACT or --model DP")
        stem = STEMS.get(task, "button" if task == "PressButton" else "research" if interface == "ee" else "marine")
        return [f"scripts/train_{stem}_{model.lower()}.py", *(["--task", task] if task not in STEMS else [])]
    if model is not None:
        raise ValueError("--model is used only for training; evaluation reads the checkpoint configuration")
    if task in STEMS:
        stem = STEMS[task]
        return [f"scripts/{'collect' if operation == 'collect' else 'evaluate'}_{stem}_visual.py"]
    script = "scripts/rollout_button_visual.py" if task == "PressButton" else "scripts/rollout_research_visual.py"
    args = [script, "--task", task, "--mode", "collect" if operation == "collect" else "policy"]
    if task != "PressButton":
        args += ["--interface", interface]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operation", choices=["collect", "train", "evaluate"])
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--model", choices=["ACT", "DP"])
    parser.add_argument("--interface", choices=["actuator", "ee"])
    parser.add_argument("--dry-run", action="store_true", help="Print the exact command without loading the simulator")
    args, remainder = parser.parse_known_args()
    # Backend mode is fixed by the public operation, not overridable in forwarded options.
    if any(x == "--mode" or x.startswith("--mode=") for x in remainder):
        parser.error("Choose collect/evaluate as the operation instead of passing --mode")
    try:
        backend = command(args.task, args.operation, args.model, args.interface)
    except ValueError as error:
        parser.error(str(error))
    root = Path(__file__).resolve().parents[2]
    python = root / ".venv/bin/python"
    argv = [str(python) if python.is_file() else sys.executable, *backend, *remainder]
    if not (root / backend[0]).is_file():
        raise FileNotFoundError(backend[0])
    if args.dry_run:
        print(json.dumps({"cwd": str(root), "command": argv}, indent=2))
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(root / ".deps/valve-policy-deps"), environment.get("PYTHONPATH", "")])
    )
    environment.setdefault("WARP_CACHE_PATH", str(root / ".deps/research-core-warp-cache"))
    raise SystemExit(subprocess.call(argv, cwd=root, env=environment))
