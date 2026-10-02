"""Read-only analysis of an existing OpenHatch rollout, without Isaac or GPU.

Print JSON by default. --output writes only the requested derived report, never
the source rollout. Raw images/checkpoint/expert state are not needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wasman.learning.hatch_diagnostics import analyze_hatch_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {
        name: args.rollout / name for name in ("metadata.json", "diagnostics.json", "proprio.npy", "sample_step.npy")
    }
    terminal_path = args.rollout / "terminal.npy"
    if terminal_path.exists():
        paths["terminal.npy"] = terminal_path
    if args.output and args.output.resolve().is_relative_to(args.rollout.resolve()):
        parser.error("Write derived reports outside the immutable source rollout directory")
    report = analyze_hatch_rollout(
        json.loads(paths["metadata.json"].read_text()),
        json.loads(paths["diagnostics.json"].read_text()),
        np.load(paths["proprio.npy"], mmap_mode="r"),
        np.load(paths["sample_step.npy"], mmap_mode="r"),
        np.load(terminal_path, mmap_mode="r") if terminal_path.exists() else None,
    )
    report["input_rollout"] = str(args.rollout)
    report["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}
    source_root = Path(__file__).resolve().parents[1]
    report["analysis_source_sha256"] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("scripts/analyze_hatch_rollout.py", "src/wasman/learning/hatch_diagnostics.py")
    }
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(args.output)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
