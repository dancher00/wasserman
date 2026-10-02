"""Hash every selected RGB byte and independently replay each core success predicate."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from wasman.learning.revision_protocol import PROTOCOL, TASKS, sha256, split_episodes


def replay(task, data, meta):
    if task in ("PressButton", "PushSlider", "PullLever"):
        if task == "PressButton":
            from button_visual_support import verify
        else:
            from wasman.controllers.marine_evidence import verify
        result = verify(data, meta["success_contract"])
        return result["first_success_step"][0]
    if task == "RotateValve":
        if meta["success_contract"]["version"] != "ambench-angle-170-v1":
            raise ValueError("Unexpected Valve success contract")
        successful = np.flatnonzero(data["angle_after"] >= math.radians(170))
        return int(successful[0] + 1) if len(successful) else -1
    if task == "OpenHatch":
        from wasman.controllers.hatch_contract import HatchContract

        contract = HatchContract(1, "cpu", 1 / meta["raw_hz"])
    elif task == "CollectShell":
        from wasman.controllers.shell_collection_contract import ShellCollectionContract

        contract = ShellCollectionContract(1, 1, "cpu", 1 / meta["raw_hz"])
    else:
        raise ValueError(task)
    inputs = {k.removeprefix("contract_"): v for k, v in data.items() if k.startswith("contract_")}
    if not inputs:
        raise ValueError("Missing independently replayable contact evidence")
    first = -1
    for step in range(meta["length"]):
        kwargs = {k: torch.as_tensor(v[step : step + 1]) for k, v in inputs.items()}
        state = contract.update(**kwargs)
        success = state["success"] if isinstance(state, dict) else state
        if bool(success[0]) != bool(data["success_after"][step]):
            raise ValueError(f"Contract mismatch at frame {step}")
        if success[0] and first < 0:
            first = step + 1
    return first


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    output = a.data / "revision_audit.json"
    if output.exists():
        raise ValueError("Preserve existing audit")
    if not a.pilot and a.episodes != 80:
        raise ValueError("Primary dataset requires 80 episodes; explicitly label smaller engineering pilots")
    metas = sorted(a.data.glob("train_batch*/seed_*/metadata.json"))
    if a.pilot and not metas:
        metas = sorted(a.data.glob("seed_*/metadata.json"))
    metas.sort(key=lambda path: json.loads(path.read_text())["seed"])
    attempts = [json.loads(path.read_text()) for path in metas]
    seeds = [row["seed"] for row in attempts]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate attempted seeds")
    if not a.pilot:
        start = 40000 + 1000 * TASKS.index(a.task)
        if seeds != list(range(start, start + len(seeds))) or len(seeds) > 400:
            raise ValueError("Training candidates must be the complete ordered prefix of the frozen stream")
        if any(row.get("asset_profile") != "open-procedural-v1" for row in attempts):
            raise ValueError("Every primary attempt must bind the open asset profile")
    accepted = [p.parent for p in metas if json.loads(p.read_text())["outcome"] == "success"][: a.episodes]
    if len(accepted) != a.episodes:
        raise ValueError(f"Need {a.episodes} successes, have {len(accepted)}")
    records = []
    for path in accepted:
        meta = json.loads((path / "metadata.json").read_text())
        data = dict(np.load(path / "trajectory.npz"))
        n = meta["length"]
        if data["terminal"].any() or data["success_after"][:-1].any() or not data["success_after"][-1]:
            raise ValueError("First-episode censoring violation")
        if any(not np.isfinite(v).all() for v in data.values()):
            raise ValueError("Nonfinite trajectory")
        if any(len(v) != n for v in data.values()):
            raise ValueError("Trajectory length mismatch")
        if not np.all(np.diff(data["camera_frame"]) == 1):
            raise ValueError("Camera not fresh every control step")
        if not np.allclose(data["time"], np.arange(n) / meta["raw_hz"], atol=1e-6, rtol=0):
            raise ValueError("Observation/action timeline mismatch")
        if (path / "wrist.rgb").stat().st_size != n * np.prod(meta["image_shape"]):
            raise ValueError("RGB byte size mismatch")
        rgb = np.memmap(path / "wrist.rgb", mode="r", dtype=np.uint8, shape=(n, *meta["image_shape"]))
        frames = np.asarray(rgb[np.linspace(0, n - 1, min(9, n)).astype(int)])
        if np.min(frames.reshape(len(frames), -1).std(-1)) < 1:
            raise ValueError("Blank image")
        if replay(a.task, data, meta) != n:
            raise ValueError("Independent success replay failed")
        record = dict(
            seed=meta["seed"],
            length=n,
            path=str(path.relative_to(a.data)),
            files={name: sha256(path / name) for name in ("metadata.json", "trajectory.npz", "wrist.rgb")},
            contract_replayed=True,
        )
        records.append(record)
    train, val = split_episodes(accepted)
    payload = dict(
        protocol=PROTOCOL,
        passed=True,
        pilot=a.pilot,
        task=a.task,
        episodes=records,
        attempted_episodes=len(metas),
        attempts=[
            dict(seed=row["seed"], outcome=row["outcome"], metadata_sha256=sha256(path))
            for row, path in zip(attempts, metas, strict=True)
        ],
        train_seeds=[json.loads((p / "metadata.json").read_text())["seed"] for p in train],
        validation_seeds=[json.loads((p / "metadata.json").read_text())["seed"] for p in val],
        source_sha256=sha256(__file__),
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(passed=True, episodes=len(records), task=a.task)), flush=True)


if __name__ == "__main__":
    main()
