"""Shared, explicit v2 split and inference contract; historical defaults stay separate."""

import hashlib
import json
from pathlib import Path

import numpy as np

PROTOCOL = "wm-open-v2-20260929"
TASKS = ("PressButton", "RotateValve", "OpenHatch", "CollectShell", "PushSlider", "PullLever")
CONTROL_STEPS = dict(zip(TASKS, (470, 2240, 1790, 7190, 1790, 1790), strict=True))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_episodes(episodes, split_seed=20260929):
    if len(episodes) < 2:
        raise ValueError("Need at least two whole episodes")
    seeds = [json.loads((Path(p) / "metadata.json").read_text())["seed"] for p in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate reset seeds")
    indices = np.random.default_rng(split_seed).permutation(len(episodes))
    val = set(indices[: max(1, round(0.05 * len(episodes)))].tolist())
    return ([p for i, p in enumerate(episodes) if i not in val], [p for i, p in enumerate(episodes) if i in val])


def verify_asset_profile(config):
    from wasman.assets.open_geometry import selected_profile

    expected = config.get("asset_profile", "historical-cad-v1")
    if selected_profile() != expected:
        raise ValueError(f"Policy asset profile {expected} != runtime {selected_profile()}")


def configure_inference(actor, config):
    """Apply only explicitly versioned behavior; reject accidental cross-profile use."""
    verify_asset_profile(config)
    if config.get("protocol") != PROTOCOL:
        return
    if not config.get("pilot"):
        validate_runtime_snapshot(config["runtime_manifest_sha256"])
    import torch

    if config["model"] == "DP":
        install_dp_transform(actor)
    actor.float().eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def install_dp_transform(model):
    from wasman.learning.revision_rgb import DPImageTransform

    for key in model.obs_encoder.rgb_keys:
        model.obs_encoder.key_transform_map[key] = DPImageTransform()


def validate_runtime_snapshot(expected_hash=None):
    root = Path(__file__).resolve().parents[3]
    path = root / "research/revision-v2-runtime.json"
    actual_hash = sha256(path)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("Checkpoint/runtime manifest mismatch")
    manifest = json.loads(path.read_text())
    if manifest["status"] != "frozen" or manifest["protocol"] != PROTOCOL:
        raise ValueError("Primary training/evaluation requires the frozen, versioned runtime")
    for name, digest in manifest["source_sha256"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Runtime source changed after freeze: {name}")
    return actual_hash


def replan_control_steps(config):
    actions = config.get("n_action_steps", 16 if config["model"] == "ACT" else 8)
    return round(actions * 30 / config["policy_hz"])


def validate_evaluation(config, task, seeds, purpose, *, steps=None):
    """Protect the untouched primary cohort and prevent pilot weights scoring as final."""
    if config.get("protocol") != PROTOCOL:
        return
    if task != config["task"] or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation task mismatch or repeated reset seeds")
    index = TASKS.index(task)
    test = set(range(60000 + 1000 * index, 60030 + 1000 * index))
    validation = set(range(50000 + 1000 * index, 50008 + 1000 * index))
    used = set(seeds)
    if used & set(config["train_seeds"] + config["validation_seeds"]):
        raise ValueError("Train/evaluation leakage")
    if purpose == "test":
        if config["pilot"] or list(seeds) != sorted(test):
            raise ValueError("Primary test requires final non-pilot weights and the complete frozen cohort")
    elif used & test:
        raise ValueError("Reserved test states cannot be used for development or other studies")
    if purpose in ("test", "validation", "research") and steps != CONTROL_STEPS[task]:
        raise ValueError("Scored evaluation requires the complete declared control horizon")
    if purpose == "validation" and list(seeds) != sorted(validation):
        raise ValueError("Closed-loop validation cohort mismatch")
