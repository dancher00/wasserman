"""Episode-safe native 30 Hz actuator chunks from synchronized 30 Hz shell recordings.

No simulator imports or object states. Absolute normalized robot targets and
measured proprioception. Images are never interpolated or taken from the future.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def interpolate_commands(command, index):
    """Linear interpolation of absolute normalized actuator targets."""
    index = np.clip(index, 0, len(command) - 1)
    lo = np.floor(index).astype(int)
    hi = np.minimum(lo + 1, len(command) - 1)
    weight = (index - lo)[..., None]
    return (command[lo] * (1 - weight) + command[hi] * weight).astype(np.float32)


def successful_episodes(root):
    episodes = []
    for path in sorted(Path(root).glob("train_batch*/seed_*/metadata.json")):
        meta = json.loads(path.read_text())
        if meta["outcome"] == "success":
            episodes.append(path.parent)
    seeds = [json.loads((p / "metadata.json").read_text())["seed"] for p in episodes]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Repeated episode seed")
    return episodes


class ShellVisualDataset(Dataset):
    def __init__(self, episodes, *, horizon=16, state_mode="absolute", policy_hz=30):
        if policy_hz not in (20, 30):
            raise ValueError("Supported policy frequencies: native30 or historical20")
        self.policy_hz = policy_hz
        if state_mode != "absolute":
            raise ValueError(state_mode)
        self.episodes, self.horizon, self.state_mode = list(map(Path, episodes)), horizon, state_mode
        self.items, self.buffers, self.trajectories = [], {}, []
        for episode, path in enumerate(self.episodes):
            meta = json.loads((path / "metadata.json").read_text())
            if meta["schema"] != "wasman-shell-rgb-actuator-v1" or meta["outcome"] != "success":
                raise ValueError(f"Not a successful, versioned expert episode: {path}")
            # Physical replay evidence is deliberately excluded from model data
            # and worker memory; it is read separately by the dataset auditor.
            with np.load(path / "trajectory.npz") as archive:
                data = {
                    k: archive[k]
                    for k in (
                        "measured",
                        "command",
                        "terminal",
                        "success_after",
                        "camera_frame",
                    )
                }
            if data["terminal"].any() or not data["success_after"][-1]:
                raise ValueError("Invalid first-episode success")
            if np.any(np.diff(data["camera_frame"]) != 1):
                raise ValueError("Camera frames are stale or skipped")
            length = meta["length"]
            if (path / "wrist.rgb").stat().st_size != length * np.prod(meta["image_shape"]):
                raise ValueError("Image file length disagrees with trajectory")
            # Native 30 Hz preserves commands exactly; legacy20 is available for diagnosis.
            # Its actual timestamp is also the anchor for the future command grid.
            if policy_hz == meta["raw_hz"]:
                anchors = np.arange(length)
            else:
                anchors = np.floor(
                    np.arange(int((length - 1) / meta["raw_hz"] * policy_hz) + 1) * meta["raw_hz"] / policy_hz + 1e-7
                ).astype(int)
            indices = anchors[:, None] + np.arange(horizon)[None, :] * meta["raw_hz"] / policy_hz
            chunks = interpolate_commands(data["command"], indices)
            state = data["measured"][anchors]
            self.trajectories.append(
                dict(meta=meta, data=data, anchors=anchors, action=chunks, state=state, pad=indices > length - 1)
            )
            self.items.extend((episode, i) for i in range(len(anchors)))
        if not self.items:
            raise ValueError("Empty dataset")

    def statistics(self):
        actions = np.concatenate([t["action"][~t["pad"]] for t in self.trajectories])
        state = np.concatenate([t["state"] for t in self.trajectories])
        return {
            k: torch.from_numpy(v)
            for k, v in dict(
                action_mean=actions.mean(0),
                action_std=np.maximum(actions.std(0), 1e-3),
                state_mean=state.mean(0),
                state_std=np.maximum(state.std(0), 1e-3),
            ).items()
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        episode, sample = self.items[index]
        t = self.trajectories[episode]
        if episode not in self.buffers:
            self.buffers[episode] = np.memmap(
                self.episodes[episode] / "wrist.rgb",
                mode="r",
                dtype=np.uint8,
                shape=(t["meta"]["length"], *t["meta"]["image_shape"]),
            )
        image = np.asarray(self.buffers[episode][t["anchors"][sample]]).copy()
        return {
            "observation.images.wrist": torch.from_numpy(image).permute(2, 0, 1),
            "observation.state": torch.from_numpy(t["state"][sample]),
            "action": torch.from_numpy(t["action"][sample]),
            "action_is_pad": torch.from_numpy(t["pad"][sample]),
        }


def normalize_act_batch(batch, statistics, device):
    result = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    image = result["observation.images.wrist"].float() / 255
    mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    result["observation.images.wrist"] = (image - mean) / std
    for key, prefix in [("observation.state", "state"), ("action", "action")]:
        if key in result:
            result[key] = (result[key] - statistics[prefix + "_mean"].to(device)) / statistics[prefix + "_std"].to(
                device
            )
    return result
