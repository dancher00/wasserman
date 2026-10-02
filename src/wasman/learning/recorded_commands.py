"""Read-only expert command playback, with optional declared policy-clock transport."""

import json
from pathlib import Path

import numpy as np

from wasman.learning.revision_protocol import sha256


class RecordedCommands:
    def __init__(self, root, seeds, *, profile, policy_hz=None, quaternion=False):
        self.commands, self.sources = [], []
        self.policy_hz, self.quaternion = policy_hz, quaternion
        for seed in seeds:
            paths = list(Path(root).glob(f"train_batch*/seed_{seed}")) + list(Path(root).glob(f"seed_{seed}"))
            if len(paths) != 1:
                raise ValueError("Playback requires exactly one matching recorded reset")
            path = paths[0]
            meta = json.loads((path / "metadata.json").read_text())
            if meta["seed"] != seed or meta["asset_profile"] != profile or meta["raw_hz"] != 30:
                raise ValueError("Playback identity/profile/clock mismatch")
            data = np.load(path / "trajectory.npz")
            self.commands.append(data["command"])
            self.sources.append(
                dict(
                    seed=seed,
                    metadata_sha256=sha256(path / "metadata.json"),
                    trajectory_sha256=sha256(path / "trajectory.npz"),
                )
            )

    def interpolate(self, commands, index):
        index = min(max(index, 0), len(commands) - 1)
        lo = int(index)
        hi = min(lo + 1, len(commands) - 1)
        a, b = commands[lo], commands[hi].copy()
        if self.quaternion and np.dot(a[3:7], b[3:7]) < 0:
            b[3:7] *= -1
        value = a + (b - a) * (index - lo)
        if self.quaternion:
            value[3:7] /= max(np.linalg.norm(value[3:7]), 1e-12)
        return value

    def at(self, step):
        values = []
        for commands in self.commands:
            if self.policy_hz is None or self.policy_hz == 30:
                value = commands[min(step, len(commands) - 1)]
            else:
                if self.policy_hz != 20:
                    raise ValueError("Only the declared native20/native30 clocks are supported")
                start = step - step % 12
                offset = (step % 12) * 20 / 30
                low = int(offset)
                pair = np.stack([self.interpolate(commands, start + j * 1.5) for j in (low, low + 1)])
                value = self.interpolate(pair, offset - low)
            values.append(value)
        result = np.stack(values).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite recorded command")
        return result
