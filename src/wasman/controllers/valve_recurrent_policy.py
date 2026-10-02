"""History-conditioned neural imitation; no teacher phase or IK at inference."""

import torch


class ValveRecurrentPolicy(torch.nn.Module):
    def __init__(self, mean, std, memory_size=128):
        super().__init__()
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization requires finite mean and positive effective standard deviation")
        self.register_buffer("mean", mean.reshape(50).clone())
        self.register_buffer("std", std.reshape(50).clone())
        self.register_buffer("increment_scale", torch.tensor([0.005] * 9 + [0.002, 0.03]))
        self.memory = torch.nn.GRU(50, memory_size, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(50 + memory_size, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 11),
        )
        self.phase_auxiliary = torch.nn.Linear(memory_size, 8)
        self.hidden = None

    def _actions(self, x, normalized, memory):
        delta = self.mlp(torch.cat((normalized, memory), -1)) * self.increment_scale
        action = x[..., 27:38] + delta
        return torch.cat((action[..., :3], torch.zeros_like(action[..., 3:6]), action[..., 6:]), -1)

    def forward_sequence(self, x):
        normalized = (x - self.mean) / self.std
        memory, _ = self.memory(normalized)
        return self._actions(x, normalized, memory), self.phase_auxiliary(memory)

    def forward(self, observations):
        x = observations["policy"]
        if self.hidden is not None and self.hidden.shape[1] != len(x):
            self.hidden = None
        normalized = (x - self.mean) / self.std
        memory, hidden = self.memory(normalized[:, None], self.hidden)
        self.hidden = hidden.detach()
        return self._actions(x, normalized, memory[:, 0])

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hidden = None
        elif self.hidden is not None:
            self.hidden[:, env_ids] = 0

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        model = cls(torch.zeros(50), torch.ones(50), **checkpoint["parameters"])
        model.load_state_dict(checkpoint["state_dict"])
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()) or (model.std <= 0).any():
            raise ValueError("Invalid recurrent checkpoint: non-finite weights or normalization")
        return model.to(device).eval()
