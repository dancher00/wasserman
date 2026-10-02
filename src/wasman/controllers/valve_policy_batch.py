"""Independent local-policy candidates sharing one vectorized simulator."""

import torch

from wasman.controllers.valve_local_policy import ValveLocalPolicy


class ValvePolicyBatch(torch.nn.Module):
    def __init__(self, policies, environments_per_policy):
        super().__init__()
        self.policies = torch.nn.ModuleList(policies)
        self.environments_per_policy = environments_per_policy
        self.num_envs = len(policies) * environments_per_policy

    def forward(self, observations):
        x = observations["policy"]
        if len(x) != self.num_envs:
            raise ValueError("Bundle environment count differs from checkpoint")
        return torch.cat(
            [
                policy({"policy": batch})
                for policy, batch in zip(self.policies, x.split(self.environments_per_policy), strict=True)
            ]
        )

    def reset(self, env_ids=None):
        for i, policy in enumerate(self.policies):
            if env_ids is None:
                policy.reset()
            else:
                local_ids = env_ids - i * self.environments_per_policy
                policy.reset(local_ids[(local_ids >= 0) & (local_ids < self.environments_per_policy)])

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        return cls(
            [ValveLocalPolicy.from_checkpoint(c, device) for c in checkpoint["policies"]],
            checkpoint["environments_per_policy"],
        ).eval()
