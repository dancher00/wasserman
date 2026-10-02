"""Learned imitation mixture: local acquisition/turn, recurrent stopping/release.

The routing threshold is fitted from demonstrations, not provided by the task
controller. Both branches see only public observations; no expert runs here.
"""

import torch

from wasman.controllers.valve_local_policy import ValveLocalPolicy
from wasman.controllers.valve_recurrent_policy import ValveRecurrentPolicy


class ValveMixturePolicy(torch.nn.Module):
    def __init__(self, local, recurrent, threshold):
        super().__init__()
        self.local, self.recurrent, self.threshold = local, recurrent, threshold
        self.maximum_angle = None

    def forward(self, observations):
        angle = observations["policy"][:, 38]
        if self.maximum_angle is None or self.maximum_angle.shape != angle.shape:
            self.maximum_angle = angle.clone()
        self.maximum_angle = torch.maximum(self.maximum_angle, angle)
        # Update neural history on every actual observation, including while
        # the other learned branch controls the robot.
        recurrent_action = self.recurrent(observations)
        local_action = self.local(observations)
        return torch.where((self.maximum_angle >= self.threshold)[:, None], recurrent_action, local_action)

    def reset(self, env_ids=None):
        self.local.reset(env_ids)
        self.recurrent.reset(env_ids)
        if env_ids is None:
            self.maximum_angle = None
        elif self.maximum_angle is not None:
            self.maximum_angle[env_ids] = -float("inf")

    @classmethod
    def from_checkpoint(cls, checkpoint, device):
        return cls(
            ValveLocalPolicy.from_checkpoint(checkpoint["local"], device),
            ValveRecurrentPolicy.from_checkpoint(checkpoint["recurrent"], device),
            checkpoint["threshold"],
        ).eval()
