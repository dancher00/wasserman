"""Learned stage transitions; no expert state, IK, or physical success override."""

import torch

from wasman.controllers.valve_mode_router import ValveModeRouter


class ValveSequentialRouter(torch.nn.Module):
    features = tuple(range(27)) + tuple(range(38, 50))
    stopping_mode = 4

    def __init__(self, trees):
        super().__init__()
        if len(trees) not in (6, 7):
            raise ValueError("Six or seven learned transitions required")
        self.trees = torch.nn.ModuleList([ValveModeRouter(tree) for tree in trees])
        self.mode = self.age = None

    def forward(self, observations):
        if self.mode is None or len(self.mode) != len(observations):
            self.mode = torch.zeros(len(observations), device=observations.device, dtype=torch.long)
            self.age = torch.zeros_like(self.mode)
        features = torch.cat((observations[:, self.features], self.age[:, None] / 30), -1).double()
        advance = torch.zeros_like(self.mode, dtype=torch.bool)
        for stage, tree in enumerate(self.trees):
            advance |= (self.mode == stage) & tree.classify(features).bool()
        self.mode += advance.long()
        self.age = torch.where(advance, 0, self.age + 1)
        return self.mode

    def reset(self, env_ids=None):
        if env_ids is None:
            self.mode = self.age = None
        elif self.mode is not None:
            self.mode[env_ids] = 0
            self.age[env_ids] = 0


def previous_stages_and_ages(phases):
    """Teacher annotation history for fitting only, with explicit [time,env] layout."""
    previous = torch.cat((torch.zeros_like(phases[:1]), phases[:-1]))
    ages = torch.zeros_like(previous)
    for step in range(1, len(phases)):
        ages[step] = torch.where(previous[step] == previous[step - 1], ages[step - 1] + 1, 0)
    return previous, ages
