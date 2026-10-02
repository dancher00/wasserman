"""Training-fitted late-motion routing from physical observations, not commands."""

import torch


class ValveModeRouter(torch.nn.Module):
    features = (38, 39, 42, 43, 47, 48, 49)

    def __init__(self, tree):
        super().__init__()
        for name in ("left", "right", "feature", "threshold", "prediction"):
            self.register_buffer(name, tree[name].clone())
        self.depth = tree["max_depth"]
        self.maximum_angle = None
        self.mode = None

    def classify(self, features):
        node = torch.zeros(len(features), device=features.device, dtype=torch.long)
        for _ in range(self.depth + 1):
            leaf = self.left[node] < 0
            value = features.gather(1, self.feature[node].clamp_min(0)[:, None])[:, 0]
            child = torch.where(value <= self.threshold[node], self.left[node], self.right[node])
            node = torch.where(leaf, node, child)
        return self.prediction[node]

    def forward(self, observations):
        angle = observations[:, 38]
        if self.maximum_angle is None or self.maximum_angle.shape != angle.shape:
            self.maximum_angle = angle.clone()
            self.mode = torch.zeros(len(angle), device=angle.device, dtype=torch.long)
        self.maximum_angle = torch.maximum(self.maximum_angle, angle)
        features = torch.cat((observations[:, self.features], self.maximum_angle[:, None]), -1).double()
        # Monotone progress is an architectural prior; all split thresholds
        # and class choices are fitted from training demonstrations.
        self.mode = torch.maximum(self.mode, self.classify(features))
        return self.mode

    def reset(self, env_ids=None):
        if env_ids is None:
            self.maximum_angle = self.mode = None
        elif self.maximum_angle is not None:
            self.maximum_angle[env_ids] = -float("inf")
            self.mode[env_ids] = 0
