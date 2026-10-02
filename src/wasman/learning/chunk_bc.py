"""Simple image/state regression baseline: ImageNet ResNet18 and an MLP chunk head."""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


class ChunkBC(nn.Module):
    def __init__(self, state_dim, action_dim, *, pretrained=True):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            norm_layer=FrozenBatchNorm2d,
        )
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512 + state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 16 * action_dim),
        )

    def get_optim_params(self):
        return self.parameters()

    def predict_action_chunk(self, batch):
        image = self.encoder(batch["observation.images.wrist"])
        features = torch.cat((image, batch["observation.state"]), dim=-1)
        return self.head(features).reshape(-1, 16, self.action_dim)

    def forward(self, batch):
        loss = (self.predict_action_chunk(batch) - batch["action"]).abs()
        valid = (~batch["action_is_pad"]).unsqueeze(-1).expand_as(loss)
        return loss[valid].mean(), {}


def make_bc(task, *, pretrained=False):
    state_dim, action_dim = (21, 10) if task == "PressButton" else (8, 8)
    return ChunkBC(state_dim, action_dim, pretrained=pretrained)
