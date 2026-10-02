"""CLIP processing with stochastic augmentation restricted to training."""

import torch
import torchvision.transforms as T


class DPImageTransform(torch.nn.Module):
    def __init__(self, size=224):
        super().__init__()
        crop = int(size * 0.95)
        self.augmentation = torch.nn.Sequential(
            T.RandomCrop(crop),
            T.Resize(size, antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        )
        self.evaluation = torch.nn.Sequential(T.CenterCrop(crop), T.Resize(size, antialias=True))
        self.normalization = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def forward(self, image):
        return self.normalization((self.augmentation if self.training else self.evaluation)(image))
