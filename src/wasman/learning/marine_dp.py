"""Reuse the pinned diffusion architecture with versioned marine episodes."""

from wasman.learning.marine_visual_dataset import MarineVisualDataset
from wasman.learning.shell_dp import ROOT, ShellDPDataset, make_dp  # noqa: F401


class MarineDPDataset(ShellDPDataset):
    def __init__(self, episodes, *, augment_start=False):
        MarineVisualDataset.__init__(self, episodes)
        self.items = [(e, i) for e, i in self.items if not self.trajectories[e]["pad"][i].any()]
        self.augment_start = False
