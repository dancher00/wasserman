"""Versioned geometry regression; old score-bearing task IDs are untouched."""

from isaaclab.utils import configclass

from wasman.assets.registered_bluerov import configure_registered_bluerov

from .pool_cfg import PoolApproachPressButtonEnvCfg
from .thruster_cfg import ThrusterPressButtonEnvCfg


@configclass
class RegisteredPressButtonEnvCfg(ThrusterPressButtonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_registered_bluerov(self)


@configclass
class RegisteredPoolApproachPressButtonEnvCfg(PoolApproachPressButtonEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_registered_bluerov(self)
