"""Versioned smooth pressing with physical, independently lagged T200 actuation."""

from isaaclab.utils import configclass

from .smooth import SmoothPressButtonEnvCfg


@configclass
class ThrusterPressButtonEnvCfg(SmoothPressButtonEnvCfg):
    use_physical_thrusters = True
    thruster_time_constant = 0.08
    thruster_command_delay_steps = 2
