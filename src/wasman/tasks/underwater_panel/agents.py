"""Separate experiment namespace; valve checkpoints are not button policies."""

from isaaclab.utils import configclass

from wasman.tasks.underwater_press_button.config.bluerov2_alpha.agents.rsl_rl_ppo_cfg import PPORunnerCfg


@configclass
class ValvePPORunnerCfg(PPORunnerCfg):
    experiment_name = "wasman_underwater_rotate_valve"

    def __post_init__(self):
        self.actor.class_name = "wasman.controllers.valve_policy:ValveResidualActor"
