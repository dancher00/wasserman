"""Experimental horizontal, flooded submarine hatch; no trained policy released."""

import gymnasium as gym

for cameras in (False, True):
    gym.register(
        id=f"Wasman-Underwater-OpenHatch{'-Cameras' if cameras else ''}-Direct",
        entry_point=f"{__name__}.env:UnderwaterHatchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{'HatchCamerasEnvCfg' if cameras else 'HatchEnvCfg'}",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "wasman.tasks.underwater_press_button.config.bluerov2_alpha.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
    )
