"""Experimental panel tasks; registration does not imply a trained policy."""

import gymnasium as gym

gym.register(
    id="Wasman-Underwater-RotateValve-Approach-T200-Direct",
    entry_point=f"{__name__}.approach:ApproachRotateValveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.approach:ApproachRotateValveEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ValvePPORunnerCfg",
    },
)

gym.register(
    id="Wasman-Underwater-RotateValve-T200-Direct",
    entry_point=f"{__name__}.valve:UnderwaterRotateValveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnderwaterRotateValveT200EnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:ValvePPORunnerCfg",
    },
)

for task in ("PushSlider", "PullLever", "RotateValve"):
    gym.register(
        id=f"Wasman-Underwater-{task}-Direct",
        entry_point="wasman.tasks.underwater_press_button.config.bluerov2_alpha.env:UnderwaterPressButtonEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:Underwater{task}EnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "wasman.tasks.underwater_press_button.config.bluerov2_alpha.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
        },
    )

# Versioned replacements leave historical scenes and checkpoints reproducible.
for task in ('PushSlider', 'PullLever'):
    gym.register(
        id=f'Wasman-Underwater-{task}-Marine-v1',
        entry_point='wasman.tasks.underwater_press_button.config.bluerov2_alpha.env:UnderwaterPressButtonEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}.linear_cfg:Marine{task}Cfg'},
    )
