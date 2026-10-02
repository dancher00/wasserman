"""Register the BlueROV2 Heavy + Alpha 5 press-button benchmark."""

import gymnasium as gym

from . import agents

for task_id, environment, configuration in (
    (
        "Wasman-Underwater-PressButton-T200-Registered-Direct",
        "smooth:SmoothPressButtonEnv",
        "RegisteredPressButtonEnvCfg",
    ),
    (
        "Wasman-Underwater-PressButton-Approach-T200-Pool-Registered-Direct",
        "approach:ApproachPressButtonEnv",
        "RegisteredPoolApproachPressButtonEnvCfg",
    ),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.{environment}",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.registered_cfg:{configuration}",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

gym.register(
    id="Wasman-Underwater-PressButton-Approach-T200-Pool-Direct",
    entry_point=f"{__name__}.approach:ApproachPressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pool_cfg:PoolApproachPressButtonEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wasman-Underwater-PressButton-Approach-T200-Direct",
    entry_point=f"{__name__}.approach:ApproachPressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.approach_cfg:ApproachPressButtonEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wasman-Underwater-PressButton-T200-Direct",
    entry_point=f"{__name__}.smooth:SmoothPressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.thruster_cfg:ThrusterPressButtonEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wasman-Underwater-PressButton-Direct",
    entry_point=f"{__name__}.env:UnderwaterPressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:UnderwaterPressButtonEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Wasman-Underwater-PressButton-Smooth-Direct",
    entry_point=f"{__name__}.smooth:SmoothPressButtonEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smooth:SmoothPressButtonEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
