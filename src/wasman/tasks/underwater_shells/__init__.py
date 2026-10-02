"""Shell-to-hoop task under physical-development acceptance; no trained policy claim."""

import gymnasium as gym

gym.register(
    id="Wasman-Underwater-CollectShell-Direct",
    entry_point=f"{__name__}.env:CollectShellEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CollectShellEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": (
            "wasman.tasks.underwater_press_button.config.bluerov2_alpha.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
        ),
    },
)
