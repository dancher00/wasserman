#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
uv run --extra isaacsim isaaclab list_envs --keyword Wasman
uv run --extra isaacsim isaaclab zero_agent \
  --task Wasman-Underwater-PressButton-Direct --num_envs 4 --max_steps 100 \
  physics=isaacsim_physx --viz none
