#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
uv run --extra isaacsim isaaclab train --rl_library rsl_rl \
  --task "${WASMAN_TASK:-Wasman-Underwater-PressButton-T200-Direct}" --num_envs 512 \
  physics=isaacsim_physx --viz none "$@"
