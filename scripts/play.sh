#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y

if [[ -n "${WASMAN_CHECKPOINT:-}" ]]; then
  checkpoint="${WASMAN_CHECKPOINT}"
else
  checkpoint="checkpoints/wasman_press_button_smooth_seed42.pt"
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "WASMAN checkpoint not found: ${checkpoint}" >&2
  echo "Train one with ./scripts/train.sh or set WASMAN_CHECKPOINT." >&2
  exit 1
fi

uv run --extra isaacsim isaaclab play --rl_library rsl_rl \
  --task "${WASMAN_TASK:-Wasman-Underwater-PressButton-Approach-T200-Pool-Registered-Direct}" --num_envs 1 \
  --checkpoint "${checkpoint}" --visualizer kit --max_visible_envs 1 --real-time \
  physics=isaacsim_physx "$@"
