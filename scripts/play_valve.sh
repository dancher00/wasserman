#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
controller="expert"
for argument in "$@"; do
  case "$argument" in
    --checkpoint|--checkpoint=*) controller="policy" ;;
  esac
done
if [[ "$controller" == "policy" ]]; then
  echo "RotateValve: learned checkpoint playback (no expert at inference)."
else
  echo "RotateValve: physical state-feedback expert (not a trained policy)."
  echo "To evaluate a learned actor, add --checkpoint <valve-checkpoint>."
fi
uv run --extra isaacsim python scripts/check_valve_expert.py \
  --visualizer kit --real-time --num-envs 1 \
  --output-dir logs/valve_playback "$@"
