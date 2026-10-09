#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "Install the release first: ./scripts/bootstrap.sh --asset-profile open-procedural-v1" >&2
  exit 1
fi
export WASMAN_ASSET_PROFILE="${WASMAN_ASSET_PROFILE:-open-procedural-v1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec .venv/bin/python scripts/teleop.py "$@"
