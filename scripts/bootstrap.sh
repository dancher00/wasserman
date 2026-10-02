#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaaclab_dir="${project_dir}/.deps/IsaacLab"
isaaclab_commit="76c7c60de65eb5196dbfacac91d3febbff6044d2"
reach_mesh="${project_dir}/src/wasman/assets/data/robots/bluerov2_alpha/meshes/alpha/M2.stl"
asset_profile="${WASMAN_ASSET_PROFILE:-open-procedural-v1}"
if [[ "$#" == 2 && "$1" == "--asset-profile" ]]; then
  asset_profile="$2"
elif [[ "$#" != 0 ]]; then
  echo "Usage: $0 [--asset-profile historical-cad-v1|open-procedural-v1]" >&2
  exit 2
fi
case "${asset_profile}" in
  historical-cad-v1|open-procedural-v1) ;;
  *) echo "Unknown asset profile: ${asset_profile}" >&2; exit 2 ;;
esac

if [[ "${asset_profile}" == "historical-cad-v1" && ! -f "${reach_mesh}" ]]; then
  echo "Reach Alpha visual meshes are missing (they are intentionally not redistributed by WASMAN)." >&2
  echo "Review THIRD_PARTY_NOTICES.md, obtain the upstream Alpha assets, then run:" >&2
  echo "  ./scripts/install_reach_meshes.sh /path/to/alpha --acknowledge-license" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

python3 "${project_dir}/scripts/restore_reproduction_sources.py"

if [[ ! -e "${isaaclab_dir}/.git" ]]; then
  if [[ -e "${isaaclab_dir}" ]]; then
    echo "Existing Isaac Lab directory is not a Git checkout; move it aside explicitly before installation." >&2
    exit 1
  fi
  mkdir -p "${isaaclab_dir}"
  git -C "${isaaclab_dir}" init
  git -C "${isaaclab_dir}" remote add origin https://github.com/isaac-sim/IsaacLab.git
  git -C "${isaaclab_dir}" fetch --depth 1 origin "${isaaclab_commit}"
  git -C "${isaaclab_dir}" checkout --detach FETCH_HEAD
fi

actual_commit="$(git -C "${isaaclab_dir}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${isaaclab_commit}" ]]; then
  echo "Expected Isaac Lab ${isaaclab_commit}, found ${actual_commit}." >&2
  echo "Refusing to modify an existing dependency checkout automatically." >&2
  exit 1
fi

cd "${project_dir}"
uv sync --locked --extra isaacsim --group dev
echo "Installed dependencies for ${asset_profile}. Set WASMAN_ASSET_PROFILE=${asset_profile} for each runtime command."
