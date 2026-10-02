#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_root="${1:-}"
acknowledgement="${2:-}"

if [[ -z "${source_root}" || "${acknowledgement}" != "--acknowledge-license" ]]; then
  echo "Usage: $0 /path/to/alpha --acknowledge-license" >&2
  echo "Review the Reach terms in THIRD_PARTY_NOTICES.md and the upstream repository first." >&2
  exit 2
fi

if [[ -d "${source_root}/alpha_description/meshes" ]]; then
  source_mesh_dir="${source_root}/alpha_description/meshes"
elif [[ -f "${source_root}/M2.stl" ]]; then
  source_mesh_dir="${source_root}"
else
  echo "Could not find alpha_description/meshes under: ${source_root}" >&2
  exit 1
fi

destination="${project_dir}/src/wasman/assets/data/robots/bluerov2_alpha/meshes/alpha"
mesh_paths=(
  "M2.stl"
  "M2-1-1.stl"
  "M2-1-3.stl"
  "M3-INLINE.stl"
  "RS1-100-101-123.stl"
  "end_effectors/RS1-124.stl"
  "end_effectors/RS1-130.stl"
  "end_effectors/RS1-139.stl"
)

for relative_path in "${mesh_paths[@]}"; do
  if [[ ! -f "${source_mesh_dir}/${relative_path}" ]]; then
    echo "Required mesh is missing: ${source_mesh_dir}/${relative_path}" >&2
    exit 1
  fi
done

for relative_path in "${mesh_paths[@]}"; do
  install -Dm0644 "${source_mesh_dir}/${relative_path}" "${destination}/${relative_path}"
done

echo "Installed ${#mesh_paths[@]} locally licensed Reach Alpha meshes into ${destination}."
