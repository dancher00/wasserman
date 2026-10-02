"""CPU-only pool task registration and playback CLI contract."""

import os
import subprocess
from pathlib import Path

import gymnasium as gym

import wasman.tasks  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
POOL_TASK = "Wasman-Underwater-PressButton-Approach-T200-Pool-Direct"
ORIGINAL_TASK = "Wasman-Underwater-PressButton-Approach-T200-Direct"


def test_pool_has_separate_cfg_and_preserves_original_task_contract():
    pool, original = gym.spec(POOL_TASK), gym.spec(ORIGINAL_TASK)
    assert pool.entry_point == original.entry_point
    assert pool.kwargs["env_cfg_entry_point"].endswith(".pool_cfg:PoolApproachPressButtonEnvCfg")
    assert original.kwargs["env_cfg_entry_point"].endswith(".approach_cfg:ApproachPressButtonEnvCfg")
    assert pool.kwargs["rsl_rl_cfg_entry_point"] == original.kwargs["rsl_rl_cfg_entry_point"]


def test_play_default_and_explicit_task_override_without_launching_sim():
    # An exported shell function captures CLI arguments instead of executing uv.
    command = 'function uv() { printf "%s\\n" "$@"; }; export -f uv; bash scripts/play.sh --headless'
    env = os.environ.copy()
    env.pop("WASMAN_TASK", None)
    env["WASMAN_CHECKPOINT"] = str(ROOT / "checkpoints/wasman_press_button_smooth_seed42.pt")
    default = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert default.stdout.splitlines()[default.stdout.splitlines().index("--task") + 1] == (
        "Wasman-Underwater-PressButton-Approach-T200-Pool-Registered-Direct"
    )
    assert "--headless" in default.stdout.splitlines()
    env["WASMAN_TASK"] = ORIGINAL_TASK
    override = subprocess.run(["bash", "-c", command], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert override.stdout.splitlines()[override.stdout.splitlines().index("--task") + 1] == ORIGINAL_TASK
