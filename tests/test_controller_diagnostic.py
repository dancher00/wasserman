"""CPU-only regression contracts for object-free controller diagnostics.

Compile only the actual production method/class AST with a stub Isaac parent;
no simulator, GPU or substituted physics implementation is involved.
"""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wasman.assets.pool_geometry import POOL_GEOMETRY, POOL_USD_PATH
from wasman.physics.boundary_effects import BlueROVBoundaryEffect, validate_boundary_scene

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "src/wasman/tasks/underwater_press_button/config/bluerov2_alpha/env.py"
DIAGNOSTIC_PATH = ROOT / "src/wasman/controller_diagnostic.py"


def production_class(path):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef))


def parent_method(name):
    return next(
        node for node in production_class(ENV_PATH).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def diagnostic_type():
    class StubParent:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg

    namespace = {"torch": torch, "UnderwaterPressButtonEnv": StubParent}
    tree = ast.Module(body=[production_class(DIAGNOSTIC_PATH)], type_ignores=[])
    exec(compile(tree, str(DIAGNOSTIC_PATH), "exec"), namespace)
    return namespace["ControllerDiagnosticEnv"]


@pytest.mark.parametrize("panel,button", [(object(), None), (None, object()), (object(), object())])
def test_diagnostic_rejects_any_manipulation_object(panel, button):
    with pytest.raises(ValueError, match="no manipulation objects"):
        diagnostic_type()(NS(scene=NS(panel=panel, button=button)))


def test_diagnostic_only_overrides_observation_reward_and_termination_not_force_path():
    cls = diagnostic_type()
    cfg = NS(scene=NS(panel=None, button=None))
    env = cls(cfg)
    assert cfg.observation_space == 13
    assert env.diagnostic_only is True
    assert not {"_apply_action", "_pre_physics_step", "_reset_idx", "_setup_scene"}.intersection(cls.__dict__)


def test_ordinary_manipulation_task_still_rejects_missing_mechanism():
    # Execute the production constructor's pre-simulator guard in isolation.
    guard = parent_method("__init__").body[0]
    wrapper = ast.parse("def guard(self, cfg):\n    pass\n").body[0]
    wrapper.body = [guard]
    namespace = {}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(ENV_PATH), "exec"),
        namespace,
    )
    cfg = NS(scene=NS(button=None))
    with pytest.raises(ValueError, match="requires its mechanism"):
        namespace["guard"](NS(), cfg)
    namespace["guard"](NS(diagnostic_only=True), cfg)
    namespace["guard"](NS(), NS(scene=NS(button=object())))


def test_extracted_mechanism_reset_is_ast_identical_to_original_sequence():
    # Captured from the original released _reset_idx button_pose...target block
    # before extraction. Comments/formatting/line numbers do not affect this.
    body = parent_method("_reset_mechanism").body[1:]  # omit new docstring
    normalized = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert (
        hashlib.sha256(normalized.encode()).hexdigest()
        == "751a53038b867bb649fa5d2adad5b932d1537900c93118ee88c7564552f13b4a"
    )
    reset = parent_method("_reset_idx")
    call_index = next(
        i
        for i, node in enumerate(reset.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.button is not None"
    )
    assert ast.unparse(reset.body[call_index].body[0]) == "self._reset_mechanism(indices, count)"
    # Preserve RNG ordering: mechanism draws still immediately precede current.
    assert ast.unparse(reset.body[call_index + 1].targets[0]) == "speed"


def test_no_object_return_is_after_robot_gripper_and_before_mechanism_actuation():
    body = parent_method("_apply_action").body
    gate = next(
        i for i, node in enumerate(body) if isinstance(node, ast.If) and ast.unparse(node.test) == "self.button is None"
    )
    assert isinstance(body[gate].body[0], ast.Return)
    assert ast.unparse(body[gate - 1]).startswith("self.robot.set_joint_position_target_index(")
    assert ast.unparse(body[gate + 1]).startswith("self.button.set_joint_position_target_index(")


@pytest.mark.parametrize("pool", [False, True])
def test_boundary_validator_and_forces_accept_true_absent_panel(pool):
    path = POOL_USD_PATH if pool else ROOT / "src/wasman/assets/data/seabed/sand.usda"
    cfg = NS(
        use_physical_thrusters=True,
        scene=NS(
            panel=None,
            seabed=NS(spawn=NS(usd_path=str(path), scale=None), init_state=NS(pos=(0, 0, 0), rot=(0, 0, 0, 1))),
        ),
    )
    geometry = validate_boundary_scene(cfg)
    assert geometry == (POOL_GEOMETRY if pool else None)
    model = BlueROVBoundaryEffect(pool_geometry=geometry)
    pose = torch.tensor([[0.0, 0.0, 0.3, 0, 0, 0, 1]])
    forces, gain, loss = model.apply(pose, torch.full((1, 8), -10.0), panel_pose_w=None)
    assert loss.shape == (1, 8, 5 if pool else 1)  # no hidden six-panel-face contribution
    assert torch.isfinite(forces).all() and torch.isfinite(gain).all()


def test_diagnostic_observations_and_safety_dones_are_batched_without_task_success():
    count = 4
    position = torch.zeros(count, 2, 3)
    position[:, :, 2] = 0.8
    quat = torch.zeros(count, 2, 4)
    quat[:, :, 3] = 1
    velocity = torch.zeros(count, 2, 6)
    cfg = NS(
        scene=NS(panel=None, button=None),
        max_valid_body_linear_speed=5,
        max_valid_body_angular_speed=10,
        max_valid_joint_speed=20,
        max_base_distance=3,
        min_base_height=0.1,
        max_base_height=2,
    )
    env = diagnostic_type()(cfg)
    env.num_envs, env.device, env._base_body_id = count, "cpu", 0
    env.scene = NS(env_origins=torch.zeros(count, 3))
    env.robot = NS(
        data=NS(
            body_link_pos_w=NS(torch=position),
            body_link_quat_w=NS(torch=quat),
            body_com_vel_w=NS(torch=velocity),
            joint_pos=NS(torch=torch.zeros(count, 5)),
            joint_vel=NS(torch=torch.zeros(count, 5)),
        )
    )
    env._invalid_state = torch.zeros(count, dtype=torch.bool)
    env.episode_length_buf, env.max_episode_length = torch.tensor([0, 0, 0, 240]), 240
    assert env._get_observations()["policy"].shape == (count, 13)
    assert not env._get_rewards().any()
    velocity[1, 1, 0] = float("nan")
    position[2, 0, 0] = 3.1
    terminated, truncated = env._get_dones()
    assert terminated.tolist() == [False, True, True, False]
    assert truncated.tolist() == [False, False, False, True]
    assert env._invalid_state.tolist() == [False, True, False, False]
