import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

from wasman.controllers.thruster_visuals import (
    DIRECTIONS,
    HANDEDNESS,
    POSITIONS,
    ThrusterDisplayState,
    allocation_matrix,
)


def test_no_outboard_shaft_extension_and_wrench_is_unchanged():
    from wasman.controllers.thruster_visuals import CAD_OFFSET, NOMINAL_POSITIONS

    positions = torch.tensor(POSITIONS, dtype=torch.float64)
    expected = torch.tensor(NOMINAL_POSITIONS, dtype=torch.float64) + torch.tensor(CAD_OFFSET, dtype=torch.float64)
    torch.testing.assert_close(positions, expected)
    direction = torch.tensor(DIRECTIONS, dtype=torch.float64)
    old_offsets = torch.tensor([0.03720, 0.03718, 0.03815, 0.03813, 0.03599, 0.03599, 0.03599, 0.03599])[:, None]
    previous = positions - direction * old_offsets
    torch.testing.assert_close(torch.linalg.cross(previous, direction), torch.linalg.cross(positions, direction))


def test_full_six_dof_allocation_and_reconstruction():
    matrix = allocation_matrix(dtype=torch.float64)
    assert torch.linalg.matrix_rank(matrix) == 6
    wrench = torch.eye(6, dtype=torch.float64)
    forces = wrench @ torch.linalg.pinv(matrix).T
    torch.testing.assert_close(forces @ matrix.T, wrench)
    # Heave uses vertical motors, yaw uses horizontal motors.
    assert forces[2, :4].abs().max() < 1e-10
    assert forces[5, 4:].abs().max() < 1e-10


def test_mesh_mount_axes_match_allocator():
    root = ET.parse(Path(__file__).parents[1] / "src/wasman/assets/data/robots/bluerov2_alpha/bluerov2_alpha.urdf")
    for index, direction in enumerate(DIRECTIONS, 1):
        visual = root.find(f".//visual[@name='thruster{index}_visual']")
        position = torch.tensor(list(map(float, visual.find("origin").get("xyz").split())))
        torch.testing.assert_close(position, torch.tensor(POSITIONS[index - 1]))
        r, p, y = map(float, visual.find("origin").get("rpy").split())
        # URDF Rz(yaw) Ry(pitch) Rx(roll) applied to native propeller +Y.
        axis = torch.tensor(
            (
                math.cos(y) * math.sin(p) * math.sin(r) - math.sin(y) * math.cos(r),
                math.sin(y) * math.sin(p) * math.sin(r) + math.cos(y) * math.cos(r),
                math.cos(p) * math.sin(r),
            )
        )
        torch.testing.assert_close(axis, torch.tensor(direction, dtype=torch.float32), atol=1e-6, rtol=0)
        filename = visual.find("geometry/mesh").get("filename")
        assert Path(filename).name == ("ccw_prop.dae" if HANDEDNESS[index - 1] == 1 else "cw_prop.dae")


def test_reverse_stop_and_independent_reset():
    state = ThrusterDisplayState(2, "cpu")
    force = torch.tensor([[0.0, 0.0, -20.0], [0.0, 0.0, 20.0]])
    for _ in range(50):
        state.update(force, torch.zeros_like(force), 1 / 120)
    torch.testing.assert_close(state.omega[0], -state.omega[1])
    assert state.omega[0, 4] > 0 > state.omega[0, 5]
    previous = state.phase[1].clone()
    state.reset(torch.tensor([0]))
    assert not state.phase[0].any()
    torch.testing.assert_close(state.phase[1], previous)
    for _ in range(300):
        state.update(torch.zeros_like(force), torch.zeros_like(force), 1 / 120)
    assert state.omega.abs().max() < 1e-7
