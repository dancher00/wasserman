"""CPU invariants for the separate, explicitly approximate Rex dynamics."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from wasman.physics.rexrov2 import (
    FRD_TO_FLU,
    SOURCE,
    HeldArmRexHydrodynamics,
    RexThrusters,
    cross_bias,
    load_parameters,
)


@pytest.fixture
def parameters():
    return load_parameters()


def test_source_evidence_is_pinned_and_licensed():
    provenance = json.loads((SOURCE / "provenance.json").read_text())
    assert provenance["license"] == "Apache-2.0"
    for entry in provenance["files"]:
        assert hashlib.sha256((SOURCE / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        assert any(revision in entry["url"] for revision in provenance["revisions"].values())
    source = (SOURCE / "HydrodynamicModel.cc").read_text()
    assert "output.Y() = -1 * output.Y();" in source
    assert "output.Z() = -1 * output.Z();" in source
    assert "this->ToNED(linVel - flowVel)" in source
    assert "this->ToNED(angVel)" in source
    assert "this->FromNED(Vec3dToGazebo(tau.head<3>()))" in source
    assert "this->FromNED(Vec3dToGazebo(tau.tail<3>()))" in source
    assert "Eigen::Vector6d added = -this->GetAddedMass() * this->filteredAcc;" in source


def test_native_composite_and_full_added_inertia(parameters):
    p = parameters
    assert p.mass[0] == 1862.87
    assert p.mass.sum() == pytest.approx(1921.9462210630736)
    assert np.allclose(p.rigid_composite[:3, :3], np.eye(3) * p.mass.sum())
    assert np.linalg.eigvalsh(p.rigid_composite).min() > 0
    assert np.linalg.eigvalsh(p.added_mass).min() > 0
    assert np.allclose(p.added_mass, p.added_mass.T)
    source = p.added_mass_source_frd
    assert np.max(np.abs(source - source.T)) == pytest.approx(0.001)
    assert np.allclose(p.added_mass, FRD_TO_FLU @ ((source + source.T) / 2) @ FRD_TO_FLU)
    assert p.added_mass[0, 2] == 103.32
    assert p.added_mass[0, 4] == 165.54
    assert p.added_mass[1, 3] == -409.44
    assert np.count_nonzero(p.added_mass - np.diag(np.diag(p.added_mass))) == 30


def test_buoyancy_and_explicit_gripper_estimates(parameters):
    p = parameters
    assert p.volume[0] == 1.83826
    assert np.allclose(p.cob[0], [0, 0, 0.3])
    assert p.volume.sum() == pytest.approx(1.85590086581329)
    for index, name in enumerate(p.names):
        if "finger" in name or name == "oberon_end_effector":
            assert p.volume[index] == pytest.approx(p.mass[index] / 2700)
            assert np.allclose(p.cob[index], p.com[index])
    hydro = HeldArmRexHydrodynamics(p, "cpu")
    quat = torch.tensor(Rotation.from_euler("x", 0.1).as_quat(), dtype=torch.float32)
    q = quat.expand(1, len(p.names), 4)
    fluid = hydro.local_fluid(torch.zeros(1, len(p.names), 6), q)
    assert fluid[0, 0, 3] < 0  # positive roll gets restoring negative roll torque
    assert fluid[0, 0, 2] > 0


def test_drag_dissipates_relative_motion(parameters):
    hydro = HeldArmRexHydrodynamics(parameters, "cpu")
    torch.manual_seed(31)
    twist = torch.randn(9, len(parameters.names), 6)
    quaternion = torch.tensor([0, 0, 0, 1.0]).expand(9, len(parameters.names), 4)
    drag = hydro.local_fluid(twist, quaternion) - hydro.local_fluid(torch.zeros_like(twist), quaternion)
    assert (torch.sum(drag * twist, dim=-1) <= 0).all()


def test_coriolis_no_power_and_algebraic_added_closure(parameters):
    hydro = HeldArmRexHydrodynamics(parameters, "cpu")
    torch.manual_seed(13)
    twist, current, external = torch.randn(12, 6), torch.randn(12, 3), torch.randn(12, 6) * 300
    bias = cross_bias(hydro.added, twist)
    assert torch.allclose((bias * twist).sum(-1), torch.zeros(12), atol=0.002)
    added, acceleration = hydro.close_added_mass(twist, current, external)
    residual = acceleration @ hydro.rigid.T + cross_bias(hydro.rigid, twist) - external - added
    assert residual.abs().max() < 0.004
    relative = twist.clone()
    relative[:, :3] -= current
    derivative = torch.cat((torch.cross(twist[:, 3:], current, dim=-1), torch.zeros_like(current)), dim=-1)
    expected = -(acceleration + derivative) @ hydro.added.T - cross_bias(hydro.added, relative)
    assert torch.allclose(added, expected)


def test_added_mass_slows_acceleration_without_delayed_feedback(parameters):
    hydro = HeldArmRexHydrodynamics(parameters, "cpu")
    force = torch.tensor([[0, 0, 1000, 0, 0, 0.0]])
    added, acceleration = hydro.close_added_mass(torch.zeros(1, 6), torch.zeros(1, 3), force)
    rigid_only = torch.linalg.solve(hydro.rigid, force.T).T
    assert 0 < acceleration[0, 2] < rigid_only[0, 2]
    assert added[0, 2] < 0
    assert acceleration[0, [0, 1, 3, 4, 5]].abs().max() > 0.001


def test_native_thruster_allocation_and_lag(parameters):
    p = parameters
    source = np.array(yaml.safe_load((SOURCE / "TAM.yaml").read_text())["tam"])
    assert np.allclose(p.allocation, source, atol=2e-6)
    assert np.linalg.matrix_rank(p.allocation) == 6
    assert np.allclose(np.linalg.norm(p.thruster_directions, axis=-1), 1)
    assert np.allclose(p.allocation[3:].T, np.cross(p.thruster_positions, p.thruster_directions))
    assert p.max_thrust == 1540 and p.motor_time_constant == 0.05
    motors = RexThrusters(p, 2, 1 / 120, "cpu")
    command = torch.tensor([[1e5, 1e5, -1e5, 1e5, 0, 0], [0, 0, 0, 0, 0, 0.0]])
    motors.step(command)
    assert motors.force.abs().max() < p.max_thrust / 10
    for _ in range(200):
        realized = motors.step(command)
        assert motors.force.abs().max() <= p.max_thrust + 0.001
    assert torch.allclose(realized, command * motors.scale, atol=0.003)
    assert torch.count_nonzero(motors.force[1]) == 0


def test_saved_free_floating_record_replays():
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/rexrov2_stationkeeping_v1"
    replay = runpy.run_path(str(root / "scripts/audit_rexrov2_stationkeeping.py"))["audit"](folder)
    assert replay["passed"]
    assert replay == json.loads((folder / "replay_audit.json").read_text())
    assert replay["trace_samples"] == 600
    assert replay["max_recorded_motor_force_N"] < 1540


def test_separate_physx_import_audit(parameters):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "artifacts/rexrov2_import_audit_v1/import_audit.json").read_text())
    assert report["passed"] and not report["fixed_base"]
    assert report["mass_matches_native_urdf"] and report["com_matches_native_urdf"]
    assert report["base_com_at_link_origin"]
    assert np.allclose(report["actual_physx_mass_kg"], parameters.mass[None], rtol=2e-6, atol=1e-6)
    assert np.allclose(report["actual_physx_com_in_link_m"], parameters.com[None], atol=1e-6)
    assert np.allclose(report["nominal_held_arm_rigid_inertia"], parameters.rigid_composite)
