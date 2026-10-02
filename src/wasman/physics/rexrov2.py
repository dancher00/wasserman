"""Native RexROV2 held-arm hydrodynamic prototype; independent of BlueROV.

FLU body coordinates, XYZW quaternions, spatial twist [v, omega]. The source
Fossen plugin explicitly flips Y/Z before/after evaluating its FRD matrix.
Full added inertia is closed algebraically, never delayed acceleration feedback.
"""

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation

from wasman.physics.hydrodynamics import quat_apply_inverse_xyzw

ASSET = Path(__file__).parents[1] / "assets/data/robots/rexrov2_oberon7"
SOURCE = Path(__file__).parent / "data/rexrov2"
XACRO = "{http://www.ros.org/wiki/xacro}"
FRD_TO_FLU = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def cross_bias(matrix: torch.Tensor, twist: torch.Tensor) -> torch.Tensor:
    """Spatial Coriolis product C(nu)nu; nu dot C(nu)nu is zero."""
    momentum = twist @ matrix.T
    velocity, omega = twist[..., :3], twist[..., 3:]
    linear, angular = momentum[..., :3], momentum[..., 3:]
    return torch.cat(
        (
            torch.cross(omega, linear, dim=-1),
            torch.cross(velocity, linear, dim=-1) + torch.cross(omega, angular, dim=-1),
        ),
        dim=-1,
    )


@dataclass
class RexParameters:
    names: list[str]
    mass: np.ndarray
    com: np.ndarray
    volume: np.ndarray
    cob: np.ndarray
    quadratic_arm: np.ndarray
    rigid_composite: np.ndarray
    added_mass_source_frd: np.ndarray
    added_mass: np.ndarray
    linear_base: np.ndarray
    quadratic_base: np.ndarray
    thruster_positions: np.ndarray
    thruster_directions: np.ndarray
    allocation: np.ndarray
    max_thrust: float
    rotor_constant: float
    motor_time_constant: float
    water_density: float = 1028.0
    gravity: float = 9.81


def load_parameters() -> RexParameters:
    """Native URDF inertias and source volumes; explicit gripper/drag estimates."""
    urdf = ET.parse(ASSET / "rexrov2_oberon7.urdf").getroot()
    model = ET.parse(ASSET / "source/rexrov2.gazebo.xacro").getroot()
    properties = {p.get("name"): p.get("value") for p in model.findall(f"{XACRO}property")}
    native_added = np.fromstring(model.find(".//added_mass").text, sep=" ").reshape(6, 6)
    # Source (2,5)=10.774 versus (5,2)=10.775 is a 0.001 transcription asymmetry.
    added = FRD_TO_FLU @ ((native_added + native_added.T) / 2) @ FRD_TO_FLU
    values = {}
    for file in ("serial_arm.xacro", "parallel_gripper.xacro"):
        root = ET.parse(ASSET / "source/oberon7_description/urdf/parameters" / file).getroot()
        for prop in root.findall(f"{XACRO}property"):
            with suppress(ValueError):
                values[prop.get("name")] = float(prop.get("value"))
    kinds = (
        "base",
        "shoulder_link",
        "upper_arm",
        "elbow_link",
        "forearm",
        "wrist",
        "gripper_base",
        "finger",
        "finger",
        "finger_tip",
        "finger_tip",
    )
    boxes = {
        kind: np.array([values[f"{kind}_length"], values[f"{kind}_width"], values[f"{kind}_height"]]) for kind in kinds
    }
    total_box = sum(np.prod(boxes[kind]) for kind in kinds)
    link_kinds = {
        "oberon_" + name: kind
        for name, kind in zip(
            (
                "base",
                "shoulder_link",
                "upper_arm",
                "elbow_link",
                "forearm",
                "wrist_link",
                "end_effector",
                "finger_left",
                "finger_right",
                "finger_tip_left",
                "finger_tip_right",
            ),
            kinds,
            strict=True,
        )
    }
    transforms = {"base_link": np.eye(4)}
    pending = list(urdf.findall("joint"))
    while pending:
        progress = False
        for joint in list(pending):
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in transforms:
                continue
            transform = np.eye(4)
            origin = joint.find("origin")
            if origin is not None:
                transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                transform[:3, :3] = Rotation.from_euler(
                    "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                ).as_matrix()
            transforms[child] = transforms[parent] @ transform
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError("Disconnected URDF")
    names, masses, centers, volumes, buoyancy_centers, arm_drag = [], [], [], [], [], []
    composite = np.zeros((6, 6))
    for link in urdf.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").get("value"))
        center = np.fromstring(inertial.find("origin").get("xyz", "0 0 0"), sep=" ")
        orientation = Rotation.from_euler(
            "xyz", np.fromstring(inertial.find("origin").get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        i = inertial.find("inertia")
        inertia = np.array(
            [
                [float(i.get("ixx")), float(i.get("ixy")), float(i.get("ixz"))],
                [float(i.get("ixy")), float(i.get("iyy")), float(i.get("iyz"))],
                [float(i.get("ixz")), float(i.get("iyz")), float(i.get("izz"))],
            ]
        )
        rotation = transforms[name][:3, :3]
        r = transforms[name][:3, 3] + rotation @ center
        block = np.zeros((6, 6))
        block[:3, :3] = mass * np.eye(3)
        block[:3, 3:], block[3:, :3] = -mass * skew(r), mass * skew(r)
        block[3:, 3:] = rotation @ orientation @ inertia @ orientation.T @ rotation.T - mass * skew(r) @ skew(r)
        composite += block
        if name == "base_link":
            volume = float(properties["volume"])
            cob = np.fromstring(properties["cob"], sep=" ")
            damping = np.zeros(6)
        else:
            kind = link_kinds[name]
            dimensions = boxes[kind]
            if kind in kinds[:6]:
                volume = values["total_volume"] * np.prod(dimensions) / total_box
                cob = np.zeros(3)  # native UUV BuoyantObject default, in link frame
            else:
                volume, cob = mass / 2700.0, center.copy()  # explicit missing-gripper assumption
            area = np.prod(dimensions) / dimensions
            # Cd=1 bluff-body estimate, not calibrated cylinder/arm hydrodynamics.
            damping = np.r_[0.5 * 1028 * area, 0.5 * 1028 * area * (dimensions.max() / 2) ** 3]
        names.append(name)
        masses.append(mass)
        centers.append(center)
        volumes.append(volume)
        buoyancy_centers.append(cob)
        arm_drag.append(damping)
    positions, directions = [], []
    for visual in urdf.find("link[@name='base_link']").findall("visual"):
        if not visual.get("name", "").startswith("thruster_"):
            continue
        origin = visual.find("origin")
        positions.append(np.fromstring(origin.get("xyz"), sep=" "))
        angles = np.fromstring(origin.get("rpy"), sep=" ")
        directions.append(Rotation.from_euler("xyz", angles).apply([1, 0, 0]))
    positions, directions = np.array(positions), np.array(directions)
    allocation = np.r_[directions.T, np.cross(positions, directions).T]
    motors = yaml.safe_load((SOURCE / "thruster_manager.yaml").read_text())["thruster_manager"]
    snippet = ET.parse(SOURCE / "rexrov2_snippets.xacro").getroot()
    motor = snippet.find(f".//{XACRO}thruster_module_first_order_basic_fcn_macro")
    return RexParameters(
        names,
        np.array(masses),
        np.array(centers),
        np.array(volumes),
        np.array(buoyancy_centers),
        np.array(arm_drag),
        composite,
        native_added,
        added,
        -np.fromstring(model.find(".//linear_damping").text, sep=" "),
        -np.fromstring(model.find(".//quadratic_damping").text, sep=" "),
        positions,
        directions,
        allocation,
        float(motors["max_thrust"]),
        float(motor.get("rotor_constant")),
        float(motor.get("dyn_time_constant")),
    )


class RexThrusters:
    """Six native axes, bounded allocation and original first-order rotor lag."""

    def __init__(self, parameters: RexParameters, num_envs: int, dt: float, device):
        self.parameters, self.dt = parameters, dt
        self.matrix = torch.tensor(parameters.allocation, dtype=torch.float32, device=device)
        self.inverse = torch.linalg.inv(self.matrix)
        self.omega = torch.zeros(num_envs, 6, device=device)
        self.force = torch.zeros_like(self.omega)
        self.scale = torch.ones(num_envs, 1, device=device)

    def step(self, wrench):
        requested = wrench @ self.inverse.T
        self.scale = (self.parameters.max_thrust / requested.abs().clamp_min(1e-6)).amin(-1, keepdim=True).clamp(max=1)
        target = requested * self.scale
        desired_omega = target.sign() * (target.abs() / self.parameters.rotor_constant).sqrt()
        alpha = 1 - np.exp(-self.dt / self.parameters.motor_time_constant)
        self.omega.lerp_(desired_omega, float(alpha))
        self.force = self.parameters.rotor_constant * self.omega.abs() * self.omega
        return self.force @ self.matrix.T


class HeldArmRexHydrodynamics:
    """Explicitly approximate held-arm closure; no claims for moving-arm dynamics."""

    def __init__(self, parameters: RexParameters, device):
        self.parameters = parameters

        def tensor(value):
            return torch.tensor(value, dtype=torch.float32, device=device)

        self.rigid = tensor(parameters.rigid_composite)
        self.added = tensor(parameters.added_mass)
        self.effective_inverse = torch.linalg.inv(self.rigid + self.added)
        self.mass, self.volume = tensor(parameters.mass), tensor(parameters.volume)
        self.com, self.cob = tensor(parameters.com), tensor(parameters.cob)
        self.arm_quadratic = tensor(parameters.quadratic_arm)
        self.linear_base, self.quadratic_base = tensor(parameters.linear_base), tensor(parameters.quadratic_base)

    def local_fluid(self, twist_relative, quaternion):
        """Per-link COM wrenches excluding base added inertia and Coriolis."""
        up = torch.zeros_like(twist_relative[..., :3])
        up[..., 2] = self.volume * self.parameters.water_density * self.parameters.gravity
        buoyancy = quat_apply_inverse_xyzw(quaternion, up)
        torque = torch.cross((self.cob - self.com).expand_as(buoyancy), buoyancy, dim=-1)
        fluid = torch.cat((buoyancy, torque), dim=-1)
        fluid -= self.arm_quadratic * twist_relative.abs() * twist_relative
        fluid[:, 0] -= self.linear_base * twist_relative[:, 0]
        fluid[:, 0] -= self.quadratic_base * twist_relative[:, 0].abs() * twist_relative[:, 0]
        return fluid

    def close_added_mass(self, body_twist, current_b, non_added_external):
        """External includes realized motors, gravity, buoyancy, drag and pulses.

        M_RB nu_dot + C_RB nu = tau_ext - M_A nu_rel_dot - C_A nu_rel.
        For steady world current, nu_rel_dot = nu_dot + [omega x current_b, 0].
        """
        relative = body_twist.clone()
        relative[..., :3] -= current_b
        current_derivative = torch.cat(
            (torch.cross(body_twist[..., 3:], current_b, dim=-1), torch.zeros_like(current_b)), dim=-1
        )
        added_bias = cross_bias(self.added, relative)
        rhs = non_added_external - cross_bias(self.rigid, body_twist) - added_bias - current_derivative @ self.added.T
        acceleration = rhs @ self.effective_inverse.T
        added_wrench = -(acceleration + current_derivative) @ self.added.T - added_bias
        return added_wrench, acceleration
