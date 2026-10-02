#!/usr/bin/env python3
"""CPU camera-envelope checks along an existing, unmodified arm recording."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from wasman.robot_camera_profiles import WRIST_AUDIT_ENVELOPE_RADIUS_M, robot_camera_profile

ARM_JOINTS = ("alpha_axis_e", "alpha_axis_d", "alpha_axis_c", "alpha_axis_b")
ROOT = Path(__file__).resolve().parents[1]


def source_path(file):
    return str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)


def transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return result


def link_transforms(root, rows):
    count = len(rows)
    result = {"base_link": np.broadcast_to(np.eye(4), (count, 4, 4)).copy()}
    pending = list(root.findall("joint"))
    positions = np.array([row["arm_joint_position_rad"] for row in rows])
    if positions.shape != (count, 4) or not np.isfinite(positions).all():
        raise ValueError("Camera clearance requires finite four-joint telemetry")
    while pending:
        progressed = False
        for joint in pending[:]:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in result:
                continue
            motion = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
            if joint.get("name") in ARM_JOINTS:
                angle = positions[:, ARM_JOINTS.index(joint.get("name"))]
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:, :3, :3] = Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
            result[child] = result[parent] @ transform(joint.find("origin")) @ motion
            pending.remove(joint)
            progressed = True
        if not progressed:
            raise ValueError("Disconnected robot kinematic tree")
    return result


def proximity(mesh, points, radius):
    """Nearest visual triangle and optional convex-solid inclusion, no AABB-only pass."""
    _, distances, _ = trimesh.proximity.closest_point(mesh, points)
    nearest = int(np.argmin(distances))
    # Collision proxies are convex; oriented plane inequalities avoid ray parity.
    if mesh.is_convex and mesh.is_watertight:
        plane_distance = np.einsum("pfi,fi->pf", points[:, None, :] - mesh.triangles_center, mesh.face_normals)
        inside = np.all(plane_distance < -1e-8, axis=1)
    elif mesh.is_watertight:
        inside = mesh.contains(points)
    else:
        inside = None
    return {
        "minimum_surface_distance_m": float(distances[nearest]),
        "minimum_envelope_clearance_m": float(distances[nearest] - radius),
        "nearest_frame": nearest,
        "surface_envelope_overlap_frames": np.flatnonzero(distances < radius).tolist(),
        "watertight": bool(mesh.is_watertight),
        "inside_solid_frames": None if inside is None else np.flatnonzero(inside).tolist(),
    }


def audit(urdf, trace, profile_name):
    data = json.loads(trace.read_text())
    rows = data.get("trace", data.get("frames"))
    root = ET.parse(urdf).getroot()
    poses = link_transforms(root, rows)
    profile = robot_camera_profile(profile_name)
    collision_files = sorted((urdf.parent / "collision").glob("part_*.obj"))
    if not collision_files:
        raise ValueError("Registered hull collision components are required for a complete clearance audit")
    results = {}
    for camera_name, mount in (("base", profile.base), ("gripper", profile.gripper)):
        parent = poses[mount.parent_link]
        centers = parent[:, :3, :3] @ np.array(mount.position_m) + parent[:, :3, 3]
        radius = WRIST_AUDIT_ENVELOPE_RADIUS_M if camera_name == "gripper" else 0.0
        checks = {}
        for link in root.findall("link"):
            name = link.get("name")
            # Body source mesh can contain cavities and decorative open surfaces.
            # Check it once; native rotor animation is not reconstructed here.
            for i, visual in enumerate(link.findall("visual")):
                source = visual.find("geometry/mesh")
                if source is None or (name == "base_link" and i > 0):
                    continue
                file = (urdf.parent / source.get("filename")).resolve()
                mesh = trimesh.load(file, force="mesh", process=True)
                mesh.apply_scale(np.fromstring(source.get("scale", "1 1 1"), sep=" "))
                mesh.apply_transform(transform(visual.find("origin")))
                pose = poses[name]
                local = np.einsum("pji,pj->pi", pose[:, :3, :3], centers - pose[:, :3, 3])
                checks[f"{name}/visual{i}"] = proximity(mesh, local, radius)
                checks[f"{name}/visual{i}"]["source"] = source_path(file)
                checks[f"{name}/visual{i}"]["source_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        collisions = []
        for file in collision_files:
            mesh = trimesh.load(file, force="mesh", process=True)
            candidate = ((centers >= mesh.bounds[0] - radius) & (centers <= mesh.bounds[1] + radius)).all(axis=1)
            if not candidate.any():
                continue
            indices = np.flatnonzero(candidate)
            check = proximity(mesh, centers[indices], radius)
            if check["inside_solid_frames"] or check["surface_envelope_overlap_frames"]:
                check["nearest_frame"] = int(indices[check["nearest_frame"]])
                for key in ("inside_solid_frames", "surface_envelope_overlap_frames"):
                    if check[key] is not None:
                        check[key] = indices[check[key]].tolist()
                collisions.append({"source": source_path(file.resolve()), **check})
        results[camera_name] = {
            "envelope_radius_m": radius,
            "visual_surfaces": checks,
            "hull_collision_conflicts": collisions,
        }
    return {
        "profile": profile.metadata(),
        "trace": str(trace),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "urdf": str(urdf),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "asset_manifest_sha256": hashlib.sha256((urdf.parent / "manifest.json").read_bytes()).hexdigest(),
        "camera_profile_source_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/wasman/robot_camera_profiles.py").read_bytes()
        ).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frames": len(rows),
        "hull_collision_components_checked": len(collision_files),
        "first_arm_pose_rad": rows[0]["arm_joint_position_rad"],
        "assumptions": (
            "Measured four arm joints; unavailable finger telemetry uses nominal closed pose. "
            "Rigid base coordinates remove common world motion. Wrist envelope is an assumed 10mm sphere, "
            "not a manufactured camera. Base lens has zero envelope, inside its source housing pivot. "
            "Open/non-watertight visual surfaces have no reliable inside/outside certificate. "
            "Convex collision interiors may fill cavities: report them, do not equate them to CAD material. "
            "No dynamic rotor-phase reconstruction or contact-physics claim."
        ),
        "cameras": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=Path("website/public/static/press_button_camera_trace.json"))
    parser.add_argument("--profile", default="geometry-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.urdf, args.trace, args.profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
