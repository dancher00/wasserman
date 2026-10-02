"""Bake the complete pinned Blue Collada scenes into an isolated metre-scale asset.

CPU only. Does not copy the restricted Reach arm meshes: the URDF refers to the
existing local installation. Generated Blue meshes retain the upstream MIT notice.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from wasman.bluerov2_registered import (
    ASSET_DIR,
    ASSET_VERSION,
    BLUE_REVISION,
    NATIVE_DIRECTIONS,
    NATIVE_POSITIONS,
    NATIVE_RPY,
    SOURCE_CAMERA_MOUNT,
    SOURCE_DIR,
    SOURCE_SHA256,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transformed_components(scene, post_transform):
    """Retain every scene instance, with complete node matrices before post-scale."""
    meshes, records = [], []
    for index, node in enumerate(scene.graph.nodes_geometry):
        transform, key = scene.graph[node]
        transform = np.asarray(post_transform) @ transform
        mesh = scene.geometry[key].copy()
        mesh.apply_transform(transform)  # Correctly flips winding for mirrored nodes.
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is None:
            mesh.visual.uv = np.zeros((len(mesh.vertices), 2))
        mesh.metadata["name"] = f"component_{index:03d}"
        meshes.append(mesh)
        records.append(
            {
                "node": node,
                "geometry": key,
                "matrix_column_vector": transform.tolist(),
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.faces),
                "mirrored": bool(np.linalg.det(transform[:3, :3]) < 0),
                "bounds_m": mesh.bounds.tolist(),
            }
        )
    if not meshes:
        raise ValueError("Source scene has no mesh instances")
    return meshes, records


def export_verified(meshes, path):
    """Audit exported OBJ vertices, triangles and instance count independently."""
    import tinyobjloader

    path.parent.mkdir(parents=True, exist_ok=True)
    payload, materials = trimesh.exchange.obj.export_obj(
        trimesh.Scene(meshes),
        include_normals=True,
        return_texture=True,
        mtl_name=path.stem + ".mtl",
        digits=10,
    )
    path.write_text(payload)
    outputs = [path]
    for name, data in materials.items():
        target = path.parent / name
        target.write_bytes(data)
        outputs.append(target)
    reader = tinyobjloader.ObjReader()
    if not reader.ParseFromFile(str(path)):
        raise RuntimeError(reader.Error())
    actual = np.asarray(reader.GetAttrib().vertices).reshape(-1, 3)
    expected = np.concatenate([m.vertices for m in meshes])
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=3e-8, rtol=0):
        raise RuntimeError("OBJ vertex/transform mismatch")
    shapes = reader.GetShapes()
    if len(shapes) != len(meshes) or sum(len(s.mesh.num_face_vertices) for s in shapes) != sum(
        len(m.faces) for m in meshes
    ):
        raise RuntimeError("OBJ component/triangle loss")
    # Independent face orientation/index audit, not merely bounds.
    expected_faces, offset = [], 0
    for mesh in meshes:
        expected_faces.extend(mesh.faces + offset)
        offset += len(mesh.vertices)
    actual_faces = np.asarray([i.vertex_index for s in shapes for i in s.mesh.indices]).reshape(-1, 3)
    if not np.array_equal(actual_faces, expected_faces):
        raise RuntimeError("OBJ triangle winding/index mismatch")
    return outputs


def build_collision_hulls(meshes, target):
    """Per-connected-solid convex envelopes, not one hull sealing the whole frame.

    Small holes within each native solid are conservatively filled; frame-level
    gaps are retained. Every source vertex must be enclosed by its own hull.
    """
    from scipy.spatial import ConvexHull

    surface = trimesh.util.concatenate(meshes)
    surface.merge_vertices(merge_tex=True, merge_norm=True)
    parts = surface.split(only_watertight=False)
    records = []
    for index, part in enumerate(parts):
        if len(part.vertices) < 4 or np.linalg.matrix_rank(part.vertices - part.vertices.mean(0), tol=1e-9) < 3:
            records.append(
                {
                    "source_component": index,
                    "source_bounds_m": part.bounds.tolist(),
                    "source_vertices": len(part.vertices),
                    "source_triangles": len(part.faces),
                    "omitted_reason": "zero-volume isolated CAD surface, visual retained",
                }
            )
            continue
        hull = part.convex_hull
        equations = ConvexHull(hull.vertices).equations
        error = float(np.max(part.vertices @ equations[:, :3].T + equations[:, 3]))
        if error > 1e-8:
            raise RuntimeError("Collision proxy does not contain original component")
        relative = f"collision/part_{index:03d}.obj"
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        hull.visual = trimesh.visual.ColorVisuals(mesh=hull)
        hull.export(path, file_type="obj", include_normals=False, include_color=False)
        records.append(
            {
                "path": relative,
                "sha256": sha(path),
                "source_component": index,
                "source_vertices": len(part.vertices),
                "source_triangles": len(part.faces),
                "source_bounds_m": part.bounds.tolist(),
                "hull_bounds_m": hull.bounds.tolist(),
                "vertices": len(hull.vertices),
                "triangles": len(hull.faces),
                "max_vertex_outside_distance_m": error,
            }
        )
    return records


def prepare(target):
    if target.resolve() == SOURCE_DIR.resolve():
        raise ValueError("Never overwrite the source asset")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Output directory must be absent or empty; preserve versioned builds")
    target.mkdir(parents=True, exist_ok=True)
    source_urdf = SOURCE_DIR / "bluerov2_alpha.urdf"
    root = ET.parse(source_urdf).getroot()
    root.set("name", ASSET_VERSION)
    report = {
        "schema_version": 1,
        "variant": ASSET_VERSION,
        "urdf": f"{ASSET_VERSION}.urdf",
        "thruster_positions_m": NATIVE_POSITIONS,
        "status": "CPU geometry preparation only; not an Isaac or policy validation",
        "source_urdf_sha256": sha(source_urdf),
        "blue_revision": BLUE_REVISION,
        "body_rule": "p_base = 0.025 * (T_scene_node * p_raw); origin zero; mesh scale one",
        "base_frame_changed": False,
        "source_arm_mount_changed": False,
        "mass_inertia_or_hydrodynamics_changed": False,
        "motor_positions_base_m": NATIVE_POSITIONS,
        "motor_directions_base": NATIVE_DIRECTIONS,
        "motor_rpy_rad": NATIVE_RPY,
        "motor_angle_note": "Native rounded RPY angles idealized to exact pi/2 and pi/3, matching legacy axes",
        "camera_mount_base_m": SOURCE_CAMERA_MOUNT,
        "camera_note": "Upstream integration mount, not manufacturer-calibrated optical extrinsic",
        "collision_note": "Connected-solid convex hulls; local holes filled, separate-solid gaps preserved",
        "restricted_arm_meshes_copied": False,
        "versions": {n: importlib.metadata.version(n) for n in ("trimesh", "pycollada", "tinyobjloader", "numpy")},
        "sources": [],
        "derived_visuals": [],
        "files": [],
    }
    for mesh in root.findall(".//mesh"):
        original = (SOURCE_DIR / mesh.get("filename")).resolve()
        # Relative references are relocatable within the installed repository.
        import os

        mesh.set("filename", os.path.relpath(original, target))
    for name, expected in SOURCE_SHA256.items():
        source = SOURCE_DIR / "meshes/blue" / name
        if sha(source) != expected:
            raise RuntimeError(f"Pinned source hash mismatch: {source}")
        scene = trimesh.load(source, force="scene", process=False)
        body = name.startswith("bluerov2")
        post = (
            np.diag([0.025, 0.025, 0.025, 1])
            if body
            else trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
        )
        meshes, components = transformed_components(scene, post)
        relative = "meshes/" + source.stem + ".obj"
        outputs = export_verified(meshes, target / relative)
        report["files"].extend({"path": str(p.relative_to(target)), "sha256": sha(p)} for p in outputs)
        report["sources"].append({"path": str(source.relative_to(SOURCE_DIR)), "sha256": expected})
        report["derived_visuals"].append(
            {
                "source": name,
                "path": relative,
                "components": components,
                "bounds_m": trimesh.Scene(meshes).bounds.tolist(),
                "vertices_verified": True,
                "instances_verified": True,
                "faces_winding_verified": True,
                "post_transform": post.tolist(),
            }
        )
        if body:
            visual = root.find("./link[@name='base_link']/visual[@name='bluerov2_heavy_visual']")
            visual.find("origin").set("xyz", "0 0 0")
            visual.find("geometry/mesh").set("filename", relative)
            visual.find("geometry/mesh").set("scale", "1 1 1")
            report["collision_components"] = build_collision_hulls(meshes, target)
        else:
            for visual in root.findall("./link[@name='base_link']/visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is not None and Path(mesh.get("filename")).name == name:
                    mesh.set("filename", relative)
                    mesh.set("scale", "1 1 1")
    base = root.find("./link[@name='base_link']")
    for index, (position, rpy) in enumerate(zip(NATIVE_POSITIONS, NATIVE_RPY, strict=True), 1):
        origin = base.find(f"visual[@name='thruster{index}_visual']/origin")
        origin.set("xyz", " ".join(map(str, position)))
        origin.set("rpy", " ".join(map(str, rpy)))
    for collision in base.findall("collision"):
        base.remove(collision)
    for item in report["collision_components"]:
        if "path" not in item:
            continue
        collision = ET.SubElement(base, "collision", name=f"body_component_{item['source_component']:03d}")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=item["path"], scale="1 1 1")
    # Do not inherit the old XML comment claiming broad redistribution rights.
    ET.indent(root, space="  ")
    urdf = target / f"{ASSET_VERSION}.urdf"
    ET.ElementTree(root).write(urdf, encoding="utf-8", xml_declaration=True)
    report["generated_urdf_sha256"] = sha(urdf)
    for name in ("BLUE_LICENSE.txt", "REACH_MESH_LICENSE.txt"):
        (target / name).write_bytes((SOURCE_DIR / name).read_bytes())
    report["passed"] = True
    (target / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    (target / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSET_DIR)
    args = parser.parse_args()
    result = prepare(args.output_dir)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output_dir),
                "collision_hulls": len(result["collision_components"]),
            },
            indent=2,
        )
    )
