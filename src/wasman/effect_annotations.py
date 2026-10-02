"""Project measured world geometry into the actual recorded USD camera.

Screen positions are not hand-placed. UI labels may offset from these anchors,
but must not change the underlying physical distances or arrow lengths.
"""

import numpy as np


def project_point(point, view_projection, viewport):
    clip = np.r_[point, 1.0] @ np.asarray(view_projection)
    if not np.isfinite(clip).all() or clip[3] <= 1e-8:
        return [0.0, 0.0], False
    ndc = clip[:3] / clip[3]
    return [(ndc[0] + 1) * viewport[0] / 2, (1 - ndc[1]) * viewport[1] / 2], bool(-1 <= ndc[2] <= 1)


def project_trajectory(points, matrix, viewport):
    """Project actual world-space samples; validity is per vertex, not a fake path.

    Pixels outside XY viewport may be clipped by the SVG viewport. Points behind
    the camera or outside its depth interval are individually invalid, so the UI
    must not connect a path across those vertices.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Trajectory contains a non-finite world point")
    projected = [project_point(point, matrix, viewport) for point in points]
    pixels, valid = ([item[index] for item in projected] for index in (0, 1))
    return {
        "points_w_m": points.tolist(),
        "points_px": pixels,
        "valid": valid,
        "current_px": pixels[-1] if pixels else None,
        "current_valid": valid[-1] if valid else False,
    }


def _project_path(points, matrix, viewport, *, closed=False):
    projected = project_trajectory(points, matrix, viewport)
    return {key: projected[key] for key in ("points_w_m", "points_px", "valid")} | {"closed": closed}


def _convex_hull_indices(points):
    """Monotone-chain silhouette of projected 3-D hemisphere samples."""
    ordered = sorted(range(len(points)), key=lambda index: tuple(points[index]))

    def turn(a, b, c):
        first, second = points[b] - points[a], points[c] - points[a]
        return first[0] * second[1] - first[1] * second[0]

    lower, upper = [], []
    for chain, sequence in ((lower, ordered), (upper, reversed(ordered))):
        for index in sequence:
            while len(chain) >= 2 and turn(chain[-2], chain[-1], index) <= 0:
                chain.pop()
            chain.append(index)
    result = lower[:-1] + upper[:-1]
    return result + result[:1]


def project_hemisphere(center, normal, matrix, viewport, *, radius_m=0.045):
    """Perspective paths from a real 3-D hemisphere, not a screen-space circle.

    This fixed-radius schematic is NOT a computed jet footprint. The base ring
    lies in the physical surface and the dome points into water. A convex hull
    of dense projected surface samples approximates its silhouette; meridians
    and ring use explicit 3-D circular arcs. No camera-dependent world rescaling.
    """
    center, normal = np.asarray(center, dtype=float), np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite([*center, *normal, radius_m]).all() or norm < 1e-8 or radius_m <= 0:
        raise ValueError("Hemisphere requires finite center, nonzero normal and positive radius")
    normal = normal / norm
    reference = np.array([1, 0, 0]) if abs(normal[0]) < 0.8 else np.array([0, 1, 0])
    tangent = np.cross(normal, reference)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    azimuth = np.linspace(0, 2 * np.pi, 25)
    ring_vectors = np.cos(azimuth)[:, None] * tangent + np.sin(azimuth)[:, None] * bitangent
    ring = center + radius_m * ring_vectors
    elevation = np.linspace(0, np.pi / 2, 9)
    surface = center + radius_m * (
        np.cos(elevation)[:, None, None] * ring_vectors[None, :-1] + np.sin(elevation)[:, None, None] * normal
    )
    surface = surface.reshape(-1, 3)
    projected = project_trajectory(surface, matrix, viewport)
    # Partial near-plane cuts require a real clipped surface, which this small
    # explanatory overlay deliberately does not pretend to resolve.
    visible = all(projected["valid"])
    indices = _convex_hull_indices(np.asarray(projected["points_px"])) if visible else []
    arc = np.linspace(0, np.pi, 17)
    meridians = [
        center + radius_m * (np.cos(arc)[:, None] * axis + np.sin(arc)[:, None] * normal)
        for axis in (tangent, bitangent)
    ]
    return {
        "radius_m": float(radius_m),
        "center_w_m": center.tolist(),
        "normal_w": normal.tolist(),
        "valid": bool(visible and len(indices) >= 4),
        "silhouette": _project_path(surface[indices], matrix, viewport, closed=True),
        "base_ring": _project_path(ring, matrix, viewport, closed=True),
        "meridians": [_project_path(points, matrix, viewport) for points in meridians],
    }


def build_annotations(
    record, matrix, viewport, mode, *, pool_bounds=None, trajectory_w=None, hemisphere_radius_m=0.045
):
    """Project measured geometry and an explicitly sized schematic cap.

    The radius changes only the explanatory hemisphere, never ray intersections
    or the boundary model. Keep the default for exact archived reproduction.
    """
    if not np.isfinite(hemisphere_radius_m) or hemisphere_radius_m <= 0:
        raise ValueError("Hemisphere radius must be finite and positive")

    def segment(start, end, **fields):
        first, first_valid = project_point(start, matrix, viewport)
        second, second_valid = project_point(end, matrix, viewport)
        return {
            "start_px": first,
            "end_px": second,
            "valid": first_valid and second_valid,
            "start_w_m": list(start),
            "end_w_m": list(end),
            **fields,
        }

    axial = "rotor_axes_w" in record
    result = {
        "schema_version": 2 if axial else 1,
        "viewport_px": list(viewport),
        "rotor_rays": [],
        "current_arrows": [],
        "commanded_line": [],
        "measured_line": [],
    }
    if axial:
        result["rotor_ray_semantics"] = "axial proximity schematic; chosen +/- axis, not signed exhaust or CFD"
        if "gripper_position_w_m" in record:
            current = np.asarray(record["gripper_position_w_m"], dtype=float)
            history = [] if trajectory_w is None else list(trajectory_w)
            if not history or not np.array_equal(history[-1], current):
                history.append(current)
            result["gripper_trajectory"] = project_trajectory(history, matrix, viewport)
        if "gripper_reference_w_m" in record:
            reference = record["gripper_reference_w_m"]
            pixel, valid = project_point(reference, matrix, viewport)
            result["gripper_reference"] = {
                "point_w_m": list(reference),
                "point_px": pixel,
                "valid": valid,
                "label": "Initial tool position",
            }
    actual = np.asarray(record["base_position_w_m"])
    target = np.asarray(record["target_position_w_m"])
    if mode == "current":
        velocity = np.asarray(record["current_w_m_s"])
        result["current_arrow_horizon_s"] = 1.0
        if np.linalg.norm(velocity) > 1e-6:
            for anchor in ((-0.55, -0.20, 0.55), (-0.55, 0.10, 0.80), (-0.55, 0.40, 1.05)):
                start = np.asarray(anchor)
                result["current_arrows"].append(segment(start, start + velocity))
        return result
    if mode not in ("seabed", "wall"):
        return result
    axis = 2 if mode == "seabed" else 0
    plane = 0.0 if mode == "seabed" else record["wall_plane_x_m"]

    def endpoint(point):
        hit = np.array(point, dtype=float)
        hit[axis] = plane
        return hit

    def surface_contains(point):
        if pool_bounds is None:
            return False
        xmin, xmax, ymin, ymax, depth = pool_bounds
        if mode == "seabed":
            return bool(xmin <= point[0] <= xmax and ymin <= point[1] <= ymax)
        return bool(ymin <= point[1] <= ymax and 0 <= point[2] <= depth)

    if axial:
        water_normal = np.array([0, 0, 1]) if mode == "seabed" else np.array([-1, 0, 0])
        for motor, (position, direction) in enumerate(
            zip(record["rotor_positions_w_m"], record["rotor_axes_w"], strict=True)
        ):
            position, direction = np.asarray(position, dtype=float), np.asarray(direction, dtype=float)
            length = float(np.linalg.norm(direction))
            if not np.isfinite([*position, *direction]).all():
                raise ValueError("Non-finite rotor origin/axis")
            if length < 1e-8:
                continue
            direction = direction / length
            normal_clearance = float((position - endpoint(position)) @ water_normal)
            incidence = float(direction @ water_normal)
            if normal_clearance <= 1e-8 or abs(incidence) <= 1e-8:
                continue  # On/behind the surface or truly parallel: no forward hit.
            # Either direction of the actual physical shaft axis is eligible.
            # Motor force sign is deliberately irrelevant to this proximity guide.
            if incidence > 0:
                direction = -direction
                incidence = -incidence
            distance = normal_clearance / -incidence
            hit = position + distance * direction
            if not surface_contains(hit):
                continue
            guide = segment(
                position,
                hit,
                motor_index=motor,
                axis_direction_w=direction.tolist(),
                ray_length_m=float(distance),
                normal_clearance_m=normal_clearance,
            )
            guide["hemisphere"] = project_hemisphere(hit, water_normal, matrix, viewport, radius_m=hemisphere_radius_m)
            result["rotor_rays"].append(guide)
    else:
        # Frozen schema 1: old recordings used normal clearance guides. Do not
        # reinterpret archived footage as if it recorded the new native axes.
        for motor, position in enumerate(record["rotor_positions_w_m"]):
            hit = endpoint(position)
            distance = float(np.linalg.norm(np.asarray(position) - hit))
            guide = segment(position, hit, motor_index=motor, clearance_m=distance)
            guide["valid"] &= surface_contains(hit)
            result["rotor_rays"].append(guide)
    for label, point in (("commanded", target), ("measured", actual)):
        hit = endpoint(point)
        distance = float(point[2]) if mode == "seabed" else float(plane - point[0])
        result[f"{label}_distance"] = segment(point, hit, value_m=distance)
        result[f"{label}_distance"]["valid"] &= surface_contains(hit)
        # Explicit 1.3m world segments in their actual clearance planes. Their
        # perspective/orientation therefore comes from the same camera as video.
        delta = np.array((0.65, 0, 0)) if mode == "seabed" else np.array((0, 0, 0.65))
        result[f"{label}_line"] = [segment(point - delta, point + delta)]
    return result
