"""Metric pool tile regions, not raised decals or separate painted objects.

World Aquatics February 2026 sections 15.13.1–15.13.4 specify geometry and
contrasting colour, not a required construction material. Dark ceramic is our
architectural choice. A 0.20 m line fits four nominal 50 mm mosaic modules.
"""

LANE_CENTRES = (-11.2, *tuple((lane - 3.5) * 2.5 for lane in range(8)), 11.2)
LINE_WIDTH_M = 0.20


def floor_tile_regions():
    """Named (u0,u1,v0,v1) rectangles in pool-local X/Y metres."""
    regions = []
    half = LINE_WIDTH_M / 2
    for lane, y in enumerate(LANE_CENTRES):
        regions.append((f"Lane{lane}", (-10.5, 10.5, y - half, y + half)))
        for x in (-10.5, 10.5):
            regions.append((f"Lane{lane}T{x:+g}", (x - half, x + half, y - 0.5, y + 0.5)))
        # 15m from either end of a 25m pool: local X = +/-2.5m.
        for x in (-2.5, 2.5):
            regions.append((f"Lane{lane}15m{x:+g}", (x - half, x + half, y - 0.25, y + 0.25)))
    return regions


def wall_tile_regions(*, end_wall, water_depth=2.5, wall_height=2.7):
    """U is horizontal along wall, V is height; targets only on end walls."""
    regions = [("Waterline", (-12.5, 12.5, water_depth - 0.05, water_depth + 0.05))]
    if end_wall:
        half = LINE_WIDTH_M / 2
        for lane, y in enumerate(LANE_CENTRES):
            regions.append((f"Target{lane}", (y - half, y + half, 0, wall_height)))
            cross_z = water_depth - 0.3
            regions.append((f"Target{lane}Cross", (y - 0.25, y + 0.25, cross_z - half, cross_z + half)))
    return regions


def partition_rectangles(bounds, regions):
    """Non-overlapping planar quads covering bounds, tagged dark/light.

    Split only at rectangle edges. This preserves an exactly coplanar surface
    and its original cuboid collision extent, avoiding decal z-fighting.
    """
    u0, u1, v0, v1 = bounds
    us, vs = {u0, u1}, {v0, v1}
    for _, (a, b, c, d) in regions:
        us.update(value for value in (a, b) if u0 < value < u1)
        vs.update(value for value in (c, d) if v0 < value < v1)
    us, vs = sorted(us), sorted(vs)
    result = []
    for a, b in zip(us[:-1], us[1:], strict=True):
        for c, d in zip(vs[:-1], vs[1:], strict=True):
            u, v = (a + b) / 2, (c + d) / 2
            dark = any(r[0] <= u <= r[1] and r[2] <= v <= r[3] for _, r in regions)
            result.append(((a, b, c, d), dark))
    return result
