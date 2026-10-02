"""Geometrically similar 4x industrial valve, separately versioned and explicit."""

import hashlib
import json
import math
from pathlib import Path

from pxr import Usd, Vt

root = Path(__file__).resolve().parents[1]
folder = root / "src/wasman/assets/data/objects/industrial_valve"
source = folder / "valve.usdc"
target = folder / "valve_bimanual_4x.usda"
stage = Usd.Stage.Open(str(source))
layer = stage.Flatten()
layer.Export(str(target))
stage = Usd.Stage.Open(str(target))
scale = 4.0
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        name = attr.GetName()
        value = attr.Get()
        if value is None:
            continue
        if name == "points":
            attr.Set(Vt.Vec3fArray([x * scale for x in value]))
        elif name == "extent":
            attr.Set(Vt.Vec3fArray([x * scale for x in value]))
        elif name in ["xformOp:translate", "physics:localPos0", "physics:localPos1", "physics:centerOfMass"]:
            if all(math.isfinite(x) for x in value):
                attr.Set(value * scale)
        elif name in ["height", "radius", "size"]:
            attr.Set(value * scale)
        elif name == "physics:mass":
            attr.Set(value * scale**3)
        elif name == "physics:diagonalInertia":
            attr.Set(value * scale**5)
stage.GetRootLayer().Save()
target.write_text(target.read_text().replace(str(folder) + "/", ""))
(folder / "valve_bimanual_4x.json").write_text(
    json.dumps(
        {
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "scale": scale,
            "wheel_diameter_m": 0.1264653 * scale,
            "rim_center_radius_m": 0.059 * scale,
            "rim_tube_radius_m": 0.0042 * scale,
            "wheel_mass_kg": 0.18 * scale**3,
            "wheel_principal_inertia_kg_m2": [x * scale**5 for x in [0.00045, 0.00025, 0.00025]],
            "wheel_displaced_volume_m3": 0.000060 * scale**3,
            "assumption": "Geometric similarity with preserved source density and shape, not an identified commercial valve. Bearing coefficients must be explicitly configured in the new study; original fixture unchanged.",
        },
        indent=2,
    )
    + "\n"
)
print(target)
