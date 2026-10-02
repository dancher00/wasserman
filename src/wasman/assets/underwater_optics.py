"""Opt-in depth-dependent underwater sensor appearance, not a full optical solver.

Separate attenuation and backscatter coefficients follow the distinction in
Akkaynak & Treibitz (CVPR 2018). Values below are illustrative, not measured water
types. RGB is transformed through linear light; camera range is in metres.
"""

import math
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WaterOptics:
    attenuation: tuple[float, float, float]
    backscatter: tuple[float, float, float]
    veil_linear: tuple[float, float, float]
    max_range_m: float = 30.0

    def __post_init__(self):
        if any(len(values) != 3 for values in (self.attenuation, self.backscatter, self.veil_linear)):
            raise ValueError("Exactly three RGB coefficients required")
        if not all(math.isfinite(v) for v in self.attenuation + self.backscatter + self.veil_linear):
            raise ValueError("Finite coefficients required")
        if not math.isfinite(self.max_range_m):
            raise ValueError("Finite maximum range required")
        if any(v < 0 for v in self.attenuation + self.backscatter) or self.max_range_m <= 0:
            raise ValueError("Nonnegative inverse-metre coefficients and positive range required")
        if any(not 0 <= v <= 1 for v in self.veil_linear):
            raise ValueError("Linear veil radiance must be in [0,1]")


WATER_PRESETS = {
    "clear": WaterOptics((0.09, 0.035, 0.018), (0.04, 0.025, 0.02), (0.008, 0.035, 0.045)),
    "coastal": WaterOptics((0.55, 0.23, 0.16), (0.30, 0.24, 0.20), (0.015, 0.075, 0.085)),
    "harbor": WaterOptics((1.20, 0.72, 0.54), (0.82, 0.66, 0.54), (0.025, 0.090, 0.080)),
}

# Existing CAD has four front lamp housings (two upper/two lower), not two.
# Coordinates extracted from disconnected mesh components 254--257 of the
# converted source DAE, with a small outward offset to avoid self-occlusion.
LAMP_MOUNTS = {
    "PortUpper": (0.312, 0.1617, 0.1100),
    "StarboardUpper": (0.3125, -0.2105, 0.1101),
    "PortLower": (0.3175, 0.1617, -0.0494),
    "StarboardLower": (0.3180, -0.2104, -0.0494),
}


def underwater_rgb(rgb: torch.Tensor, range_m: torch.Tensor, water: WaterOptics, *, illumination=1.0):
    """Return uint8 RGB with range-dependent attenuation and veiling radiance.

    Accept (...,H,W,3) sRGB uint8 and matching (...,H,W[,1]) Euclidean range.
    ``illumination`` scales ambient veiling radiance, avoiding self-luminous
    water when all lights are off. Lamp-path attenuation, multiple scattering,
    suspended particles and sensor noise are deliberately not modeled.
    """
    if rgb.dtype != torch.uint8 or rgb.shape[-1] != 3:
        raise ValueError("Expected channels-last uint8 RGB")
    if range_m.ndim == rgb.ndim - 1:
        range_m = range_m.unsqueeze(-1)
    if range_m.shape != rgb.shape[:-1] + (1,):
        raise ValueError("RGB/range shapes must agree")
    if not 0 <= illumination <= 1:
        raise ValueError("Illumination must be in [0,1]")
    distance = torch.nan_to_num(range_m.float(), nan=water.max_range_m, posinf=water.max_range_m, neginf=0)
    distance = distance.clamp(0, water.max_range_m)
    srgb = rgb.float() / 255
    linear = torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055).pow(2.4))
    beta = torch.tensor(water.attenuation, device=rgb.device)
    beta_b = torch.tensor(water.backscatter, device=rgb.device)
    veil = torch.tensor(water.veil_linear, device=rgb.device) * illumination
    output = linear * torch.exp(-distance * beta) + veil * (-torch.expm1(-distance * beta_b))
    output = output.clamp(0, 1)
    srgb = torch.where(output <= 0.0031308, 12.92 * output, 1.055 * output.pow(1 / 2.4) - 0.055)
    return (srgb.clamp(0, 1) * 255).round().to(torch.uint8)


def mount_robot_spotlights(stage, base_path, *, intensity=18000.0):
    """Attach emitters at front lamp locations; no duplicate lamp/prop geometry.

    USD lights emit along local -Z: -65 deg around Y points forward and 25 deg
    down in the robot frame. Pose follows the base link, never the world.
    Radiometric output is an illustration parameter, not hardware calibration.
    """
    from pxr import Gf, UsdGeom, UsdLux

    if not stage.GetPrimAtPath(base_path).IsValid():
        raise ValueError(f"Missing robot base: {base_path}")
    paths = []
    for side, position in LAMP_MOUNTS.items():
        path = f"{base_path}/TaskLamp{side}"
        light = UsdLux.SphereLight.Define(stage, path)
        light.CreateRadiusAttr(0.008)
        light.CreateIntensityAttr(intensity)
        light.CreateColorAttr(Gf.Vec3f(0.87, 0.96, 1.0))
        light.CreateNormalizeAttr(True)
        shape = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shape.CreateShapingConeAngleAttr(42.0)
        shape.CreateShapingConeSoftnessAttr(0.35)
        xform = UsdGeom.Xformable(light)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddRotateYOp().Set(-65.0)
        paths.append(path)
    return paths


def set_scene_lighting(stage, lamp_paths, *, ambient: bool, lamps: bool, intensity=18000.0):
    """Explicitly switch existing ambient/key lights and robot-mounted emitters."""
    from pxr import Sdf, UsdLux

    # RTX also has a non-USD ambient term. Turning off scene lights alone leaves
    # the sand illuminated. Preserve/disable it when running inside Kit; pure
    # USD unit tests do not import or initialize the renderer.
    carb = sys.modules.get("carb")
    if carb is not None and hasattr(carb, "settings"):
        settings = carb.settings.get_settings()
        world = stage.GetPrimAtPath("/World")
        saved = world.GetAttribute("wasman:originalRtxAmbient")
        if not saved:
            saved = world.CreateAttribute("wasman:originalRtxAmbient", Sdf.ValueTypeNames.Float, custom=True)
            saved.Set(settings.get("/rtx/sceneDb/ambientLightIntensity") or 0.0)
        settings.set_float("/rtx/sceneDb/ambientLightIntensity", saved.Get() if ambient else 0.0)

    for path in lamp_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdLux.LightAPI):
            raise ValueError(f"Missing robot light: {path}")
    # Save authored values on each prim before switching. This works for custom
    # scenes too, and disables every non-robot light rather than two guessed ones.
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdLux.LightAPI) or str(prim.GetPath()) in lamp_paths:
            continue
        light = UsdLux.LightAPI(prim)
        saved = prim.GetAttribute("wasman:originalIntensity")
        if not saved:
            saved = prim.CreateAttribute("wasman:originalIntensity", Sdf.ValueTypeNames.Float, custom=True)
            saved.Set(light.GetIntensityAttr().Get())
        light.GetIntensityAttr().Set(saved.Get() if ambient else 0.0)
    for path in lamp_paths:
        UsdLux.LightAPI(stage.GetPrimAtPath(path)).GetIntensityAttr().Set(intensity if lamps else 0.0)
