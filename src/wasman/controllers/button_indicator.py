"""Mechanical button indicator; purely visual, never part of reward/success."""

import torch


class ButtonIndicatorState:
    def __init__(self, count, device, on_depth=0.004, off_depth=0.003):
        self.on_depth, self.off_depth = on_depth, off_depth
        self.pressed = torch.zeros(count, dtype=torch.bool, device=device)

    def update(self, depth):
        self.pressed.copy_(torch.where(self.pressed, depth > self.off_depth, depth >= self.on_depth))

    def reset(self, indices):
        self.pressed[indices] = False


class ButtonIndicator:
    def __init__(self, stage, num_envs, device, max_visible_envs=16):
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        self.state = ButtonIndicatorState(min(num_envs, max_visible_envs), device)
        self.visuals = []
        self.last = [None] * len(self.state.pressed)
        for index in range(len(self.state.pressed)):
            root = f"/World/envs/env_{index}/Button"
            material = UsdShade.Material.Define(stage, root + "/IndicatorMaterial")
            shader = UsdShade.Shader.Define(stage, root + "/IndicatorMaterial/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            emission = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.28)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            plunger = stage.GetPrimAtPath(root + "/plunger/plunger")
            if not plunger:
                raise RuntimeError("Button indicator requires a real plunger mesh")
            UsdShade.MaterialBindingAPI.Apply(plunger).Bind(material, UsdShade.Tokens.strongerThanDescendants)
            light = UsdLux.SphereLight.Define(stage, root + "/plunger/IndicatorGlow")
            light.CreateRadiusAttr(0.025)
            light.CreateColorAttr(Gf.Vec3f(0.08, 1.0, 0.25))
            light.CreateIntensityAttr(0.0)
            UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(-0.012, 0, 0))
            self.visuals.append((diffuse, emission, light.GetIntensityAttr()))
        self._render()

    def _render(self):
        for i, on in enumerate(self.state.pressed.tolist()):
            if self.last[i] == on:
                continue
            diffuse, emission, intensity = self.visuals[i]
            diffuse.Set((0.025, 0.65, 0.12) if on else (0.55, 0.018, 0.012))
            emission.Set((0.08, 3.0, 0.35) if on else (0.0, 0.0, 0.0))
            intensity.Set(12.0 if on else 0.0)
            self.last[i] = on

    def update(self, depth, render):
        self.state.update(depth[: len(self.visuals)])
        if render:
            self._render()

    def reset(self, indices):
        if indices is None:
            indices = slice(None)
        else:
            indices = torch.as_tensor(indices, device=self.state.pressed.device, dtype=torch.long)
            indices = indices[indices < len(self.visuals)]
        self.state.reset(indices)
        self._render()
