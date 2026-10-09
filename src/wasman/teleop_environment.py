"""Live water controls and a depth-based local turbidity model for manual piloting."""

import math
import time
from dataclasses import asdict, dataclass, replace

import torch


@dataclass
class WaterSettings:
    current: tuple[float, float, float] = (0.0, 0.0, 0.0)
    density: float = 1025.0
    viscosity: float = 1.0  # mPa s; relative to the benchmark's nominal linear damping.
    resistance: float = 1.0
    turbidity: float = 0.0  # illustrative optical coefficient, inverse metres.
    radius: float = 3.0

    def validate(self):
        if len(self.current) != 3 or not all(math.isfinite(v) and abs(v) <= 0.6 for v in self.current):
            raise ValueError("Current components must be finite and in [-0.6, 0.6] m/s")
        for name, low, high in (
            ("density", 900.0, 1200.0),
            ("viscosity", 0.25, 4.0),
            ("resistance", 0.25, 4.0),
            ("turbidity", 0.0, 2.0),
            ("radius", 0.5, 8.0),
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and in [{low}, {high}]")


class WaterControls:
    """Apply controls to the same hydrodynamics used by the vehicle's thrust loop.

    Nominal empirical coefficients are retained as the reference. Viscosity
    scales linear drag; density scales buoyancy, quadratic drag and added mass.
    The separate resistance multiplier scales both drag terms. This is a
    parameterized damping model, not a Reynolds-number/temperature solver.
    """

    def __init__(self, env):
        self.env = env
        hydro = env._hydrodynamics
        self.nominal_density = hydro.water_density
        self.linear = hydro.linear_damping.clone()
        self.quadratic = hydro.quadratic_damping.clone()
        self.added = hydro.added_mass.clone()
        self.defaults = WaterSettings(
            current=tuple(float(v) for v in env._mean_current_w[0]), density=hydro.water_density
        )
        self.settings = replace(self.defaults)
        self.dirty = True
        self.changes = 0
        self.apply()

    def set(self, **values):
        settings = replace(self.settings, **values)
        settings.validate()
        self.settings = settings
        self.dirty = True

    def rotate_current(self):
        x, y, z = self.settings.current
        self.set(current=(-y, x, z))

    def reverse_current(self):
        self.set(current=tuple(-v for v in self.settings.current))

    def calm(self):
        self.set(current=(0.0, 0.0, 0.0))

    def restore_defaults(self):
        self.settings = replace(self.defaults)
        self.dirty = True

    def apply(self, *, force=False):
        if not self.dirty and not force:
            return
        self.settings.validate()
        settings, env = self.settings, self.env
        hydro = env._hydrodynamics
        density_ratio = settings.density / self.nominal_density
        hydro.water_density = env.cfg.water_density = settings.density
        hydro.linear_damping.copy_(self.linear * settings.viscosity * settings.resistance)
        hydro.quadratic_damping.copy_(self.quadratic * density_ratio * settings.resistance)
        hydro.added_mass.copy_(self.added * density_ratio)
        env._mean_current_w[:] = torch.as_tensor(settings.current, device=env.device, dtype=torch.float32)
        env._turbulent_current_w.zero_()
        # A chosen vector is exact; no hidden random drift in the controls.
        env.cfg.turbulence_sigma = 0.0
        hydro.reset()
        self.dirty = False
        self.changes += 1

    def metadata(self):
        return asdict(self.settings)


def cloud_path_length(depth, rays, camera_offset, radius):
    """Length of each camera ray inside a sphere centred on the vehicle.

    Rays are unit world vectors, depth is Euclidean range in metres. Occluders
    before the cloud entry receive no haze; geometry beyond its exit receives
    only the finite path through the local cloud.
    """
    dot = (rays * camera_offset).sum(-1)
    discriminant = dot.square() - (camera_offset.square().sum() - radius**2)
    root = discriminant.clamp_min(0).sqrt()
    entry = (-dot - root).clamp_min(0)
    exit_distance = (-dot + root).clamp_min(0)
    distance = torch.nan_to_num(depth.squeeze(-1), nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0)
    length = (torch.minimum(distance, exit_distance) - entry).clamp_min(0)
    return torch.where(discriminant >= 0, length, torch.zeros_like(length)).unsqueeze(-1)


class BaseCameraFeed:
    """Sample the robot-mounted camera and apply local absorption/backscatter."""

    def __init__(self, env, water):
        self.env, self.water = env, water
        self.camera = env.scene["base_camera"]
        self.local_rays = None
        self.frames = 0
        self.last_raw = self.last_image = None
        self.last_path_max = 0.0

    def image(self):
        from wasman.assets.underwater_optics import WaterOptics, underwater_rgb

        data = self.camera.data
        raw = data.output["rgb"].torch[0, ..., :3]
        settings = self.water.settings
        if self.local_rays is None:
            height, width = raw.shape[:2]
            intrinsic = data.intrinsic_matrices.torch[0]
            v, u = torch.meshgrid(
                torch.arange(height, device=raw.device), torch.arange(width, device=raw.device), indexing="ij"
            )
            # Camera world convention: forward +X, image right -Y, up +Z.
            rays = torch.stack(
                (
                    torch.ones_like(u),
                    -(u - intrinsic[0, 2]) / intrinsic[0, 0],
                    -(v - intrinsic[1, 2]) / intrinsic[1, 1],
                ),
                -1,
            ).float()
            self.local_rays = rays / rays.norm(dim=-1, keepdim=True)
        q = data.quat_w_world.torch[0]
        cross = torch.linalg.cross(q[:3].expand_as(self.local_rays), self.local_rays)
        world_rays = self.local_rays + 2 * (q[3] * cross + torch.linalg.cross(q[:3].expand_as(cross), cross))
        vehicle = self.env.robot.data.body_link_pos_w.torch[0, self.env._base_body_id]
        offset = data.pos_w.torch[0] - vehicle
        path = cloud_path_length(data.output["distance_to_camera"].torch[0], world_rays, offset, settings.radius)
        t = settings.turbidity
        if t == 0:
            image = raw
        else:
            optics = WaterOptics(
                (1.4 * t, t, 0.75 * t), (t, 0.85 * t, 0.7 * t), (0.025, 0.10, 0.12), max_range_m=2 * settings.radius
            )
            image = underwater_rgb(raw, path, optics)
        self.frames += 1
        self.last_raw, self.last_image = raw, image
        self.last_path_max = float(path.max())
        return image

    def metadata(self):
        data = self.camera.data
        return {
            "frames": self.frames,
            "sensor_frames": int(self.camera.frame.torch[0]),
            "resolution": list(self.last_image.shape[:2]) if self.frames else None,
            "raw_std": float(self.last_raw.float().std()) if self.frames else None,
            "local_path_max_m": self.last_path_max,
            "camera_position": data.pos_w.torch[0].cpu().tolist(),
            "world_quaternion_xyzw": data.quat_w_world.torch[0].cpu().tolist(),
        }


class WaterPanel:
    """Kit UI callbacks only edit settings; physics changes happen between steps."""

    def __init__(self, water):
        import omni.ui as ui

        self.water, self.models = water, {}
        self.syncing = False
        self.window = ui.Window("WasserMan - Water", width=410, height=440, position_x=1010, position_y=40)
        with self.window.frame, ui.VStack(spacing=6):
            ui.Label("Current - world axes (m/s)", height=23)
            for axis, name in enumerate("XYZ"):
                self.row(f"Current {name}", f"current_{axis}", -0.6, 0.6, 0.05, water.settings.current[axis])
            with ui.HStack(height=28, spacing=5):
                ui.Button("Rotate 90 deg", clicked_fn=lambda: self.action(water.rotate_current))
                ui.Button("Reverse", clicked_fn=lambda: self.action(water.reverse_current))
                ui.Button("Calm", clicked_fn=lambda: self.action(water.calm))
            ui.Separator(height=4)
            self.row("Density, kg/m3", "density", 900, 1200, 25, water.settings.density)
            self.row("Viscosity, mPa.s", "viscosity", 0.25, 4, 0.25, 1)
            self.row("Drag multiplier", "resistance", 0.25, 4, 0.25, 1)
            self.row("Turbidity, 1/m", "turbidity", 0, 2, 0.1, 0)
            self.row("Cloud radius, m", "radius", 0.5, 8, 0.5, 3)
            ui.Label("Local turbidity affects the base camera", height=23)
            ui.Button("Reset water settings", height=28, clicked_fn=lambda: self.action(water.restore_defaults))

    def row(self, label, name, low, high, step, value):
        import omni.ui as ui

        model = ui.SimpleFloatModel(value)
        self.models[name] = model
        model.add_value_changed_fn(lambda m: self.changed(name, m.as_float))
        with ui.HStack(height=29, spacing=4):
            ui.Label(label, width=170)
            ui.Button("-", width=25, clicked_fn=lambda: model.set_value(max(low, model.as_float - step)))
            ui.FloatSlider(model, min=low, max=high, step=step, format="%.2f", tooltip=label)
            ui.Button("+", width=25, clicked_fn=lambda: model.set_value(min(high, model.as_float + step)))

    def changed(self, name, value):
        if self.syncing:
            return
        if name.startswith("current_"):
            current = list(self.water.settings.current)
            current[int(name[-1])] = min(0.6, max(-0.6, value))
            self.water.set(current=tuple(current))
        else:
            self.water.set(**{name: value})

    def action(self, callback):
        callback()
        self.syncing = True
        try:
            for name, model in self.models.items():
                value = (
                    self.water.settings.current[int(name[-1])]
                    if name.startswith("current_")
                    else getattr(self.water.settings, name)
                )
                model.set_value(value)
        finally:
            self.syncing = False

    def close(self):
        self.window.destroy()


class CameraPanel:
    def __init__(self):
        import omni.ui as ui

        self.window = ui.Window(
            "WasserMan - Base camera",
            width=530,
            height=345,
            position_x=15,
            position_y=350,
            raster_policy=ui.RasterPolicy.NEVER,
        )
        self.provider = ui.ByteImageProvider()
        self.refresh_time, self.refresh_updates = time.monotonic(), 0
        with self.window.frame, ui.VStack(spacing=4):
            ui.ImageWithProvider(self.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
            self.status = ui.Label("Vehicle camera | 640 x 360", height=24)

    def update(self, image):
        rgba = torch.cat((image, torch.full_like(image[..., :1], 255)), -1).contiguous().cpu().numpy()
        self.provider.set_bytes_data(rgba.flatten().data, [rgba.shape[1], rgba.shape[0]], strict=True)
        self.refresh_updates += 1
        now = time.monotonic()
        if now - self.refresh_time >= 1:
            self.status.text = (
                f"Vehicle camera | 640 x 360 | {self.refresh_updates / (now - self.refresh_time):.1f} fps"
            )
            self.refresh_time, self.refresh_updates = now, 0

    def close(self):
        self.window.destroy()
