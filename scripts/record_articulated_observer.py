"""Camera-only observer for the articulated PressButton development-seed take."""

import argparse
import json
import runpy
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--robot", choices=["blue", "rex"], required=True)
args, remaining = p.parse_known_args()
output = Path(remaining[remaining.index("--output-dir") + 1])
original_launcher_init = AppLauncher.__init__


def launch(self, namespace, *positional, **kwargs):
    namespace.enable_cameras = True
    original_launcher_init(self, namespace, *positional, **kwargs)
    import imageio_ffmpeg
    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    original_reset = sim_utils.SimulationContext.reset
    original_step = sim_utils.SimulationContext.step
    original_close = self.app.close
    from wasman.physics.rexrov2 import RexThrusters
    from rex_rotor_presentation import RexRotorPresentation
    rotor = None
    original_thruster_step = RexThrusters.step
    def motor_step(motors, command):
        nonlocal rotor
        result = original_thruster_step(motors, command)
        if rotor is None:
            rotor = RexRotorPresentation(sim_utils.get_current_stage(), motors.parameters)
        rotor.update(motors)
        return result
    if args.robot == "rex":
        RexThrusters.step = motor_step
    camera = None
    writer = None
    steps = 0
    frames = 0

    def reset(sim, *a, **kw):
        nonlocal camera, writer
        if camera is not None:
            return original_reset(sim, *a, **kw)
        from pxr import UsdGeom
        # Visibility only: retain the negative control in physics and raw telemetry.
        envs=sim_utils.get_current_stage().GetPrimAtPath("/World/envs")
        for env_prim in envs.GetChildren():
            if env_prim.GetName()!="env_0":UsdGeom.Imageable(env_prim).MakeInvisible()
        support=sim_utils.CuboidCfg(size=(.06,.8,1.3),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.21,.34,.38),roughness=.85))
        from pxr import Usd, UsdPhysics
        origin=UsdGeom.XformCache().GetLocalToWorldTransform(envs.GetChild("env_0")).ExtractTranslation()
        support.func("/World/ObserverFixtureSupport",support,translation=(3.08+origin[0],origin[1],.65+origin[2]))
        for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath("/World/ObserverFixtureSupport")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if args.robot == "rex":
            light = sim_utils.DomeLightCfg(intensity=2200, color=(0.75, 0.87, 0.95))
            light.func("/World/ObserverLight", light)
            ground = sim_utils.CuboidCfg(
                size=(40, 40, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.23, 0.24), roughness=0.9),
            )
            ground.func("/World/ObserverGround", ground, translation=(0, 0, 0))
            from pxr import Usd, UsdPhysics

            for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath("/World/ObserverGround")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)

        from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

        eye = torch.tensor([[5.0, -5.0, 3.8]], device=sim.device)
        target = torch.tensor([[1.6, 0.0, 1.1]], device=sim.device)
        quaternion = quat_from_matrix(create_rotation_matrix_from_view(eye, target, "Z", device=sim.device))[0]
        camera = Camera(
            CameraCfg(
                prim_path="/World/envs/env_0/StudyObserver"
                if args.robot == "rex"
                else "/World/envs/env_[0-9]+/StudyObserver",
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(eye[0].tolist()), rot=tuple(quaternion.tolist()), convention="opengl"
                ),
                width=1280,
                height=720,
                update_period=0,
                data_types=["rgb"],
                renderer_cfg=IsaacRtxRendererCfg(),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18, horizontal_aperture=20.955, clipping_range=(0.02, 11)
                ),
            )
        )
        result = original_reset(sim, *a, **kw)
        writer = imageio_ffmpeg.write_frames(
            str(output / "observer.mp4"),
            (1280, 720),
            fps=30,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            quality=8,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        return result

    def step(sim, *a, **kw):
        nonlocal steps, frames
        steps += 1
        record = steps % round(1 / 30 / namespace.dt) == 0
        if record and camera is not None:
            from pxr import UsdGeom

            origin = (
                UsdGeom.XformCache()
                .GetLocalToWorldTransform(sim_utils.get_current_stage().GetPrimAtPath("/World/envs/env_0"))
                .ExtractTranslation()
            )
            time = steps * namespace.dt
            u = min(1.0, max(0.0, (time - 3) / 9))
            u = u * u * (3 - 2 * u)
            start = (5.0, -5.0, 3.8) if args.robot == "rex" else (4.2, -2.7, 2.2)
            end = (2.55, -0.95, 1.45)
            eye = [x + (y - x) * u + origin[i] for i, (x, y) in enumerate(zip(start, end, strict=True))]
            target = [1.6 + 1.3 * u + origin[0], origin[1], 1.1 + origin[2]]
            if args.robot == "blue":
                target[0] = 2.5 + 0.4 * u + origin[0]
            camera.set_world_poses_from_view(
                torch.tensor([eye], device=sim.device), torch.tensor([target], device=sim.device), env_ids=[0]
            )
        kw["render"] = record
        result = original_step(sim, *a, **kw)
        if record and camera is not None:
            sim.forward()
            sim.render()
            camera.update(1 / 30, force_recompute=True)
            rgb = camera.data.output["rgb"].torch[0, ..., :3].contiguous().cpu().numpy()
            if frames in (60, 420):
                from PIL import Image

                Image.fromarray(rgb).save(output / f"observer_{frames}.png")
            if rotor is not None: rotor.record()
            writer.send(rgb)
            frames += 1
        return result

    def close(*a, **kw):
        if writer is not None:
            writer.close()
        if output.exists():
            (output / "observer.json").write_text(
                json.dumps(
                    dict(
                        rotor_presentation=rotor.report() if rotor is not None else None,
                        frames=frames,
                        fps=30,
                        overlays=False,
                        kind="separate common-IK live take; preselected development seed 85000",
                        visual_ground_only=args.robot == "rex",
                        sampling="post-physics 30 Hz; raw Rex telemetry samples before each control interval",
                        camera_scope="env_0; other environments hidden from rendering only",
                        visual_fixture_support="noncolliding grounded support behind the fixed button",
                        observer_source=Path(__file__).read_text(),
                    ),
                    indent=2,
                )
                + "\n"
            )
        return original_close(*a, **kw)

    sim_utils.SimulationContext.reset = reset
    sim_utils.SimulationContext.step = step
    self.app.close = close


AppLauncher.__init__ = launch
sys.argv = [f"scripts/probe_{args.robot}_articulated.py", *remaining]
runpy.run_path(sys.argv[0], run_name="__main__")
