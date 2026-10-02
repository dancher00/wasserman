"""Add synchronized robot cameras to a fresh expert presentation recording.

The supplied driver retains its controller, dynamics and success contract.
These new takes are demonstrations, never replacements for scored episodes.
"""

import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--driver", type=Path, required=True)
args, remaining = parser.parse_known_args()
if not ("--expert" in remaining or ("--mode" in remaining and remaining[remaining.index("--mode") + 1] == "expert")):
    parser.error("Only an expert driver is allowed")
output = Path(remaining[remaining.index("--output-dir") + 1])
recorder_source = Path(__file__).read_bytes()
original_launch = AppLauncher.__init__


def launch(self, namespace, *positional, **kwargs):
    namespace.enable_cameras = True
    original_launch(self, namespace, *positional, **kwargs)
    import gymnasium as gym
    import imageio_ffmpeg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from PIL import Image

    from wasman.assets.robot_cameras import add_robot_cameras

    original_make = gym.make

    def make(task, **options):
        cfg = options["cfg"]
        add_robot_cameras(cfg, width=384, height=384, profile="geometry-v2")
        for name in ("base_camera", "gripper_camera"):
            camera = getattr(cfg.scene, name)
            camera.update_period = 0.0
            camera.renderer_cfg = IsaacRtxRendererCfg()
        env = original_make(task, **options)
        raw = env.unwrapped
        original_step, original_close = env.step, env.close
        writers = {}
        count = 0
        calls = 0
        timestamps = []

        def step(action):
            nonlocal count, calls
            result = original_step(action)
            calls += 1
            if getattr(namespace, "mode", None) == "expert" and calls == 1:
                return result
            if not writers:
                for name in ("base", "gripper"):
                    writer = imageio_ffmpeg.write_frames(
                        str(output / f"{name}.mp4"),
                        (384, 384),
                        fps=1 / raw.step_dt,
                        codec="libx264",
                        pix_fmt_in="rgb24",
                        pix_fmt_out="yuv420p",
                        quality=7,
                        output_params=["-movflags", "+faststart", "-threads", "2"],
                    )
                    writer.send(None)
                    writers[name] = writer
            raw.sim.forward()
            raw.sim.render()
            for name, writer in writers.items():
                camera = raw.scene[name + "_camera"]
                camera.update(raw.step_dt, force_recompute=True)
                frame = camera.data.output["rgb"].torch[0, ..., :3].contiguous().cpu().numpy()
                writer.send(frame)
                if count == 60:
                    Image.fromarray(frame).save(output / f"{name}.png")
            timestamps.append((count + 1) * raw.step_dt)
            count += 1
            return result

        def close():
            for writer in writers.values():
                writer.close()
            if writers:
                (output / "recorder.py").write_bytes(recorder_source)
                (output / "multiview.json").write_text(
                    json.dumps(
                        {
                            "kind": "fresh expert presentation take; separate from benchmark scores",
                            "same_episode": True,
                            "sampling": "post-physics, same step as external observer",
                            "frames": count,
                            "fps": 1 / raw.step_dt,
                            "duration_s": count * raw.step_dt,
                            "timestamps_s": timestamps,
                            "camera_profile": "geometry-v2",
                            "driver": str(args.driver),
                            "driver_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
                            "recorder_sha256": hashlib.sha256(recorder_source).hexdigest(),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            return original_close()

        env.step, env.close = step, close
        return env

    gym.make = make


AppLauncher.__init__ = launch
sys.argv = [str(args.driver), *remaining]
runpy.run_path(str(args.driver), run_name="__main__")
