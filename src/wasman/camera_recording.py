"""Same-step robot-camera recording for finite, reset-free expert demonstrations."""


def configure_robot_streams(cfg, steps):
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    from wasman.assets.robot_cameras import add_robot_cameras

    add_robot_cameras(cfg, width=256, height=256, profile=getattr(cfg, "robot_camera_profile", "legacy-v1"))
    for name in ("base_camera", "gripper_camera"):
        camera = getattr(cfg.scene, name)
        camera.update_period = 0.0
        camera.renderer_cfg = IsaacRtxRendererCfg()
    cfg.video_recorders[0].video_length = steps + 1


class RobotCameraRecording:
    """Initialize at a recorded hold step, then stream post-physics RGB at policy Hz."""

    def __init__(self, env, app, output, prefix):
        import imageio_ffmpeg
        import torch

        self.env, self.raw = env, env.unwrapped
        self.writers, self.times = {}, []
        raw = self.raw
        hold = torch.zeros((1, raw.cfg.action_space), device=raw.device)
        hold[:, :3] = (
            raw.robot.data.body_link_pos_w.torch[:, raw._base_body_id]
            - raw.scene.env_origins
            - raw._base_target_nominal
        ) / raw._base_target_position_scale
        if hold.shape[1] > 10:
            hold[:, 10] = -1
        self.observation, _, done, timeout, _ = env.step(hold)
        if (done | timeout).any():
            raise ValueError("Camera initialization terminated")
        for name in ("base_camera", "gripper_camera"):
            _ = raw.scene[name].data
        for _ in range(8):
            raw.sim.forward()
            raw.scene.update(0.0)
            app.update()
            for name in ("base_camera", "gripper_camera"):
                raw.scene[name].update(0.0, force_recompute=True)
        output.mkdir(parents=True, exist_ok=True)
        for name in ("base", "gripper"):
            writer = imageio_ffmpeg.write_frames(
                str(output / f"{prefix}_{name}.mp4"),
                (256, 256),
                fps=1 / raw.step_dt,
                codec="libx264",
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                quality=8,
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            self.writers[name] = writer

    def capture(self, step):
        import warp as wp

        times = []
        for name, writer in self.writers.items():
            camera = self.raw.scene[f"{name}_camera"]
            rgb = camera.data.output["rgb"].torch[0, ..., :3].contiguous()
            if step == 0 and rgb.float().std() < 1:
                raise ValueError(f"Blank initial {name} camera")
            writer.send(rgb.cpu().numpy())
            times.append(float(wp.to_torch(camera._timestamp_last_update)[0]) - self.raw.step_dt)
        if abs(times[0] - times[1]) > 1e-5 or abs(times[0] - (step + 1) * self.raw.step_dt) > 0.02:
            raise ValueError("Robot camera timestamps differ from telemetry")
        self.times.append(times)

    def metadata(self):
        return {
            "enabled": True,
            "names": list(self.writers),
            "fps": 1 / self.raw.step_dt,
            "size": [256, 256],
            "frames": len(self.times),
            "observer_initial_frames_to_trim": 1,
            "initial_hold_s": self.raw.step_dt,
            "timestamps_s": self.times,
        }

    def close(self):
        for writer in self.writers.values():
            writer.close()
