"""Two-view RGB + robot proprioception behavioral-cloning baseline.

No mechanism state, target pose, expert phase or elapsed time enters the actor.
Station keeping and joint servos remain the task's existing low-level control.
"""

import torch
from torch import nn

PROPRIO_FIELDS = (
    ("base_odometry_xyz", 3),
    ("base_quaternion_xyzw", 4),
    ("base_linear_velocity_world", 3),
    ("base_angular_velocity_world", 3),
    ("arm_joint_positions", 4),
    ("arm_joint_velocities", 4),
    ("gripper_joint_positions", 3),
    ("gripper_joint_velocities", 3),
    ("previous_command", 11),
)
PROPRIO_DIM = sum(size for _, size in PROPRIO_FIELDS)


def robot_proprioception(env, previous_action):
    """Robot-only state; simulator odometry is assumed available, not vision-only SLAM."""
    data, body = env.robot.data, env._base_body_id
    if not hasattr(env, "_vision_gripper_joint_ids"):
        env._vision_gripper_joint_ids = env.robot.find_joints(
            ["alpha_axis_a", "alpha_left_finger_joint", "alpha_right_finger_joint"], preserve_order=True
        )[0]
    grip_ids = env._vision_gripper_joint_ids
    return torch.cat(
        (
            data.body_link_pos_w.torch[:, body] - env.scene.env_origins,
            data.body_link_quat_w.torch[:, body],
            data.body_com_lin_vel_w.torch[:, body],
            data.body_com_ang_vel_w.torch[:, body],
            data.joint_pos.torch[:, env._arm_joint_ids],
            data.joint_vel.torch[:, env._arm_joint_ids],
            data.joint_pos.torch[:, grip_ids],
            data.joint_vel.torch[:, grip_ids],
            previous_action,
        ),
        dim=-1,
    )


class HatchVisionActor(nn.Module):
    """Small shared CNN, spatial features and receding-horizon action chunks."""

    def __init__(
        self,
        chunk_size=9,
        representation="incremental",
        *,
        state_clip=None,
        hold_level=False,
        state_noise=0.0,
        previous_command_dropout=0.0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        if representation not in ("incremental", "absolute"):
            raise ValueError("Unknown action representation")
        self.representation = representation
        if state_clip is not None and state_clip <= 0:
            raise ValueError("state_clip must be positive")
        if state_noise < 0 or not 0 <= previous_command_dropout <= 1:
            raise ValueError("Invalid state augmentation")
        self.state_clip = state_clip
        self.hold_level = hold_level
        self.state_noise = state_noise
        self.previous_command_dropout = previous_command_dropout
        self.register_buffer("proprio_mean", torch.zeros(PROPRIO_DIM))
        self.register_buffer("proprio_std", torch.ones(PROPRIO_DIM))
        self.register_buffer("increment_scale", torch.tensor([0.04] * 6 + [0.15] * 4 + [0.30]))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.Conv2d(24, 48, 3, 2, 1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * 64 * 16 + PROPRIO_DIM, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, chunk_size * 11),
        )

    def forward(self, rgb, proprio):
        # Inputs B × 2 × 3 × H × W, uint8 RGB in [0,255].
        batch = rgb.shape[0]
        images = rgb.reshape(-1, *rgb.shape[2:]).float().div(255).sub(0.5)
        features = self.encoder(images).reshape(batch, -1)
        state = (proprio - self.proprio_mean) / self.proprio_std
        if self.training:
            if self.state_noise:
                state = state + self.state_noise * torch.randn_like(state)
            if self.previous_command_dropout:
                state = state.clone()
                keep = torch.rand((batch, 1), device=state.device) >= self.previous_command_dropout
                state[:, -11:] *= keep
        if self.state_clip is not None:
            state = state.clamp(-self.state_clip, self.state_clip)
        residual = self.head(torch.cat((features, state), -1)).reshape(batch, self.chunk_size, 11)
        output = (
            residual if self.representation == "absolute" else proprio[:, None, -11:] + self.increment_scale * residual
        )
        if self.hold_level:
            # Same upright heading as the demonstrations, enforced by the existing
            # station-keeping controller. No mechanism state or teacher queries.
            output = output.clone()
            output[..., 3:6] = 0
        return output
