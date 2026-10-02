# ruff: noqa: E402
"""AM-Bench's pinned CLIP/UNet diffusion policy with WASMAN episode inputs."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / ".deps/valve-policy-deps"),
    str(ROOT / ".deps/ambench-60bf5b7/source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface"),
]

import numpy as np
import torch
import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from scipy.spatial.transform import Rotation
from timm.models.vision_transformer import checkpoint_filter_fn

from wasman.learning.valve_visual_dataset import ValveVisualDataset, relative_trajectory


def rotation6d(quaternion):
    shape = quaternion.shape[:-1]
    return Rotation.from_quat(quaternion.reshape(-1, 4)).as_matrix()[..., :2, :].reshape(*shape, 6).astype(np.float32)


def dp_lowdim(measured_history, measured_start):
    relative = relative_trajectory(measured_history, measured_history[-1])
    relative_start = relative_trajectory(measured_history, measured_start)
    return {
        "robot0_eef_pos": relative[..., :3],
        "robot0_eef_rot_axis_angle": rotation6d(relative[..., 3:7]),
        "robot0_gripper_width": measured_history[..., 7:],
        "robot0_eef_rot_axis_angle_wrt_start": rotation6d(relative_start[..., 3:7]),
    }


class ValveDPDataset(ValveVisualDataset):
    def __init__(self, episodes, *, augment_start=True, policy_hz=20):
        super().__init__(episodes, policy_hz=policy_hz)
        # AM-Bench's DP dataset rejects action windows extending past the episode.
        self.items = [(e, i) for e, i in self.items if not self.trajectories[e]["pad"][i].any()]
        self.augment_start = augment_start

    def lowdim(self, episode, sample):
        t = self.trajectories[episode]
        indices = t["anchors"][[max(sample - 1, 0), sample]]
        measured = t["data"]["measured"][indices]
        start = t["data"]["measured"][0].copy()
        if self.augment_start:
            # Match the upstream UMI start-pose orientation noise (0.05 rad).
            start[3:7] = Rotation.from_rotvec(
                Rotation.from_quat(start[3:7]).as_rotvec() + np.random.normal(scale=0.05, size=3)
            ).as_quat()
        obs = dp_lowdim(measured, start)
        action = t["action"][sample]
        action = np.concatenate((action[..., :3], rotation6d(action[..., 3:7]), action[..., 7:]), -1)
        return obs, action, indices

    def __getitem__(self, index):
        episode, sample = self.items[index]
        t = self.trajectories[episode]
        obs, action, indices = self.lowdim(episode, sample)
        if episode not in self.buffers:
            self.buffers[episode] = np.memmap(
                self.episodes[episode] / "wrist.rgb",
                mode="r",
                dtype=np.uint8,
                shape=(t["meta"]["length"], *t["meta"]["image_shape"]),
            )
        rgb = torch.from_numpy(np.asarray(self.buffers[episode][indices]).copy()).permute(0, 3, 1, 2).float() / 255
        rgb = T.functional.resize(rgb, [224, 224], antialias=True)
        return {
            "obs": {"camera0_rgb": rgb, **{k: torch.from_numpy(v) for k, v in obs.items()}},
            "action": torch.from_numpy(action),
        }

    def normalizer(self):
        previous = self.augment_start
        self.augment_start = False
        all_obs, all_action = {}, []
        for e, i in self.items:
            obs, action, _ = self.lowdim(e, i)
            for key, value in obs.items():
                all_obs.setdefault(key, []).append(value)
            all_action.append(action)
        self.augment_start = previous
        normalizer = LinearNormalizer()
        for key, arrays in all_obs.items():
            normalizer[key] = (
                SingleFieldLinearNormalizer.create_identity()
                if "rot_axis_angle" in key
                else SingleFieldLinearNormalizer.create_fit(np.concatenate(arrays))
            )
        normalizer["camera0_rgb"] = SingleFieldLinearNormalizer.create_identity()
        action_norm = SingleFieldLinearNormalizer.create_fit(np.concatenate(all_action))
        with torch.no_grad():
            action_norm.params_dict["scale"][3:9] = 1
            action_norm.params_dict["offset"][3:9] = 0
        normalizer["action"] = action_norm
        normalizer.requires_grad_(False)
        return normalizer


def make_dp(*, pretrained=True):
    shape = dict(
        obs={
            "camera0_rgb": dict(shape=[3, 224, 224], horizon=2, type="rgb"),
            "robot0_eef_pos": dict(shape=[3], horizon=2, type="low_dim"),
            "robot0_eef_rot_axis_angle": dict(shape=[6], horizon=2, type="low_dim"),
            "robot0_gripper_width": dict(shape=[1], horizon=2, type="low_dim"),
            "robot0_eef_rot_axis_angle_wrt_start": dict(shape=[6], horizon=2, type="low_dim"),
        },
        action=dict(shape=[10], horizon=16),
    )
    encoder = TimmObsEncoder(
        shape,
        model_name="vit_base_patch16_clip_224.openai",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=[
            SimpleNamespace(type="RandomCrop", ratio=0.95),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
        ],
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True,
        feature_aggregation="attention_pool_2d",
        position_encording="sinusoidal",
    )
    if pretrained:
        backbone = encoder.key_model_map["camera0_rgb"]
        weights = torch.load(ROOT / ".deps/valve-weights/clip_vit_b16_timm.bin", map_location="cpu", weights_only=True)
        weights = checkpoint_filter_fn(weights, backbone)
        weights = {k: v for k, v in weights.items() if not k.startswith("head.")}
        backbone.load_state_dict(weights, strict=True)
    scheduler = DDIMScheduler(
        num_train_timesteps=50,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    )
    return DiffusionUnetTimmPolicy(
        shape,
        scheduler,
        encoder,
        num_inference_steps=16,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        input_pertub=0.1,
        train_diffusion_n_samples=1,
    )
