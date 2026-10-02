# ruff: noqa: E402
"""AM-Bench's pinned CLIP/UNet diffusion policy with WASMAN episode inputs."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / ".deps/valve-policy-deps"),
    str(ROOT / ".deps/ambench-60bf5b7/source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface"),
]

import torch
import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy
from timm.models.vision_transformer import checkpoint_filter_fn

from wasman.learning.marine_dp import MarineDPDataset  # noqa: F401


def make_dp(*, pretrained=True):
    shape = dict(
        obs={
            "camera0_rgb": dict(shape=[3, 224, 224], horizon=2, type="rgb"),
            "robot_state": dict(shape=[21], horizon=2, type="low_dim"),
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
