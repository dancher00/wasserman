"""Two opt-in RGB sensors attached to moving links, independent of policy observations."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from wasman.robot_camera_profiles import robot_camera_profile

BASE_PATH = "{ENV_REGEX_NS}/Robot/Geometry/base_link"
JAW_PATH = BASE_PATH + (
    "/alpha_m3_inline_link/alpha_m2_1_1_link/alpha_m2_joint_link/alpha_m2_1_2_link/alpha_m1_link/alpha_jaw_base_link"
)


def add_robot_cameras(cfg, *, width=384, height=384, profile="legacy-v1"):
    """Call before gym.make, with AppLauncher enable_cameras=True.

    RGB buffers are scene['base_camera'/'gripper_camera'].data.output['rgb'].
    Existing state-based policy/checkpoint dimensions are intentionally unchanged.
    """

    selected = robot_camera_profile(profile)

    def camera(path, mount):
        return CameraCfg(
            prim_path=path,
            update_period=1 / 20,
            width=width,
            height=height,
            data_types=["rgb"],
            update_latest_camera_pose=True,
            renderer_cfg=MultiBackendRendererCfg(default=IsaacRtxRendererCfg(), isaacsim_rtx=IsaacRtxRendererCfg()),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=selected.focal_length_mm,
                horizontal_aperture=selected.horizontal_aperture_mm,
                clipping_range=selected.clipping_range_m,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=mount.position_m, rot=mount.quaternion_xyzw, convention=selected.orientation_convention
            ),
        )

    cfg.scene.base_camera = camera(BASE_PATH + "/BaseCamera", selected.base)
    cfg.scene.gripper_camera = camera(JAW_PATH + "/GripperCamera", selected.gripper)
    return cfg
