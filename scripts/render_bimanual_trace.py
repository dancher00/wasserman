"""Render measured states only: no dynamics, commands or new success evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

try:
    import imageio_ffmpeg
except ModuleNotFoundError as exc:
    raise SystemExit('Install recording dependencies: uv pip install --python .venv/bin/python '
                     '--no-deps -r research/media-requirements.txt') from exc

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
os.environ.setdefault('ACCEPT_EULA', 'Y')
ROOT = Path(__file__).resolve().parents[1]
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--snapshots-only', action='store_true', help='Preview framing; never accepted as a final film')
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=['none'], enable_cameras=True)
args = parser.parse_args()
RUN, OUT = args.run.resolve(), args.output_dir.resolve()
VALVE = ROOT / 'src/wasman/assets/data/objects/industrial_valve/valve_bimanual_4x.usda'
POOL = ROOT / 'src/wasman/assets/data/environments/standard_pool/pool.usda'
PANEL = ROOT / 'src/wasman/assets/data/panels/ship_green/panel.usda'

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from wasman.physics.rexrov2_bimanual import URDF, load_bimanual_parameters

report = json.loads((RUN / 'report.json').read_text())
audit = json.loads((RUN / 'geometry-audit.json').read_text())
assert report['mode'] in ('two-hands', 'support', 'free') and report['seconds'] == 180
assert not audit['episodes'][0]['cad_collisions']
replay = json.loads((RUN / 'independent-replay.json').read_text())
assert replay['complete'] and replay['finite'] and replay['recorded_success_agrees']
with np.load(RUN / 'trace.npz') as a:
    data = {k: a[k] for k in ['time', 'position', 'quaternion', 'joint_position', 'valve_angle',
                              'motor_force', 'left_tool_position', 'tool_position']}
model = pin.buildModelFromUrdf(str(URDF))
pd = model.createData()
qids = [model.joints[model.getJointId(n)].idx_q for n in report['joint_names']]
params = load_bimanual_parameters()
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'render_bimanual_trace.py').write_bytes(Path(__file__).read_bytes())

launcher = AppLauncher(args)
import carb
import omni.usd
import omni.replicator.core as rep
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
sys.path.insert(0, str(ROOT / 'scripts'))
from rex_rotor_presentation import RexRotorPresentation

settings = carb.settings.get_settings()
settings.set('/rtx/hydra/readTransformsFromFabricInRenderDelegate', False)
settings.set('/rtx/rendermode', 'RaytracedLighting')
settings.set('/rtx/post/aa/op', 2)  # temporal AA; rendering-only setting
omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
converter = UrdfConverter(UrdfConverterCfg(
    asset_path=str(URDF), usd_dir=str(OUT / 'robot_usd'), fix_base=False,
    merge_fixed_joints=False, self_collision=True,
    robot_type='Mobile Manipulators', run_multi_physics_conversion=False,
))
ROBOT = Path(converter.usd_path)
robot = stage.DefinePrim('/World/envs/env_0/Robot', 'Xform')
robot.GetReferences().AddReference(str(ROBOT))
valve = stage.DefinePrim('/World/Valve', 'Xform')
valve.GetReferences().AddReference(str(VALVE))
pool = stage.DefinePrim('/World/Pool', 'Xform')
pool.GetReferences().AddReference(str(POOL))
UsdGeom.Xformable(pool).AddTranslateOp(opSuffix='presentation').Set(Gf.Vec3d(-8, 7.5, 0))
panel = stage.DefinePrim('/World/Panel', 'Xform')
panel.GetReferences().AddReference(str(PANEL))
UsdGeom.Xformable(panel).AddTranslateOp(opSuffix='presentation').Set(Gf.Vec3d(4.06, 0, 1.1))
slab = UsdGeom.Mesh(stage.GetPrimAtPath('/World/Panel/Surface'))
slab.GetPointsAttr().Set([Gf.Vec3f(p[0], p[1] * 2.4 / 1.15, p[2] * 2.2 / 1.15) for p in slab.GetPointsAttr().Get()])
uv = UsdGeom.PrimvarsAPI(slab).GetPrimvar('st')
uv.Set([Gf.Vec2f(p[0] * (2.4 / 1.15 if i < 8 or i >= 16 else 1),
                  p[1] * (2.2 / 1.15 if i < 16 else 1)) for i, p in enumerate(uv.Get())])

def world_op(prim):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp(opSuffix='measured_pose')
    xf.SetResetXformStack(True)
    return op

links = {}
for prim in list(stage.Traverse()):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if str(prim.GetPath()).startswith(str(robot.GetPath())):
            links[prim.GetName()] = world_op(prim)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
assert len(links) == 23
assert not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse())
valve_base = world_op(stage.GetPrimAtPath('/World/Valve/base_link'))
valve_wheel = world_op(stage.GetPrimAtPath('/World/Valve/handle_link'))

def matrix(r, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return Gf.Matrix4d(m.T.tolist())

valve_base.Set(matrix(np.eye(3), [3.7754058, 0, 1.1]))
def cube(path, size, center, color):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1)
    obj.AddTranslateOp().Set(Gf.Vec3d(*center))
    obj.AddScaleOp().Set(Gf.Vec3f(*size))
    obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
if report['mode'] != 'two-hands':
    rail = UsdGeom.Cylinder.Define(stage, '/World/Rail')
    rail.CreateAxisAttr('Z')
    rail.CreateHeightAttr(.28)
    rail.CreateRadiusAttr(.018)
    rail.AddTranslateOp().Set(Gf.Vec3d(3, .5, 1.1))
    rail.CreateDisplayColorAttr([Gf.Vec3f(.35, .45, .5)])
    for i, z in enumerate([.96, 1.24]):
        cube(f'/World/RailBracket{i}', [1.02, .035, .035], [3.51, .5, z], [.35, .45, .5])
light = UsdLux.DomeLight.Define(stage, '/World/Light')
light.CreateIntensityAttr(1800)
light.CreateColorAttr(Gf.Vec3f(.75, .87, .95))
camera = UsdGeom.Camera.Define(stage, '/World/Camera')
camera.CreateFocalLengthAttr(18)
camera.CreateHorizontalApertureAttr(20.955)
camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 50))
cam_op = camera.AddTransformOp()
render_product = rep.create.render_product(str(camera.GetPath()), (1280, 720))
annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
annotator.attach([render_product])
rotor = RexRotorPresentation(stage, params)
phase = np.zeros(6)
max_fk_error = 0.0
clearance = {'floor_m': float('inf'), 'water_surface_m': float('inf'), 'pool_wall_m': float('inf')}

def pose(index):
    global max_fk_error
    q = np.zeros(model.nq)
    q[qids] = data['joint_position'][index, 0]
    pin.framesForwardKinematics(model, pd, q)
    rotation = Rotation.from_quat(data['quaternion'][index, 0]).as_matrix()
    position = data['position'][index, 0]
    for name, op in links.items():
        frame = pd.oMf[model.getFrameId(name)]
        op.Set(matrix(rotation @ frame.rotation, position + rotation @ frame.translation))
    for side, key in [('left', 'left_tool_position'), ('right', 'tool_position')]:
        frame = pd.oMf[model.getFrameId(side + '_oberon_end_effector')]
        tcp = position + rotation @ (frame.translation + frame.rotation @ np.array([.16, 0, 0]))
        max_fk_error = max(max_fk_error, float(np.linalg.norm(tcp - data[key][index, 0])))
    valve_wheel.Set(matrix(Rotation.from_rotvec([data['valve_angle'][index, 0], 0, 0]).as_matrix(), [3, 0, 1.1]))
    u = np.clip((data['time'][index] - 3) / 9, 0, 1)
    u = u*u*(3-2*u)
    eye = np.array([.3, -4.4, 2.3])*(1-u) + np.array([1.5, -2.5, 2.05])*u
    target = np.array([1.2, 0, 1.15])*(1-u) + np.array([2.8, 0, 1.15])*u
    camera.GetFocalLengthAttr().Set(float(15 * (1-u) + 18 * u))
    cam_op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse())

pose(0)
for _ in range(15):
    launcher.app.update()
writer = imageio_ffmpeg.write_frames(str(OUT / 'observer.mp4'), (1280, 720), fps=30,
    codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', quality=7,
    output_params=['-movflags', '+faststart', '-threads', '4'])
writer.send(None)
start = time.monotonic()
try:
    selected = [0, 300, 900, 1350, 2400, 3300, 4200, 4800, 5399] if args.snapshots_only else range(len(data['time']))
    for i in selected:
        pose(i)
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(robot).ComputeAlignedRange()
        low, high = bounds.GetMin(), bounds.GetMax()
        clearance['floor_m'] = min(clearance['floor_m'], float(low[2]))
        clearance['water_surface_m'] = min(clearance['water_surface_m'], float(2.5 - high[2]))
        clearance['pool_wall_m'] = min(clearance['pool_wall_m'], float(low[0] + 20.5), float(4.5 - high[0]), float(low[1] + 5), float(20 - high[1]))
        force = data['motor_force'][i, 0]
        speed = np.sign(force) * 4 * np.sqrt(np.minimum(np.abs(force) / 1540, 1))
        phase[:] = (phase + speed * 360 / 30) % 360
        for op, value in zip(rotor.ops, phase):
            op.Set(float(value))
        # Timeline remains paused: app updates draw authored USD poses only.
        launcher.app.update()
        rgb = np.asarray(annotator.get_data())
        assert rgb.shape[:2] == (720, 1280), rgb.shape
        rgb = np.ascontiguousarray(rgb[..., :3])
        if i % 150 == 0 or args.snapshots_only:
            Image.fromarray(rgb).save(OUT / f'observer_{i}.png')
            print(json.dumps({'frame': i, 'time_s': float(data['time'][i]), 'elapsed_s': time.monotonic()-start}), flush=True)
        writer.send(rgb)
finally:
    writer.close()
assert max_fk_error < 1e-4
assert min(clearance.values()) > 0, ('Rendered robot crosses the finite pool or water surface', clearance)
result = dict(kind='rendered replay of recorded physical states in the standard pool; no new simulation',
    source_run=str(RUN), seed=report['seeds'][0], development_seeds=report['seeds'],
    mode=report['mode'], fps=30, frames=len(selected), complete=not args.snapshots_only, overlays=False, audio=False,
    physics_steps=0, speed='real time', max_rendered_tcp_error_m=max_fk_error,
    source_trace_sha256=hashlib.sha256((RUN/'trace.npz').read_bytes()).hexdigest(),
    final_angle_deg=float(np.rad2deg(data['valve_angle'][-1, 0])),
    success=bool(report['successes'][0]),
    presentation='standard-pool-ship-green-v1', presentation_only=True,
    pool_center_m=[-8, 7.5, 0], water_depth_m=2.5, pool_dimensions_m=[25, 25],
    panel_bottom_m=0, panel_top_m=2.2,
    robot_pool_clearance=clearance, clearance_sampling='all rendered frames; visual geometry world bounds',
    render_assets_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in [POOL.parent, PANEL.parent] for p in sorted(folder.rglob('*')) if p.is_file()},
    urdf_sha256=hashlib.sha256(URDF.read_bytes()).hexdigest(),
    rotor_presentation='Recorded signed motor forces; display speed capped at 4 turns/s, not physical RPM',
    elapsed_s=time.monotonic()-start)
(OUT/'observer.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
launcher.app.close()
