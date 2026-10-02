"""Capture synchronized presentation cameras of a bounded expert rollout.

Adds sensors only; task geometry, physics, commands and success are unchanged.
These snapshots illustrate tasks, not a new benchmark cohort.
"""
import argparse,json,os,hashlib
from pathlib import Path
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES');os.environ.setdefault('ACCEPT_EULA','Y')
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--task',choices=['CollectShell','PushSlider','PullLever'],required=True)
p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--capture-steps',type=int,nargs='+',required=True)
AppLauncher.add_app_launcher_args(p);p.set_defaults(enable_cameras=True,visualizer=['none'])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);app=AppLauncher(a)
import torch,gymnasium as gym
import isaaclab.sim as sim
from isaaclab.sensors import CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_tasks.utils import resolve_task_config
from PIL import Image
import wasman.tasks
from wasman.assets.robot_cameras import add_robot_cameras
name=f'Wasman-Underwater-{a.task}-'+('Direct' if a.task=='CollectShell' else 'Marine-v1')
cfg,_=resolve_task_config(name,'',overrides=('physics=isaacsim_physx',))
cfg.scene.num_envs=1;cfg.seed=a.seed;cfg.sim.render_interval=100000
add_robot_cameras(cfg,width=480,height=480,profile='geometry-v2')
for key in ['base_camera','gripper_camera']:
 c=getattr(cfg.scene,key);c.update_period=0.;c.renderer_cfg=IsaacRtxRendererCfg()
cfg.scene.overview=CameraCfg(prim_path='{ENV_REGEX_NS}/PaperOverview',width=960,height=720,update_period=0.,data_types=['rgb'],renderer_cfg=IsaacRtxRendererCfg(),spawn=sim.PinholeCameraCfg(focal_length=24,horizontal_aperture=36,clipping_range=(.02,20)))
env=gym.make(name,cfg=cfg);r=env.unwrapped;env.reset(seed=a.seed)
r.seed(a.seed);r._reset_idx(torch.tensor([0],device=r.device,dtype=torch.int32));r.scene.write_data_to_sim();r.sim.forward()
if a.task=='CollectShell':
 from wasman.controllers.shell_collection import ShellCollectionExpert
 expert=ShellCollectionExpert(r);eye=[1.25,-1.5,1.05];target=[.28,.05,.22]
else:
 from wasman.controllers.marine_mechanism import MarineMechanismExpert
 expert=MarineMechanismExpert(r,a.task);eye=[-.35,-1.1,1.05];target=[.79,-.03,.67]
r.scene['overview'].set_world_poses_from_view(eyes=r.scene.env_origins+r.scene.env_origins.new_tensor(eye),targets=r.scene.env_origins+r.scene.env_origins.new_tensor(target))
records=[]
try:
 with torch.inference_mode():
  for step in range(1,max(a.capture_steps)+1):
   action=expert.actions();_,_,done,timeout,info=env.step(action)
   if (done|timeout).any():raise RuntimeError('Unexpected terminal/reset during presentation')
   if step in a.capture_steps:
    folder=a.output/f'step-{step}';folder.mkdir()
    root_before=r.robot.data.root_state_w.torch.clone()
    for _ in range(8):
     r.sim.forward();r.sim.render();r.scene.update(0.)
     for key in ['overview','base_camera','gripper_camera']:r.scene[key].update(0.,force_recompute=True)
    assert torch.equal(root_before,r.robot.data.root_state_w.torch)
    files={}
    for key,outname in [('overview','external'),('base_camera','base'),('gripper_camera','gripper')]:
     rgb=r.scene[key].data.output['rgb'].torch[0,...,:3].cpu().numpy();path=folder/f'{outname}.png';Image.fromarray(rgb).save(path)
     files[outname]={'path':str(path.relative_to(a.output)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    row={'step':step,'time_s':step*r.step_dt,'phase':expert.phase.cpu().tolist(),'observed_success':info['wasman_success'].cpu().tolist(),'files':files}
    records.append(row);print(json.dumps(row),flush=True)
  report={'task':a.task,'seed':a.seed,'profile':os.environ.get('WASMAN_ASSET_PROFILE','historical-cad-v1'),'cameras':'external, base and gripper from the same frozen post-step state','scope':'Expert illustration; no completed-cohort success claim','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records}
  (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
finally:env.close();app.app.close()
