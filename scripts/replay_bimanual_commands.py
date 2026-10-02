"""Replay recorded 30 Hz base/joint targets with native low-level feedback intact."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import numpy as np
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--commands',type=Path,required=True)
a,remaining=p.parse_known_args()
source=json.loads((a.commands/'report.json').read_text())
with np.load(a.commands/'trace.npz') as archive:
 recorded={k:archive[k] for k in ['time','joint_target','base_reference']}
original_init=AppLauncher.__init__
def launch(self,namespace,*args,**kwargs):
 assert namespace.seeds==source['seeds'] and namespace.mode==source['mode'] and namespace.fixture==source['fixture']
 assert namespace.seconds<=source['seconds']
 original_init(self,namespace,*args,**kwargs)
 import torch
 from wasman.controllers.bimanual_control import BimanualController
 def targets(controller,now,goals,rotations,grip):
  index=round(now*30)
  assert abs(float(recorded['time'][index])-now)<1e-6
  ref=torch.as_tensor(recorded['joint_target'][index],device=controller.robot.device)
  base=torch.as_tensor(recorded['base_reference'][index],device=controller.robot.device)
  origin=controller.base-controller.base[0]
  return base+origin,ref.clone()
 BimanualController.targets=targets
 original_close=self.app.close
 def close(*args,**kwargs):
  output=namespace.output_dir
  evidence={'kind':'fixed 30 Hz high-level command replay; base PID and native joint drives remain live',
   'command_source':str(a.commands.resolve()),'command_sha256':hashlib.sha256((a.commands/'trace.npz').read_bytes()).hexdigest(),
   'source_report_sha256':hashlib.sha256((a.commands/'report.json').read_bytes()).hexdigest(),
   'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'phase_clock_is_diagnostic_only':True,'no_state_overwrites_after_reset':True}
  if output.exists():
   (output/'command-replay.json').write_text(json.dumps(evidence,indent=2)+'\n')
   (output/'replay_bimanual_commands.py').write_bytes(Path(__file__).read_bytes())
   if (output/'report.json').exists():
    report=json.loads((output/'report.json').read_text());report['command_replay']=evidence
    report['control_mode']=evidence['kind'];(output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
  return original_close(*args,**kwargs)
 self.app.close=close
AppLauncher.__init__=launch
sys.argv=['scripts/probe_bimanual_valve.py',*remaining]
runpy.run_path(sys.argv[0],run_name='__main__')
