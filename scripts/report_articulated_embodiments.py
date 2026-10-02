"""Independent success replay and descriptive two-system metrics; never overwrite evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def summarize(folder,robot):
 r=json.loads((folder/'report.json').read_text());t=dict(np.load(folder/'trace.npz',allow_pickle=False));n=len(r['seeds'])
 assert all(np.isfinite(v).all() for v in t.values()),'non-finite telemetry'
 assert len(t['time'])==600 and np.allclose(np.diff(t['time']),1/30), 'incomplete trace'
 assert r['control_mode']=='ik' and 0<r['stroke_m']<=.2
 att=t['attitude' if robot=='blue' else 'attitude_error']
 pred={'travel':t['button_travel']>=.004,'distance':t['distance']<.13,'alignment':t['tool_alignment']>.7,'attitude':att<.25,'angular_speed':t['angular_speed']<.35}
 qualified=np.logical_and.reduce(list(pred.values()));counter=np.zeros(n+1,int);success=np.zeros(n+1,bool);first=np.full(n+1,np.nan)
 for i,valid in enumerate(qualified):
  counter=np.where(valid,counter+1,0);new=(counter>=4)&~success;first[new]=t['time'][i];success|=counter>=4
 assert success.tolist()==r['successes'],'30 Hz replay differs from live success'
 assert not success[-1] and np.count_nonzero(t['motor_force'][:,-1])==0,'negative control failed'
 delta=t['tool_position'][:,:n]-t['target_tool'][:,:n];error=np.linalg.norm(delta,axis=-1)*1000
 base=t['base_position' if robot=='blue' else 'position'][:,:n];excursion=np.linalg.norm(base-base[0],axis=-1)*1000
 rows=json.loads((ROOT/'src/wasman/physics/data/t200_16v.json').read_text())['samples']
 low,high=(min(row[2] for row in rows),max(row[2] for row in rows)) if robot=='blue' else (-1540.,1540.)
 force=t['motor_force'][:,:n];util=np.abs(force)/np.where(force>=0,high,-low)
 assert util.max()<=1.00001,'native motor limits exceeded'
 if robot=='blue':
  assert np.abs(t['action'][:,:n]).max()<=1.00001,'hidden adapter clipping'
  if r.get('hydro_variant')=='geometry-scaled':assert np.count_nonzero(t['hydro_clip_counts'][-1,:n])==0,'hydrodynamic clipping occurred'
 phases={}
 for label,start,end in [('settling',0,5),('approach',5,13),('contact',13,20.001)]:
  mask=(t['time']>=start)&(t['time']<end)
  phases[label]={'tcp_rmse_mm':float(np.sqrt(np.square(error[mask]).mean())),'base_excursion_rms_mm':float(np.sqrt(np.square(excursion[mask]).mean())),'base_attitude_rms_deg':float(np.rad2deg(np.sqrt(np.square(att[mask,:n]).mean()))),'motor_utilization_peak':float(util[mask].max()),'qualified_fraction':float(qualified[mask,:n].mean())}
 series=[]
 for i in range(0,len(t['time']),3):
  lo,hi=np.quantile(error[i],[.1,.9]);series.append({'time_s':float(t['time'][i]),'error_mean_mm':float(error[i].mean()),'error_low_mm':float(lo),'error_high_mm':float(hi),'travel_mean_mm':float(t['button_travel'][i,:n].mean()*1000),'attitude_mean_deg':float(np.rad2deg(att[i,:n]).mean()),'motor_utilization_mean':float(util[i].max(-1).mean())})
 record={'robot':robot,'label':'BlueROV2 + Reach Alpha' if robot=='blue' else 'RexROV2 + Oberon7','seeds':r['seeds'],'episodes':n,'successes':int(success[:n].sum()),'per_episode_success':success[:n].tolist(),'first_success_s':[None if np.isnan(v) else float(v) for v in first[:n]],'tool_tracking_rmse_mm':float(np.sqrt(np.square(error).mean())),'phases':phases,'series':series,'negative_control_passed':True,'maximum_joint_excursion_rad':float(np.abs(t['joint_position'][:,:n]-t['joint_position'][0,:n]).max()),'saturation_fraction':float((t['motor_scale'][:,:n]<.99999).mean()) if 'motor_scale' in t else None,'source_hashes':{f:digest(folder/f) for f in ['trace.npz','report.json','run_script.py']},'late_predicate_pass_fraction':{k:float(v[t['time']>=15,:n].mean()) for k,v in pred.items()}}
 if 'normal_contact_by_body' in t:
  f=t['normal_contact_by_body'][:,:n]
  record['normal_contact_peak_by_body_N']={name:{target:float(np.linalg.norm(f[:,:,j,k],axis=-1).max()) for k,target in enumerate(r['contact_targets'])} for j,name in enumerate(r['contact_body_names'])}
 return record,r

def report(blue,rex):
 b,br=summarize(blue,'blue');r,rr=summarize(rex,'rex')
 assert br.get('offset_frame','world')==rr.get('offset_frame','world')
 assert br.get('solver_type',1)==rr.get('solver_type',1) and br['dt']==rr['dt'], 'Common numeric configuration required'
 assert br['seeds']==rr['seeds'] and br['initial_offsets_m']==rr['initial_offsets_m']
 assert br['stroke_m']==rr['stroke_m'] and br.get('ik_mode','feedback')==rr.get('ik_mode','feedback') and br.get('hold_travel_m',0)==rr.get('hold_travel_m',0)
 difference=float(np.abs(np.array(br['initial_tool_position_m'])-np.array(rr['initial_tool_position_m'])).max())
 assert difference<1e-4
 final=br['seeds']==list(range(86000,86030))
 return {'complete':final,'evaluation_split':'final' if final else 'development','configuration':{'offset_frame':br.get('offset_frame','world'),'solver_type':br.get('solver_type',1),'ik_mode':br.get('ik_mode','feedback'),'stroke_m':br['stroke_m'],'hold_travel_m':br.get('hold_travel_m',0),'blue_hydro_variant':br.get('hydro_variant','legacy'),'physics_dt':br['dt']},'study':'articulated-ee-v3','robots':[b,r],'initial_tcp_max_difference_m':difference,'note':'Same scripted EE objective and reference-state IK with measured arm-offset compensation in the commanded level-base frame; 6 mm progress holds native commands. Native low-level gains, geometry and tool frames are retained. This compares complete systems, not isolated morphology or learned transfer.','numerical_limit':'Rex base-only added mass with current articulated rigid inertia and lagged measured joint acceleration/contact. The frozen development protocol includes numerical motion/contact checks. No arm added mass or hardware calibration.','sample_timing':'Blue samples after control interval; Rex at its beginning. Times in seconds preserve this offset. Success independently replayed at 30 Hz; bands are descriptive reset percentiles, not training uncertainty.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('blue',type=Path);p.add_argument('rex',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=report(a.blue,a.rex)
 with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({v['robot']:{k:v[k] for k in ['successes','episodes','phases','saturation_fraction']} for v in r['robots']},indent=2))
