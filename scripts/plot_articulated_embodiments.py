"""Build shareable plots from a verified final summary; no simulator or training."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('summary',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
r=json.loads(a.summary.read_text());assert r['complete'] and r['evaluation_split']=='final' and all(v['episodes']==30 for v in r['robots'])
a.output_dir.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#163e52','axes.labelcolor':'#163e52'})
fig,axes=plt.subplots(2,2,figsize=(7.05,4.8));fig.subplots_adjust(left=.085,right=.985,top=.89,bottom=.10,wspace=.28,hspace=.42)
colors=['#163e52','#087fbd'];metrics=[('travel_mean_mm','A  Button travel','mm',1),('error_mean_mm','B  TCP goal error','mm',1),('attitude_mean_deg','C  Base attitude error','degrees',1),('motor_utilization_mean','D  Motor force / native limit','%',100)]
for ax,(key,title,unit,scale) in zip(axes.flat,metrics):
 for robot,color in zip(r['robots'],colors):
  s=[v for v in robot['series'] if v['time_s']>=5];t=np.array([v['time_s'] for v in s]);y=np.array([v[key]*scale for v in s])
  ax.plot(t,y,color=color,lw=1.5,label=f"{robot['label']} ({robot['successes']}/{robot['episodes']})")
  if key=='error_mean_mm':ax.fill_between(t,[v['error_low_mm'] for v in s],[v['error_high_mm'] for v in s],color=color,alpha=.12,linewidth=0)
 ax.set_title(title,loc='left',fontweight='bold');ax.set_ylabel(unit);ax.set_xlim(5,20);ax.set_xticks([5,10,15,20]);ax.set_ylim(bottom=0);ax.grid(axis='y',color='#dfe7eb',lw=.5)
 if key=='travel_mean_mm':ax.set_ylim(0,10);ax.axhline(4,color='#697e89',ls='--',lw=.8);ax.text(19.8,4.3,'4 mm threshold',ha='right',fontsize=7,color='#697e89')
 if ax in axes[1]:ax.set_xlabel('Time (s)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.52,.995))
for ext in ['pdf','svg','png']:fig.savefig(a.output_dir/f'embodiments-v2.{ext}',dpi=300,bbox_inches='tight',pad_inches=.04)
plt.close(fig)
manifest={'source_summary_sha256':hashlib.sha256(a.summary.read_bytes()).hexdigest(),'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'window_s':[5,20],'band':'TCP goal-error 10th–90th percentiles across paired resets; not training confidence intervals','comparability':'Whole robot/controller/fluid-model systems. Physics-model variants differ from historical held-arm v1.','figure_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output_dir.iterdir()}}
(a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
