"""Blue, data-bound current-study figures; no interpolation beyond measured levels."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from current_contact_study import ROOT,digest
P=ROOT/'research/current-contact-5090';R=P/'results';F=P/'figures'

def main():
    F.mkdir(exist_ok=True)
    result=json.loads((R/'summary.json').read_text());rows=list(csv.DictReader((R/'training-seeds.csv').open()))
    tasks=['OpenHatch','RotateValve'];speeds=[0.,.1,.2];seeds=[17,43,101]
    colors=['#123B66','#176BA0','#4285AD'];markers=['o','s','^'];styles=['-','--',':']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    exports=[]
    def save(fig,name):
        for ext in ['pdf','svg','png']:
            path=F/f'{name}.{ext}';fig.savefig(path,dpi=300);exports.append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.35),layout='constrained',sharey=True)
    success_ranges=[0.,100.]
    for ax,task in zip(axes,tasks):
        values=[]
        for seed,col,marker,style in zip(seeds,colors,markers,styles):
            vals=[next(float(r['success_rate'])*100 for r in rows if r['task']==task and int(r['training_seed'])==seed and float(r['speed_m_s'])==speed) for speed in speeds]
            values.append(vals);ax.plot(speeds,vals,color=col,marker=marker,linestyle=style,label=f'Training seed {seed}',linewidth=1.3,markersize=5)
        v=np.array(values);success_ranges.extend((v.mean(0)-v.std(0,ddof=1)).tolist());success_ranges.extend((v.mean(0)+v.std(0,ddof=1)).tolist());ax.errorbar(np.array(speeds)+.004,v.mean(0),yerr=v.std(0,ddof=1),fmt='D',color='#092540',capsize=3,markersize=4,label='Mean ± training SD')
        ax.set(title=task,xlabel='Transverse current (m/s)',xticks=speeds,ylim=(-5,105));ax.grid(axis='y',alpha=.2)
    for ax in axes:ax.set_ylim(min(success_ranges)-5,max(success_ranges)+5)
    axes[0].set_ylabel('Task success (%)');axes[1].legend(fontsize=8,loc='best')
    save(fig,'success-by-current')
    definitions=[('success_rate','Success difference (pp)',100),('signed_progress_deg','Early progress difference (°)',1),('opposing_contact_fraction','Opposing contact difference (pp)',100),('contact_lost','Contact-loss incidence difference (pp)',100),('station_position_rms_m','Position-error RMS difference (mm)',1000),('motor_force_rms_N','Motor-force RMS difference (N)',1)]
    fig,axes=plt.subplots(3,2,figsize=(7.1,8.1),layout='constrained')
    labels=[f'{t}\n{v:.2f} − 0 m/s' for t in tasks for v in [.1,.2]]
    for ax,(key,title,scale) in zip(axes.flat,definitions):
        for j,(task,speed) in enumerate((t,s) for t in tasks for s in [.1,.2]):
            c=next(r for r in result['comparisons'] if r['task']==task and r['reference_speed_m_s']==0 and r['speed_m_s']==speed)['contrasts'][key]
            if c['difference'] is None:
                ax.text(0,j,'No observed exposure');continue
            x=c['difference']*scale;lo,hi=np.array(c['descriptive_paired_crossed_95'])*scale
            ax.plot([lo,hi],[j,j],color=colors[0 if speed==.1 else 1],linewidth=2)
            ax.plot(x,j,marker='o' if speed==.1 else 's',color=colors[0 if speed==.1 else 1])
        ax.axvline(0,color='.5',linewidth=.8,linestyle='--');ax.set(yticks=range(4),yticklabels=labels,xlabel=title,ylim=(3.5,-.5));ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.15)
    save(fig,'paired-current-effects')
    # Standalone dynamics figure exposes state/actuator behavior even at saturated task success.
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.2),layout='constrained')
    for ax,(key,title,scale) in zip(axes.flat,[('base_displacement_rms_m','Base displacement RMS (mm)',1000),('world_attitude_rms_deg','World-identity attitude RMS (°)',1),('motor_force_rms_N','Motor-force RMS (N)',1),('saturation_fraction','Allocation saturation (%)',100)]):
        for task,col,marker in zip(tasks,colors[:2],markers[:2]):
            means=[];sd=[]
            for speed in speeds:
                v=next(g for g in result['groups'] if g['task']==task and g['speed_m_s']==speed)[key];means.append(v['mean']*scale);sd.append(v['sd']*scale)
            ax.errorbar(speeds,means,yerr=sd,color=col,marker=marker,linestyle='-' if task==tasks[0] else '--',capsize=3,label=task)
        ax.set(xlabel='Transverse current (m/s)',ylabel=title,xticks=speeds);ax.grid(alpha=.15)
        if key=='saturation_fraction' and all(g[key]['mean']==0 and g[key]['sd']==0 for g in result['groups']):
            ax.set(ylim=(-.005,.10),yticks=[0,.05,.10])
            ax.text(.5,.55,'0% in every observed\nfirst-20-second window',transform=ax.transAxes,ha='center',va='center',fontsize=9,color=colors[0])
    axes[0,0].legend(fontsize=8);save(fig,'vehicle-response')
    captions='''# Figure definitions and accessible descriptions

`success-by-current`: Each thin line is one of three independent trained DP
checkpoints, evaluated on the same 30 ordered task resets at each current level.
Diamonds and whiskers give equal-training mean and sample SD, not confidence
intervals. Mean markers are shifted +0.004 m/s horizontally solely for visibility;
all observations were collected at 0, 0.10 or 0.20 m/s. Lines connect only the three tested conditions; they do not estimate
intermediate currents. Source: `results/training-seeds.csv`.

`paired-current-effects`: Points are current-minus-zero paired differences.
Horizontal intervals are descriptive paired crossed 95% percentile bootstrap
intervals (20,000 draws, three training and 30 reset indices, shared between
conditions). Early metrics use each pair's common observed prefix, at most
20 seconds; success uses the complete task horizon. Contact-loss incidence
uses all episodes, so never-acquired contact is not counted as a loss.
Intervals are marginal, not multiplicity-adjusted. Source: `results/summary.json`.

`vehicle-response`: Equal-training means and sample SD of individual observed
first-20-second episode metrics. The exact exposure per episode is in
`results/episodes.csv`. Displacement is measured from the initial base position
and includes commanded approach motion; it must not be read as current-induced drift.
attitude is relative to world identity, not a commanded attitude. Allocation
saturation is a control-sample frequency. Source: `results/summary.json`.

All panels retain zero, negative and unresolved results. Color is redundant with
markers/line styles and direct labels. PDF/SVG are vector exports; PNG is 300 DPI.
Widths are 7.1 inches for provisional two-column manuscript use, without claiming
venue certification. Underlying tables and registered metric definitions are
provided. No smoothing, fitted trend, discarded seed or hidden failure is used.
'''
    (F/'README.md').write_text(captions)
    (F/'manifest.json').write_text(json.dumps(dict(source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [R/'summary.json',R/'training-seeds.csv']},plot_script_sha256=digest(__file__),files=exports),indent=2)+'\n')

if __name__=='__main__':main()
