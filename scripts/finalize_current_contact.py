"""Verify selected video reruns and write the compact scientific handoff."""
import json,subprocess
from pathlib import Path
import torch
from current_contact_study import ROOT,digest
from revision_scoring_evidence import check_embedded_report,checked_tensor_outcomes
P=ROOT/'research/current-contact-5090';A=ROOT/'artifacts/current-contact-5090'

def main():
    result=json.loads((P/'results/summary.json').read_text())
    selection=json.loads((P/'video-selection.json').read_text())
    clips=[]
    for selected in selection['clips']:
        task=selected['task'];speed=selected['speed_m_s'];seed=selected['training_seed'];j=selected['batch_index']
        label=f'video-{task}-DP-{seed}-u{speed:g}';folder=A/'video'/label
        report=json.loads((folder/'summary.json').read_text());c=json.loads((folder/'current-condition.json').read_text())
        data=torch.load(folder/'rollout.pt',map_location='cpu',weights_only=True)
        check_embedded_report(data['summary'],report);checked_tensor_outcomes(task,report,data['trace'])
        assert report['seeds'][j]==selected['reset_id']
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        probe=subprocess.run([ffmpeg,'-hide_banner','-i',str(folder/'wrist.mp4')],capture_output=True,text=True)
        assert ': Video:' in probe.stderr and ': Audio:' not in probe.stderr
        frames, duration=imageio_ffmpeg.count_frames_and_secs(str(folder/'wrist.mp4'))
        assert frames==c['video_frames']
        success=bool(report['success_per_seed'][j]);reset=report['first_reset_step'][j]
        expected=report['first_success_step'][j] if success else reset if reset>0 else report['evaluated_steps']
        assert frames==expected,(label,frames,expected)
        clips.append(dict(**selected,path=str((folder/'wrist.mp4').relative_to(ROOT)),sha256=digest(folder/'wrist.mp4'),
            frames=frames,fps=30,audio_streams=0,overlaid_text=False,video_success=success,
            agrees_with_scored_outcome=success==selected['scored_success'],video_first_reset_step=reset,
            report_sha256=digest(folder/'summary.json'),physical_rescoring_passed=True,
            note='Full ordered 30-environment rerun; only the selected slot recorded. Rerun is illustrative, never replacement evidence.'))
    (P/'video-manifest.json').write_text(json.dumps(dict(study=result['study'],clips=clips),indent=2)+'\n')
    groups={(g['task'],g['speed_m_s']):g for g in result['groups']}
    lines=['# Current/contact study: text for manuscript integration','',
        '## Setup','',
        'We evaluate fixed diffusion policies on OpenHatch and RotateValve under '
        'constant world-frame currents (0, 0, 0), (0, 0.10, 0), and (0, 0.20, 0) m/s. '
        'Three independently trained checkpoints per task share 30 ordered, previously unused '
        'research resets across conditions (540 episodes). Current starts immediately before '
        'the camera-warmup step. Pre-current physical states and randomized parameters match; '
        'controller gains, coefficients, thruster limits, rendering, action decoding and '
        'success criteria remain fixed. Policies remain in feedback, so commands can change '
        'with the observations. Baseline trials are newly executed in this series. '
        'Before policy evaluation, an unchanged five-second expert pilot on disjoint '
        'development states verified the intervention and finite responses; it did '
        'not establish full-horizon expert feasibility under current.','',
        '## Numerical results','',
        '| Task | Current (m/s) | Successes/30, seeds 17/43/101 | Mean ± training SD (%) |',
        '| --- | ---: | --- | ---: |']
    for task in ['OpenHatch','RotateValve']:
        for speed in [0.,.1,.2]:
            g=groups[task,speed];m=g['success_rate'];counts='/'.join(map(str,g['per_training_successes']))
            lines.append(f"| {task} | {speed:.2f} | {counts} | {100*m['mean']:.1f} ± {100*m['sd']:.1f} |")
    lines+=['','Current-minus-zero differences below use paired crossed bootstrap intervals '
        '(20,000 draws over three training identities and 30 reset identities, shared between '
        'conditions). They are descriptive marginal 95% intervals, not multiplicity-adjusted.','']
    for c in result['comparisons']:
        if not c['primary']:continue
        r=c['contrasts']['success_rate'];lo,hi=r['descriptive_paired_crossed_95']
        lines.append(f"- {c['task']}, {c['speed_m_s']:.2f} m/s: success difference {r['difference']*100:+.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}].")
        for key,title,scale,unit in [('signed_progress_deg','early signed progress',1,'degrees'),('opposing_contact_fraction','opposing-contact occupancy',100,'pp'),('contact_lost','contact-loss incidence',100,'pp'),('station_position_rms_m','position-error RMS',1000,'mm'),('motor_force_rms_N','motor-force RMS',1,'N')]:
            v=c['contrasts'][key]
            if v['difference'] is not None:
                low,high=v['descriptive_paired_crossed_95'];lines.append(f"  {title}: {v['difference']*scale:+.3f} {unit} [{low*scale:+.3f}, {high*scale:+.3f}].")
    hatch0=groups['OpenHatch',0.]['motor_force_rms_N']['mean']
    hatch2=groups['OpenHatch',.2]['motor_force_rms_N']['mean']
    valve_loss=[groups['RotateValve',v]['contact_lost']['mean']*100 for v in [0.,.1,.2]]
    dose=next(c for c in result['comparisons'] if c['task']=='RotateValve' and c['reference_speed_m_s']==.1)['contrasts']['success_rate']
    dose_lo,dose_hi=dose['descriptive_paired_crossed_95']
    assert all(g['exposure_s']['mean']==20 and g['saturation_fraction']['mean']==0 for g in result['groups'])
    lines+=['','## Main interpretation','',
        f'All three Valve checkpoints completed fewer episodes under both currents than '
        f'under their own zero-current baseline. The observed response is not a monotonic '
        f'dose–response: the 0.20-minus-0.10 m/s success contrast is {100*dose["difference"]:+.1f} pp with a '
        f'descriptive interval of [{100*dose_lo:+.1f}, {100*dose_hi:+.1f}] pp. Early contact-loss incidence across '
        f'all evaluated episodes was {valve_loss[0]:.1f}%, {valve_loss[1]:.1f}% and '
        f'{valve_loss[2]:.1f}% at 0, 0.10 and 0.20 m/s. These joint observations do '
        f'not establish contact loss as a causal mediator of task failure. '
        f'OpenHatch retained 30/30 completion in every checkpoint/condition, while '
        f'early motor-force RMS increased from {hatch0:.3f} to {hatch2:.3f} N '
        f'under 0.20 m/s (+{100*(hatch2/hatch0-1):.1f}%). Thus completion alone '
        f'misses this change in actuator effort. Every evaluated episode supplied '
        f'the entire 20-second early window; no allocation saturation was recorded '
        f'in that window. These results support task-dependent simulated current '
        f'sensitivity, not equivalence or universal robustness.','',
        '## Interpretation and limits','',
        'The contrasts estimate sensitivity of these declared simulated policy–controller '
        'systems to an imposed transverse water velocity, including its onset during '
        'camera warmup rather than an independently equilibrated steady-flow condition. '
        'Flow enters the existing robot-link relative-velocity hydrodynamics; '
        'the task mechanisms retain their original buoyancy/contact dynamics, without '
        'a newly modeled flow load on the hatch or valve. Early metrics use matched observed '
        'prefixes of at most 20 seconds; exact exposures are retained. Contact loss requires '
        'prior established opposing contact and is not necessarily an unintended failure. '
        'Conditional loss among acquired grasps is descriptive because current can affect '
        'acquisition. Success-conditional times use different survivor sets; capped completion '
        'time is separately labeled. Three trainings provide limited information about training '
        'variability. The intervals do not separately estimate variability from repeated '
        'executions of the same checkpoint/reset; illustrative video reruns are not an '
        'additional replication study. Boundary intervals can be degenerate and do not imply certainty. '
        'No causal mediation by contact, calibrated hydrodynamics, CFD agreement, hardware '
        'transfer, or robustness to other current directions/speeds is established.','',
        'The four selected clips follow the registered rule; their manifest distinguishes '
        'scored and recreated outcomes. Final scientific interpretation requires accountable '
        'human author verification. The shared manuscript and site were not edited.']
    (P/'paper-text.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'clips':len(clips),'video_outcome_agreements':sum(c['agrees_with_scored_outcome'] for c in clips)}))

if __name__=='__main__':main()
