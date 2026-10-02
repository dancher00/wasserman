"""Predeclared contact events, exposure-aware metrics and paired uncertainty."""
import numpy as np

DT=1/30
PREFIX=600


def contact_events(contact):
    """Confirm acquisition/loss with 3 samples; no acquisition means no loss."""
    established=False; positive=negative=losses=0
    first_acquisition=first_loss=None
    for j, present in enumerate(np.asarray(contact,dtype=bool)):
        if present:
            positive+=1;negative=0
            if not established and positive>=3:
                established=True
                if first_acquisition is None:first_acquisition=(j+1)*DT
        else:
            negative+=1;positive=0
            if established and negative>=3:
                losses+=1;established=False
                if first_loss is None:first_loss=(j+1)*DT
    return dict(contact_acquired=int(first_acquisition is not None),contact_lost=int(losses>0),
        contact_loss_count=losses,first_acquisition_s=first_acquisition,first_loss_s=first_loss)


def arrays(payload,physical):
    trace=payload['trace'];extra=physical['trace']
    if len(trace)!=len(extra):raise ValueError('Physical instrumentation and scoring traces differ in length')
    keys=['active','reset','grasped','angle','station_position_error','base_attitude','motor_force','motor_saturation_scale','contact_force_vectors']
    a={k:np.stack([r[k].numpy() for r in trace]) for k in keys}
    a.update({k:np.stack([r[k].numpy() for r in extra]) for k in ['base_position','base_linear_velocity','base_angular_velocity']})
    a['initial_position']=physical['initial']['before_current_and_warmup']['base_position'].numpy()
    a['initial_angle']=physical['initial']['after_warmup']['mechanism_joints'].numpy()[:,0]
    a['valid']=a['active'] & ~a['reset']
    f=a['contact_force_vectors'];mag=np.linalg.norm(f,axis=-1)
    unit=f/np.maximum(mag[...,None],1e-6)
    a['opposing']=(mag>.1).all(-1)&((unit[:,:,0]*unit[:,:,1]).sum(-1)<-.25)
    return a


def episode_metrics(a,index,mask,limits):
    mask=np.asarray(mask,dtype=bool)
    ids=np.flatnonzero(mask)
    if ids.size and not np.array_equal(ids,np.arange(ids.size)):
        raise ValueError('Expected a continuous observed prefix')
    n=int(ids.size)
    if not n:return dict(exposure_s=0.,metrics_available=False)
    def get(k):return a[k][:len(mask),index][mask].astype(float)
    motor=get('motor_force');base=get('base_position')-a['initial_position'][index]
    angle=get('angle');prev=np.r_[a['initial_angle'][index],angle[:-1]]
    contact=get('opposing').astype(bool)
    util=np.abs(motor)/np.where(motor>=0,limits[1],-limits[0])
    result=dict(exposure_s=n*DT,metrics_available=True,
        opposing_contact_fraction=float(contact.mean()),native_grasp_fraction=float(get('grasped').mean()),
        signed_progress_deg=float(np.rad2deg(angle[-1]-a['initial_angle'][index])),
        max_progress_deg=float(np.rad2deg(np.max(angle)-a['initial_angle'][index])),
        progress_during_contact_deg=float(np.rad2deg(((angle-prev)*contact).sum())),
        base_displacement_rms_m=float(np.sqrt(np.mean(np.sum(base**2,axis=-1)))),
        station_position_rms_m=float(np.sqrt(np.mean(np.sum(get('station_position_error')**2,axis=-1)))),
        world_attitude_rms_deg=float(np.rad2deg(np.sqrt(np.mean(get('base_attitude')**2)))),
        base_linear_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(get('base_linear_velocity')**2,axis=-1)))),
        base_angular_speed_rms_rad_s=float(np.sqrt(np.mean(np.sum(get('base_angular_velocity')**2,axis=-1)))),
        motor_force_rms_N=float(np.sqrt(np.mean(motor**2))),motor_force_peak_N=float(np.abs(motor).max()),
        motor_peak_utilization=float(util.max()),
        saturation_fraction=float((get('motor_saturation_scale').reshape(n,-1).min(-1)<1-1e-6).mean()),
        **contact_events(contact))
    return result


def paired_interval(x,y,draws=20000):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.shape!=y.shape or x.shape!=(3,30):raise ValueError('Expected 3 trainings × 30 paired resets')
    delta=x-y;valid=np.isfinite(delta)
    if not valid.any():return dict(difference=None,descriptive_paired_crossed_95=None,finite_pairs=0)
    # Equal-training means; a completely absent training is reported, not silently included as zero.
    def average(v):
        count=np.isfinite(v).sum(-1);means=np.divide(np.nansum(v,axis=-1),count,out=np.full(count.shape,np.nan,dtype=float),where=count>0)
        count2=np.isfinite(means).sum(-1)
        return np.divide(np.nansum(means,axis=-1),count2,out=np.full(count2.shape,np.nan,dtype=float),where=count2>0)
    rng=np.random.default_rng(20261001)
    trains=rng.integers(0,3,size=(draws,3));resets=rng.integers(0,30,size=(draws,30))
    distribution=average(delta[trains[:,:,None],resets[:,None,:]])
    finite=distribution[np.isfinite(distribution)]
    return dict(difference=float(average(delta)),descriptive_paired_crossed_95=np.percentile(finite,[2.5,97.5]).tolist(),
        finite_pairs=int(valid.sum()),finite_bootstrap_draws=len(finite),draws=draws)
