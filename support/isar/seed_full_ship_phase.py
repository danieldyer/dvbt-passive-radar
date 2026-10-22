"""Measure whole-capture carrier paths on an acquired physical candidate.

The fixed observation grid is unchanged. Templates include actual native
epochs/weights within each retained bin. Smoothing uses training data only;
unwrapped curves are diagnostic initializers, not adopted motion solutions.
No per-channel retiming or correction is applied to recorded samples.
"""
import argparse,json
from pathlib import Path
import numpy as np
from full_ship_field import load
from joint_ship_motion import JointMotion
from motion import C,enu_basis,excess_path
from rigid_patch import profile_static
from gpga import Isolation
from matched_phase import history
from scipy.interpolate import CubicHermiteSpline
from rigid_patch import RigidPatch
from bistatic_path_lift import lift


def bin_carrier(time,rate,native_time,native_information,k):
    epochs=native_time.reshape(-1,16);weights=native_information.reshape(-1,16);total=weights.sum(axis=1)
    phase=np.exp(-1j*k*rate[:,None]*(epochs-time[:,None]))
    return np.divide(np.sum(weights*phase,axis=1),total,out=np.ones(len(time),complex),where=total>0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--field',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--bandwidth-hz',type=float,default=2.);p.add_argument('--coarse-track',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    fields,coordinates,description=load(a.field);candidate=json.loads(a.candidate.read_text());motion=JointMotion(Path(candidate['motion']));mid=candidate['mid_s']
    s=motion.base(np.array([mid]));sample=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.)
    local=enu_basis(sample['latitude'][0],sample['longitude'][0]);gauge=s['body_to_ecef'][0].T@local;point=gauge@np.asarray(candidate['points'])[0]
    track=None
    if a.coarse_track:
        archive=np.load(a.coarse_track);tt=archive['time_s'];rr=archive['path_m'];vv=archive['rate_mps']
        track=CubicHermiteSpline(tt,rr,vv,extrapolate=False)
    arrays={};reports=[]
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,12),layout='constrained',sharex=True)
    for c,(field,coordinate) in enumerate(zip(fields,coordinates)):
        t=field.time;physical=motion.state(t);position=physical['position_ecef_m']+np.einsum('tij,j->ti',physical['body_to_ecef'],point)
        velocity=physical['velocity_ecef_mps']+np.einsum('tij,j->ti',physical['frame_velocity'],point)
        if track is not None:
            clamped=np.clip(t,tt[0],tt[-1]);target=track(clamped);speed=track(clamped,1)
            # Only an explicit constant-velocity state propagation is used
            # outside acquired epochs. No unconstrained spline extrapolation.
            target+=speed*(t-clamped)
            kin=RigidPatch(t,157.,position,velocity,motion.local,motion.tx[3],motion.rx)
            lifted=lift(position,velocity,kin.direction,kin.direction_rate,target,speed,motion.tx[3],motion.rx)
            position=lifted['position_ecef_m'];velocity=lifted['velocity_ecef_mps']
        length,gradient=excess_path(position,motion.tx[c],motion.rx);rate=np.sum(gradient*velocity,axis=1)
        raw=field.raw_trajectory(position,velocity,transmitter=motion.tx[c],receiver=motion.rx,position_offset=position-field.data['centre_ecef_m'])
        factor=bin_carrier(t,rate,coordinate['native_time_s'],coordinate['native_information'],field.k)
        template=field.apply(raw*factor[:,None]);model=np.zeros_like(template);data=np.zeros_like(template);train=coordinate['training']
        for mask in [train,~train]:
            groups=coordinate.get('stationary_groups')
            model[mask]=profile_static(template,field.root_information,mask,groups)[mask];data[mask]=profile_static(field.y,field.root_information,mask,groups)[mask]
        z,info=history(data,model);valid=info>0
        smoother=Isolation(t,info,train&valid,a.bandwidth_hz);smooth,_=smoother.apply(z);phase=np.unwrap(np.angle(smooth[:,0]))
        phase-=np.interp(162.,t,phase);residual=-phase/field.k
        for key,value in dict(time_s=t,value=z[:,0],information=info,training=train,smooth=smooth[:,0],unwrapped_residual_path_m=residual,
                predicted_path_m=length,predicted_rate_mps=rate,bin_carrier=factor).items():arrays[f'{key}_{c}']=value
        name=['Seven','SBS','Nine','Ten','ABC'][c];axes[0].plot(t,residual,label=name);axes[1].plot(t,10*np.log10(np.maximum(abs(smooth[:,0])**2*info,1e-30)),label=name)
        axes[2].plot(t,length,label=name)
        reports.append(dict(channel=c,rows=len(t),phase_seed_span_m=float(np.ptp(residual)),maximum_bin_attenuation_db=float(-20*np.log10(max(abs(factor).min(),1e-30)))))
        print(json.dumps(reports[-1]),flush=True)
    axes[0].set(ylabel='Measured carrier-path residual (m)',title='One constant origin at 162 s; individual unwraps may contain slips');axes[0].legend(ncol=5)
    axes[1].set(ylabel='Smoothed matched information (dB)',title='Training-only display smoothing; weak/null intervals remain visible')
    axes[2].set(ylabel='Candidate physical excess path (m)',xlabel='Capture time (s)')
    fig.suptitle('Whole-capture measured phase with one fixed observation reference\nActual independent OFDM epochs; no channel retiming; known native-bin attenuation included; initializer only')
    fig.savefig(a.output/'full-phase-seed.png',dpi=140);plt.close(fig);np.savez_compressed(a.output/'histories.npz',point_body_m=point,**arrays)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,candidate=str(a.candidate),field=str(a.field),bands=reports,
        bandwidth_hz=a.bandwidth_hz,coarse_track=str(a.coarse_track) if a.coarse_track else None,adopted_as_motion=False),indent=2)+'\n')
if __name__=='__main__':main()
