"""Compare measured point-history gradients with rigid-motion velocity modes.

This is an instantaneous diagnostic conditional on the current pose, not a
motion correction. Gaussian isolation uses training epochs only. Histories
remain correlated and may contain the same unresolved bright return at several
candidate points; a high regression score is therefore NOT a heading estimate
or independent-scatterer certificate. In particular a fitted angular increment
that cancels the assumed rotation can be a point-collapse ambiguity.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bistatic_path_lift import lift
from full_ship_field import load
from full_path_point import PointBand
from gpga import Isolation
from joint_ship_motion import JointMotion
from motion import excess_path


def velocity_modes(direction,lever,value,weight):
    x=np.column_stack([direction,lever]);weight=np.asarray(weight)
    keep=np.isfinite(value)&np.isfinite(weight)&(weight>0)
    if np.count_nonzero(keep)<3:return None
    a=x[keep]*np.sqrt(weight[keep,None]);y=value[keep]*np.sqrt(weight[keep])
    norm=np.linalg.norm(a,axis=0)
    if np.min(norm)<=0:return None
    u,s,vh=np.linalg.svd(a/norm,full_matrices=False)
    if s[-1]<s[0]*1e-3:return None
    parameter=(vh.T@((u.T@y)/s))/norm
    common=np.dot(a[:,0],y)/np.dot(a[:,0],a[:,0])
    rss0=float(np.sum((y-a[:,0]*common)**2));rss1=float(np.sum((y-a@parameter)**2))
    return dict(path_rate_mps=float(parameter[0]),angular_rate_radps=float(parameter[1]),
        normalized_condition=float(s[0]/s[-1]),common_path_rss=rss0,rigid_velocity_rss=rss1,
        spatial_gradient_fraction_explained=1-rss1/rss0 if rss0>0 else None,points=int(keep.sum()))


def isolated_gradients(time,value,information,training,norm_ratio,bandwidth):
    smoother=Isolation(time,information,training,bandwidth)
    z,dz=smoother.apply(value)
    variance=norm_ratio[training]/information[training,None]
    vz=smoother.kernel.power(2)@variance;vd=smoother.derivative.power(2)@variance
    covariance=smoother.kernel.multiply(smoother.derivative)@variance
    slope=np.divide(dz,z,out=np.zeros_like(dz),where=abs(z)>0)
    noise=np.maximum(vd+abs(slope)**2*vz-2*slope.real*covariance,0.)
    gradient_variance=np.divide(noise,2*abs(z)**2,out=np.full_like(noise,np.inf),where=abs(z)>0)
    snr=np.divide(abs(z)**2,vz,out=np.zeros_like(vz),where=vz>0)
    return slope.imag,gradient_variance,snr


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--histories',type=Path,nargs='+',required=True);p.add_argument('--bandwidth-hz',type=float,default=.25)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']));points=np.asarray(d['scene_points_body_m'])
    fields,coordinates,_=load(Path(d['field']));reference_point=np.asarray(d['point_body_m']);results=[];arrays={}
    from matplotlib import pyplot as plt
    times=[158.,161.,164.,168.,175.];fig,axes=plt.subplots(len(a.histories),len(times),figsize=(21,3.5*len(a.histories)),layout='constrained',squeeze=False)
    summary,summary_axes=plt.subplots(3,1,figsize=(15,11),layout='constrained',sharex=True)
    names=['Seven','SBS','Nine','Ten','ABC']
    for row,root in enumerate(a.histories):
        meta=json.loads((root/'analysis.json').read_text())
        if Path(meta['source_gpga_motion'])!=a.motion:raise ValueError('Histories use another frozen physical motion')
        c=meta['gpga_training_channel'];z=np.load(root/'histories.npz');f=fields[c].windowed(*d['interval_s']);indices=np.searchsorted(fields[c].time,f.time)
        coordinate=coordinates[c];coord={k:coordinate[k].reshape(-1,16)[indices].ravel() for k in ['native_time_s','native_information']};coord['training']=coordinate['training'][indices]
        b=PointBand(f,coord,motion,reference_point,np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),d['anchor_path_time_s'],0.,d['orientation'])
        np.testing.assert_array_equal(b.t,z['time_s'])
        physical=lift(b.anchor,b.velocity,b.direction,b.direction_rate,b.base_path,b.base_rate,motion.tx[3],motion.rx)
        displacement=np.einsum('tij,pj->tpi',b.frame,points-reference_point)
        position=physical['position_ecef_m'][:,None,:]+displacement
        spin_offset=np.c_[-(points-reference_point)[:,1],(points-reference_point)[:,0],np.zeros(len(points))]
        angular_position=np.einsum('tij,pj->tpi',b.frame,spin_offset)
        direction=[];lever=[]
        for point in range(len(points)):
            _,g=excess_path(position[:,point],motion.tx[c],motion.rx)
            direction.append(np.sum(g*physical['position_path_jacobian'],axis=1))
            lever.append(np.sum(g*angular_position[:,point],axis=1))
        direction=np.asarray(direction).T;lever=np.asarray(lever).T
        gradient,variance,snr=isolated_gradients(b.t,z['value'],z['information'],z['training'],z['norm_ratio'],a.bandwidth_hz)
        rate=-gradient/f.k;rate_variance=variance/f.k**2
        weight=np.divide(1.,rate_variance,out=np.zeros_like(variance),where=(rate_variance>0)&(snr>9))
        trials=[velocity_modes(g,l,y,w) for g,l,y,w in zip(direction,lever,rate,weight)]
        angular=np.asarray([np.nan if q is None else q['angular_rate_radps'] for q in trials]);path=np.asarray([np.nan if q is None else q['path_rate_mps'] for q in trials])
        fraction=np.asarray([np.nan if q is None else q['spatial_gradient_fraction_explained'] for q in trials])
        spin=np.einsum('tji,tjk->tik',b.frame,b.frame_rate);assumed=.5*(spin[:,1,0]-spin[:,0,1])
        summary_axes[0].plot(b.t,path,label=names[c]);summary_axes[1].plot(b.t,np.degrees(angular),label=names[c]);summary_axes[2].plot(b.t,fraction,label=names[c])
        if row==0:summary_axes[1].plot(b.t,-np.degrees(assumed),'k--',label='Cancels assumed rotation: possible collapse ambiguity')
        samples=[]
        for column,t in enumerate(times):
            at=int(np.argmin(abs(b.t-t)));ax=axes[row,column];good=weight[at]>0
            ax.scatter(lever[at],rate[at],c=np.where(good,snr[at],0),cmap='viridis',s=np.clip(snr[at],8,60))
            trial=trials[at]
            if trial is not None:
                predicted=direction[at]*trial['path_rate_mps']+lever[at]*trial['angular_rate_radps'];order=np.argsort(lever[at]);ax.plot(lever[at,order],predicted[order],color='#c90040',lw=1)
                ax.set_title(f'{names[c]} · {b.t[at]:.1f} s\nΔspin {np.degrees(trial["angular_rate_radps"]):+.3f}°/s; fit {trial["spatial_gradient_fraction_explained"]:.2f}')
            else:ax.set_title(f'{names[c]} · {b.t[at]:.1f} s · unresolved modes')
            ax.set(xlabel='Physical angular lever (m/rad)',ylabel='Measured residual path rate (m/s)')
            samples.append(dict(time_s=float(b.t[at]),fit=trial,lever_m_per_rad=lever[at].tolist(),rate_mps=rate[at].tolist(),snr=snr[at].tolist()))
        results.append(dict(channel=c,samples=samples))
        arrays.update({f'time_s_{c}':b.t,f'measured_rate_mps_{c}':rate,f'lever_m_per_rad_{c}':lever,
            f'conditional_gradient_weight_{c}':weight,f'path_rate_mps_{c}':path,f'angular_rate_radps_{c}':angular,f'assumed_rate_radps_{c}':assumed})
    fig.suptitle('Measured gradients versus instantaneous physical velocity modes\nTraining-only isolation; correlated/unresolved histories are diagnostics, not independent point phases')
    fig.savefig(a.output/'instantaneous-phase-rigidity.png',dpi=140);plt.close(fig)
    summary_axes[0].set_ylabel('Common path-rate increment (m/s)');summary_axes[1].set_ylabel('Angular-rate increment (deg/s)');summary_axes[2].set(ylabel='Spatial gradient fit fraction',xlabel='Capture time (s)',ylim=[-.02,1.02])
    for ax in summary_axes:ax.legend(fontsize=8,ncol=3)
    summary.suptitle('Physical velocity-gradient diagnostic; no correction adopted\nA regression can identify the same unresolved return in several pixels; point collapse is explicitly shown')
    summary.savefig(a.output/'phase-rigidity-timeseries.png',dpi=140);plt.close(summary)
    np.savez_compressed(a.output/'histories.npz',**arrays)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),bandwidth_hz=a.bandwidth_hz,
        channels=results,heading_estimated=False,correction_applied=False),indent=2)+'\n')


if __name__=='__main__':main()
