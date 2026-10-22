"""GPGA initializer from the extended ship's exact matched histories.

Only the nominated training band estimates the phase gradient. The estimated
phase is converted to one physical path correction, not separate band screens.
It initializes the later exact multiband waveform likelihood. Gaussian
isolation, approximate common history variances and spline projection are
initializer operations; none replaces the final original-epoch likelihood.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse.linalg import lsqr

from full_ship_field import load
from full_path_point import PointBand
from full_path_scene import SceneBand
from gpga import fit
from joint_ship_motion import JointMotion
from path_dynamics import project_initial
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--channel',type=int,default=3);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']))
    fields,coordinates,_=load(Path(d['field']));c=a.channel;full=fields[c];coordinate=coordinates[c]
    f=full.windowed(*d['interval_s']);indices=np.searchsorted(full.time,f.time)
    coord={k:coordinate[k].reshape(-1,16)[indices].ravel() for k in ['native_time_s','native_information']};coord['training']=coordinate['training'][indices]
    if 'stationary_groups' in coordinate:coord['stationary_groups']=coordinate['stationary_groups'][indices]
    anchor=d.get('anchor_path_time_s',162.);b=PointBand(f,coord,motion,d['point_body_m'],np.asarray(d['path_knots_s']),np.asarray(d['path_coefficients_m']),anchor,0.,d['orientation'])
    band=SceneBand(b,np.asarray(d['scene_points_body_m']));h=band.design(np.zeros(b.b.shape[1]),c,False)
    history=np.zeros((len(b.t),band.phase_count),complex);normal=np.zeros(history.shape)
    for mask in [b.training,~b.training]:
        hp=profile_static(h,f.root_information,mask,b.groups);y=profile_static(f.y,f.root_information,mask,b.groups)
        history[mask]=np.einsum('trp,tr->tp',hp[mask].conj(),y[mask]);normal[mask]=np.sum(abs(hp[mask])**2,axis=1)
    w=f.root_information**2
    normalized_norm=np.divide(normal,w[:,None],out=np.zeros_like(normal),where=w[:,None]>0)
    column_scale=np.sqrt(np.average(normalized_norm[b.training],axis=0,weights=w[b.training]))
    value=np.divide(history,w[:,None]*column_scale,out=np.zeros_like(history),where=w[:,None]>0)
    result=fit(b.t,value,w,b.training,bandwidths_hz=[2.,1.,.5,.25,.125],iterations_per_bandwidth=20)
    phase=result.pop('phase_radians');correction=-phase/f.k
    coefficient=lsqr(b.b[b.training],correction[b.training],atol=1e-12,btol=1e-12,iter_lim=500)[0]
    base=np.asarray(d['path_coefficients_m']);origin=BSpline(d['path_knots_s'],np.eye(len(base)),3)(anchor)
    change=np.r_[coefficient,0.];coeff=base+change-origin@change
    projected,projection=project_initial(d['path_knots_s'],coeff,d['max_acceleration_mps2'],anchor)
    curve=BSpline(d['path_knots_s'],projected-base,3)
    # Remove a constant phase before measuring projection error: source/band
    # phases are re-estimated, using training observations only, by the fitter.
    error=phase+f.k*curve(b.t);error-=np.average(error[b.training],weights=w[b.training])
    output=dict(d);output.update(path_coefficients_m=projected.tolist(),scope=__doc__,source_gpga_motion=str(a.motion),
        gpga_training_channel=c,gpga_diagnostics=result,physical_projection=projection,
        training_phase_projection_rms_deg=float(np.degrees(np.sqrt(np.average(error[b.training]**2,weights=w[b.training])))),
        initializer_only=True,numerical_convergence=False,accepted_as_vessel_motion=False)
    np.savez_compressed(a.output/'histories.npz',time_s=b.t,training=b.training,value=value,information=w,
        estimated_phase_radians=phase,physical_path_change_m=curve(b.t),norm_ratio=normalized_norm/column_scale[None,:]**2)
    (a.output/'analysis.json').write_text(json.dumps(serial(output),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained',sharex=True)
    axes[0].plot(b.t,np.degrees(phase),label='Training-only GPGA estimate');axes[0].plot(b.t,-np.degrees(f.k*curve(b.t)),label='Bounded physical spline initializer');axes[0].legend();axes[0].set_ylabel('Phase correction (degrees)')
    for point in range(min(6,value.shape[1])):
        mask=~b.training;before=value[mask,point];after=before*np.exp(-1j*phase[mask]);axes[1].scatter(b.t[mask],np.degrees(np.angle(after)),s=2,alpha=.35)
    axes[1].set(xlabel='Capture time (s)',ylabel='Excluded GPGA residual phase (deg)',ylim=[-180,180])
    fig.suptitle('Extended-scene phase-gradient initializer; no independent-band phase screen\nApproximate histories acquire a seed; exact physical multiband likelihood must decide whether it transfers')
    fig.savefig(a.output/'gpga-path-initializer.png',dpi=140);plt.close(fig)
    print(json.dumps(serial({k:output[k] for k in ['gpga_training_channel','training_phase_projection_rms_deg','physical_projection','gpga_diagnostics']})),flush=True)


if __name__=='__main__':main()
