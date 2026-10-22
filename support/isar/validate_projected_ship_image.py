"""Known-covariance clutter-projected ship image and native point-response control.

P is the orthogonal constant-profile nuisance projection after fixed range
whitening. The scalar image statistic is |h^H P y|²/(h^H P h). Native source
rows, calibration and physical motion are frozen. No fitted image alignment,
adaptive colour normalization or local background estimate enters the image.
The covariance and scattering model remain explicit conditional assumptions.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from amplitude_arc import linear_basis
from field import average
from midpoint_model import representative
from native_folds import training_mask
from recorded_injection_fields import load_fields, read_profiles, common_projection
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial
from rigid_patch import RigidPatch
from template_information import gpu as gpu_information, reference as reference_information,bin_moments
from waveform_kernels import KernelField


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation/ais-near-three-point-30s'))
    p.add_argument('--profiles',type=Path,default=Path('build/isar-20261018/current-native-full/profiles.bin'))
    p.add_argument('--point-profiles',type=Path,default=Path('build/isar-20261018/ais-near-point-spread/brightness-profiles.bin'))
    p.add_argument('--interval',type=float,nargs=2,default=[142.04,171.96])
    p.add_argument('--spacing-m',type=float,default=4.)
    p.add_argument('--extent-m',type=float,nargs=2,default=[320.,500.])
    p.add_argument('--brightness',action='store_true',help='Use the training-fitted real visibility of the anchor as a conditional temporal template')
    p.add_argument('--fold',choices=['all','training','testing'],default='all')
    p.add_argument('--resume',action='store_true',help='Reuse a byte-verified completed information model after an interrupted later check')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=args.resume)
    saved=json.loads((args.fit/'analysis.json').read_text());fit=json.loads((args.fit/'render-fit.json').read_text())
    first,last=args.interval;mid=fit['midpoint_s'];state=np.asarray(fit['coarse_state_enu']);parameters=np.asarray(fit['rigid_parameters'])
    fields,rotation,rx,tx,_,_=load_fields(args.profiles,offsets=np.arange(-12,13),fixed_epoch_s=mid,interval_s=[first,last])
    header,point=read_profiles(args.point_profiles)
    gain=np.asarray(fit['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0];gain=gain[:,0]+1j*gain[:,1]
    brightness=saved['phase_histories']['brightness_model']
    native=[];bands=[];point_y=[];selected=[];amplitude=[];native_amplitude=[];source_amplitude=[]
    for c,f in enumerate(fields):
        ch=point[c];np.testing.assert_array_equal(ch['rows'][:,1],f.data['row_time'])
        np.testing.assert_array_equal(ch['information']*ch['rows'][:,4],f.data['row_information'])
        point_y.append(common_projection(f,ch,header['first_range_sample']).reshape(f.y.shape))
        mean=representative(f);blocks=f.pulse.row_blocks.reshape(len(f.time),16)
        if np.any(blocks!=blocks[:,0,None]):raise ValueError('Pulse identity changed within an averaging group')
        mean=KernelField(mean,f.pulse.kernels,blocks[:,0])
        keep=np.ones(len(f.time),bool) if args.fold=='all' else training_mask(f.time,saved['interval_s'][0])
        if args.fold=='testing':keep=~keep
        selected.append(keep)
        coefficient=np.asarray(brightness['bands'][c]['coefficients'])[0];knots=np.asarray(brightness['bands'][c]['knots_s'])
        amplitude.append(linear_basis(mean.data['row_time'],knots)@coefficient/coefficient.mean() if args.brightness else np.ones(len(mean.time)))
        source_amplitude.append(linear_basis(f.data['row_time'],knots)@coefficient/coefficient.mean())
        native_amplitude.append(source_amplitude[-1] if args.brightness else np.ones(len(f.data['row_time'])))
        for field,target in [(f,native),(mean,bands)]:
            dt=field.data['row_time']-mid
            position=rx+(state[0]+dt[:,None]*state[1]+.5*dt[:,None]**2*state[2])@rotation
            velocity=(state[1]+dt[:,None]*state[2])@rotation
            kin=RigidPatch(field.data['row_time'],mid,position,velocity,rotation.T,
                np.asarray(fit['reference_transmitter_ecef_m']),rx,translation_direction=fit.get('translation_direction_ecef'))
            target.append(SimpleNamespace(field=field,kinematics=kin,training=keep,tx=tx[c],rx=rx,origin=0.))
    source_checks=[]
    for c,b in enumerate(native):
        f=b.field;offset,velocity=b.kinematics.point_state(parameters,np.zeros(3),origin=f.data['centre_ecef_m'])
        raw=f.raw_trajectory(f.data['centre_ecef_m']+offset,velocity,position_offset=offset)*source_amplitude[c][:,None]*gain[c]
        expected=f.apply(average(raw,f.data['row_information'],16)[0]);actual=point_y[c]
        inner=np.vdot(expected,actual);norm=np.vdot(expected,expected).real*np.vdot(actual,actual).real
        loss=float(10*np.log10(norm/abs(inner)**2));phase=float(np.degrees(np.angle(inner)))
        source_checks.append(dict(channel=c,native_waveform_coherence_loss_db=max(0.,loss),constant_phase_error_deg=phase,
            relative_complex_error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))))
        print('Source waveform check',NAMES[c],source_checks[-1],flush=True)
    (out/'source-checks.json').write_text(json.dumps(source_checks,indent=2)+'\n')
    if max(v['native_waveform_coherence_loss_db'] for v in source_checks)>.05 or max(abs(v['constant_phase_error_deg']) for v in source_checks)>.1:
        raise ValueError('Native point synthesis must match its forward waveform before image interpretation')
    east=args.spacing_m*np.arange(-np.ceil(args.extent_m[0]/2/args.spacing_m),np.ceil(args.extent_m[0]/2/args.spacing_m)+1)
    north=args.spacing_m*np.arange(-np.ceil(args.extent_m[1]/2/args.spacing_m),np.ceil(args.extent_m[1]/2/args.spacing_m)+1)
    xx,yy=np.meshgrid(east,north);pixels=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    print('Evaluating known-template information',len(pixels),'pixels',flush=True)
    moments=[bin_moments(b,a) for b,a in zip(native,native_amplitude)]
    information,observed,psf=gpu_information(bands,parameters,pixels,out/'information-worker',selected=selected,
        moments=moments,score_values=[[b.field.y for b in bands],point_y],reuse=args.resume)
    probe=np.linspace(0,len(pixels)-1,9,dtype=int);probe=np.unique(np.r_[probe,np.argmin(np.linalg.norm(pixels,axis=1))])
    checks=[]
    for c,(b,n) in enumerate(zip(bands,native)):
        expected=reference_information(b,parameters,pixels[probe],selected=selected[c],moments=moments[c])
        original=reference_information(n,parameters,pixels[probe],selected=selected[c],amplitude=native_amplitude[c])
        gpu_error=np.max(abs(information[c,probe,:2]-expected)/expected[:,0,None])
        native_error=np.max(abs(original-expected)/original[:,0,None])
        if gpu_error>5e-4:raise ValueError(f'GPU information check failed: {gpu_error}')
        if native_error>1e-3:raise ValueError(f'Representative/native information check failed: {native_error}')
        checks.append(dict(channel=c,gpu_error_relative_to_raw_information=float(gpu_error),
            representative_native_error_relative_to_raw_information=float(native_error),
            cpu=expected,native=original,pixels=pixels[probe]))
        print('Information verified',NAMES[c],gpu_error,native_error,flush=True)
    # Audit the GPU score at the same probes, using the native original rows.
    score_checks=[]
    for c,(b,n) in enumerate(zip(bands,native)):
        from fixed_range_field import stationary_profile_design
        from stationary import Stationary
        selected_flat=np.repeat(selected[c],len(n.field.offsets));nu=Stationary(stationary_profile_design(n.field),selected_flat)
        clean=nu.remove(n.field.y.ravel()[selected_flat]);reference=[]
        for q in pixels[probe]:
            offset,velocity=n.kinematics.point_state(parameters,q,origin=n.field.data['centre_ecef_m'])
            raw=n.field.raw_trajectory(n.field.data['centre_ecef_m']+offset,velocity,position_offset=offset)*native_amplitude[c][:,None]
            h=n.field.apply(average(raw,n.field.data['row_information'],16)[0]).ravel()[selected_flat]
            reference.append(np.vdot(h,clean))
        reference=np.asarray(reference);error=np.linalg.norm(observed[c,probe]-reference)/np.linalg.norm(reference)
        if error>2e-3:raise ValueError(f'Native score check failed: {error}')
        score_checks.append(dict(channel=c,native_score_relative_error=float(error)))
    raw_info,projected_info=information[:,:,0],information[:,:,1]
    # No target-dependent regularisation or variance floor. Round-off-invalid
    # information is explicitly unavailable rather than filled with a scale.
    valid=projected_info>0
    data={}
    for name,score in [('observed',observed),('point',psf)]:
        independent=np.divide(abs(score)**2,projected_info,out=np.full_like(projected_info,np.nan),where=valid)
        combined_score=np.einsum('c,cp->p',gain.conj(),score)
        combined_info=np.einsum('c,cp->p',abs(gain)**2,projected_info)
        combined=abs(combined_score)**2/combined_info
        data[name]=np.r_[independent,combined[None]].reshape(6,*xx.shape)
    np.savez(out/'images.npz',east_m=east,north_m=north,observed_statistic=data['observed'],point_statistic=data['point'],
        observed_score=observed,point_score=psf,information=information,gains=gain,pixels_east_north_up_m=pixels)
    fig,axes=plt.subplots(2,3,figsize=(13,12),sharex=True,sharey=True,layout='constrained')
    high=10*np.log10(np.nanmax(data['observed']))+.1;low=high-45
    for c,ax in enumerate(axes.ravel()):
        im=ax.imshow(10*np.log10(np.maximum(data['observed'][c],1e-30)),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],vmin=low,vmax=high,cmap='magma')
        ax.set(title=(NAMES+['Coherent five-band template'])[c],xlabel='East from fitted anchor (m)',ylabel='North (m)',aspect='equal')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Conditional nuisance-projected matched statistic (dB)',shrink=.7)
    fig.suptitle('Known covariance and constant-profile nuisance: |hᴴ P y|² / (hᴴ P h)\nFixed physical motion and band calibration; '+('training-fitted real visibility' if args.brightness else 'constant-strength temporal template'))
    fig.savefig(out/'projected-image.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for ax,value,title in zip(axes,[data['observed'][5],data['point'][5],(projected_info/raw_info)[3].reshape(xx.shape)],
            ['Actual recorded scene','One physical point with fitted visibility','Ten: fraction of template information retained']):
        fraction=ax is axes[2]
        im=ax.imshow(value if fraction else 10*np.log10(np.maximum(value,1e-30)),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],
            cmap='viridis' if fraction else 'magma',vmin=0 if fraction else low,vmax=1 if fraction else high)
        ax.set(title=title,xlabel='East from fitted anchor (m)',ylabel='North (m)',aspect='equal')
        fig.colorbar(im,ax=ax,shrink=.55,label='Information fraction' if fraction else 'Conditional matched statistic (dB)')
    fig.suptitle('Same native support, motion, visibility control, waveform response, covariance and nuisance\nPoint response is a validation control; no point is inserted into the measured image')
    fig.savefig(out/'point-comparison.png',dpi=150);plt.close(fig)
    result=dict(scope=__doc__,complete=True,focused_ship_certified=False,source_profiles=str(args.profiles),
        interval_s=[first,last],fold=args.fold,brightness_template=args.brightness,
        conditioned_on_covariance_and_upstream_motion=True,information_checks=checks,score_checks=score_checks,
        nonpositive_projected_information_cells=int((~valid).sum()),retention_min_max=[float(np.min(projected_info/raw_info)),float(np.max(projected_info/raw_info))],
        image_scale_db=[low,high],image_registration_applied=False,source_checks=source_checks)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print('Wrote',out/'projected-image.png',flush=True)


if __name__=='__main__':main()
