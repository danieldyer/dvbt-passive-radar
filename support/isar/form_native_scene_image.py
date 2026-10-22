"""Known-covariance native scene BP with an explicit competing return.

Data and templates share the stationary-profile projection. A frozen moving
source is then profiled with an exact rank-one Schur update of both numerator
and information. Per-band images and a common-phase positive-amplitude GLRT
retain complex calibrated band scores; no empirical floor or image blending
enters formation. Native forward checks are mandatory before writing images.
"""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from amplitude_arc import linear_basis
from coherent_amplitude_glrt import statistic,fixed_spectrum_statistic
from coordinated_patch import CoordinatedPatch
from field import average
from fixed_range_field import stationary_profile_design
from frozen_native_scene import component
from midpoint_model import representative
from native_folds import training_mask
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial
from rigid_patch import RigidPatch
from stationary import Stationary,rank_one_projection
from template_information import gpu,bin_moments
from waveform_kernels import KernelField


def templates(band,parameters,points,amplitude):
    f=band.field;columns=[]
    for q in points:
        offset,velocity=band.kinematics.point_state(parameters,q,origin=f.data['centre_ecef_m'])
        raw=f.raw_trajectory(f.data['centre_ecef_m']+offset,velocity,position_offset=offset)*amplitude[:,None]
        columns.append(f.apply(average(raw,f.data['row_information'],f.metadata['block_rows'])[0]).ravel())
    return np.column_stack(columns)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--competing-fit',type=Path);p.add_argument('--competing-component',type=int,default=3)
    p.add_argument('--temporal-component',type=int,help='Conditional positive visibility from this fitted source; otherwise constant strength')
    p.add_argument('--fold',choices=['all','training','excluded'],default='all',help='Use the continuous aperture for physical image review; periodic validation folds have strong temporal grating lobes')
    p.add_argument('--spacing-m',type=float,default=1.);p.add_argument('--extent-m',type=float,nargs=2,default=[260.,400.])
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    saved=json.loads((args.fit/'analysis.json').read_text());render=json.loads((args.fit/'render-fit.json').read_text())
    if not saved['complete']:raise ValueError('Finish the native fit before formation')
    competitor=args.competing_fit or (Path(saved['nuisance_fit']) if saved.get('nuisance_fit') else None)
    if competitor is None:raise ValueError('Declare the frozen competing source explicitly')
    nc=json.loads((competitor/'analysis.json').read_text());nr=json.loads((competitor/'render-fit.json').read_text())
    if nc['profiles']!=saved['profiles']:raise ValueError('Competing scene comes from a different observation archive')
    first,last=saved['interval_s'];mid=render['midpoint_s'];state=np.asarray(render['coarse_state_enu']);parameters=np.asarray(render['rigid_parameters'])
    coordinated=render.get('motion_model')=='coordinated-turn'
    if coordinated:parameters=np.r_[np.zeros(4),render['turn_rate_degps']]
    observed=saved['observed_range_samples'];fields,r,rx,tx,_,_=load_fields(Path(saved['profiles']),offsets=np.arange(observed[0],observed[1]+1),fixed_epoch_s=mid,interval_s=[first,last])
    brightness=saved['phase_histories']['brightness_model'];points=np.asarray(saved['refined_points_east_north_up_m'])
    control_index=0 if args.temporal_component is None else args.temporal_component;control_point=points[control_index]
    native=[];fast=[];selected=[];amplitudes=[];competitors=[];point_data=[];phase=[];spectrum=[];nuisance_information=[];nuisance_score=[]
    for c,f in enumerate(fields):
        keep=training_mask(f.time,first)
        keep=np.ones(len(f.time),bool) if args.fold=='all' else (keep if args.fold=='training' else ~keep)
        selected.append(keep);band=brightness['bands'][c];a=np.asarray(band['coefficients'])[control_index];mean=a.mean()
        if mean<=0:raise ValueError('The declared control source has no positive brightness support')
        visibility=linear_basis(f.data['row_time'],np.asarray(band['knots_s']))@a/mean
        amplitudes.append(np.ones(len(f.data['row_time'])) if args.temporal_component is None else visibility)
        phase.append(band['band_phase_rad']);spectrum.append(mean*np.exp(1j*band['band_phase_rad']))
        n=component(f,r,rx,nr,nc['phase_histories']['brightness_model'],nc['refined_points_east_north_up_m'],args.competing_component).reshape(f.y.shape)
        competitors.append(n)
        representative_field=representative(f);block=f.metadata['block_rows'];ids=f.pulse.row_blocks.reshape(len(f.time),block)
        if np.any(ids!=ids[:,:1]):raise ValueError('Native pulse identity changes within a representative epoch')
        representative_field=KernelField(representative_field,f.pulse.kernels,ids[:,0])
        for field,target in [(f,native),(representative_field,fast)]:
            time=field.data['row_time'];dt=time-mid
            kin=RigidPatch(time,mid,rx+(state[0]+dt[:,None]*state[1]+.5*dt[:,None]**2*state[2])@r,
                (state[1]+dt[:,None]*state[2])@r,r.T,np.asarray(render['reference_transmitter_ecef_m']),rx,
                translation_direction=render.get('translation_direction_ecef'))
            if coordinated:kin=CoordinatedPatch(time,mid,state[0],state[1],r,rx)
            target.append(SimpleNamespace(field=field,kinematics=kin,training=keep,tx=tx[c],rx=rx,origin=0.))
        control=templates(native[-1],parameters,[control_point],amplitudes[-1])[:,0]
        gain=mean*np.exp(1j*(band['band_phase_rad']+brightness['point_phase_rad'][control_index]));point_data.append((control*gain).reshape(f.y.shape))
        use=np.repeat(keep,len(f.offsets));nu=Stationary(stationary_profile_design(f),use)
        nn=nu.remove(n.ravel()[use]);yy=nu.remove(f.y.ravel()[use])
        nuisance_information.append(np.vdot(nn,nn).real);nuisance_score.append(np.vdot(nn,yy))
    east=args.spacing_m*np.arange(-np.ceil(args.extent_m[0]/2/args.spacing_m),np.ceil(args.extent_m[0]/2/args.spacing_m)+1)
    north=args.spacing_m*np.arange(-np.ceil(args.extent_m[1]/2/args.spacing_m),np.ceil(args.extent_m[1]/2/args.spacing_m)+1)
    xx,yy=np.meshgrid(east,north);pixels=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    moments=[bin_moments(b,a) for b,a in zip(native,amplitudes)]
    print('Forming',len(pixels),'physical pixels',args.fold,flush=True)
    info,z,cross=gpu(fast,parameters,pixels,out/'observed-worker',selected=selected,moments=moments,
        score_values=[[f.y for f in fields],competitors])
    point_info,point_z,point_cross=gpu(fast,parameters,pixels,out/'point-worker',selected=selected,moments=moments,
        score_values=[point_data,competitors])
    np.testing.assert_array_equal(point_info,info);np.testing.assert_array_equal(point_cross,cross)
    norms=np.asarray(nuisance_information);base_info=info[:,:,1]
    score,information=rank_one_projection(z,base_info,cross,np.asarray(nuisance_score)[:,None],norms[:,None])
    control_scores=[]
    for c,f in enumerate(fields):
        use=np.repeat(selected[c],len(f.offsets));nu=Stationary(stationary_profile_design(f),use)
        control_scores.append(np.vdot(nu.remove(competitors[c].ravel()[use]),nu.remove(point_data[c].ravel()[use])))
    point_score,point_information=rank_one_projection(point_z,base_info,cross,np.asarray(control_scores)[:,None],norms[:,None])
    np.testing.assert_array_equal(point_information,information)
    if np.any(information<=0):raise ValueError('Unidentifiable pixel after competing-source projection; do not floor its information')
    probe=np.unique(np.r_[np.linspace(0,len(pixels)-1,5,dtype=int),np.argmin(np.linalg.norm(pixels-control_point,axis=1))])
    checks=[]
    for c,b in enumerate(native):
        f=b.field;use=np.repeat(selected[c],len(f.offsets));design=np.c_[stationary_profile_design(f),competitors[c].ravel()]
        nu=Stationary(design,use);h=nu.remove(templates(b,parameters,pixels[probe],amplitudes[c])[use])
        yy=nu.remove(f.y.ravel()[use]);pp=nu.remove(point_data[c].ravel()[use])
        expected_i=np.sum(abs(h)**2,axis=0);expected_z=h.conj().T@yy;expected_p=h.conj().T@pp
        i_error=float(np.max(abs(information[c,probe]-expected_i)/expected_i))
        z_error=float(np.linalg.norm(score[c,probe]-expected_z)/np.linalg.norm(expected_z))
        p_error=float(np.linalg.norm(point_score[c,probe]-expected_p)/np.linalg.norm(expected_p))
        checks.append(dict(channel=c,native_information_relative_error=i_error,native_observed_score_relative_error=z_error,native_point_score_relative_error=p_error))
        if max(i_error,z_error,p_error)>3e-3:raise ValueError('Native Schur-projected image validation failed: '+str(checks[-1]))
        print('Native image verified',NAMES[c],checks[-1],flush=True)
    data={};fixed={};calibration=np.exp(-1j*np.asarray(phase))[:,None]
    for label,values in [('observed',score),('point',point_score)]:
        independent=abs(values)**2/information;coherent,_=statistic(values*calibration/np.sqrt(information))
        data[label]=np.r_[independent,coherent[None]].reshape(6,*xx.shape)
        fixed[label]=fixed_spectrum_statistic(values,information,spectrum)[0].reshape(xx.shape)
    np.savez(out/'images.npz',east_m=east,north_m=north,pixels_east_north_up_m=pixels,observed_statistic=data['observed'],
        point_statistic=data['point'],observed_score=score,point_score=point_score,information=information,base_information=base_info,
        band_phase_rad=np.asarray(phase),control_point_enu_m=control_point,
        observed_fixed_spectrum_statistic=fixed['observed'],point_fixed_spectrum_statistic=fixed['point'],fixed_spectrum_gain=np.asarray(spectrum))
    high=10*np.log10(max(np.max(data['observed']),np.max(data['point']),np.max(fixed['observed']),np.max(fixed['point'])))+.1;low=0.
    for label,title in [('observed','Recorded scene'),('point','One physical point: known-pulse forward control')]:
        for mode in ('profiled-brightness','fixed-spectrum'):
            fig,axes=plt.subplots(2,3,figsize=(13,12),sharex=True,sharey=True,layout='constrained')
            for c,ax in enumerate(axes.ravel()):
                power=fixed[label] if c==5 and mode=='fixed-spectrum' else data[label][c]
                image=ax.imshow(10*np.log10(np.maximum(power,np.finfo(float).tiny)),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],cmap='magma',vmin=low,vmax=high,aspect='equal')
                combined='Coherent: reference-source spectrum' if mode=='fixed-spectrum' else 'Common phase; positive band amplitudes profiled'
                ax.set(title=(NAMES+[combined])[c],xlabel='East from aperture anchor (m)',ylabel='North (m)')
            fig.colorbar(image,ax=axes.ravel().tolist(),label='Conditional profile likelihood statistic (dB)',shrink=.7)
            fig.suptitle(title+' · '+args.fold+' samples\nKnown range covariance; stationary plus frozen competing-source projection; '+('constant temporal template' if args.temporal_component is None else f'component {control_index} visibility'))
            suffix='-fixed-spectrum-image.png' if mode=='fixed-spectrum' else '-image.png'
            fig.savefig(out/(label+suffix),dpi=150);plt.close(fig)
    result=dict(scope=__doc__,complete=True,source_fit=str(args.fit),fold=args.fold,interval_s=[first,last],
        competing_fit=str(competitor),competing_component=args.competing_component,temporal_component=args.temporal_component,
        fixed_spectrum_scope='Relative band amplitudes/phases from the declared reference component; conditional matched filter, not a universal spectrum for every ship reflector.',
        native_checks=checks,control_point_enu_m=control_point,noise_floor_estimated_from_image=False,
        information_retention_min_max=[float(np.min(information/base_info)),float(np.max(information/base_info))],image_scale_db=[low,high],
        point_control_scope='Analytic known-native-pulse source in the observed coordinates; independent physical replay controls remain separate.',
        geographic_or_resolved_ship_certified=False)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print('Wrote',out/'observed-image.png',flush=True)


if __name__=='__main__':main()
