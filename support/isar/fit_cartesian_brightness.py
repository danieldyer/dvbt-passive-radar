"""Joint planar CV translation, rigid spin and constant inter-band calibration.

Position and velocity are both free in east/north. Height, antenna geometry
and the declared source support are conditional. Real brightness splines
and stationary complex profiles are eliminated on every available sample. Four
relative band phases are constant throughout the aperture; none supplies an
independent time-dependent trajectory or phase screen.
"""
import argparse,json,time
from pathlib import Path

import numpy as np

from cartesian_patch import CartesianPatch
from coordinated_patch import CoordinatedPatch
from cv_rigid_patch import CvRigidPatch
from cartesian_brightness_scene import CartesianBrightnessBand,CartesianBrightnessFit
from fixed_range_field import stationary_profile_design
from midpoint_model import representative,discrepancy
from motion import excess_path
from native_folds import fitting_mask
from recorded_injection_fields import load_fields
from report_measured_scatterer_phases import serial
from waveform_kernels import KernelField


def seed_phases(model,seed):
    """Acquire constant phase intercepts on fitting samples, not phase histories."""
    p=seed.copy();coefficient=[]
    for b in model.bands:
        raw,_=b.responses(p,False)
        h=np.column_stack([b.nuisance.remove(b.reduced(u[:,:,None])[b.selected])[:,0] for u in raw])
        coefficient.append(np.linalg.lstsq(h,b.clean,rcond=1e-10)[0])
    z=np.asarray(coefficient);n=z.shape[1];m=model.bands[0].motion_size;ref=model.bands[0].reference_channel
    if model.bands[0].fixed_calibration:
        p[m:m+n]=np.angle(np.sum(z,axis=0));return p
    phi=np.angle(z[ref]);beta=np.zeros(len(z))
    for _ in range(12):
        beta=np.angle(np.sum(z*np.exp(-1j*phi),axis=1))
        phi=np.angle(np.sum(z*np.exp(-1j*beta[:,None]),axis=0))
        common=beta[ref];phi+=common;beta-=common
    p[m:m+n]=phi
    for b in model.bands:
        if b.calibration_index is not None:p[b.calibration_index]=beta[b.channel]
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-point',type=Path,required=True)
    parser.add_argument('--points-from',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,default=Path('build/isar-20261018/current-native-full/profiles.bin'))
    parser.add_argument('--interval',type=float,nargs=2)
    parser.add_argument('--range-samples',type=int,nargs=2,default=[-12,12])
    parser.add_argument('--knot-s',type=float,default=1.)
    parser.add_argument('--refine-positions',action='store_true')
    parser.add_argument('--free-rotation-centre',action='store_true',help='Fit the anchor offset from the CV centre; do not force the anchor itself to have zero acceleration')
    parser.add_argument('--motion-family',choices=['constant-velocity','coordinated-turn'],default='constant-velocity',help='A coordinated turn couples the centre course rate and rigid rotation instead of imposing straight translation')
    parser.add_argument('--fixed-band-calibration',action='store_true',help='Preserve the declared physical relative band phases when position changes; do not fit calibration nuisance phases')
    parser.add_argument('--nuisance-fit',type=Path);parser.add_argument('--nuisance-components',type=int,nargs='+',default=[])
    parser.add_argument('--initial-fit',type=Path,help='Start from a checked Cartesian fit with identical data, support, priors and knots')
    parser.add_argument('--position-bound-m',type=float,default=60.)
    parser.add_argument('--spin-degps',type=float,nargs='+',default=[.2,0.])
    parser.add_argument('--transverse-seeds-m',type=float,nargs='+',default=[0.])
    parser.add_argument('--representative-iterations',type=int,default=200)
    parser.add_argument('--native-iterations',type=int,default=120)
    parser.add_argument('--information',action='store_true')
    parser.add_argument('--check-gradient',action='store_true')
    parser.add_argument('--phase-histories',action='store_true')
    parser.add_argument('--fit-samples',choices=['all','diagnostic-subset'],default='all',help='Use all samples; periodic withholding is a legacy diagnostic only')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    if args.free_rotation_centre and args.motion_family=='coordinated-turn':raise ValueError('A coordinated-turn point already has a physical turning centre; no CV lever-arm model is needed')
    if bool(args.nuisance_fit)!=bool(args.nuisance_components):raise ValueError('Declare both the competing scene and its source indices')
    reference=json.loads((args.reference_point/'analysis.json').read_text())
    first,last=args.interval or reference['interval_s'];mid=(first+last)/2
    state=np.asarray(reference['physical_state_enu']);dt=mid-reference['midpoint_s']
    state=np.array([state[0]+dt*state[1]+.5*dt*dt*state[2],state[1]+dt*state[2],np.zeros(3)])
    render=json.loads((args.reference_point/'render-fit.json').read_text())
    g=np.asarray(render['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0]
    gain=g[:,0]+1j*g[:,1]
    points=np.load(args.points_from)
    if points.ndim!=2 or points.shape[1]!=3 or np.any(points[0]) or not np.isfinite(points).all():
        raise ValueError('Finite ENU support with an explicit origin anchor is required')
    fields,rotation,rx,tx,ais,_=load_fields(args.profiles,offsets=np.arange(args.range_samples[0],args.range_samples[1]+1),
        fixed_epoch_s=mid,interval_s=(first,last))
    _,gradient=excess_path((rx+(state[0]+points)@rotation),tx[2],rx)
    gauge=np.mean([f.k for f in fields])*(gradient@rotation.T)[:,:2] if args.refine_positions else None
    native=[];fast=[]
    competing=None;competing_render=None
    if args.nuisance_fit:
        competing=json.loads((args.nuisance_fit/'analysis.json').read_text())
        competing_render=json.loads((args.nuisance_fit/'render-fit.json').read_text())
        if competing['profiles']!=str(args.profiles) or first<competing['interval_s'][0] or last>competing['interval_s'][1]:
            raise ValueError('The frozen competing scene must cover these original observations')
    for c,f in enumerate(fields):
        keep=fitting_mask(f.time,first,args.fit_samples)
        knots=np.r_[f.data['row_time'][0],np.arange(first+args.knot_s,last,args.knot_s),f.data['row_time'][-1]]
        mean=representative(f);blocks=f.pulse.row_blocks.reshape(len(f.time),f.metadata['block_rows'])
        if np.any(blocks!=blocks[:,0,None]):raise ValueError('Representative waveform identities change within a bin')
        mean=KernelField(mean,f.pulse.kernels,blocks[:,0])
        for field,target in [(f,native),(mean,fast)]:
            kind=CoordinatedPatch if args.motion_family=='coordinated-turn' else (CvRigidPatch if args.free_rotation_centre else CartesianPatch)
            kin=kind(field.data['row_time'],mid,state[0],state[1],rotation,rx)
            design=stationary_profile_design(field)
            if competing is not None:
                from frozen_native_scene import component
                design=np.column_stack([design,*[component(field,rotation,rx,competing_render,competing['phase_histories']['brightness_model'],
                    competing['refined_points_east_north_up_m'],j) for j in args.nuisance_components]])
            target.append(CartesianBrightnessBand(field,keep,knots,kin,points,np.angle(gain[c]),
                design,position_gauge=gauge,channel=c,fixed_calibration=args.fixed_band_calibration,
                reference_rf_hz=fields[3].metadata['rf_hz'],reference_transmitter=tx[3]))
    accelerated=CartesianBrightnessFit(fast);exact=CartesianBrightnessFit(native)
    result=dict(scope=__doc__,complete=False,interval_s=[first,last],midpoint_s=mid,profiles=str(args.profiles),
        reference_point=str(args.reference_point),reference_state_enu=state,observed_range_samples=args.range_samples,
        points_east_north_up_m=points,point_source=str(args.points_from),fit_samples=args.fit_samples,
        point_support_provenance='Previously selected radar candidates; not an independent test set',
        motion_model='cartesian-'+args.motion_family,motion_family=args.motion_family,transverse_motion_conditioned_on_ais=False,
        geographic_localisation_certified=False,fixed_band_calibration=args.fixed_band_calibration,
        band_phase_nuisance='Fixed physical relative calibration' if args.fixed_band_calibration else 'Four constant relative phases, Ten reference',
        nuisance_fit=None if args.nuisance_fit is None else str(args.nuisance_fit),nuisance_components=args.nuisance_components,
        initial_band_phases_rad=np.angle(gain),position_phase_gauge=gauge,refine_positions=args.refine_positions,
        position_bound_m=args.position_bound_m,knot_s=args.knot_s,trials=[],free_rotation_centre=args.free_rotation_centre)
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    tangent=np.array([-gradient[0]@rotation[1],gradient[0]@rotation[0]])
    tangent/=np.linalg.norm(tangent)
    for displacement in args.transverse_seeds_m:
        for spin in args.spin_degps:
            seed=np.zeros(exact.size);seed[:2]=displacement*tangent;seed[4]=spin
            seed=seed_phases(accelerated,seed)
            if args.initial_fit:
                initial=json.loads(args.initial_fit.read_text())
                for key in ['interval_s','midpoint_s','profiles','reference_point','point_source','refine_positions','knot_s','observed_range_samples']:
                    if initial[key]!=result[key]:raise ValueError('Initial fit settings differ: '+key)
                np.testing.assert_array_equal(initial['reference_state_enu'],state)
                np.testing.assert_array_equal(initial['points_east_north_up_m'],points)
                old=np.asarray(initial['fit']['parameters']);m=native[0].motion_size
                old_m=7 if initial.get('free_rotation_centre',False) else 5
                old_end=old_m+len(points)+(2*(len(points)-1) if args.refine_positions else 0)
                seed[:5]=old[:5];seed[m:m+old_end-old_m]=old[old_m:old_end]
                if not args.fixed_band_calibration and not initial.get('fixed_band_calibration',False):seed[-4:]=old[-4:]
                if old_m==m==7:seed[5:7]=old[5:7]
                result['initial_fit']=str(args.initial_fit)
            started=time.monotonic()
            trial=accelerated.fit(seed,max_iterations=args.representative_iterations,position_bound_m=args.position_bound_m)
            trial.update(seed_transverse_displacement_m=displacement,seed_spin_degps=spin,elapsed_s=time.monotonic()-started)
            result['trials'].append(trial);save();print('Representative trial',serial(trial),flush=True)
    best=min(result['trials'],key=lambda q:q['objective']);started=time.monotonic()
    p=np.asarray(best['parameters'])
    fit=(exact.fit(p,max_iterations=args.native_iterations,position_bound_m=args.position_bound_m) if args.native_iterations else
        exact.assess(p,position_bound_m=args.position_bound_m))
    fit['elapsed_s']=time.monotonic()-started;result['fit']=fit;save();p=np.asarray(fit['parameters'])
    print('Native fit',serial(fit),flush=True)
    if args.check_gradient:result['gradient_check']=exact.gradient_check(p);save()
    if args.information:result['conditional_information']=exact.information(p);save()
    result['residual_diagnostics']=[b.residual_diagnostics(p) for b in native]
    result['prediction']=[] if args.fit_samples=='all' else [b.prediction(p) for b in native]
    result['component_transfer']=[b.component_transfer(p) for b in native]
    save()
    if args.phase_histories:
        from native_scene_histories import audit
        result['phase_histories']=audit(native,p,first,out/'phase-histories',physical_phases=exact.physical_calibration(p))
    refined=points.copy();n=len(points);m=native[0].motion_size
    if gauge is not None:refined[1:,:2]+=p[m+n:m+n+2*(n-1)].reshape(-1,2)
    result['operator_comparison']=[]
    for b,a in zip(native,fast):
        u=b.evaluate_scene(p,full=True);v=a.evaluate_scene(p,full=True)
        result['operator_comparison'].append(discrepancy(u['prediction'],v['design']@u['amplitude']))
    beta,phi=exact.physical_calibration(p);result['physical_band_phases_rad']=beta;result['physical_point_phases_rad']=phi
    physical=state.copy();physical[0,:2]+=p[:2];physical[1,:2]+=p[2:4]
    if args.motion_family=='coordinated-turn':physical[2,:2]=np.radians(p[4])*np.array([-physical[1,1],physical[1,0]])
    result['physical_anchor_state_enu']=physical.copy()
    if args.free_rotation_centre:
        physical,offset=native[0].kinematics.centre_state(p)
        result['physical_anchor_state_enu'][2,:2]=-np.radians(p[4])**2*offset[:2]
        result['rotation_centre_source_shift_enu_m']=offset
        refined+=offset
    result['refined_points_east_north_up_m']=refined
    result['physical_state_enu']=physical
    gains=[]
    for c,b in enumerate(native):
        coefficients=b.evaluate_scene(p,full=True)['amplitude'].reshape(n,len(b.knots))
        gains.append(np.mean(coefficients[0])*np.exp(1j*(beta[c]+phi[0])))
    exported=dict(midpoint_s=mid,coarse_state_enu=physical,interval_s=[first,last],motion_model='rigid-patch',
        reference_transmitter_ecef_m=tx[2],rigid_parameters=np.array([0.,0.,0.,p[4],0.]),
        fits=dict(constant_velocity_gls=dict(parameters=np.zeros(6),coefficients_real_imag=[[[z.real,z.imag]] for z in gains])),
        angular_motion_fitted=True,fit_samples=args.fit_samples,
        note='Cartesian CV translation fit; four constant relative band phases jointly estimated; geographic uniqueness not certified.')
    if args.motion_family=='coordinated-turn':
        exported.update(motion_model='coordinated-turn',turn_rate_degps=float(p[4]),
            note='Coordinated-turn translation and rotation; physical relative band calibration as declared. Geographic uniqueness not certified.')
    (out/'render-fit.json').write_text(json.dumps(serial(exported),indent=2)+'\n')
    (out/'camera.json').write_text(json.dumps(dict(state_enu=physical.tolist(),marker_east_north_m=(np.asarray(ais)[0,:2]-physical[0,:2]).tolist(),
        marker_label='AIS antenna prior',viewport_label='Cartesian CV fitted point'),indent=2)+'\n')
    result['complete']=True;save()
    print('Fitting-sample residual fractions',[q['explained_fraction'] for q in result['residual_diagnostics']],flush=True)
    if result['prediction']:print('Unused-sample prediction',[q['explained_fraction'] for q in result['prediction']],flush=True)


if __name__=='__main__':main()
