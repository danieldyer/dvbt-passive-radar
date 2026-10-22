"""Full-aperture native-row rigid autofocus, with fixed band calibration.

The short-arc radial fit supplies a conditional motion prior. Only common
path-rate/acceleration/jerk and physical angular rate/acceleration are refined.
Image axes remain N/E. Known-waveform representative epochs accelerate search;
The final objective and gradient use every available original native symbol.
Legacy sample withholding is available only as an explicit diagnostic.
"""
import argparse,json
from pathlib import Path
import numpy as np
from block_pulse import BlockPulse
from fixed_range_field import stationary_profile_design
from midpoint_model import representative,discrepancy
from native_folds import fitting_mask,fit_sample_selection
from recorded_injection_fields import load_fields
from rigid_patch import RigidPatch
from rigid_brightness_scene import RigidBrightnessBand,RigidBrightnessFit
from waveform_kernels import KernelField
from motion import excess_path
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refine-positions',action='store_true')
    p.add_argument('--fix-motion',action='store_true',help='Refine point coordinates/phases while holding the declared motion hypothesis fixed')
    saved_fit=p.add_mutually_exclusive_group()
    saved_fit.add_argument('--replay-fit',type=Path,help='Audit a saved fit at identical data/operator settings without optimising again')
    saved_fit.add_argument('--polish-fit',type=Path,help='Verify a saved fit, then refine it using the native-symbol objective')
    saved_fit.add_argument('--refit-fit',type=Path,help='Use all saved scene parameters to initialise a new native fit, with all samples by default')
    p.add_argument('--fit-samples',choices=['all','diagnostic-subset'],help='Default: all samples for new fits; identical-likelihood replays preserve the saved selection')
    p.add_argument('--point-indices',type=int,nargs='+',help='Explicit indices in the candidate list')
    p.add_argument('--points-from',type=Path,help='Declared N by 3 ENU support coordinates, in a NumPy file')
    p.add_argument('--initial-fit',type=Path,help='Use shared motion from a previous fit, acquiring new point phases on the fitting samples')
    p.add_argument('--motion-model',choices=['radial-cubic','constant-velocity','constant-translation'],default='radial-cubic',
        help='Constant velocity fixes all point accelerations and rotation; constant translation also permits a constant physical angular rate')
    p.add_argument('--position-bound-m',type=float,default=6.)
    p.add_argument('--fixed-point-count',type=int,default=1,help='Keep this many leading support coordinates fixed during refinement')
    p.add_argument('--interval',type=float,nargs=2,help='Fit another aperture using the declared support and coarse physical prior')
    p.add_argument('--profiles',type=Path,default=Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    p.add_argument('--knot-s',type=float,default=1.)
    p.add_argument('--native-iterations',type=int,default=100,help='Native optimisation budget; zero checks the exact objective/gradient and operator mismatch without native optimisation')
    p.add_argument('--residual-proposals',action='store_true',help='Generate checked Vulkan residual source proposals after the native fit')
    p.add_argument('--phase-histories',action='store_true',help='Diagnose jointly deblended phase histories without applying any local correction')
    p.add_argument('--reference-point',type=Path,default=Path('docs/data/isar_20261018/native-shared-radial-point-6s'),
        help='Declared physical point prior and constant band calibration')
    p.add_argument('--range-samples',type=int,nargs=2,help='Inclusive observed range offsets; defaults to the reference point crop')
    p.add_argument('--check-gradient',action='store_true',help='Audit central differences and inner NNLS optimality at the final checkpoint')
    p.add_argument('--information',action='store_true',help='Audit local motion information after brightness, point-phase and position nuisance projection')
    p.add_argument('--nuisance-fit',type=Path,help='Previously fitted competing scene; selected source templates remain fixed while target motion varies')
    p.add_argument('--nuisance-components',type=int,nargs='+',default=[],help='Source indices in the competing scene to profile as constant-complex nuisance columns')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    saved_path=args.replay_fit or args.polish_fit or args.refit_fit
    saved=None if saved_path is None else json.loads(saved_path.read_text())
    selection=fit_sample_selection(args.fit_samples,saved=saved,reproduce=bool(args.replay_fit or args.polish_fit))
    if args.initial_fit and saved_path:raise ValueError('Choose a motion initializer or a saved scene, not both')
    source=args.reference_point
    if source!=Path('docs/data/isar_20261018/native-shared-radial-point-6s') and args.points_from is None:
        raise ValueError('A new physical reference requires explicitly declared support coordinates')
    data=json.loads((source/'analysis.json').read_text());mid=data['midpoint_s'];first,last=data['interval_s'];state=np.asarray(data['physical_state_enu'])
    old_mid=mid
    if args.interval:
        first,last=args.interval;mid=(first+last)/2;dt=mid-old_mid
        state=np.array([state[0]+dt*state[1]+.5*dt**2*state[2],state[1]+dt*state[2],state[2]])
    if args.motion_model!='radial-cubic':state[2]=0.
    render=json.loads((source/'render-fit.json').read_text());g=np.asarray(render['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0];gain=g[:,0]+1j*g[:,1]
    spatial=Path('docs/data/isar_20261018/native-spatial-validation/spatial-transfer')
    points=np.load(args.points_from if args.points_from else spatial/'points.npy')
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Points must be finite ENU triples')
    point_indices=np.flatnonzero(np.linalg.norm(points[:,:2],axis=1)<250) if args.point_indices is None else np.asarray(args.point_indices)
    if point_indices[0]!=0 or len(np.unique(point_indices))!=len(point_indices):raise ValueError('Keep the anchor first, with distinct candidates')
    points=points[point_indices]
    if not 1<=args.fixed_point_count<=len(points):raise ValueError('The fixed leading support must include the anchor')
    if args.interval and args.points_from is None:
        angle=np.radians(.22678571428571473*(mid-old_mid));c,s=np.cos(angle),np.sin(angle)
        points=points@np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    observed=args.range_samples or data.get('observed_range_samples',[0,24])
    fields,rotation,rx,tx,ais_state,_=load_fields(args.profiles,offsets=np.arange(observed[0],observed[1]+1),fixed_epoch_s=mid,interval_s=args.interval)
    translation_direction=None
    if args.motion_model!='radial-cubic':
        _,gradient=excess_path((rx+state[0]@rotation)[None,:],tx[2],rx)
        up=rotation[2];gradient=gradient[0]-np.dot(gradient[0],up)*up
        translation_direction=gradient/np.dot(gradient,gradient)
    position_gauge=None
    if args.refine_positions:
        positions=rx+(state[0]+points)@rotation
        _,gradient=excess_path(positions,tx[2],rx)
        kref=2*np.pi*np.mean([f.metadata['rf_hz'] for f in fields])/299792458.
        position_gauge=kref*(gradient@rotation.T)[:,:2]
    native=[];fast=[]
    nuisance=None;nuisance_render=None
    if bool(args.nuisance_fit)!=bool(args.nuisance_components):raise ValueError('Specify both a competing scene and its nuisance components')
    if args.nuisance_fit:
        nuisance=json.loads((args.nuisance_fit/'analysis.json').read_text())
        nuisance_render=json.loads((args.nuisance_fit/'render-fit.json').read_text())
        if nuisance['profiles']!=str(args.profiles) or abs((first-nuisance['interval_s'][0])-round(first-nuisance['interval_s'][0]))>1e-9:
            raise ValueError('Competing source must use the same input and compatible observation epochs')
        if first<nuisance['interval_s'][0] or last>nuisance['interval_s'][1]:raise ValueError('Competing source does not cover the target aperture')
    for c,original in enumerate(fields):
        f=original.windowed(first,last);keep=fitting_mask(f.time,first,selection)
        knots=np.r_[f.data['row_time'][0],np.arange(first+args.knot_s,last,args.knot_s),f.data['row_time'][-1]]
        mean=representative(f);blocks=f.pulse.row_blocks.reshape(len(f.time),16)
        if np.any(blocks!=blocks[:,0,None]):raise ValueError('Representative kernel identities disagree')
        mean=KernelField(mean,f.pulse.kernels,blocks[:,0])
        for field,target in [(f,native),(mean,fast)]:
            t=field.data['row_time'];dt=t-mid
            position=rx+(state[0]+dt[:,None]*state[1]+.5*dt[:,None]**2*state[2])@rotation
            velocity=(state[1]+dt[:,None]*state[2])@rotation
            kin=RigidPatch(t,mid,position,velocity,rotation.T,tx[2],rx,translation_direction=translation_direction)
            design=stationary_profile_design(field)
            if nuisance is not None:
                from frozen_native_scene import component
                extra=[component(field,rotation,rx,nuisance_render,nuisance['phase_histories']['brightness_model'],
                    nuisance['refined_points_east_north_up_m'],j) for j in args.nuisance_components]
                design=np.column_stack([design,*extra])
            target.append(RigidBrightnessBand(field,keep,knots,kin,points,np.angle(gain[c]),design,position_gauge=position_gauge))
    n=len(points);result=dict(scope=__doc__,complete=False,interval_s=[first,last],midpoint_s=mid,
        reference_state_enu=state,points_east_north_up_m=points,band_phases_rad=np.angle(gain),
        fit_samples=selection,point_support_provenance='Previously selected radar candidates; not an independent test set',
        transverse_motion_conditioned_on_ais=True,geographic_localisation_certified=False,
        refine_positions=args.refine_positions,motion_fixed=args.fix_motion,point_indices=point_indices,
        position_phase_gauge=position_gauge,position_bound_m=args.position_bound_m,
        profiles=str(args.profiles),knot_s=args.knot_s,trials=[],reference_point=str(source),observed_range_samples=observed)
    result.update(nuisance_fit=None if args.nuisance_fit is None else str(args.nuisance_fit),nuisance_components=args.nuisance_components,
        nuisance_scope='Stationary observed-profile columns plus explicitly declared frozen competing sources; unknown complex coefficients are profiled on fitting samples. Target templates receive the same projection.')
    result.update(motion_model=args.motion_model,translation_direction_ecef=translation_direction,
        point_source=str(args.points_from) if args.points_from else str(spatial/'points.npy'))
    result['fixed_point_count']=args.fixed_point_count
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    def checkpoint(value):
        if value['iterations']==1 or value['iterations']%10==0:
            (out/'native-progress.json').write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')
            print('Native iteration',value['iterations'],'evaluations',value['evaluations'],'objective',value['objective'],flush=True)
    accelerated=RigidBrightnessFit(fast);exact=RigidBrightnessFit(native)
    offsets=np.zeros((n-1,2));phases=np.zeros(n)
    if args.refine_positions and args.points_from is None:
        measured=json.loads((spatial/'analysis.json').read_text())
        coefficient=np.asarray(measured['training_coefficient_real_imag'])[:,point_indices]
        coefficient=coefficient[:,:,0]+1j*coefficient[:,:,1]
        positions=rx+(state[0]+points)@rotation
        gradients=np.asarray([excess_path(positions,antenna,rx)[1]@rotation.T for antenna in tx])
        k=np.asarray([f.k for f in fields])
        for j in range(1,n):
            direction=gradients[2,j,:2]/np.dot(gradients[2,j,:2],gradients[2,j,:2])
            candidates=np.linspace(-4.,4.,161)[:,None]*direction
            correction=candidates@(k[:,None]*gradients[:,j,:2]-position_gauge[j]).T
            z=coefficient[:,j]*np.exp(-1j*np.angle(gain))
            cross=np.exp(1j*correction)@z;chosen=np.argmax(abs(cross))
            offsets[j-1]=candidates[chosen];phases[j]=np.angle(cross[chosen])
    free=np.arange(5,accelerated.size) if args.fix_motion else None
    if not args.fix_motion and args.motion_model!='radial-cubic':
        free=np.r_[0,([3] if args.motion_model=='constant-translation' else []),np.arange(5,accelerated.size)].astype(int)
    if args.refine_positions and args.fixed_point_count>1:
        free=np.setdiff1d(np.arange(accelerated.size) if free is None else free,
            np.arange(5+n,5+n+2*(args.fixed_point_count-1)))
    if saved_path:
        if saved.get('nuisance_fit')!=result['nuisance_fit'] or saved.get('nuisance_components',[])!=args.nuisance_components:
            raise ValueError('Saved inference nuisance model differs')
        if saved['refine_positions']!=args.refine_positions or saved['motion_fixed']!=args.fix_motion:
            raise ValueError('Saved inference subspace differs')
        if saved.get('motion_model','radial-cubic')!=args.motion_model:raise ValueError('Saved motion model differs')
        np.testing.assert_array_equal(saved['points_east_north_up_m'],points)
        np.testing.assert_allclose(saved['reference_state_enu'],state,rtol=0,atol=1e-9)
        if (saved['profiles']!=str(args.profiles) or saved['interval_s']!=[first,last] or saved['midpoint_s']!=mid or
                saved['knot_s']!=args.knot_s or saved['observed_range_samples']!=list(observed)):
            raise ValueError('Saved scene uses different observations, epochs or amplitude basis')
        np.testing.assert_allclose(saved['band_phases_rad'],np.angle(gain),rtol=0,atol=1e-12)
        if saved.get('position_phase_gauge') is not None:np.testing.assert_allclose(saved['position_phase_gauge'],position_gauge,rtol=0,atol=1e-12)
        fitted=saved['fit'];value,_=exact.objective(np.asarray(fitted['parameters']))
        if args.refit_fit:
            result.update(refitted_from=str(saved_path),initial_native_objective=value,
                previous_fit_samples=saved.get('fit_samples','diagnostic-subset'))
            save();print('Refitting native scene',selection,'initial objective',value,flush=True)
        else:
            if abs(value-fitted['objective'])>1e-10:raise ValueError('Saved native likelihood changed')
            result['replayed_fit']=str(saved_path)
        if args.polish_fit or args.refit_fit:
            options=dict(free=free,position_bound_m=args.position_bound_m)
            if args.native_iterations<0:raise ValueError('Native iteration budget cannot be negative')
            fitted=(exact.fit(np.asarray(fitted['parameters']),max_iterations=args.native_iterations,progress=checkpoint,**options)
                if args.native_iterations else exact.assess(np.asarray(fitted['parameters']),**options))
            if args.polish_fit:result['native_polished_from_replay']=True
    else:
        rates=[0.] if args.motion_model=='constant-velocity' else ([.22678571428571473] if args.fix_motion else [.22678571428571473,0.,-.22678571428571473])
        initial=None
        if args.initial_fit:
            initial=json.loads(args.initial_fit.read_text())
            np.testing.assert_allclose(initial['reference_state_enu'],state,rtol=0,atol=1e-9)
            if initial['midpoint_s']!=mid:raise ValueError('Initial motion must use the same reference epoch')
            rates=[float(initial['fit']['parameters'][3])]
        for rate in rates:
            seed=np.r_[np.zeros(3),rate,0.,phases,offsets.ravel() if args.refine_positions else np.empty(0)]
            if initial is not None:seed[:5]=initial['fit']['parameters'][:5]
            if args.points_from:
                coefficient=[]
                for b in fast:
                    raw,_=b.responses(seed,False)
                    h=np.column_stack([b.nuisance.remove(b.reduced(u[:,:,None])[b.selected])[:,0] for u in raw])
                    coefficient.append(np.linalg.lstsq(h,b.clean,rcond=1e-10)[0])
                seed[5:5+n]+=np.angle(np.sum(coefficient,axis=0))
            trial=accelerated.fit(seed,max_iterations=160,free=free,position_bound_m=args.position_bound_m);result['trials'].append(trial);save()
            print('Representative trial',serial(trial),flush=True)
        best=min(result['trials'],key=lambda q:q['objective'])
        native_options=dict(free=free,position_bound_m=args.position_bound_m)
        if args.native_iterations<0:raise ValueError('Native iteration budget cannot be negative')
        fitted=(exact.fit(np.asarray(best['parameters']),max_iterations=args.native_iterations,progress=checkpoint,**native_options) if args.native_iterations else
            exact.assess(np.asarray(best['parameters']),**native_options))
    result['fit']=fitted;save();print('Native fit',serial(fitted),flush=True)
    if args.check_gradient:
        result['gradient_check']=exact.gradient_check(np.asarray(fitted['parameters']),free=free);save()
        print('Gradient check',serial(result['gradient_check']),flush=True)
    if args.information:
        result['conditional_information']=exact.information(np.asarray(fitted['parameters']));save()
    result['residual_diagnostics']=[b.residual_diagnostics(np.asarray(fitted['parameters'])) for b in native]
    result['prediction']=[] if selection=='all' else [b.prediction(np.asarray(fitted['parameters'])) for b in native]
    result['component_transfer']=[b.component_transfer(np.asarray(fitted['parameters'])) for b in native]
    if args.phase_histories:
        from native_scene_histories import audit
        result['phase_histories']=audit(native,np.asarray(fitted['parameters']),first,out/'phase-histories')
    refined=points.copy()
    if args.refine_positions:refined[1:,:2]+=np.asarray(fitted['parameters'])[5+n:].reshape(-1,2)
    result['refined_points_east_north_up_m']=refined
    if args.residual_proposals:
        from native_residual_proposals import propose
        result['residual_proposals']=propose(native,fast,np.asarray(fitted['parameters']),refined,out/'residual-proposals')
    result['operator_comparison']=[]
    for b,a in zip(native,fast):
        u=b.evaluate_scene(np.asarray(fitted['parameters']),full=True)
        v=a.evaluate_scene(np.asarray(fitted['parameters']),full=True)
        result['operator_comparison'].append(discrepancy(u['prediction'],v['design']@u['amplitude']))
    gains=[]
    for c,b in enumerate(native):
        a=b.evaluate_scene(np.asarray(fitted['parameters']),full=True)['amplitude'].reshape(n,len(b.knots))
        gains.append(np.mean(a[0])*np.exp(1j*(np.angle(gain[c])+fitted['parameters'][5])))
    gains=np.asarray(gains)
    result['complete']=True;save()
    render=dict(midpoint_s=mid,coarse_state_enu=state,interval_s=[first,last],motion_model='rigid-patch',
        reference_transmitter_ecef_m=tx[2],rigid_parameters=np.asarray(fitted['parameters'])[:5],
        translation_direction_ecef=translation_direction,
        fits=dict(constant_velocity_gls=dict(parameters=np.zeros(6),coefficients_real_imag=[[[z.real,z.imag]] for z in gains])),
        angular_motion_fitted=not args.fix_motion,
        note='Native rigid autofocus, fixed band phase calibration, positive brightness, conditional transverse AIS motion.',
        fit_samples=selection)
    (out/'render-fit.json').write_text(json.dumps(serial(render),indent=2)+'\n')
    antenna=np.asarray(data['prior_mean_state_enu'])[0] if args.interval is None else np.asarray(ais_state)[0]
    (out/'camera.json').write_text(json.dumps(dict(state_enu=state.tolist(),marker_east_north_m=(antenna[:2]-state[0,:2]).tolist(),
        marker_label='AIS antenna prior',viewport_label='fitted point'),indent=2)+'\n')
    print('Fitting-sample residual fractions',[q['explained_fraction'] for q in result['residual_diagnostics']],flush=True)
    if result['prediction']:print('Unused-sample prediction',[q['explained_fraction'] for q in result['prediction']],flush=True)


if __name__=='__main__':main()
