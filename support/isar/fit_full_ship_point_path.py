"""Fit a continuous radio path against the full original complex range field.

This deliberately restrictive one-point model is a falsifiable motion probe.
It does not assert vessel identity or unique geographic localization. All
estimation uses training epochs; excluded predictions freeze amplitudes,
phase and motion. No per-band or per-window time-varying phase is fitted.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize,LinearConstraint,OptimizeResult
from scipy.sparse.linalg import lsqr
from full_ship_field import load
from full_path_point import PointBand,PointPath,initial_curve
from joint_ship_motion import JointMotion
from report_measured_scatterer_phases import serial
from full_path_scene import SceneBand,ScenePath,RefinedScenePath,warm_phases
from motion import enu_basis
from path_dynamics import project_initial,constraint,certificate


def compact(value):
    return {k:v for k,v in value.items() if k in ['cost','energy','explained_fraction']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--field',type=Path,required=True);p.add_argument('--phase-seed',type=Path,required=True)
    p.add_argument('--coarse-track',type=Path,required=True);p.add_argument('--spacing-s',type=float,default=3.)
    p.add_argument('--amplitude-spacing-s',type=float,default=2.);p.add_argument('--initial',choices=['zero','ten-phase'],default='zero')
    p.add_argument('--orientation',choices=['existing','reported-course','fixed'],default='existing')
    p.add_argument('--angular-rate-offset-degps',type=float,default=0.,help='Declared physical rigid-frame rate correction, shared by all points and bands')
    p.add_argument('--angular-acceleration-offset-degps2',type=float,default=0.)
    p.add_argument('--interval',type=float,nargs=2,help='Isolated aperture test; underlying reference-motion spline support is preserved')
    p.add_argument('--scene',type=Path,help='Optional shared persistent scatterers from a local complex-data fit')
    p.add_argument('--refine-points-m',type=float,help='Bounded common horizontal source refinement around training-selected positions')
    p.add_argument('--phase-coordinate-preconditioner',action='store_true',help='Invertible carrier-phase pivot coordinates; physical likelihood and bounds unchanged')
    p.add_argument('--noise-objective',action='store_true',help='Numerical objective is the Gaussian log-likelihood difference per retained noise observation; scientific weights unchanged')
    p.add_argument('--information-preconditioner',action='store_true',help='Invertible frozen local Gaussian-information coordinates; requires noise objective and phase pivot')
    p.add_argument('--phase-acquisition',action='store_true',help='Short-subaperture common cubic-phase grid initializer before exact physical refinement')
    p.add_argument('--phase-acquisition-points',type=int,nargs='+')
    p.add_argument('--phase-acquisition-seconds',type=float,default=5.)
    p.add_argument('--phase-acquisition-range-offsets',type=int,nargs=2,help='Fixed raw range-cell gate used only by coarse acquisition; full-field refinement is retained')
    p.add_argument('--range-offsets',type=int,nargs=2,help='Declared fixed range patch for a local autofocus fit; native image formation keeps all ranges')
    p.add_argument('--fit-path-offset',action='store_true',help='Estimate constant physical envelope delay as well as the anchored path variation; RF phase gauges remain separate')
    p.add_argument('--fixed-relative-band-phases',action='store_true',help='Retain upstream band-phase differences; fit only one common phase and the shared physical path')
    p.add_argument('--resume-motion',type=Path,help='Use a completed continuous path as the new reference')
    p.add_argument('--resume-parameters',type=Path,help='Resume an optimizer checkpoint against the same reference curve')
    p.add_argument('--max-acceleration-mps2',type=float,help='Exact global bound on the measured physical path acceleration')
    p.add_argument('--max-path-correction-m',type=float,default=60.,help='Declared bounded path search around the acquisition initializer')
    p.add_argument('--phase-warmup-iterations',type=int,default=0,help='Pre-fit only the scene/band constant phases at the initial physical path')
    p.add_argument('--phases-only',action='store_true',help='Hold physical motion fixed after phase/brightness initialization; still report the full physical KKT residual')
    p.add_argument('--iterations',type=int,default=120);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    if a.fixed_relative_band_phases and (not a.scene or not a.resume_motion or a.max_acceleration_mps2 is None):
        raise ValueError('Fixed relative band phases require a scene, upstream motion and constrained physical-path fit')
    fields,coordinates,description=load(a.field);motion=JointMotion(Path(description['motion']));seed=np.load(a.phase_seed/'histories.npz');point=seed['point_body_m']
    if a.range_offsets:
        if not a.interval:raise ValueError('A restricted autofocus patch requires an explicit local aperture')
        from full_ship_field import select_range
        fields=[select_range(f,*a.range_offsets) for f in fields]
    anchor=162.
    if a.interval:
        first,last=a.interval
        if not motion.interval[0]<=first<last<=motion.interval[1]:raise ValueError('Requested fit interval exceeds reference-motion support')
        subset=[];epochs=[]
        for f,c in zip(fields,coordinates):
            selected=f.windowed(first,last);indices=np.searchsorted(f.time,selected.time);np.testing.assert_array_equal(f.time[indices],selected.time)
            coordinate={k:c[k].reshape(-1,16)[indices].ravel() for k in ['native_time_s','native_information']}
            coordinate['training']=c['training'][indices]
            if 'stationary_groups' in c:coordinate['stationary_groups']=c['stationary_groups'][indices]
            subset.append(selected);epochs.append(coordinate)
        fields,coordinates=subset,epochs;motion.interval=[first,last];anchor=(first+last)/2
    track=np.load(a.coarse_track);grid,base=initial_curve(track['time_s'],track['path_m'],track['rate_mps'],motion.interval,a.spacing_s)
    resumed=None
    if a.resume_motion:
        resumed=json.loads(a.resume_motion.read_text())
        if np.array_equal(grid,np.asarray(resumed['path_knots_s'])):base=np.asarray(resumed['path_coefficients_m'])
        else:
            previous=BSpline(resumed['path_knots_s'],resumed['path_coefficients_m'],3,extrapolate=False)
            t=np.linspace(*motion.interval,4001);grid,base=initial_curve(t,previous(t),previous(t,1),motion.interval,a.spacing_s)
    projected_initial=None
    if a.max_acceleration_mps2 is not None:
        if a.max_acceleration_mps2<=0:raise ValueError('Acceleration bound must be positive')
        base,projected_initial=project_initial(grid,base,a.max_acceleration_mps2,anchor)
    bands=[PointBand(f,c,motion,point,grid,base,anchor,a.amplitude_spacing_s,a.orientation,angular_rate_offset_degps=a.angular_rate_offset_degps,angular_acceleration_offset_degps2=a.angular_acceleration_offset_degps2,free_path_offset=a.fit_path_offset) for f,c in zip(fields,coordinates)]
    point_bands=bands;scene=None;points=None
    if a.scene:
        scene=json.loads(a.scene.read_text());at=np.array([scene['mid_s']]);state=motion.state(at);sample=motion.track.sample(motion.epoch+at,max_gap_s=400.)
        gauge=state['body_to_ecef'][0].T@enu_basis(sample['latitude'][0],sample['longitude'][0])
        points=np.asarray(scene['points'])@gauge.T
        bands=[SceneBand(b,points,scene.get('scene_window_edges_s'),a.amplitude_spacing_s,scene.get('active_points')) for b in bands]
        likelihood=(RefinedScenePath if a.refine_points_m is not None else ScenePath)(bands)
    else:likelihood=PointPath(bands)
    if a.refine_points_m is not None and (scene is None or a.refine_points_m<=0):raise ValueError('Positive source-refinement bounds require a shared scene')
    if a.phase_coordinate_preconditioner and scene is None:raise ValueError('Carrier-phase pivot coordinates require a declared scene')
    phase_end=likelihood.n+5+(likelihood.p-1 if points is not None else 0)
    initial=np.zeros(phase_end+(2*likelihood.p if a.refine_points_m is not None else 0))
    if a.initial=='ten-phase':
        b=point_bands[3];mask=b.training
        residual=np.interp(b.t,seed['time_s_3'],seed['unwrapped_residual_path_m_3'])
        initial[:likelihood.n]=lsqr(b.b[mask],residual[mask],atol=1e-11,btol=1e-11,iter_lim=2000)[0]
    for c,b in enumerate(point_bands):initial[likelihood.n+c]=np.angle(np.interp(anchor,seed[f'time_s_{c}'],seed[f'smooth_{c}']))
    if resumed:initial[likelihood.n:likelihood.n+5]=np.angle([complex(*z) for z in resumed['physical_gains_real_imag']])
    if scene:
        phi=np.asarray(scene['phase_parameters'])[4:].ravel();initial[likelihood.n+5:phase_end]=phi[1:]-phi[0]
        if 'band_phase_initial_radians' in scene:initial[likelihood.n:likelihood.n+5]=scene['band_phase_initial_radians']
    if a.resume_parameters:
        restored=np.asarray(json.loads(a.resume_parameters.read_text())['parameters'])
        if restored.shape!=initial.shape:raise ValueError('Checkpoint has a different motion/scene layout')
        initial=restored
    fixed_band_offsets=None
    if a.fixed_relative_band_phases:
        calibration=np.angle([complex(*z) for z in resumed['physical_gains_real_imag']])
        fixed_band_offsets=calibration-calibration[3]
        common=np.angle(np.exp(1j*(initial[likelihood.n:likelihood.n+5]-fixed_band_offsets)).sum())
        initial[likelihood.n:likelihood.n+5]=fixed_band_offsets+common
    acquisition=None
    if a.phase_acquisition:
        if scene is None or points.ndim!=2 or a.resume_parameters or a.initial!='zero':raise ValueError('Phase acquisition requires one local fixed scene and a zero correction to its resumed reference curve')
        if motion.interval[1]-motion.interval[0]>60:raise ValueError('Use local phase acquisition on at most 60 seconds before joining a continuous full-recording path')
        from polynomial_phase_acquisition import acquire
        mid=float(np.mean(motion.interval));half=(motion.interval[1]-motion.interval[0])/2
        indices=a.phase_acquisition_points
        if indices is not None and (min(indices)<0 or max(indices)>=len(points) or len(set(indices))!=len(indices)):raise ValueError('Acquisition source indices must identify distinct scene points')
        acquisition_bands=bands
        if a.phase_acquisition_range_offsets:
            from full_ship_field import select_range
            acquisition_bands=[SceneBand(PointBand(select_range(f,*a.phase_acquisition_range_offsets),c,motion,point,grid,base,anchor,a.amplitude_spacing_s,a.orientation,
                angular_rate_offset_degps=a.angular_rate_offset_degps,angular_acceleration_offset_degps2=a.angular_acceleration_offset_degps2,free_path_offset=a.fit_path_offset),points,amplitude_spacing=a.amplitude_spacing_s) for f,c in zip(fields,coordinates)]
        amplitude=[np.ones((len(b.point.t),len(points))) for b in acquisition_bands]
        polynomial,acquisition=acquire(acquisition_bands,mid,half,amplitude_weights=amplitude,subaperture_s=a.phase_acquisition_seconds,source_indices=indices)
        acquisition['range_gate_offsets']=a.phase_acquisition_range_offsets
        acquisition['final_likelihood_uses_full_range_field']=True
        sample=np.linspace(*motion.interval,4001);u=(sample-mid)/half
        delta=np.linalg.lstsq(BSpline.design_matrix(sample,grid,3).toarray(),np.c_[u,u*u,u*u*u]@polynomial,rcond=None)[0]
        if a.max_acceleration_mps2 is not None:
            total,projection=project_initial(grid,base+delta,a.max_acceleration_mps2,anchor);delta=total-base
            acquisition['physical_initializer_projection']=projection
        initial[:len(base)-1]=delta[:-1]-delta[-1]
        if a.fit_path_offset:initial[likelihood.n-1]=float(BSpline(grid,np.eye(len(base)),3)(anchor)@delta)
        (a.output/'phase-acquisition.json').write_text(json.dumps(serial(acquisition),indent=2)+'\n')
        print(json.dumps(dict(stage='polynomial-phase-acquisition',polynomial_m=polynomial.tolist(),zero_score=acquisition['zero_correction_score_fraction'],acquired_score=acquisition['best_score_fraction'],seconds=time.monotonic()-start)),flush=True)
    phase_warmup=None
    if a.phase_warmup_iterations:
        if scene is None:raise ValueError('Phase-block initialization requires the persistent-scene model')
        phase_report_time=time.monotonic()
        def phase_progress(record):
            nonlocal phase_report_time
            if time.monotonic()-phase_report_time<30:return
            phase_report_time=time.monotonic();parameters=initial.copy();parameters[likelihood.n:phase_end]=record['phase_parameters']
            (a.output/'phase-checkpoint.json').write_text(json.dumps(serial(dict(parameters=parameters,scope='Best evaluated phase-block training point; physical path held fixed',objective=record['objective'])),indent=2)+'\n')
            print(json.dumps(dict(stage='phase-block-progress',calls=record['calls'],objective=record['objective'],maximum_gradient=record['maximum_gradient'],seconds=time.monotonic()-start)),flush=True)
        phase_initial,phase_warmup=warm_phases(ScenePath(bands),initial[:phase_end],a.phase_warmup_iterations,phase_progress,fixed_band_offsets)
        initial[:phase_end]=phase_initial
        print(json.dumps(dict(stage='phase-block-initialization',**phase_warmup,seconds=time.monotonic()-start)),flush=True)
    elif a.phases_only:raise ValueError('Phase-only mode requires a nonzero phase-block iteration limit')
    def validation(parameters,models):
        if points is None:return [b.evaluate(parameters[:likelihood.n],parameters[likelihood.n+c],c,training=False,derivatives=False,frozen_amplitude=model['amplitude']) for c,(b,model) in enumerate(zip(bands,models))]
        if isinstance(likelihood,RefinedScenePath):likelihood.set_points(parameters)
        phi=np.r_[0.,parameters[likelihood.n+5:phase_end]]
        return [b.evaluate(parameters[:likelihood.n],phi+parameters[likelihood.n+c],c,training=False,derivatives=False,frozen_amplitude=model['amplitude']) for c,(b,model) in enumerate(zip(bands,models))]
    before,gradient,models=likelihood.objective(initial,True)
    unused_before=validation(initial,models)
    print(json.dumps(dict(stage='initial',objective=before,unused_explained_fraction=1-sum(v['cost'] for v in unused_before)/sum(v['energy'] for v in unused_before),seconds=time.monotonic()-start)),flush=True)
    reports=[];calls=0;last_report=time.monotonic();cache=[None,None]
    noise_degrees=sum((np.count_nonzero(b.training&(b.field.root_information>0))-len(np.unique(b.groups[b.training&(b.field.root_information>0)])))*len(b.field.offsets) for b in point_bands)
    if noise_degrees<=0:raise ValueError('No residual Gaussian noise observations remain after stationary projection')
    numerical_scale=likelihood.energy/noise_degrees if a.noise_objective else 1.
    def objective(x):
        nonlocal calls,last_report
        if cache[0] is None or not np.array_equal(x,cache[0]):
            value,g=likelihood.objective(x);cache[:]=[x.copy(),(value,g)];calls+=1
            if time.monotonic()-last_report>30:
                print(json.dumps(dict(stage='evaluating',calls=calls,objective=value,maximum_gradient=float(max(abs(g))),seconds=time.monotonic()-start)),flush=True);last_report=time.monotonic()
        return cache[1]
    def callback(x):
        value,g=objective(x);report=dict(iteration=len(reports)+1,objective=value,maximum_gradient=float(max(abs(g))),seconds=time.monotonic()-start)
        if bound is not None:report.update(certificate(x,g*numerical_scale,bound))
        reports.append(report);print(json.dumps(report),flush=True)
        (a.output/'checkpoint.json').write_text(json.dumps(serial(dict(parameters=x,gradient=g,optimizer_likelihood_scale=numerical_scale,reports=reports)),indent=2)+'\n')
    bound=None;coordinate_transform=None
    if a.phase_coordinate_preconditioner:
        from scene_phase_coordinates import transform
        coordinate_transform=transform(likelihood,len(initial))
        if a.information_preconditioner:
            if not a.noise_objective:raise ValueError('Information coordinates require the noise-scaled objective')
            from full_scene_information import information,coordinates,aspect_information
            if len(initial)>128:
                def information_progress(record):print(json.dumps(dict(stage='aspect-information',window=record['window'],seconds=time.monotonic()-start)),flush=True)
                fisher,details=aspect_information(likelihood,initial,information_progress)
                error=None
            else:
                fisher,checked_gradient,details=information(likelihood,initial)
                reference_gradient=likelihood.objective(initial)[1]
                error=float(max(abs(checked_gradient-reference_gradient)))
                if error>1e-6:raise ValueError(f'Projected Gaussian Jacobian disagrees with the original likelihood gradient: {error}')
            coordinate_transform,diagnostic=coordinates(fisher,coordinate_transform,noise_degrees)
            np.save(a.output/'conditional-local-information.npy',fisher)
            diagnostic.update(maximum_original_gradient_error=error,channels=details)
            diagnostic['aspect_block_approximation']=len(initial)>128
            (a.output/'information-coordinates.json').write_text(json.dumps(serial(diagnostic),indent=2)+'\n')
            print(json.dumps(dict(stage='information-coordinates',gradient_error=error,seconds=time.monotonic()-start)),flush=True)
        np.save(a.output/'optimizer-coordinate-transform.npy',coordinate_transform)
    elif a.information_preconditioner:raise ValueError('Information coordinates require the phase pivot')
    def physical(q):return q if coordinate_transform is None else coordinate_transform@q
    def optimizer_objective(q):
        value,g=objective(physical(q))
        value=(value-1)*numerical_scale if a.noise_objective else value
        g=g*numerical_scale
        return value,g if coordinate_transform is None else g@coordinate_transform
    optimizer_initial=initial if coordinate_transform is None else np.linalg.solve(coordinate_transform,initial)
    def optimizer_callback(q):callback(physical(q))
    if a.max_acceleration_mps2 is None:
        bounds=None if a.refine_points_m is None else [(None,None)]*phase_end+[(-a.refine_points_m,a.refine_points_m)]*(len(initial)-phase_end)
        result=OptimizeResult(x=optimizer_initial,success=True,message='Phase/brightness block only; physical parameters held fixed',nit=0) if a.phases_only else minimize(optimizer_objective,optimizer_initial,jac=True,method='L-BFGS-B',callback=optimizer_callback,
            bounds=bounds,
            options=dict(maxiter=a.iterations,gtol=1e-8,ftol=1e-12,maxcor=30,maxls=30))
    else:
        bound=constraint(grid,base,len(initial),a.max_acceleration_mps2,a.max_path_correction_m,anchor,a.fit_path_offset)
        if fixed_band_offsets is not None:
            free=np.array([0,1,2,4]);phase_matrix=np.zeros((4,len(initial)))
            phase_matrix[np.arange(4),likelihood.n+free]=1.;phase_matrix[:,likelihood.n+3]=-1.
            bound=LinearConstraint(np.r_[bound.A,phase_matrix],np.r_[bound.lb,fixed_band_offsets[free]],np.r_[bound.ub,fixed_band_offsets[free]])
        if a.refine_points_m is not None:
            position_matrix=np.eye(len(initial))[phase_end:]
            bound=LinearConstraint(np.r_[bound.A,position_matrix],np.r_[bound.lb,np.full(len(position_matrix),-a.refine_points_m)],np.r_[bound.ub,np.full(len(position_matrix),a.refine_points_m)])
        def scaled_objective(x):
            value,g=optimizer_objective(x)
            factor=1. if a.noise_objective else 100.
            return value*factor,g*factor
        optimizer_bound=bound if coordinate_transform is None else LinearConstraint(bound.A@coordinate_transform,bound.lb,bound.ub)
        result=OptimizeResult(x=optimizer_initial,success=True,message='Phase/brightness block only; physical parameters held fixed',nit=0) if a.phases_only else minimize(scaled_objective,optimizer_initial,jac=True,method='SLSQP',constraints=[optimizer_bound],callback=optimizer_callback,
            options=dict(maxiter=a.iterations,ftol=1e-11 if a.noise_objective else 1e-9))
    result.x=physical(result.x)
    value,g,models=likelihood.objective(result.x,True)
    stationarity=certificate(result.x,g*numerical_scale,bound) if bound is not None else dict(maximum_kkt_residual=float(max(abs(g*numerical_scale))),maximum_constraint_violation=0.,active_linear_constraints=0)
    unused=validation(result.x,models)
    origin=np.asarray(__import__('scipy').interpolate.BSpline(grid,np.eye(len(base)),3)(anchor))
    correction=np.r_[result.x[:len(base)-1],0.];final_coeff=base+correction-np.dot(origin,correction)
    if a.fit_path_offset:final_coeff+=result.x[likelihood.n-1]
    arrays={};figures=[]
    for c,(b,model) in enumerate(zip(point_bands,models)):
        arrays[f'time_s_{c}']=b.t;arrays[f'path_m_{c}']=b.base_path+b.b@result.x[:likelihood.n]
        arrays[f'rate_mps_{c}']=b.base_rate+b.bd@result.x[:likelihood.n];arrays[f'amplitude_{c}']=b.amplitude@model['amplitude']
        arrays[f'amplitude_coefficients_{c}']=model['amplitude']
    np.savez_compressed(a.output/'trajectory.npz',**arrays)
    gains=np.exp(1j*result.x[likelihood.n:likelihood.n+5])
    output=dict(complete=True,scope=__doc__,motion=str(motion.path),field=str(a.field),coarse_track=str(a.coarse_track),phase_seed=str(a.phase_seed),
        model_kind='whole-capture-radar-scene-path' if scene else 'whole-capture-radar-point-path',point_body_m=point,interval_s=motion.interval,path_knots_s=grid,path_coefficients_m=final_coeff,
        physical_gains_real_imag=np.c_[gains.real,gains.imag],parameters=result.x,path_spacing_s=a.spacing_s,amplitude_spacing_s=a.amplitude_spacing_s,
        band_calibration='Constant phase only; positive per-scatterer brightness absorbs band magnitude. Unit response is a declared combined-image hypothesis.',
        initial=a.initial,orientation=a.orientation,initial_objective=before,objective=value,maximum_gradient=float(max(abs(g))),iterations=result.nit,calls=calls,
        message=str(result.message),numerical_convergence=bool(result.success and stationarity['maximum_kkt_residual']<1e-6 and stationarity['maximum_constraint_violation']<1e-8),seconds=time.monotonic()-start,
        unused_before=list(map(compact,unused_before)),unused=list(map(compact,unused)),
        unused_explained_fraction=1-sum(v['cost'] for v in unused)/sum(v['energy'] for v in unused),
        training=list(map(compact,models)),anchor_path_time_s=anchor,anchor_path_is_coarse=not a.fit_path_offset,constant_physical_path_offset_fitted=a.fit_path_offset,unique_geographic_localization=False)
    if a.fit_path_offset:output['physical_path_offset_change_m']=float(result.x[likelihood.n-1])
    if a.angular_rate_offset_degps or a.angular_acceleration_offset_degps2:
        output.update(angular_rate_offset_degps=a.angular_rate_offset_degps,angular_acceleration_offset_degps2=a.angular_acceleration_offset_degps2)
    if scene:
        final_points=likelihood.set_points(result.x) if isinstance(likelihood,RefinedScenePath) else points
        output.update(source_scene=str(a.scene),scene_points_body_m=final_points,scene_point_phase_radians=np.r_[0.,result.x[likelihood.n+5:phase_end]])
        if a.refine_points_m is not None:output.update(source_refinement_bound_m=a.refine_points_m,initial_scene_points_body_m=points)
    if scene and 'scene_window_edges_s' in scene:output['scene_window_edges_s']=scene['scene_window_edges_s']
    if scene and 'active_points' in scene:output['scene_active_points']=scene['active_points']
    if resumed:output['reference_continuous_motion']=str(a.resume_motion)
    if fixed_band_offsets is not None:
        output['fixed_relative_band_phase_offsets_radians']=fixed_band_offsets
        output['band_calibration']='Upstream relative band phases fixed; one common constant phase and shared-source phases fitted, with positive per-band brightness'
    output.update(stationarity,max_acceleration_mps2=a.max_acceleration_mps2,max_path_correction_m=a.max_path_correction_m if bound is not None else None,projected_initializer=projected_initial,phase_warmup=phase_warmup)
    output.update(noise_objective=a.noise_objective,optimizer_likelihood_scale=numerical_scale,noise_observation_degrees=noise_degrees,phase_acquisition_used=bool(acquisition))
    output['optimizer_objective_multiplier']=1. if a.noise_objective else 100.
    output['optimization_scope']='Phase and real brightness only; motion held fixed' if a.phases_only else 'Joint physical path and constant scene/band phases; real brightness profiled'
    if a.range_offsets:
        output['fit_range_offsets']=a.range_offsets
        output['range_selection_scope']='Fixed raw range selection for local motion estimation only; conditional covariance recomputed; original profiles and native image ranges retained'
    if coordinate_transform is not None:output['optimizer_coordinates']='Invertible frozen carrier-phase pivot transform; original physical likelihood, constraints and exported parameters retained'
    if a.information_preconditioner:output['optimizer_coordinates']='Invertible frozen local Gaussian-information transform after active real-brightness projection; original likelihood, constraints and physical parameters retained'
    (a.output/'analysis.json').write_text(json.dumps(serial(output),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,12),layout='constrained')
    b=point_bands[3];axes[0].plot(b.t,b.base_path,label='Starting continuous path');axes[0].plot(b.t,arrays['path_m_3'],label='Complex-data fitted path');axes[0].legend();axes[0].set(ylabel='Ten excess path (m)')
    axes[1].plot(b.t,arrays['rate_mps_3']);axes[1].set(ylabel='Same physical path derivative (m/s)')
    for c in range(5):
        amp=arrays[f'amplitude_{c}'];amp=amp if amp.ndim==1 else amp.sum(axis=1)
        axes[2].plot(point_bands[c].t,amp,label=['Seven','SBS','Nine','Ten','ABC'][c])
    axes[2].legend(ncol=5);axes[2].set(xlabel='Capture time (s)',ylabel='Positive brightness')
    count=1 if points is None else likelihood.p
    fig.suptitle(f'One continuous physical path; {count} declared reflector supports; constant band calibration\nExcluded complex energy predicted {100*output["unused_explained_fraction"]:.2f}%; geographic/extended-scene validation remains separate')
    fig.savefig(a.output/'physical-point-path.png',dpi=140);plt.close(fig)
    print(json.dumps(serial({k:v for k,v in output.items() if k in ['objective','unused_explained_fraction','maximum_gradient','seconds','numerical_convergence']})),flush=True)

if __name__=='__main__':main()
