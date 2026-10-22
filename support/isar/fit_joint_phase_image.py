"""Fine common-path acquisition with an unknown physical-pixel sparse image.

Coarse range/Doppler, attitude and constant inter-band calibration are given.
The image is re-estimated at every motion trial. The phase approximation is
checked afterwards against the exact physical waveform, with no alignment.
Synthetic sources are off the estimator mesh and are never supplied to it.
This is an offline acquisition experiment, not an accepted production fit.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from full_scene_context import open_scene
from fit_distributed_physical_scene import operator_fingerprint
from moment_phase_scene import MomentPhaseScene
from sparse_profiled_phase_scene import SparseSceneSolver
from run_residual_scene_reconstruction import simulate
from rigid_patch import profile_static


def shifted_bands(bands, bases, rates, parameters):
    result=[]
    for band,basis,rate in zip(bands,bases,rates):
        other=copy.copy(band)
        other.base_path=band.base_path+basis@parameters
        other.base_rate=band.base_rate+rate@parameters
        result.append(other)
    return result


def exact_predictions(bands, points, coefficient, spectrum, beta):
    active=np.flatnonzero(abs(coefficient)>0)
    predictions=[]
    for c,band in enumerate(bands):
        raw=np.zeros_like(band.field.y)
        for j in active:
            raw+=band.waveform(np.zeros(band.b.shape[1]),c,False,points[j]-band.reference_point)*coefficient[j]
        predictions.append(raw*spectrum[c]*np.exp(1j*beta[c]))
    return predictions


def prediction_score(bands,predictions,masks):
    records=[]
    for band,raw,mask in zip(bands,predictions,masks):
        f=band.field
        y=profile_static(f.y,f.root_information,mask,band.groups)*mask[:,None]
        prediction=profile_static(raw,f.root_information,mask,band.groups)*mask[:,None]
        cost=float(np.vdot(y-prediction,y-prediction).real)
        energy=float(np.vdot(y,y).real)
        records.append(dict(cost=cost,energy=energy,explained_fraction=1-cost/energy))
    cost=sum(r['cost'] for r in records);energy=sum(r['energy'] for r in records)
    return dict(cost=cost,energy=energy,explained_fraction=1-cost/energy,channels=records)


def scalar_report(report):
    result={k:v for k,v in report.items() if k not in ['coefficient','physical_coefficient']}
    if 'sparse_optimization' in result:
        result['sparse_optimization']={k:v for k,v in result['sparse_optimization'].items() if k!='trace'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input',choices=['synthetic-hull','recorded'],required=True)
    p.add_argument('--truth-path-m',type=float,nargs=3,default=[.11,-.07,.03])
    p.add_argument('--point-snr-db',type=float,default=30.)
    p.add_argument('--strength-relative',type=float,default=.1)
    p.add_argument('--prior-from',type=Path,help='Freeze the absolute prior chosen in an earlier inner-validation result')
    p.add_argument('--fit-band-phases',action='store_true',help='Refine one constant relative calibration phase per band; Ten fixes the gauge')
    p.add_argument('--maximum-iterations',type=int,default=60)
    p.add_argument('--coefficient-bound-m',type=float,default=.5)
    p.add_argument('--fixed-linear-path-m',type=float,help='Fix the linear phase/image-shift gauge at the declared coarse acquisition value')
    p.add_argument('--review-only',action='store_true',help='Render matched diagnostics for an existing fit without refitting')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.review_only);started=time.monotonic();previous=None
    if a.review_only:
        previous=json.loads((a.output/'analysis.json').read_text())
        if previous['input']!=a.input or Path(previous['modes'])!=a.modes:raise ValueError('Review input differs from saved fit')
        if previous['truth'] is not None:
            a.truth_path_m=previous['truth']['truth_residual_path_coefficients_m']
            a.point_snr_db=previous['truth']['per_point_integrated_snr_db']
        a.fixed_linear_path_m=previous['fixed_linear_path_m']
    description=json.loads((a.modes/'analysis.json').read_text())
    if not description['complete']:raise ValueError('Incomplete physical operator modes')
    geometry=Path(description['geometry']);fit,_,scene,_=open_scene(Path(description['motion']))
    bands=[s.point for s in scene];points=np.load(a.modes/'points-body.npy');v=np.load(a.modes/'spatial-basis.npy')
    beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']])
    if operator_fingerprint(bands,points,beta,Path(description['motion']))!=description['operator_fingerprint']:
        raise ValueError('Known physical operator fingerprint changed')
    mid=np.mean(fit['interval_s']);half=np.diff(fit['interval_s'])[0]/2
    basis=[];rates=[]
    for band in bands:
        u=(band.t-mid)/half
        basis.append(np.column_stack([u,u*u,u*u*u]))
        rates.append(np.column_stack([np.ones_like(u),2*u,3*u*u])/half)
    truth=None;true_bands=None
    if a.input=='synthetic-hull':
        truth_parameters=np.asarray(a.truth_path_m)
        true_bands=shifted_bands(bands,basis,rates,truth_parameters)
        x=np.arange(-200.,80.1,.75);y=np.arange(-80.,-19.9,.75);xx,yy=np.meshgrid(x,y)
        fine_points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
        observed,truth=simulate(true_bands,beta,fine_points,'hull',a.point_snr_db,spectral_model='shared')
        truth.update(grid_aligned_to_estimator=False,motion_given=False,truth_residual_path_coefficients_m=truth_parameters.tolist())
        for band,synthetic in zip(bands,observed):
            band.field=copy.copy(band.field);band.field.y=synthetic.field.y
        spectrum=np.ones(5,complex)
    else:
        with np.load(geometry/'scene.npz') as f:spectrum=f['spectrum_gains']
    initial_spectrum=spectrum.copy()
    if a.review_only:
        with np.load(a.output/'scene.npz') as f:spectrum=f['spectrum_gains']
    designs=[np.load(a.modes/f'channel-{c}-design.npy',mmap_mode='r') for c in range(5)]
    projected=[];original_point_information=[]
    with np.load(geometry/'statistics.npz') as cache:
        if str(cache['operator_fingerprint'])!=description['operator_fingerprint']:raise ValueError('Physical statistics identity changed')
        for c in range(5):
            original_point_information.append(cache[f'original_gram_{c}'].diagonal().real.copy())
            target=a.modes/f'channel-{c}-original-modal-gram.npy'
            if target.exists():g=np.load(target)
            else:
                g=v.conj().T@(cache[f'original_gram_{c}']@v)
                g=(g+g.conj().T)/2;np.save(target,g)
            projected.append(g)
            print(json.dumps(dict(stage='modal-normal',channel=c,seconds=time.monotonic()-started)),flush=True)
    degree=64;moment_paths=[a.modes/f'channel-{c}-original-chebyshev-{degree}-means.npy' for c in range(5)]
    moments=[np.load(path,mmap_mode='r') for path in moment_paths] if all(path.exists() for path in moment_paths) else None
    model=MomentPhaseScene(designs,[b.field.y for b in bands],[b.field.root_information for b in bands],
        [b.groups for b in bands],[b.training for b in bands],basis,[b.field.k for b in bands],1.,spectrum,
        degree=degree,projected_gram=projected,operator_moments=moments,
        progress=lambda item:print(json.dumps(dict(stage='original-epoch-moments',seconds=time.monotonic()-started,**item)),flush=True))
    if moments is None:
        for path,value in zip(moment_paths,model.mean_moments):np.save(path,value)
    gram,rhs,*_=model.normal_with_derivatives(np.zeros(3))
    strength=a.strength_relative*float(max(abs(v@rhs)))
    if a.prior_from is not None:
        prior=json.loads(a.prior_from.read_text())
        if Path(prior['motion'])!=Path(description['motion']):raise ValueError('Prior selection used a different coarse motion')
        strength=float(prior['strength'])
    solver=SparseSceneSolver(v,strength,description['discarded_operator_eigenvalue'])
    model.strength=strength;model.image_solver=solver
    raw_point_information=[q+np.sum(abs(mean[0]@v.conj().T)**2,axis=0) for q,mean in zip(original_point_information,model.mean_moments)]
    if a.review_only:
        from phase_image_review import save_review
        initial=np.zeros(3)
        if a.fixed_linear_path_m is not None:initial[0]=a.fixed_linear_path_m
        save_review(a.output,model,v,points,raw_point_information,initial,np.asarray(previous['parameters_m']),truth=truth,
            scale_db=(0.,35.) if truth else (0.,55.),initial_spectrum=initial_spectrum)
        print(json.dumps(dict(stage='matched-review',output=str(a.output),seconds=time.monotonic()-started)),flush=True)
        return
    calibrated=None
    if a.fit_band_phases:
        from constant_band_calibration import ConstantBandCalibration
        calibrated=ConstantBandCalibration(model)
    objective=model if calibrated is None else calibrated
    last=None;history=[]
    def evaluate(theta):
        nonlocal last
        if last is None or not np.array_equal(last[0],theta):
            value,gradient,report=objective.objective(theta,True);last=(theta.copy(),value,gradient,report)
            compact=dict(evaluation=len(history),parameters_m=theta[:3].tolist(),parameters=theta.tolist(),objective=value,gradient=gradient.tolist(),
                seconds=time.monotonic()-started,**scalar_report(report))
            history.append(compact)
            (a.output/'checkpoint.json').write_text(json.dumps(dict(complete=False,latest=compact),indent=2)+'\n')
            print(json.dumps({k:compact[k] for k in ['evaluation','parameters_m','objective','gradient','seconds']}),flush=True)
        return last[1],last[2]
    initial=np.zeros(3 if calibrated is None else 3+len(calibrated.channels))
    if a.fixed_linear_path_m is not None:initial[0]=a.fixed_linear_path_m
    evaluate(initial)
    initial_report=scalar_report(last[3]);initial_modal=last[3]['coefficient'].copy()
    initial_outer=model.score(initial[:3],initial_modal,[~b.training for b in bands])
    bounds=[(-a.coefficient_bound_m,a.coefficient_bound_m)]*3
    if a.fixed_linear_path_m is not None:bounds[0]=(a.fixed_linear_path_m,a.fixed_linear_path_m)
    if calibrated is not None:bounds.extend([(-np.pi,np.pi)]*len(calibrated.channels))
    result=minimize(evaluate,initial,jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=a.maximum_iterations,gtol=1e-7,ftol=1e-12,maxls=25,maxcor=10))
    evaluate(result.x);report=last[3];coefficient=report['physical_coefficient'];modal=report['coefficient']
    parameters=result.x[:3];spectrum=model.spectrum.copy()
    outer=model.score(parameters,modal,[~b.training for b in bands])
    fitted_bands=shifted_bands(bands,basis,rates,parameters)
    raw=exact_predictions(fitted_bands,points,coefficient,spectrum,beta)
    physical_outer=prediction_score(bands,raw,[~b.training for b in bands])
    approximation=[]
    for c,(band,design) in enumerate(zip(bands,designs)):
        approximate=np.einsum('trp,p->tr',design,modal)*(spectrum[c]*np.exp(-1j*band.field.k*(basis[c]@parameters)))[:,None]
        error=raw[c]-approximate
        approximation.append(dict(channel=c,relative_squared_error=float(np.vdot(error,error).real/max(np.vdot(raw[c],raw[c]).real,1e-300))))
    known=[];noiseless_prediction=[]
    if truth is not None:
        for c,(true,fitted) in enumerate(zip(true_bands,fitted_bands)):
            signal=np.zeros_like(true.field.y)
            for j,point in enumerate(np.asarray(truth['points_body_m'])):
                h=true.waveform(np.zeros(true.b.shape[1]),c,False,point-true.reference_point)
                signal+=h*truth['source_magnitudes_by_band'][c][j]*np.exp(1j*(beta[c]+truth['source_phase_radians'][j]))
                q=fitted.waveform(np.zeros(fitted.b.shape[1]),c,False,point-fitted.reference_point)
                h=profile_static(h,true.field.root_information,np.ones(len(true.t),bool),true.groups)
                q=profile_static(q,fitted.field.root_information,np.ones(len(fitted.t),bool),fitted.groups)
                norm=np.sqrt(np.vdot(h,h).real*np.vdot(q,q).real);z=np.vdot(h,q)/norm
                known.append(dict(channel=c,point=j,coherence_power=float(abs(z)**2),coherence_loss_db=float(-10*np.log10(abs(z)**2)),
                    complex_phase_error_degrees=float(np.angle(z,deg=True))))
            mask=~true.training;r=true.field.root_information
            target=profile_static(signal,r,mask,true.groups)*mask[:,None]
            predicted=profile_static(raw[c],r,mask,true.groups)*mask[:,None]
            e=float(np.vdot(target,target).real);q=float(np.vdot(predicted,predicted).real);cross=np.vdot(target,predicted)
            noiseless_prediction.append(dict(channel=c,true_signal_energy=e,predicted_energy=q,
                unaligned_relative_squared_error=float(np.vdot(target-predicted,target-predicted).real/e),
                complex_coherence_power=float(abs(cross)**2/(e*q)) if q else 0.,phase_error_degrees=float(np.angle(cross,deg=True))))
    output=dict(complete=True,accepted_physical_fit=False,scope=__doc__,input=a.input,modes=str(a.modes),motion=description['motion'],
        fixed_coarse_motion=True,relative_band_calibration_fitted=a.fit_band_phases,
        image_source_positions_supplied=False,truth=truth,parameters_m=parameters.tolist(),
        band_phase_corrections_radians=report.get('band_phase_corrections_radians'),prior_from=str(a.prior_from) if a.prior_from else None,
        fixed_linear_path_m=a.fixed_linear_path_m,
        objective=float(result.fun),gradient=result.jac.tolist(),optimizer_success=bool(result.success),optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),initial=initial_report,final=scalar_report(report),strength_relative=a.strength_relative,strength=strength,
        initial_outer=initial_outer,outer=outer,exact_physical_outer=physical_outer,physical_approximation=approximation,
        known_point_coherence=known,noiseless_outer_prediction=noiseless_prediction,history=history,seconds=time.monotonic()-started)
    np.savez_compressed(a.output/'scene.npz',points_body_m=points,coefficient=coefficient,modal_coefficient=modal,
        spectrum_gains=spectrum,band_phase_radians=beta+np.angle(spectrum),parameters_m=parameters)
    (a.output/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    from phase_image_review import save_review
    save_review(a.output,model,v,points,raw_point_information,initial[:3],parameters,truth=truth,scale_db=(0.,35.) if truth else (0.,55.),initial_spectrum=initial_spectrum)
    print(json.dumps(dict(stage='finished',parameters_m=parameters.tolist(),initial_outer=initial_outer,outer=outer,
        physical_outer=physical_outer,maximum_approximation_error=max(r['relative_squared_error'] for r in approximation),
        maximum_known_point_coherence_loss_db=max((r['coherence_loss_db'] for r in known),default=None),seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
