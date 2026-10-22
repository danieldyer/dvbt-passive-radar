"""Joint antenna/motion refits using all native epochs and the jerk prior.

Each discrete site hypothesis receives the same physical model, location
prior, unknown constant band phases and nonnegative reflectivity envelopes.
No midpoint position/velocity is fixed. Four common source locations remain
conditional; these fits are not automatically accepted antenna calibrations.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,LinearConstraint,linprog,OptimizeResult
from antenna_motion import Antennas
from continuous_ship_fit import ContinuousScene
from continuous_ship_motion import ContinuousShipMotion
from full_path_scene import SceneBand,warm_phases
from full_ship_field import load
from joint_continuous_ship import JointCoordinates,JointPointBand,JointPrior
from kinematic_prior import KinematicPrior
from motion import excess_path
from scene_dictionary import fingerprint


def feasible_improvement(q,g,matrix,lo,hi,lower,upper):
    result=linprog(g,A_ub=np.r_[matrix,-matrix],b_ub=np.r_[hi-matrix@q,matrix@q-lo],
        bounds=list(zip(np.maximum(-1,lower-q),np.minimum(1,upper-q))),method='highs')
    return float(max(0.,-result.fun)) if result.success else None


def location_information(scene,q):
    """Local active-face information; shared source positions stay fixed."""
    gram=scene.mean_information(q,profile_brightness=True);model=scene.model
    interest=np.arange(model.location_start,model.n)
    nuisance=np.setdiff1d(np.arange(len(q)),interest)
    a=gram[np.ix_(interest,interest)];b=gram[np.ix_(interest,nuisance)];n=gram[np.ix_(nuisance,nuisance)]
    # The jerk/height priors regularise nuisance motion, not antenna data.
    prior=np.zeros_like(gram);prior[:model.nmotion,:model.nmotion]=scene.prior.motion_prior.gram
    prior[model.nmotion,model.nmotion]=1.;n+=prior[np.ix_(nuisance,nuisance)]
    scale=np.sqrt(np.maximum(np.diag(n),1e-30));unit=n/scale[:,None]/scale[None,:];cross=b/scale[None,:]
    trials=[]
    for cutoff in [1e-8,1e-10,1e-12]:
        schur=a-cross@np.linalg.pinv(unit,rcond=cutoff,hermitian=True)@cross.T;schur=(schur+schur.T)/2
        eigen=np.linalg.eigvalsh(schur)
        trials.append(dict(relative_rank_cutoff=cutoff,prior_standardized_location_information_eigenvalues=eigen.tolist(),directions_weaker_than_location_prior=int(np.sum(eigen<1))))
    return dict(trials=trials,scope='Local mean information on the active positive-brightness face: brightness, motion, height and phase nuisance eliminated; four source coordinates remain fixed. Jerk/height prior included explicitly; antenna prior and hard bounds excluded. Not calibrated antenna error bars or a source-location uncertainty certificate.')


def fit_case(fields,coordinates,source,priors,assignment,out,iterations,workers,*,fixed=False,seed=None,checkpoint=None,evaluate_only=False,fixed_antenna_up=False):
    motion=copy.copy(source);motion.tx=np.asarray([priors['sites'][code]['ecef_m'] for code in assignment])
    antennas=Antennas(motion.rotation,list(assignment));model=JointCoordinates(motion,antennas)
    source_fit=source.result['radar_fit'];edges=np.asarray(source_fit['aspect_edges_s']);points=np.asarray(source.result['reference_points_east_north_up_m'])
    bands=[]
    for c,(f,coordinate) in enumerate(zip(fields,coordinates)):
        point=JointPointBand(f,coordinate,model,c,source_fit['amplitude_spacing_s'])
        bands.append(SceneBand(point,np.broadcast_to(points,(len(edges)-1,*points.shape)),edges,source_fit['amplitude_spacing_s'],adjoint_gradient=True))
    settings=source_fit['motion_prior']
    prior=JointPrior(model,KinematicPrior(model.base,acceleration_change_mps2=settings['acceleration_change_mps2'],
        angular_rate_change_degps=settings['angular_rate_change_degps'],timescale_s=settings['timescale_s']))
    # Geometry changes have an arbitrary constant RF origin. Re-express the
    # old constant calibration in the new origin before releasing its phases.
    anchor=model.gauge_point[None];beta=np.angle(source.gains).copy()
    for c in range(5):beta[c]+=2*np.pi*motion.rf[c]/299792458.*float(excess_path(anchor,motion.tx[c],motion.rx)[0][0]-excess_path(anchor,source.tx[c],source.rx)[0][0])
    scene=ContinuousScene(bands,model,beta,workers,prior,fit_band_phases=True,profile_brightness=True)
    at=np.r_[np.zeros(model.n),np.asarray(source_fit['parameters'])[-scene.p:],np.zeros(len(scene.free_bands))]
    if seed is not None:
        previous=np.asarray(seed['parameters']);at[:model.nmotion]=previous[:model.nmotion]
        at[model.nmotion]=previous[model.nmotion];at[model.n:model.n+scene.p]=previous[model.n:model.n+scene.p]
        at[model.n+scene.p:]=previous[model.n+scene.p:]
        # Refit constant origins for each site; do not transfer correlated
        # location coordinates between incompatible site assignments.
    if checkpoint is not None:
        if checkpoint['assignment']!=assignment or checkpoint['fixed_antenna_locations']!=fixed or len(checkpoint['parameters'])!=len(at):
            raise ValueError('A continuation must preserve the site hypothesis and parameterization')
        at=np.asarray(checkpoint['parameters']).copy()
    matrix,lo,hi=model.constraints();matrix=np.pad(matrix,((0,0),(0,len(at)-model.n)))
    base=model.base
    bounds=[(-150.,150.)]*base.np+[(-50.,50.)]*base.np+[(-np.radians(20)/.01,np.radians(20)/.01)]*(base.na-1)
    bounds += [(-3.,3.)]+([(0.,0.)] if fixed else [(-3.,3.)])*antennas.count+[(None,None)]*(scene.p+len(scene.free_bands))
    if fixed_antenna_up:
        # The covariance has no EN/U cross terms. Fixing all six whitened
        # U coordinates therefore fixes the physical local-up offsets too.
        for c in range(6):bounds[model.location_start+3*c+2]=(0.,0.)
    lower=np.array([-np.inf if a is None else a for a,b in bounds]);upper=np.array([np.inf if b is None else b for a,b in bounds])
    if max(np.max(lo-matrix@at),np.max(matrix@at-hi),np.max(lower-at),np.max(at-upper),0.)>1e-7:
        raise ValueError('The initializer violates this case\'s physical constraints or fixed coordinates')
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();trace=[]
    if checkpoint is not None:
        reproduced=scene.objective(at)[0]
        if abs(reproduced-checkpoint['objective'])>2e-10:
            raise ValueError(f'Continuation does not reproduce its original likelihood: {reproduced} versus {checkpoint["objective"]}')
    # Solve the inexpensive reflectivity/constant-phase block of this same
    # likelihood first. Moving geometry to compensate a poor constant-phase
    # initializer wastes expensive full-record waveform evaluations.
    temporary=scene.phase_coordinates(at)
    def phase_progress(r):
        if r['calls']==1 or r['calls']%10==0:
            print(json.dumps(dict(case=out.name,stage='constant-phase-initialization',**{k:v for k,v in r.items() if k!='phase_parameters'})),flush=True)
    if evaluate_only:fitted,phase_report=temporary,dict(scope='Saved feasible point evaluated without further optimization')
    else:fitted,phase_report=warm_phases(scene,temporary,40,phase_progress,workers=workers)
    at=scene.from_phase_coordinates(fitted)
    (out/'phase-initialization.json').write_text(json.dumps(phase_report,indent=2)+'\n')
    initial=scene.objective(at)[0]
    print(json.dumps(dict(case=out.name,stage='initial-objective',objective=initial,evaluations=scene.calls,seconds=time.monotonic()-started)),flush=True)
    origin=at.copy()
    if evaluate_only:transform,conditioning=np.eye(len(at)),dict(scope='No optimization or preconditioning in evaluation-only mode')
    else:transform,conditioning=scene.preconditioner(at)
    best=initial;best_q=at.copy()
    (out/'best-progress.json').write_text(json.dumps(dict(assignment=assignment,parameters=at.tolist(),history=[],objective=initial),indent=2)+'\n')
    (out/'preconditioning.json').write_text(json.dumps(conditioning,indent=2)+'\n')
    def expand(x):return origin+transform@x
    def objective(x):
        nonlocal best,best_q
        q=expand(x);value,g=scene.objective(q)
        violation=float(max(np.max(lo-matrix@q),np.max(matrix@q-hi),np.max(lower-q),np.max(q-upper),0.))
        if np.isfinite(value) and value<best and violation<1e-7:
            best=value;best_q=q.copy()
            (out/'best-progress.json').write_text(json.dumps(dict(assignment=assignment,parameters=q.tolist(),history=trace,
                objective=value,evaluations=scene.calls),indent=2)+'\n')
        return 1000*value,1000*g@transform
    equality=np.isfinite(lower)&(lower==upper);limited=(np.isfinite(lower)|np.isfinite(upper))&~equality
    constraints=[LinearConstraint(matrix@transform,lo-matrix@origin,hi-matrix@origin),LinearConstraint(transform[limited],lower[limited]-origin[limited],upper[limited]-origin[limited])]
    if equality.any():constraints.append(LinearConstraint(transform[equality],lower[equality]-origin[equality],upper[equality]-origin[equality]))
    class StationaryPoint(Exception):
        def __init__(self,x):self.x=x.copy()
    def callback(x):
        q=expand(x);value,g,values=scene.objective(q,True)
        violation=float(max(np.max(lo-matrix@q),np.max(matrix@q-hi),np.max(lower-q),np.max(q-upper),0.))
        record=dict(iteration=len(trace)+1,objective=value,radar_rss_fraction=sum(v['cost'] for v in values)/scene.energy,
            maximum_kinematic_or_box_violation=violation,maximum_scaled_gradient=float(max(abs(g))),evaluations=scene.calls,seconds=time.monotonic()-started)
        gap=feasible_improvement(q,g,matrix,lo,hi,lower,upper) if violation<1e-7 else None
        record['maximum_linearized_feasible_improvement']=gap
        trace.append(record);saved=dict(assignment=assignment,parameters=q.tolist(),history=trace)
        (out/'progress.json').write_text(json.dumps(saved,indent=2)+'\n')
        print(json.dumps(dict(case=out.name,**record)),flush=True)
        if gap is not None and gap<1e-7 and violation<1e-7 and value<=initial:raise StationaryPoint(x)
    try:
        if evaluate_only:result=OptimizeResult(x=np.zeros_like(at),success=False,nit=0,message='Saved feasible point re-evaluated; no convergence claim')
        else:result=minimize(objective,np.zeros_like(at),jac=True,method='SLSQP',constraints=constraints,callback=callback,options=dict(maxiter=iterations,ftol=1e-10))
    except StationaryPoint as stop:result=OptimizeResult(x=stop.x,success=True,nit=len(trace),message='Independent feasible first-order gate satisfied')
    q=expand(result.x);value,g,values=scene.objective(q,True)
    returned_value=value;selected_best=bool(best<value)
    if selected_best:q=best_q.copy();value,g,values=scene.objective(q,True)
    violation=float(max(np.max(lo-matrix@q),np.max(matrix@q-hi),np.max(lower-q),np.max(q-upper),0.))
    gap=feasible_improvement(q,g,matrix,lo,hi,lower,upper)
    phases=beta.copy()
    for i,c in enumerate(scene.free_bands):phases[c]+=q[model.n+scene.p+i]
    phases+=np.array([2*np.pi*motion.rf[c]/299792458.*model.reference_phase(q[:model.n],c) for c in range(5)])
    info=location_information(scene,q) if not fixed else None
    report=dict(complete=True,assignment=assignment,fixed_antenna_locations=fixed,initial_objective=initial,objective=value,radar_rss_fraction=sum(v['cost'] for v in values)/scene.energy,
        solver_success=bool(result.success),message=str(result.message),numerical_convergence=bool(result.success and violation<1e-7 and gap is not None and gap<1e-7),
        maximum_linearized_feasible_improvement=gap,maximum_kinematic_or_box_violation=violation,iterations=result.nit,seconds=time.monotonic()-started,
        parameters=q.tolist(),motion_parameters=model.nmotion,geometry_parameters=antennas.count,height_parameter=True,common_source_phases=scene.p,free_constant_band_phases=scene.free_bands,
        prior=prior.diagnostics(q[:model.n]),per_channel=[dict(cost=v['cost'],energy=v['energy'],explained_fraction=v['explained_fraction'],maximum_amplitude_kkt=v['maximum_amplitude_kkt']) for v in values],
        antenna_bound_hits=int(np.sum(abs(q[model.location_start:model.n])>3-1e-5)),motion_bound_hits=int(np.sum(np.isfinite(lower[:model.nmotion])&((q[:model.nmotion]-lower[:model.nmotion]<1e-5)|(upper[:model.nmotion]-q[:model.nmotion]<1e-5)))),
        physical_band_phases_rad=phases.tolist(),conditional_location_information=info,scope=__doc__,accepted_as_calibration=False,midpoint_position_or_velocity_fixed=False,source_support_fixed=True,
        initialized_from_fixed_location_control=seed is not None,selected_best_feasible_evaluation=selected_best,returned_solver_objective=returned_value,evaluation_only=evaluate_only,antenna_up_offsets_fixed=fixed_antenna_up,
        continuation=None if checkpoint is None else dict(previous_objective=checkpoint['objective'],previous_iterations=checkpoint['iterations'],previous_feasible_gap=checkpoint['maximum_linearized_feasible_improvement']))
    (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');model.export(q[:model.n],out/'motion.json',report)
    d=json.loads((out/'motion.json').read_text());gain=abs(motion.gains)*np.exp(1j*phases)
    d['calibration_gains_real_imag']=np.c_[gain.real,gain.imag].tolist();(out/'motion.json').write_text(json.dumps(d,indent=2)+'\n')
    for c,v in enumerate(values):np.save(out/f'brightness-{c}.npy',v['amplitude'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['motion','field','priors','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--assignments',nargs='+',default=['00002','02002','00000','00102','12112']);p.add_argument('--iterations',type=int,default=100);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--initial-control',type=Path,help='Continue a completed finite-iteration control before initializing all site hypotheses')
    p.add_argument('--initial-joint',type=Path,help='Polish one completed site case with a freshly evaluated invertible preconditioner')
    p.add_argument('--saved-progress',type=Path,help='Use the best saved feasible evaluation from that completed joint case')
    p.add_argument('--evaluate-only',action='store_true',help='Verify and export a saved case without changing its parameters')
    p.add_argument('--fixed-antenna-up',action='store_true',help='Sensitivity check: hold Tx/Rx local-up offsets at zero; retain horizontal and vessel-height freedom')
    a=p.parse_args();source=ContinuousShipMotion(a.motion);priors=json.loads(a.priors.read_text())
    allowed=[{'0','1'},{'0','2'},{'0','1'},{'0','1'},{'0','2'}]
    for assignment in a.assignments:
        if len(assignment)!=5 or any(c not in allowed[i] for i,c in enumerate(assignment)):raise ValueError('Assignment exceeds the user-selected primary/backup set')
    identity=dict(motion_sha256=fingerprint(a.motion),priors_sha256=fingerprint(a.priors),field=str(a.field),assignments=a.assignments,iterations=a.iterations)
    if a.fixed_antenna_up:identity['antenna_up_offsets_fixed']=True
    checkpoint=None
    if (a.saved_progress is not None or a.evaluate_only) and a.initial_joint is None:raise ValueError('A completed source case is required')
    if a.initial_joint is not None:
        checkpoint=json.loads(a.initial_joint.read_text())
        if a.assignments!=[checkpoint['assignment']] or checkpoint['fixed_antenna_locations'] or not (checkpoint['complete'] or checkpoint.get('snapshot_only',False)):
            raise ValueError('Select exactly one finished joint case or explicitly labelled interruption snapshot')
        identity['initial_joint_sha256']=fingerprint(a.initial_joint)
        if a.saved_progress is not None:
            saved=json.loads(a.saved_progress.read_text())
            if saved['assignment']!=checkpoint['assignment'] or len(saved['parameters'])!=len(checkpoint['parameters']):raise ValueError('Saved point belongs to another site case')
            checkpoint=dict(checkpoint,parameters=saved['parameters'],objective=saved['objective'] if 'objective' in saved else saved['history'][-1]['objective'])
            identity['saved_progress_sha256']=fingerprint(a.saved_progress)
        if a.evaluate_only:identity['evaluate_only']=True
    a.output.mkdir(parents=True,exist_ok=True);manifest=a.output/'experiment.json';report=dict(identity=identity,complete=False,cases=[],scope=__doc__)
    if manifest.exists():
        report=json.loads(manifest.read_text())
        if report['identity']!=identity:raise ValueError('Cannot resume different science inputs')
    manifest.write_text(json.dumps(report,indent=2)+'\n');fields,coordinates,_=load(a.field)
    control=None if a.initial_control is None else json.loads(a.initial_control.read_text())
    for assignment,fixed in [('00002',True),*[(x,False) for x in a.assignments]]:
        out=a.output/(('control-' if fixed else 'joint-')+assignment)
        if (out/'analysis.json').exists():result=json.loads((out/'analysis.json').read_text())
        else:
            print(json.dumps(dict(stage='starting-case',assignment=assignment,fixed_antenna_locations=fixed)),flush=True)
            result=fit_case(fields,coordinates,source,priors,assignment,out,a.iterations,a.workers,fixed=fixed,seed=control,checkpoint=None if fixed else checkpoint,evaluate_only=a.evaluate_only and not fixed,fixed_antenna_up=a.fixed_antenna_up)
        if fixed:control=result
        if not any(x['case']==out.name for x in report['cases']):report['cases'].append(dict(case=out.name,assignment=assignment,objective=result['objective'],radar_rss_fraction=result['radar_rss_fraction'],numerical_convergence=result['numerical_convergence'],antenna_bound_hits=result['antenna_bound_hits'],motion_bound_hits=result['motion_bound_hits']))
        manifest.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;manifest.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
