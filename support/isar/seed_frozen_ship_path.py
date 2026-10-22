"""Acquire a shared residual path before exact physical waveform refinement."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize,LinearConstraint
from full_scene_context import open_scene,prediction
from frozen_carrier_path import FrozenCarrier
from full_scene_information import coordinates
from path_dynamics import constraint,certificate
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=300);p.add_argument('--max-correction-m',type=float,default=3.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,motion,bands,saved=open_scene(a.motion)
    models=[prediction(b,saved,fit,c) for c,b in enumerate(bands)]
    kernel=FrozenCarrier([b.point for b in bands],models);n=bands[0].point.b.shape[1]
    initial=np.zeros(n);fisher=kernel.information();matrix,diagnostic=coordinates(fisher,np.eye(n),kernel.degrees)
    bound=constraint(fit['path_knots_s'],fit['path_coefficients_m'],n,fit['max_acceleration_mps2'],a.max_correction_m,fit['anchor_path_time_s'])
    transformed=LinearConstraint(bound.A@matrix,bound.lb,bound.ub);scale=kernel.energy/kernel.degrees;trace=[]
    def objective(q):
        value,g=kernel.objective(matrix@q);return (value-1)*scale,(g@matrix)*scale
    def callback(q):
        value,g=kernel.objective(matrix@q);trace.append(dict(iteration=len(trace)+1,objective=value,seconds=time.monotonic()-started))
        if len(trace)%10==0:print(json.dumps(trace[-1]),flush=True)
    result=minimize(objective,initial,jac=True,method='SLSQP',constraints=[transformed],callback=callback,
        options=dict(maxiter=a.iterations,ftol=1e-11))
    delta=matrix@result.x;value,g=kernel.objective(delta);stationarity=certificate(delta,g*scale,bound)
    physical=[]
    for c,band in enumerate(bands):
        b=band.point;model=prediction(band,saved,fit,c,delta);record=dict(channel=c)
        for label,mask in [('training',b.training),('excluded',~b.training)]:
            residual=profile_static(model,b.field.root_information,mask,b.groups)*mask[:,None]-b.data[label=='training']
            record[label+'_rss']=float(np.vdot(residual,residual).real);record[label+'_energy']=b.energy[label=='training']
            residual=profile_static(models[c],b.field.root_information,mask,b.groups)*mask[:,None]-b.data[label=='training']
            record[label+'_rss_before']=float(np.vdot(residual,residual).real)
        physical.append(record)
    origin=BSpline(fit['path_knots_s'],np.eye(n+1),3)(fit['anchor_path_time_s'])
    change=np.r_[delta,0.];coeff=np.asarray(fit['path_coefficients_m'])+change-origin@change
    output=dict(fit);output.pop('parameters',None)
    output.update(scope=__doc__,path_coefficients_m=coeff,initializer_only=True,source_frozen_carrier_motion=str(a.motion),
        numerical_convergence=False,accepted_as_vessel_motion=False,frozen_carrier_path_correction=delta,
        frozen_carrier_fit=dict(objective=value,initial_objective=kernel.objective(initial)[0],message=str(result.message),iterations=result.nit,
            numerical_convergence=bool(result.success and stationarity['maximum_kkt_residual']<1e-6),trace=trace,**stationarity),
        exact_physical_fixed_brightness_validation=physical,optimizer_information=diagnostic,seconds=time.monotonic()-started)
    output['exact_training_cost_improved']=bool(sum(r['training_rss'] for r in physical)<=sum(r['training_rss_before'] for r in physical))
    for c,band in enumerate(bands):
        b=band.point;saved[f'path_m_{c}']=b.base_path+b.b@delta;saved[f'rate_mps_{c}']=b.base_rate+b.bd@delta
    np.savez_compressed(a.output/'trajectory.npz',**saved)
    (a.output/'analysis.json').write_text(json.dumps(serial(output),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(15,5),layout='constrained')
    b=bands[3].point;ax.plot(b.t,b.b@delta);ax.set(xlabel='Capture time (s)',ylabel='One shared path correction (m)',
        title='Frozen-carrier initializer only; exact physical validation recorded separately; original observations unchanged')
    fig.savefig(a.output/'shared-residual-path.png',dpi=145);plt.close(fig)
    print(json.dumps(serial(dict(frozen_carrier_fit=output['frozen_carrier_fit'],exact_physical_fixed_brightness_validation=physical,seconds=output['seconds']))),flush=True)


if __name__=='__main__':main()
