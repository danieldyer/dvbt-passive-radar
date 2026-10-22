"""Blinded constant-velocity recovery and controlled acceleration mismatch.

The zero-acceleration model is physical Cartesian ENU motion, not a zero
correction to an accelerating AIS prior. Only the coarse position and velocity
are supplied to acquisition. Truth is used by independent synthesis/scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from correlated_likelihood import crossvalidated_metric, MetricExperiment
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment, SCALE
from report_measured_scatterer_phases import serial
from validate_recorded_injection import metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018/separated-500m'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--acceleration-en',type=float,nargs=2,default=[0.,0.])
    parser.add_argument('--training-seconds',type=float,default=19.5)
    parser.add_argument('--velocity-halfwidth-mps',type=float,default=1.)
    parser.add_argument('--gain-db',type=float,default=0.)
    parser.add_argument('--profiles',type=Path,default=Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    parser.add_argument('--level',choices=['analytic','linear','iq'],default='analytic')
    args=parser.parse_args();root=args.root;out=args.output
    out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((root/'analytic-profile-fit/analysis.json').read_text())
    fields,rotation,rx,tx,state,mid=load_fields(args.profiles)
    # Both estimators use this declared CV coarse prior; neither reads true motion.
    state[2]=np.zeros(3)
    split=142.+args.training_seconds
    train=[f.time<split for f in fields]
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    truth=np.asarray(prior['truth_parameters']).copy();truth[4:]=np.asarray(args.acceleration_en)/SCALE[4:]
    gains=np.asarray(prior['gains_real_imag']);gains=gains[:,0]+1j*gains[:,1]
    gains*=10**(args.gain_db/20)
    clean=[ex.independent_signal(truth,c,[g]) for c,g in enumerate(gains)]
    background=[f.y.ravel() for f in fields]
    data=[b+s for b,s in zip(background,clean)] if args.level=='analytic' else background
    result=dict(scope=__doc__,midpoint_s=mid,training_end_s=split,
        injected_physical_acceleration_en_mps2=args.acceleration_en,
        additional_point_power_db=args.gain_db,
        injection_level=args.level,source_profiles=str(args.profiles),
        truth_parameters=truth,complete=False,acquisition=[],fits={})
    def save(): (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save()
    proposals=ex.acquire(data,np.zeros(6),count=3,constant_velocity=True)
    result['acquisition']=proposals;save()
    print('Acquired CV candidates',serial(proposals),flush=True)
    trials=[]
    for proposal in proposals:
        fit,_=ex.fit(data,np.asarray(proposal['parameters']),120,free=[0,1,2,3],position_halfwidth_m=1000.,velocity_halfwidth_mps=args.velocity_halfwidth_mps)
        trials.append(fit)
        result['cv_trials']=trials;save()
        print('CV trial',serial(fit),flush=True)
    result['cv_trials']=trials
    cv=min(trials,key=lambda f:f['training_rss'])
    result['fits']['constant_velocity']=cv;save()
    print('CV parameters',serial(cv['parameters']),flush=True)
    free,_=ex.fit(data,np.asarray(cv['parameters']),120,position_halfwidth_m=1000.,velocity_halfwidth_mps=args.velocity_halfwidth_mps)
    result['fits']['free_acceleration']=free;save()
    # Feasible GLS covariance and its spectral resolution use observed residuals.
    _,_,_,prediction=ex.profile(np.asarray(cv['parameters']),data,derivative=False)
    metric=[]
    for f,mask,y,p in zip(fields,train,data,prediction):
        residual=(y-p).reshape(f.y.shape)[mask]
        metric.append(crossvalidated_metric(residual,np.median(np.diff(f.time))))
    gls=MetricExperiment(ex,metric)
    fit,_=gls.fit(data,np.asarray(cv['parameters']),120,free=[0,1,2,3],position_halfwidth_m=1000.,velocity_halfwidth_mps=args.velocity_halfwidth_mps)
    result['fits']['constant_velocity_gls']=fit
    result['covariance']=[m.description for m in metric]
    result['scores']={}
    for name,fit in result['fits'].items():
        p=np.asarray(fit['parameters']);err=(p-truth)*SCALE
        channels=[]
        for c,f in enumerate(fields):
            alpha=np.asarray(fit['coefficients_real_imag'][c])[0];alpha=complex(*alpha)
            predicted=ex.templates(p,c)[:,0]*alpha
            keep=np.repeat(~train[c],len(f.offsets))
            channels.append(dict(full=metrics(predicted,clean[c]),excluded=metrics(predicted[keep],clean[c][keep])))
        time=np.linspace(142.,172.,301)-mid
        error=err[:2]+time[:,None]*err[2:4]+.5*time[:,None]**2*err[4:6]
        result['scores'][name]=dict(physical_error=err,horizontal_error_at_epoch_m=np.linalg.norm(err[:2]),
            velocity_error_at_epoch_mps=np.linalg.norm(err[2:4]),trajectory_error_max_m=np.linalg.norm(error,axis=1).max(),
            channels=channels)
    result['complete']=True;save()
    np.savez(out/'observations.npz',**{f'{name}_{c}':value for c in range(5)
        for name,value in [('data',data[c]),('injected',clean[c])]})
    print(json.dumps(serial(result['scores'])),flush=True)


if __name__=='__main__':main()
