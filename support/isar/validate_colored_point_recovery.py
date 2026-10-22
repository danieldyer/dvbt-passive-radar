"""Blind nonlinear point recovery under a measured-covariance surrogate.

Covariance is learned once from the recorded mixed-scene residual, without
truth. Independent proper-complex Gaussian realizations then test estimator
efficiency under that declared covariance. This tests the inference algorithm;
it does not assert that the actual background follows a Gaussian model.
"""
import argparse,json
from pathlib import Path
import numpy as np

from correlated_likelihood import FourierMetric,MetricExperiment,crossvalidated_metric
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment,SCALE
from report_measured_scatterer_phases import serial
from validate_recorded_injection import metrics as compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/injection_20261018/constant-velocity'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trials',type=int,default=6)
    parser.add_argument('--seed',type=int,default=1901)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=json.loads((args.source/'analysis.json').read_text())
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    state[2]=np.zeros(3);train=[f.time<source['training_end_s'] for f in fields]
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    with np.load(args.source/'observations.npz') as saved:
        observed=[saved[f'data_{c}'] for c in range(5)]
    coarse=np.asarray(source['fits']['constant_velocity']['parameters'])
    _,_,_,prediction=ex.profile(coarse,observed,derivative=False)
    metric=[crossvalidated_metric((y-p).reshape(f.y.shape)[keep],np.median(np.diff(f.time)))
        for y,p,f,keep in zip(observed,prediction,fields,train)]
    gls=MetricExperiment(ex,metric)
    # Independent Cartesian/quadrature source. Truth is inaccessible to acquisition/fit.
    truth=np.asarray(source['truth_parameters'])
    original=json.loads(Path('docs/data/injection_20261018/separated-500m/analytic-profile-fit/analysis.json').read_text())
    alpha=np.asarray(original['gains_real_imag']);alpha=alpha[:,0]+1j*alpha[:,1]
    alpha*=10**(source.get('additional_point_power_db',0.)/20)
    clean=[ex.independent_signal(truth,c,[a]) for c,a in enumerate(alpha)]
    _,jac,_,_=gls.profile(truth,clean);jac=jac[:,:4]
    covariance=.5*np.linalg.inv(jac.T@jac)
    physical_cov=covariance*SCALE[:4,None]*SCALE[None,:4]
    result=dict(scope=__doc__,seed=args.seed,trials=[],complete=False,
        physical_parameter_crlb=physical_cov,covariance=[m.description for m in metric])
    np.savez(out/'frozen-covariance.npz',**{f'{name}_{c}':a for c,m in enumerate(metric)
        for name,a in [('labels',m.labels),('covariance',m.covariance)]})
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save();rng=np.random.default_rng(args.seed)
    roots=[]
    for m in metric:
        e,v=np.linalg.eigh(m.covariance);roots.append((v*np.sqrt(e[:,None,:]))@v.conj().transpose(0,2,1))
    for trial in range(args.trials):
        data=[]
        for c,(f,keep,m,root) in enumerate(zip(fields,train,metric,roots)):
            z=(rng.normal(size=(keep.sum(),len(f.offsets)))+1j*rng.normal(size=(keep.sum(),len(f.offsets))))/np.sqrt(2)
            noise=np.fft.ifft(np.einsum('nrs,ns->nr',root[m.labels],z),axis=0,norm='ortho')
            y=clean[c].reshape(f.y.shape).copy();y[keep]+=noise
            y[~keep]+=(rng.normal(size=y[~keep].shape)+1j*rng.normal(size=y[~keep].shape))/np.sqrt(2)
            data.append(y.ravel())
        proposal=ex.acquire(data,np.zeros(6),count=1,constant_velocity=True)[0]
        fit,_=gls.fit(data,np.asarray(proposal['parameters']),100,free=[0,1,2,3],position_halfwidth_m=1000.)
        truth_residual,_,_,_=gls.profile(truth,data,derivative=False)
        error=(np.asarray(fit['parameters'])[:4]-truth[:4])*SCALE[:4]
        predictions=[];true=[]
        for c,f in enumerate(fields):
            a=np.asarray(fit['coefficients_real_imag'][c])[0];a=complex(*a)
            keep=np.repeat(~train[c],len(f.offsets))
            predictions.append(ex.templates(np.asarray(fit['parameters']),c)[:,0][keep]*a);true.append(clean[c][keep])
        record=dict(trial=trial,acquisition=proposal,fit=fit,physical_error=error,
            horizontal_error_m=np.linalg.norm(error[:2]),velocity_error_mps=np.linalg.norm(error[2:]),
            mahalanobis_squared=float(error@np.linalg.solve(physical_cov,error)),
            twice_profile_log_likelihood_ratio=2*(truth_residual@truth_residual-fit['training_rss']),
            excluded_five_band=compare(np.concatenate(predictions),np.concatenate(true)))
        result['trials'].append(record);save()
        print('Trial',trial,'error',serial(error),'Mahalanobis squared',record['mahalanobis_squared'],flush=True)
    result['complete']=True;save()


if __name__=='__main__':main()
