"""Check nonlinear likelihood and observable-coordinate errors in the trials.

A narrow curved likelihood ridge need not be a Cartesian Gaussian ellipse.
Both local CRLB coordinates and exact range/rate coordinates are reported;
neither a local approximation nor a favourable image alone certifies recovery.
"""
import argparse,json
from pathlib import Path
import numpy as np

from correlated_likelihood import FourierMetric,MetricExperiment
from recorded_injection_fields import load_fields
from recovery_experiment import Experiment,SCALE
from report_measured_scatterer_phases import serial


def observables(ex,p):
    position=ex.state[0].copy();position[:2]+=p[:2]*SCALE[:2]
    velocity=ex.state[1].copy();velocity[:2]+=p[2:4]*SCALE[2:4]
    x=ex.receiver+position@ex.rotation;v=velocity@ex.rotation
    value=[];derivative=[]
    for c in [2,4]:
        path=-np.linalg.norm(ex.transmitters[c]-ex.receiver);g=np.zeros(3);h=np.zeros((3,3))
        for antenna in [ex.transmitters[c],ex.receiver]:
            w=x-antenna;r=np.linalg.norm(w);u=w/r
            path+=r;g+=u;h+=(np.eye(3)-u[:,None]*u[None,:])/r
        g=g@ex.rotation.T;hv=(h@v)@ex.rotation.T
        value.extend([path,np.dot(g,velocity)])
        derivative.extend([np.r_[g[:2],0.,0.],np.r_[hv[:2],g[:2]]])
    return np.asarray(value),np.asarray(derivative)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();root=args.directory;x=json.loads((root/'analysis.json').read_text())
    source=Path('docs/data/injection_20261018/constant-velocity');original=json.loads((source/'analysis.json').read_text())
    fields,rotation,rx,tx,state,mid=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    state[2]=np.zeros(3);train=[f.time<original['training_end_s'] for f in fields]
    ex=Experiment(fields,rotation,rx,tx,state,mid,training=train)
    with np.load(root/'frozen-covariance.npz') as saved:
        metric=[FourierMetric(saved[f'labels_{c}'],saved[f'covariance_{c}']) for c in range(5)]
    gls=MetricExperiment(ex,metric);truth=np.asarray(original['truth_parameters'])
    with np.load(source/'observations.npz') as saved:clean=[saved[f'injected_{c}'] for c in range(5)]
    mu,g=observables(ex,truth);covariance=g@np.asarray(x['physical_parameter_crlb'])@g.T
    roots=[]
    for m in metric:
        e,v=np.linalg.eigh(m.covariance);roots.append((v*np.sqrt(e[:,None,:]))@v.conj().transpose(0,2,1))
    result=dict(scope=__doc__,source_complete=x['complete'],records=[])
    rng=np.random.default_rng(x['seed'])
    for trial in x['trials']:
        data=[];noise_energy=0.
        for c,(f,keep,m,covariance_root) in enumerate(zip(fields,train,metric,roots)):
            z=(rng.normal(size=(keep.sum(),len(f.offsets)))+1j*rng.normal(size=(keep.sum(),len(f.offsets))))/np.sqrt(2)
            noise=np.fft.ifft(np.einsum('nrs,ns->nr',covariance_root[m.labels],z),axis=0,norm='ortho')
            noise_energy+=np.vdot(m.apply(noise.ravel()),m.apply(noise.ravel())).real
            y=clean[c].reshape(f.y.shape).copy();y[keep]+=noise
            y[~keep]+=(rng.normal(size=y[~keep].shape)+1j*rng.normal(size=y[~keep].shape))/np.sqrt(2)
            data.append(y.ravel())
        r,_,_,_=gls.profile(truth,data,derivative=False)
        fitted=np.asarray(trial['fit']['parameters']);delta=observables(ex,fitted)[0]-mu
        q=delta@np.linalg.solve(covariance,delta)
        result['records'].append(dict(trial=trial['trial'],profile_likelihood_ratio=2*(r@r-trial['fit']['training_rss']),
            observable_mahalanobis_squared=q,cartesian_mahalanobis_squared=trial['mahalanobis_squared'],
            observable_delta=delta,whitened_noise_energy=noise_energy))
    (root/'diagnostics.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    print(json.dumps(serial(result)),flush=True)


if __name__=='__main__':main()
