"""Full complex range/slow-time likelihood for offline point-target calibration."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_triangular

from complex_likelihood import gate, projected_profile
from fit_geometry import observations
from geographic_io import read_profiles
from phase_history import C, FS
from phase_fit import load_histories
from polar_fit import Problem, joint


class Pulse:
    def __init__(self, autocorrelation):
        self.samples=np.asarray(autocorrelation)
        grid=np.arange(-32,32+1/64,1/32)
        lags=np.arange(-64,65)
        delta=grid[:,None]-lags
        window=np.i0(8*np.sqrt(np.maximum(0,1-(delta/32)**2)))/np.i0(8)
        weights=np.sinc(delta)*window*(abs(delta)<=32)
        weights/=weights.sum(axis=1)[:,None]
        self.curve=CubicSpline(grid,weights@self.samples,extrapolate=False)

    def __call__(self, lag, derivative=0):
        if np.any(abs(lag)>32):
            raise ValueError('Candidate moved outside retained template range support')
        return self.curve(lag,derivative)


class Field:
    def __init__(self, data, pulse, seed, bandwidth_hz=1., selected=None):
        self.data=data;self.seed=seed;self.pulse=pulse
        self.time=data['time'];self.offsets=data['offset_samples']
        self.k=2*np.pi*seed['rf_hz']/C
        self.selected=np.ones(len(self.time),bool) if selected is None else np.asarray(selected,bool)
        self.seed_path=np.polynomial.polynomial.polyval((self.time-seed['epoch'])/seed.get('scale',15),seed['path'])
        variance=data['variance'][:,len(self.offsets)//2]
        self.q,_=gate(self.time,variance,bandwidth_hz,seed['epoch'],seed.get('duration_s',30.),selected)
        difference=self.offsets[:,None]-self.offsets[None,:]
        covariance=pulse.samples[difference+64]/pulse.samples[64].real
        covariance=(covariance+covariance.conj().T)/2
        self.range_eigenvalues=np.linalg.eigvalsh(covariance)
        self.whiten=solve_triangular(np.linalg.cholesky(covariance),np.eye(len(self.offsets)),lower=True)
        self.y=self.apply(data['value']).ravel()

    def windowed(self,start,end):
        keep=(self.time>=start)&(self.time<end)
        result=copy.copy(self)
        result.time=self.time[keep];result.seed_path=self.seed_path[keep]
        result.selected=self.selected[keep]
        result.data={**self.data,**{key:self.data[key][keep] for key in ('time','value','variance')}}
        result.q,_=gate(result.time,result.data['variance'][:,len(self.offsets)//2],
                        1.,(start+end)/2,end-start,result.selected)
        result.y=result.apply(result.data['value']).ravel()
        return result

    def apply(self, values):
        return (self.q@values)@self.whiten.T

    def template(self, path, jacobian=None,range_bias=0.,bias_derivative=False):
        lag=self.offsets[None,:]-(path+range_bias-self.seed_path)[:,None]*FS/C
        pulse=self.pulse(lag)
        phase=np.exp(-1j*self.k*(path-(self.seed_path-self.seed['path'][0])))
        u=self.apply(phase[:,None]*pulse).ravel()
        if jacobian is None:
            return u,None
        derivative=phase[:,None]*(-1j*self.k*pulse-FS/C*self.pulse(lag,1))
        n=jacobian.shape[1];m=len(self.offsets)
        raw=derivative[:,:,None]*jacobian[:,None,:]
        if bias_derivative:
            raw=np.concatenate([raw,(-FS/C*phase[:,None]*self.pulse(lag,1))[:,:,None]],axis=2)
            n+=1
        du=self.q@raw.reshape(len(path),m*n)
        du=np.einsum('ab,fbp->fap',self.whiten,du.reshape(len(self.q),m,n)).reshape(-1,n)
        return u,du

    def evaluate(self,path,jacobian=None):
        u,du=self.template(path,jacobian)
        return projected_profile(self.y,u,du)


class RangeProblem(Problem):
    def __init__(self, channels, coarse, sites, fields, **kwargs):
        super().__init__(channels,coarse,sites,**kwargs)
        self.fields=fields
        self.range_origins=np.zeros(len(fields))

    def channel_data(self,channel):
        return self.fields[channel].y

    def channel_model(self,p,channel,free_sites=False,jacobian=False,range_bias=0.,bias_derivative=False):
        field=self.fields[channel]
        path,j=self.model.paths(p,field.time,np.full(len(field.time),channel),self.sites,free_sites)
        return field.template(path,j if jacobian else None,range_bias+self.range_origins[channel],bias_derivative)

    def auxiliary(self,p,free_sites=False,jacobian=False,columns=None,with_motion=True,with_coarse=False):
        return super().auxiliary(p,free_sites,jacobian,columns,with_motion,False)


def load_fields(patches,label,description,records):
    fields=[]
    for ch,seed in enumerate(description['channels']):
        with np.load(patches/f'{label}-{ch}.npz') as source:
            data={k:source[k] for k in ('time','value','variance','offset_samples')}
        keep=(records['channel']==ch)&(records['time']>=description['start'])&(records['time']<=description['end'])&(records['information']>0)
        if np.count_nonzero(keep)<100 or records[keep]['time'][-1]<description['end']-.1:
            raise ValueError('Template export does not cover the candidate interval')
        psf=np.average(records[keep]['psf'],axis=0,weights=records[keep]['information'])
        fields.append(Field(data,Pulse(psf),seed))
    return fields


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories',type=Path,required=True)
    parser.add_argument('--patches',type=Path,required=True)
    parser.add_argument('--templates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--labels',nargs='+',default=['A','B','C','E'])
    parser.add_argument('--reuse',action='store_true')
    parser.add_argument('--iterations',type=int,default=350)
    parser.add_argument('--name',default='joint-range')
    args=parser.parse_args()
    _,records=read_profiles(args.templates,True)
    sites=np.array(json.loads((args.output/'geometry-prior.json').read_text())['enu_transmitters_m'])
    problems,starts=[],[]
    for label,coarse in zip(args.labels,observations(args.histories,args.labels)):
        channels,description=load_histories(args.histories,label,1.)
        fields=load_fields(args.patches,label,description,records)
        problem=RangeProblem(channels,coarse,sites,fields)
        problems.append(problem)
        path=args.output/f'{label}-range.json'
        if args.reuse and path.exists():
            starts.append(np.array(json.loads(path.read_text())['fixed_state']));continue
        start=np.array(json.loads((args.output/f'{label}-polar.json').read_text())['fixed_state'])
        fixed=problem.fit(start,max_nfev=args.iterations)
        free=problem.fit(fixed.x,True,max_nfev=args.iterations)
        starts.append(fixed.x)
        result={'label':label,'fixed_state':fixed.x.tolist(),'free_state':free.x.tolist(),
            'fixed_cost':fixed.cost,'free_cost':free.cost,'fixed_converged':bool(fixed.success),
            'free_converged':bool(free.success),'fixed':problem.describe(fixed.x),'free':problem.describe(free.x,True),
            'range_covariance_eigenvalues':[f.range_eigenvalues.tolist() for f in fields],
            'transmitter_corrections_enu_m':(free.x[-9:].reshape(3,3)*[30,30,75]).tolist(),
            'template_model':'actual reconstructed waveform autocorrelation averaged over candidate interval',
            'status':'conditional range/phase fit; no metadata update'}
        path.write_text(json.dumps(result,indent=2))
        print(label,'cost fixed/free',fixed.cost,free.cost,'converged',fixed.success,free.success,
              'range/rate',result['fixed']['range_rms_m'],result['fixed']['path_rate_rms_mps'],flush=True)
    if len(problems)>1:
        fit=joint(problems,starts,max_nfev=args.iterations)
        result={'labels':args.labels,'state':fit.x.tolist(),'cost':fit.cost,'converged':bool(fit.success),
                'transmitter_corrections_enu_m':(fit.x[:9].reshape(3,3)*[30,30,75]).tolist(),
                'status':'conditional range/phase fit; no metadata update'}
        (args.output/(args.name+'.json')).write_text(json.dumps(result,indent=2))
        print('joint',fit.cost,fit.success,result['transmitter_corrections_enu_m'],flush=True)


if __name__=='__main__':
    main()
