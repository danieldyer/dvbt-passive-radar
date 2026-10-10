#!/usr/bin/env python3
"""Source-divided noise differences retain coherent narrowband interference."""
import argparse
import json
from pathlib import Path

import numpy as np
from sensitivity import K, NFFT, blocks, profiles, map_at


def run(path, output):
    original=[];weighted=[];noise=[];statistics=[]
    for b in blocks(path):
        sl=slice(b['first'],b['last']);good=np.asarray(b['valid'][sl],bool).all(axis=1)
        x=np.asarray(b['s'][sl][good],complex)*np.exp(-2j*np.pi*K*b['reference_origin']/NFFT)
        y=np.asarray(b['y'][sl][good],complex);h=np.asarray(b['h'],complex)
        p=abs(x)**2;r=y/x
        modes,_=np.linalg.qr(np.column_stack((h,1j*K/NFFT*h)))
        # Noise estimate only: keep the measurement and its phase untouched.
        innovation=r-(r@modes.conj())@modes.T
        delta=np.diff(innovation,axis=0)
        d=np.sum(abs(delta)**2,axis=0)/np.sum(1/p[1:]+1/p[:-1],axis=0)
        noise.append(d)
        error=y-h*x
        pilot=np.where(abs(p-16/9)<1e-5,9/16,1)
        gain=np.sum(abs(h)**2*p*pilot/d)/len(x)/NFFT
        baseline_gain=np.sum(h*p)/len(x)/NFFT
        original.append(profiles(error*x.conj())/baseline_gain)
        weighted.append(profiles(error*x.conj()*h.conj()/d*pilot)/gain)
        statistics.append(dict(batch=b['index'],strongest_carrier=int(np.argmax(d)),
                               strongest_to_median_noise=float(np.max(d)/np.median(d))))
    old=np.concatenate(original);new=np.concatenate(weighted);results=[]
    for seconds in (1,3,10):
        before,f=map_at(old,seconds,remove_dc=True);after,_=map_at(new,seconds,remove_dc=True)
        keep=(abs(f)>5)&(abs(f)<=200)
        p0=np.median(abs(before[keep,100:780])**2,axis=1)/np.log(2)
        p1=np.median(abs(after[keep,100:780])**2,axis=1)/np.log(2)
        results.append(dict(cpi_s=seconds,
            background_reduction_db=float(10*np.log10(np.median(p0)/np.median(p1))),
            original_row_db_quantiles=np.quantile(10*np.log10(p0/np.median(p0)),[0,.05,.5,.95,1]).tolist(),
            weighted_row_db_quantiles=np.quantile(10*np.log10(p1/np.median(p1)),[0,.05,.5,.95,1]).tolist()))
    np.savez(output.with_suffix('.npz'),original=old,weighted=new,noise_variance=np.mean(noise,axis=0))
    result=dict(source=str(path),noise_estimate='source-divided first difference, two common modes removed only for variance estimation',
                covariance='diagonal proxy; current-batch weights for diagnosis only',statistics=statistics,results=results)
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.input,a.output)
