"""Exact phase-bank scores after collapsing the fixed range response.

For each trial h[t,r] exp(-i k (v t+a t²/2)), range contractions are computed
once. Stationary-nuisance projection remains exact for every chirp. This is
algebraically the full matched-chirp bank, with no trial-by-range allocation.
"""
import numpy as np


def scores(template,data,nuisance,time,k,rate,acceleration,*,batch=512):
    h,y,q=np.asarray(template),np.asarray(data),np.asarray(nuisance)
    time,rate,acceleration=map(np.asarray,(time,rate,acceleration))
    if h.ndim!=2 or y.shape!=h.shape or q.shape[0]!=h.size or time.shape!=(len(h),) or rate.shape!=acceleration.shape:
        raise ValueError('Need matching time/range data, template, nuisance and chirp coordinates')
    matched=np.sum(h.conj()*y,axis=1)
    projected=np.einsum('trk,tr->tk',q.reshape(*h.shape,-1).conj(),h)
    norm=np.vdot(h,h).real
    result=np.empty(rate.size)
    for first in range(0,len(rate),batch):
        r,a=rate[first:first+batch],acceleration[first:first+batch]
        phase=np.exp(-1j*k*(time[:,None]*r+.5*time[:,None]**2*a))
        denominator=norm-np.sum(abs(projected.T@phase)**2,axis=0)
        result[first:first+len(r)]=abs(phase.conj().T@matched)**2/np.maximum(denominator,1e-100)
    return result
