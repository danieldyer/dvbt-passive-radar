"""Conditional tone likelihood after one constant complex profile nuisance.

Each range cell uses its known diagonal variance. The same weighted projection
acts on data and candidate tones, and the denominator is retained information.
There is no Doppler notch, taper or fitted noise normalization.
"""
import numpy as np


def projected_periodogram(time, values, information, wavenumber, rates, variance_scale, *, acceleration=0., epoch_s=None):
    t,y,w,r=map(np.asarray,(time,values,information,rates))
    if (t.ndim!=1 or y.ndim!=2 or y.shape[0]!=len(t) or w.shape!=t.shape or
        np.any(np.diff(t)<=0) or np.any(w<0) or not np.isfinite(w).all() or
        wavenumber<=0 or variance_scale<=0 or not np.isfinite(variance_scale)):
        raise ValueError('Matching ordered observations and positive known variance required')
    weight=8192*w/variance_scale;total=weight.sum()
    if total<=0:raise ValueError('No observed information')
    dt=t-(t.mean() if epoch_s is None else epoch_s)
    h=np.exp(-1j*wavenumber*(dt[:,None]*r[None,:]+.5*acceleration*dt[:,None]**2))
    mean=weight@h/total
    projected=h-mean
    retained=np.sum(weight[:,None]*abs(projected)**2,axis=0)
    clean=y-(weight@y/total)[None,:]
    cross=(projected.conj()*weight[:,None]).T@clean
    score=np.divide(abs(cross)**2,retained[:,None],out=np.zeros_like(abs(cross)**2),
        where=retained[:,None]>1e-12*total)
    return score,retained/total
