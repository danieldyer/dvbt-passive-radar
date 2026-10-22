"""Conditional information cost of locally estimated stationary clutter.

These diagnostics keep the fitted echo strengths, active amplitude set and
forward operator fixed. They compare known clutter with jointly estimated
clutter; known clutter is an upper bound, not a claim about recorded data.
"""
import numpy as np


def remove_columns(values,columns):
    if not columns.shape[1]:return values.copy()
    norms=np.linalg.norm(columns,axis=0);keep=norms>0
    if not np.any(keep):return values.copy()
    scaled=columns[:,keep]/norms[keep]
    u,s,_=np.linalg.svd(scaled,full_matrices=False)
    q=u[:,s>np.finfo(float).eps*max(scaled.shape)*s[0]]
    return values-q@(q.T@values)


def efficient_mean(mean,brightness,stationary=None):
    """Real mean derivative after complex clutter and real brightness nuisance."""
    if stationary is not None:
        mean=stationary.remove(mean);brightness=stationary.remove(brightness)
    real=np.r_[mean.real,mean.imag]
    return remove_columns(real,np.r_[brightness.real,brightness.imag])


def pair_mean(pair,parameters,*,estimate_stationary):
    value=pair.evaluate(parameters);b=pair.base;k=len(b.knots)
    responses=pair.response_blocks(parameters)
    strength=value['amplitude'].reshape(2,k)
    mean=np.concatenate([b.reduced(d*(b.amplitude_basis@a)[:,None,None])[b.selected]
                         for (_,d),a in zip(responses,strength)],axis=1)
    brightness=value['design'][b.selected][:,value['amplitude']>0]
    return efficient_mean(mean,brightness,b.nuisance if estimate_stationary else None)


def radial_coordinates(band,parameters):
    """Midpoint delay (m), phase (rad), path rate (m/s), curvature (m/s²).

    Position, velocity and acceleration change along the instantaneous excess
    path gradient. PhysicalAmplitudeArc already references midpoint phase, so
    position changes preserve that explicit phase coordinate. The exact
    response derivative carries the remaining geometric terms.
    """
    g=band.geometry;position,_,_,tx,rx=g.state(parameters[:g.count]);c=band.channel
    first=position-tx[c];second=position-rx
    gradient=first/np.linalg.norm(first)+second/np.linalg.norm(second)
    direction=gradient/(gradient@gradient)
    change=np.zeros((g.count+5,4))
    change[18:21,0]=direction/g.nuisance_sigma[:3]
    change[g.count+c,1]=1.
    change[21:24,2]=direction/g.nuisance_sigma[3:6]
    change[24:27,3]=direction/g.nuisance_sigma[6:9]
    return change


def pair_information(pair,parameters,*,estimate_stationary):
    j=pair_mean(pair,parameters,estimate_stationary=estimate_stationary)
    count=pair.primary_count
    primary=remove_columns(j[:,:count],j[:,count:])@radial_coordinates(pair.base,parameters)
    return 2*primary.T@primary
