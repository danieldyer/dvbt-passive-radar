"""Coherent Gaussian GLRT with unknown positive per-band brightness.

Inputs are noise-whitened, phase-calibrated complex matched amplitudes z_c.
Under H1, E[z_c]=a_c exp(i phi), a_c>=0. Profiling the amplitudes gives
max_phi sum_c max(Re(exp(-i phi) z_c),0)^2. A single unknown reflector phase
is shared by every band; there are no freely fitted band phases here.

The 2C sign boundaries partition the circle. Inside each interval the score
is a 2x2 quadratic form. Checking its stationary directions and all interval
boundaries yields the global maximum without a discretised phase search.
"""
import numpy as np


def fixed_spectrum_statistic(scores,information,gain):
    """Matched likelihood for a declared complex band spectrum, after whitening.

    Each band supplies h_c^H P_c y_c and h_c^H P_c h_c. The common target
    coefficient is unknown, but its relative spectrum gain_c is fixed.
    This is the matched filter for the concatenated physical templates;
    no per-pixel choice of participating bands or fitted image scale occurs.
    """
    z,i,g=np.asarray(scores,complex),np.asarray(information,float),np.asarray(gain,complex)
    if (z.ndim<1 or i.shape!=z.shape or g.shape!=(len(z),) or
            not np.isfinite(z).all() or not np.isfinite(i).all() or not np.isfinite(g).all() or np.any(i<=0)):
        raise ValueError('Finite band scores, positive information and one declared gain per band are required')
    g=g.reshape((len(g),)+(1,)*(z.ndim-1))
    numerator=np.sum(g.conj()*z,axis=0);denominator=np.sum(abs(g)**2*i,axis=0)
    if np.any(denominator<=0):raise ValueError('The declared band spectrum has no information')
    return abs(numerator)**2/denominator,numerator/denominator


def statistic(values):
    z=np.asarray(values,dtype=complex)
    if z.ndim<1 or not len(z) or not np.isfinite(z).all():raise ValueError('Need finite band amplitudes')
    shape=z.shape[1:];z=z.reshape(len(z),-1);angles=np.mod(np.angle(z)+np.pi/2,2*np.pi)
    boundary=np.sort(np.r_[angles,np.mod(angles+np.pi,2*np.pi)],axis=0)
    ends=np.r_[boundary[1:],boundary[:1]+2*np.pi];best=np.zeros(z.shape[1]);phase=np.zeros(z.shape[1])
    def update(theta,selected=None):
        nonlocal best,phase
        projection=z.real*np.cos(theta)+z.imag*np.sin(theta)
        score=np.sum(np.maximum(projection,0.)**2,axis=0)
        better=score>best
        if selected is not None:better&=selected
        best=np.where(better,score,best);phase=np.where(better,theta,phase)
    for first,last in zip(boundary,ends):
        mid=(first+last)/2;active=(z.real*np.cos(mid)+z.imag*np.sin(mid))>0
        xx=np.sum(active*z.real**2,axis=0);yy=np.sum(active*z.imag**2,axis=0);xy=np.sum(active*z.real*z.imag,axis=0)
        eigen_angle=.5*np.arctan2(2*xy,xx-yy)
        update(first)
        for theta in [eigen_angle,eigen_angle+np.pi]:
            theta=first+np.mod(theta-first,2*np.pi);update(theta,(theta<=last+1e-12))
    return best.reshape(shape),np.mod(phase.reshape(shape)+np.pi,2*np.pi)-np.pi
