"""Smooth planar constant-turn motion, matching p/v/a at the midpoint.

Speed changes linearly while course rate is constant. The six planar midpoint
coordinates are the same as the quadratic model; no extra motion degree is
introduced. The zero-acceleration limit is exactly straight CV.
"""
import numpy as np


def trajectory(time,position,velocity,acceleration):
    t=np.asarray(time);p,v,a=map(np.asarray,(position,velocity,acceleration))
    speed2=np.dot(v[:2],v[:2])
    if abs(speed2)<1e-12:raise ValueError('Coordinated turn requires a nonzero planar speed')
    growth=np.dot(v[:2],a[:2])/speed2
    rate=(v[0]*a[1]-v[1]*a[0])/speed2
    z=rate*t
    sinc=np.sinc(z/np.pi);half=np.sinc(z/(2*np.pi))
    A=t*sinc;B=.5*rate*t*t*half*half
    D=np.empty_like(z,dtype=np.result_type(z,float));small=abs(z)<1e-3
    D[small]=z[small]/3-z[small]**3/30+z[small]**5/840-z[small]**7/45360
    D[~small]=(np.sin(z[~small])-z[~small]*np.cos(z[~small]))/z[~small]**2
    C=t*t*(sinc-.5*half*half);D*=t*t
    perpendicular=np.array([-v[1],v[0]])
    value=np.empty((len(t),3),dtype=np.result_type(p,v,a,t));derivative=np.empty_like(value)
    value[:,:2]=p[:2]+(A+growth*C)[:,None]*v[:2]+(B+growth*D)[:,None]*perpendicular
    derivative[:,:2]=(1+growth*t)[:,None]*(np.cos(z)[:,None]*v[:2]+np.sin(z)[:,None]*perpendicular)
    value[:,2]=p[2]+t*v[2]+.5*t*t*a[2];derivative[:,2]=v[2]+t*a[2]
    return value,derivative,rate
