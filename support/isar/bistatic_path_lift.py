"""Lift one measured bistatic path into a declared physical position family.

The transverse trajectory remains an explicit prior. An exact scalar solve
along a fixed direction family supplies the radial coordinate. Velocity and
path/rate Jacobians differentiate that same constraint; phase and migration
cannot acquire independent motion histories through this operation.
"""
import numpy as np
from motion import excess_path


def curvature(position,vector,tx,rx):
    result=np.zeros_like(vector,dtype=float)
    for antenna in [tx,rx]:
        ray=position-antenna;distance=np.linalg.norm(ray,axis=1);unit=ray/distance[:,None]
        result+=(vector-unit*np.sum(unit*vector,axis=1)[:,None])/distance[:,None]
    return result


def lift(anchor,anchor_velocity,direction,direction_rate,path,path_rate,tx,rx):
    anchor,anchor_velocity,direction,direction_rate=map(np.asarray,[anchor,anchor_velocity,direction,direction_rate])
    path,path_rate=np.asarray(path),np.asarray(path_rate)
    if not (anchor.shape==anchor_velocity.shape==direction.shape==direction_rate.shape==(len(path),3)) or path_rate.shape!=path.shape:raise ValueError('Each physical epoch needs one path, rate, anchor and direction')
    shift=np.zeros(len(path))
    for _ in range(6):
        position=anchor+direction*shift[:,None];measured,gradient=excess_path(position,tx,rx);slope=np.sum(gradient*direction,axis=1)
        if np.min(abs(slope))<1e-7:raise ValueError('Declared radial direction is tangent to the bistatic isorange')
        change=(path-measured)/slope;shift+=change
        if max(abs(path-measured))<1e-8:break
    position=anchor+direction*shift[:,None];actual,gradient=excess_path(position,tx,rx);slope=np.sum(gradient*direction,axis=1)
    if max(abs(actual-path))>1e-6:raise ValueError('Physical path inversion did not converge')
    before=anchor_velocity+direction_rate*shift[:,None]
    shift_rate=(path_rate-np.sum(gradient*before,axis=1))/slope
    velocity=before+direction*shift_rate[:,None]
    dp=direction/slope[:,None];dg=curvature(position,dp,tx,rx)
    dslope=np.sum(dg*direction,axis=1)
    numerator=-np.sum(dg*before,axis=1)-np.sum(gradient*direction_rate,axis=1)/slope
    dshift_rate=numerator/slope-shift_rate*dslope/slope
    dv=direction_rate/slope[:,None]+direction*dshift_rate[:,None]
    return dict(position_ecef_m=position,velocity_ecef_mps=velocity,shift_m=shift,
        position_path_jacobian=dp,velocity_path_jacobian=dv,velocity_rate_jacobian=dp)
