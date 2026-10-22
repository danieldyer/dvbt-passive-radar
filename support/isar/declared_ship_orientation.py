"""Explicit rigid-frame hypotheses, separate from radar translation fitting."""
import numpy as np
from imaging import physical_state


def rotate_frame(value,time,anchor,rate_degps=0.,acceleration_degps2=0.):
    """Apply one physical in-plane angular history, about the reference point.

    The anchor orientation is a coordinate gauge. This changes both the
    rigid frame and its time derivative; it never rotates a formed image.
    Translation is left unchanged so callers can retain a measured reference
    scatterer's path while rotating the other scatterers about it.
    """
    if rate_degps==0 and acceleration_degps2==0:return value
    dt=np.atleast_1d(time)-anchor
    theta=np.radians(rate_degps*dt+.5*acceleration_degps2*dt**2)
    rate=np.radians(rate_degps+acceleration_degps2*dt)
    c,s=np.cos(theta),np.sin(theta);q=np.zeros((len(dt),3,3));dq=np.zeros_like(q)
    q[:,0,0]=q[:,1,1]=c;q[:,1,0]=s;q[:,0,1]=-s;q[:,2,2]=1
    dq[:,0,0]=dq[:,1,1]=-s*rate;dq[:,1,0]=c*rate;dq[:,0,1]=-c*rate
    result=dict(value)
    result['body_to_ecef']=value['body_to_ecef']@q
    result['frame_velocity']=value['frame_velocity']@q+value['body_to_ecef']@dq
    return result


def state(motion,time,mode='existing',course_time_offset_s=0.):
    value=motion.state(time)
    if mode=='existing':return value
    at=np.array([157.]);reference=motion.state(at)['body_to_ecef'][0]
    if mode=='reported-course':
        a=physical_state(motion.track,motion.epoch+at+course_time_offset_s,'reported-course-prior',max_gap_s=400.)
        gauge=a['body_to_ecef'][0].T@reference
        current=physical_state(motion.track,motion.epoch+np.atleast_1d(time)+course_time_offset_s,'reported-course-prior',max_gap_s=400.)
        value['body_to_ecef']=current['body_to_ecef']@gauge;value['frame_velocity']=current['frame_velocity']@gauge
    elif mode=='fixed':
        value['body_to_ecef']=np.broadcast_to(reference,(len(np.atleast_1d(time)),3,3));value['frame_velocity']=np.zeros_like(value['body_to_ecef'])
    else:raise ValueError('Unknown declared rigid-frame hypothesis')
    return value
