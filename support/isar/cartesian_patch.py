"""Physical planar constant-velocity centre, with a common rigid angular rate.

Parameters are east/north position corrections, east/north velocity
corrections and counterclockwise degrees per second. No instantaneous
bistatic gradient constrains the unknown transverse translation.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class CartesianPatch:
    parameter_count=5
    time:np.ndarray
    epoch:float
    position_enu:np.ndarray
    velocity_enu:np.ndarray
    ecef_to_enu:np.ndarray
    receiver:np.ndarray

    def __post_init__(self):
        self.time=np.asarray(self.time);self.dt=self.time-self.epoch
        self.position_enu,self.velocity_enu,self.ecef_to_enu,self.receiver=map(np.asarray,
            (self.position_enu,self.velocity_enu,self.ecef_to_enu,self.receiver))
        if (self.position_enu.shape!=(3,) or self.velocity_enu.shape!=(3,) or self.receiver.shape!=(3,) or
                self.ecef_to_enu.shape!=(3,3) or not np.isfinite(self.dt).all()):
            raise ValueError('A physical Cartesian state, local frame and finite times are required')

    def orientation(self,parameters):
        w=np.radians(parameters[4]);angle=w*self.dt;c,s=np.cos(angle),np.sin(angle)
        r=np.zeros((len(self.time),3,3));d=np.zeros_like(r)
        r[:,0,0]=r[:,1,1]=c;r[:,1,0]=s;r[:,0,1]=-s;r[:,2,2]=1.
        d[:,0,0]=d[:,1,1]=-w*s;d[:,1,0]=w*c;d[:,0,1]=-w*c
        return self.ecef_to_enu.T@r,self.ecef_to_enu.T@d

    def point_state(self,parameters,point,*,derivatives=False,origin=None):
        p=np.asarray(parameters);q=np.asarray(point);dt=self.dt;deg=np.pi/180
        w=p[4]*deg;angle=w*dt;c,s=np.cos(angle),np.sin(angle)
        local=np.c_[c*q[0]-s*q[1],s*q[0]+c*q[1],np.full(len(dt),q[2])]
        normal=np.c_[-local[:,1],local[:,0],np.zeros(len(dt))]
        change=np.r_[p[:2],0.]+dt[:,None]*np.r_[p[2:4],0.]
        base=self.receiver if origin is None else self.receiver-np.asarray(origin)
        position=base+(self.position_enu+dt[:,None]*self.velocity_enu+change+local)@self.ecef_to_enu
        velocity=(self.velocity_enu+np.r_[p[2:4],0.]+w*normal)@self.ecef_to_enu
        if not derivatives:return position,velocity
        dp=np.zeros((len(dt),3,5));dv=np.zeros_like(dp)
        dp[:,:2,:2]=np.eye(2);dp[:,:2,2:4]=dt[:,None,None]*np.eye(2)
        dv[:,:2,2:4]=np.eye(2)
        dp[:,:,4]=(deg*dt)[:,None]*normal
        dv[:,:,4]=deg*normal-(w*deg*dt)[:,None]*np.c_[local[:,:2],np.zeros(len(dt))]
        return position,velocity,np.einsum('nij,ik->nkj',dp,self.ecef_to_enu),np.einsum('nij,ik->nkj',dv,self.ecef_to_enu)

    def midpoint_path_change(self,parameters,transmitter):
        """Stable path-intercept coordinate gauge, not a temporal phase screen."""
        local_tx=(np.asarray(transmitter)-self.receiver)@self.ecef_to_enu.T
        displacement=np.r_[np.asarray(parameters)[:2],0.];delta=0.;gradient=np.zeros(3)
        for antenna in [local_tx,np.zeros(3)]:
            old=self.position_enu-antenna;new=old+displacement
            length=np.linalg.norm(new)
            delta+=(2*np.dot(old,displacement)+np.dot(displacement,displacement))/(length+np.linalg.norm(old))
            gradient+=new/length
        return float(delta),gradient[:2]
