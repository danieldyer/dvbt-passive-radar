"""A common constant-speed turn for translation and rigid orientation.

Five parameters: east/north position and velocity corrections, then the
counterclockwise turn rate in degrees/s. Unlike a rotating body with a
straight-moving centre, this hypothesis gives the centre the centripetal
acceleration required by its changing course. Zero turn is exactly CV.
"""
import numpy as np
from cartesian_patch import CartesianPatch


class CoordinatedPatch(CartesianPatch):
    def point_state(self,parameters,point,*,derivatives=False,origin=None):
        p=np.asarray(parameters);q=np.asarray(point);dt=self.dt;deg=np.pi/180
        rate=p[4]*deg;angle=rate*dt;c,s=np.cos(angle),np.sin(angle)
        sinc=np.sinc(angle/np.pi);half=np.sinc(angle/(2*np.pi))
        a=dt*sinc;b=.5*rate*dt**2*half**2
        velocity=self.velocity_enu+np.r_[p[2:4],0.]
        tangent=np.array([-velocity[1],velocity[0]])
        local=np.c_[c*q[0]-s*q[1],s*q[0]+c*q[1],np.full(len(dt),q[2])]
        normal=np.c_[-local[:,1],local[:,0],np.zeros(len(dt))]
        drift=np.c_[a[:,None]*velocity[:2]+b[:,None]*tangent,dt*velocity[2]]
        rotated=np.c_[c*velocity[0]-s*velocity[1],s*velocity[0]+c*velocity[1],np.full(len(dt),velocity[2])]
        base=self.receiver if origin is None else self.receiver-np.asarray(origin)
        position=base+(self.position_enu+np.r_[p[:2],0.]+drift+local)@self.ecef_to_enu
        speed=(rotated+rate*normal)@self.ecef_to_enu
        if not derivatives:return position,speed
        small=abs(angle)<1e-3
        da=-angle/3+angle**3/30-angle**5/840+angle**7/45360
        np.divide(angle*c-s,angle**2,out=da,where=~small)
        da*=dt**2
        db=dt**2*(sinc-.5*half**2)
        dp=np.zeros((len(dt),3,5));dv=np.zeros_like(dp)
        dp[:,:2,:2]=np.eye(2)
        dp[:,0,2]=dp[:,1,3]=a;dp[:,0,3]=-b;dp[:,1,2]=b
        dv[:,0,2]=dv[:,1,3]=c;dv[:,0,3]=-s;dv[:,1,2]=s
        dp[:,:2,4]=deg*(da[:,None]*velocity[:2]+db[:,None]*tangent)
        dp[:,:,4]+=(deg*dt)[:,None]*normal
        dv[:,:,4]=deg*(dt[:,None]*np.c_[-rotated[:,1],rotated[:,0],np.zeros(len(dt))]+normal-
            (rate*dt)[:,None]*np.c_[local[:,:2],np.zeros(len(dt))])
        return position,speed,np.einsum('nij,ik->nkj',dp,self.ecef_to_enu),np.einsum('nij,ik->nkj',dv,self.ecef_to_enu)
