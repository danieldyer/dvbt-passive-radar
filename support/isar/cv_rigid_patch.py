"""CV rotation centre without forcing a selected reflector onto that centre.

The first five parameters describe the anchor's instantaneous position,
velocity and common spin at the reference epoch. The last two are the
anchor's horizontal offset from the CV rotation centre, in metres. This is
exact rigid-body kinematics, not an independent fitted phase polynomial.
"""
import numpy as np
from cartesian_patch import CartesianPatch


class CvRigidPatch(CartesianPatch):
    parameter_count=7

    def point_state(self,parameters,point,*,derivatives=False,origin=None):
        p=np.asarray(parameters);u=np.r_[p[5:7],0.];deg=np.pi/180
        base=super().point_state(p[:5],point,derivatives=derivatives,origin=origin)
        w=p[4]*deg;theta=w*self.dt;s=np.sin(theta);cm1=-2*np.sin(theta/2)**2;smtheta=s-theta
        residual=np.c_[cm1*u[0]-smtheta*u[1],smtheta*u[0]+cm1*u[1],np.zeros(len(theta))]
        jchange=np.c_[-s*u[0]-cm1*u[1],cm1*u[0]-s*u[1],np.zeros(len(theta))]
        position=base[0]+residual@self.ecef_to_enu
        velocity=base[1]+(w*jchange)@self.ecef_to_enu
        if not derivatives:return position,velocity
        dp=np.zeros((len(theta),3,7));dv=np.zeros_like(dp);dp[:,:,:5]=base[2];dv[:,:,:5]=base[3]
        dp[:,:,4]+=(deg*self.dt[:,None]*jchange)@self.ecef_to_enu
        rotated=np.c_[(cm1+1)*u[0]-s*u[1],s*u[0]+(cm1+1)*u[1],np.zeros(len(theta))]
        dv[:,:,4]+=(deg*(jchange-w*self.dt[:,None]*rotated))@self.ecef_to_enu
        local=np.zeros((len(theta),3,2));rate=np.zeros_like(local)
        local[:,0,0]=local[:,1,1]=cm1;local[:,0,1]=-smtheta;local[:,1,0]=smtheta
        rate[:,0,0]=rate[:,1,1]=-w*s;rate[:,0,1]=-w*cm1;rate[:,1,0]=w*cm1
        dp[:,:,5:]=np.einsum('nij,ik->nkj',local,self.ecef_to_enu)
        dv[:,:,5:]=np.einsum('nij,ik->nkj',rate,self.ecef_to_enu)
        return position,velocity,dp,dv

    def centre_state(self,parameters):
        """Physical CV centre and source-coordinate shift for ordinary rendering."""
        p=np.asarray(parameters);offset=np.r_[p[5:7],0.]
        position=self.position_enu+np.r_[p[:2],0.]-offset
        velocity=self.velocity_enu+np.r_[p[2:4],0.]-np.radians(p[4])*np.array([-offset[1],offset[0],0.])
        return np.array([position,velocity,np.zeros(3)]),offset
