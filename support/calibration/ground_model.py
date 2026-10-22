"""Ground-target hypothesis with exact excess path and a terrain constraint."""
import numpy as np
from numpy.polynomial import legendre as leg

from geometry import unit
from polar_model import PolarTrajectory


class GroundTrajectory(PolarTrajectory):
    def __init__(self,seed,terrain,reference_channel=1,**kwargs):
        super().__init__(seed,reference_channel,**kwargs)
        self.terrain=terrain
        self.count=self.radial_count+self.angular_degree+2

    def positions(self,p,time,sites,free_sites=False):
        base,basis=self.length_basis(time);length=base+basis@p[:self.radial_count]
        angular=leg.legvander((np.asarray(time)-self.seed['epoch'])/15,self.angular_degree)
        theta=angular@p[self.radial_count:self.count-1]
        direction=np.c_[np.cos(theta),np.sin(theta),np.zeros(len(theta))]
        perpendicular=np.c_[-np.sin(theta),np.cos(theta),np.zeros(len(theta))]
        tx=sites.copy()
        if free_sites:
            tx+=p[self.count:].reshape(3,3)*[30,30,75]
        ref=tx[self.reference_site]+self.reference_offset;baseline=np.linalg.norm(ref);s=length+baseline
        h=p[self.count-1]*10
        radius=length*(length+2*baseline)/(2*(s-direction@ref))
        z=np.full(len(time),h-15.)
        for iteration in range(14):
            x=radius[:,None]*direction;x[:,2]=z
            g=unit(x-ref)+unit(x)
            clear,gh=self.terrain.clearance(x,True)
            error=np.c_[np.linalg.norm(x-ref,axis=1)+np.linalg.norm(x,axis=1)-baseline-length,clear-h]
            matrix=np.stack([np.c_[np.sum(g*direction,axis=1),g[:,2]],
                             np.c_[np.sum(gh*direction,axis=1),gh[:,2]]],axis=1)
            step=np.linalg.solve(matrix,error[:,:,None])[:,:,0]
            radius-=step[:,0];z-=step[:,1]
            if np.max(abs(error))<1e-6:
                break
        else:
            raise ValueError('Ground-ellipse intersection did not converge')
        if np.any(radius<=0):
            raise ValueError('Ground intersection crossed the receive-bearing branch')
        x=radius[:,None]*direction;x[:,2]=z
        rhs=np.zeros((len(time),2,len(p)))
        rhs[:,0,:self.radial_count]=basis
        dtheta=radius[:,None]*perpendicular
        rhs[:,0,self.radial_count:self.count-1]=-np.sum(g*dtheta,axis=1)[:,None]*angular
        rhs[:,1,self.radial_count:self.count-1]=-np.sum(gh*dtheta,axis=1)[:,None]*angular
        rhs[:,1,self.count-1]=10.
        if free_sites:
            first=self.count+3*self.reference_site
            rhs[:,0,first:first+3]=(unit(x-ref)+unit(ref))*[30,30,75]
        derivatives=np.linalg.solve(matrix,rhs)
        j=direction[:,:,None]*derivatives[:,0,None,:]
        j[:,2,:]+=derivatives[:,1,:]
        j[:,:,self.radial_count:self.count-1]+=dtheta[:,:,None]*angular[:,None,:]
        return x,j,tx

    def bounds(self,free_sites=False):
        lo=np.full(self.count,-1.);hi=np.full(self.count,1.)
        lo[:self.radial_count],hi[:self.radial_count]=-100.,100.
        lo[self.radial_count],hi[self.radial_count]=-2*np.pi,2*np.pi
        lo[-1],hi[-1]=0.,3. # A 0..30 m scattering height above terrain.
        if free_sites:
            lo,hi=np.r_[lo,np.full(9,-3.)],np.r_[hi,np.full(9,3.)]
        return lo,hi
