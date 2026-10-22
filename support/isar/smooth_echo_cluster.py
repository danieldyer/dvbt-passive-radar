"""A Gaussian white-jerk motion prior on the shared physical path basis.

The prior is one half integral(path'''²)/q dt, with q in m²/s⁵. Its nullspace
contains every quadratic path: it penalizes changing acceleration, not speed
or constant acceleration. The same path still drives delay, carrier and Doppler.
"""
import numpy as np

from flexible_echo_cluster import FlexibleEchoCluster


def jerk_factor(curve):
    """Exact square-root integrated jerk Gram matrix for cubic B-splines.

    Gauss-Legendre integration also covers polynomial test curves. Integrating
    each knot span separately avoids numerical differentiation or knot jumps.
    """
    first,last=curve.epoch-curve.scale,curve.epoch+curve.scale
    edges=np.unique(np.r_[first,getattr(getattr(curve,'spline',None),'t',[]),last])
    edges=edges[(edges>=first)&(edges<=last)]
    node,weight=np.polynomial.legendre.leggauss(max(4,curve.degree+1))
    time=np.concatenate([(a+b)/2+(b-a)*node/2 for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([(b-a)*weight/2 for a,b in zip(edges[:-1],edges[1:])])
    basis=curve(time,3);gram=basis.T@(weights[:,None]*basis)
    values,vectors=np.linalg.eigh((gram+gram.T)/2)
    cutoff=max(0.,values[-1])*1e-10
    keep=values>max(cutoff,1e-14)
    return np.sqrt(values[keep])[:,None]*vectors[:,keep].T


class SmoothEchoCluster(FlexibleEchoCluster):
    def __init__(self,*args,jerk_std,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(jerk_std) or jerk_std<=0:raise ValueError('Positive white-jerk standard-deviation scale required')
        self.jerk_std=float(jerk_std);factor=jerk_factor(self.curve)/jerk_std
        paths=np.unique(np.stack([p[s] for p in self.local_path_indices for s in self.selected_components]),axis=0)
        self.path_prior=np.zeros((len(paths)*len(factor),self.count))
        for i,path in enumerate(paths):self.path_prior[i*len(factor):(i+1)*len(factor),path]=factor

    def evaluate(self,parameters):
        # The superclass cache holds only data residuals, so repeated calls
        # never append or count the prior twice.
        residual,jacobian=super().evaluate(parameters)
        scale=np.sqrt(2*self.energy)
        return np.r_[residual,self.path_prior@parameters/scale],np.r_[jacobian,self.path_prior/scale]

    def prior_cost(self,parameters):
        value=self.path_prior@parameters
        return float(value@value/2)
