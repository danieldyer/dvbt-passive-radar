"""Exact spline integrals for a physically scaled motion prior.

The Cartesian white-noise-on-jerk analogue of Tang, Yoon and Barfoot (2019),
https://arxiv.org/abs/1809.06518 . This is a finite-basis MAP approximation,
not an additional measurement and not an image-domain correction.

For proper complex unit-noise radar residuals the data negative log likelihood
is ||r||^2. The motion prior is 1/(2 q) integral ||d^3 p/dt^3||^2 dt.
q has units m^2/s^5 and gives Var[a(t+T)-a(t)] = q T per Cartesian axis.
The scalar yaw-rate prior similarly penalizes squared angular acceleration.
"""
import numpy as np
from scipy.interpolate import BSpline


def derivative_quadrature(grid,degree,order):
    """Integrate products of spline derivatives exactly on each knot span."""
    grid=np.asarray(grid,float);breaks=np.unique(grid);left,right=breaks[:-1],breaks[1:]
    nodes,weight=np.polynomial.legendre.leggauss(degree-order+1)
    time=((left+right)[:,None]+(right-left)[:,None]*nodes)/2
    weight=((right-left)[:,None]*weight/2).ravel()
    design=BSpline(grid,np.eye(len(grid)-degree-1),degree,extrapolate=False)(time.ravel(),order)
    return time.ravel(),weight,design


class KinematicPrior:
    def __init__(self,model,*,acceleration_change_mps2=None,angular_rate_change_degps=None,timescale_s=30.):
        if not np.isfinite(timescale_s) or timescale_s<=0:raise ValueError('Motion-prior timescale must be positive')
        n=model.n;matrices=[];offsets=[];self.components=[]
        def add(matrix,offset,density,kind,units):
            if not np.isfinite(density) or density<=0:raise ValueError('Motion process density must be positive')
            matrices.append(matrix/np.sqrt(2*density));offsets.append(offset/np.sqrt(2*density))
            self.components.append(dict(kind=kind,density=density,density_units=units,matrix=matrix,offset=offset))
        if acceleration_change_mps2 is not None:
            if acceleration_change_mps2<=0:raise ValueError('Acceleration-change scale must be positive')
            _,w,d=derivative_quadrature(model.pk,3,3);root=np.sqrt(w)
            matrix=np.zeros((len(w),3,n))
            matrix[:,:,:model.np]=d[:,None,:]*model.axes[0][None,:,None]
            matrix[:,:,model.np:2*model.np]=d[:,None,:]*model.axes[1][None,:,None]
            matrix*=model.scale[None,None,:]*root[:,None,None]
            offset=(d@model.pc)*root[:,None]
            add(matrix.reshape(-1,n),offset.ravel(),acceleration_change_mps2**2/timescale_s,'translation_jerk','m^2/s^5')
        if angular_rate_change_degps is not None:
            if angular_rate_change_degps<=0:raise ValueError('Angular-rate-change scale must be positive')
            _,w,d=derivative_quadrature(model.ak,3,2);matrix=np.zeros((len(w),n))
            # The anchored angle's constant reference has zero derivative.
            matrix[:,2*model.np:]=d[:,:-1]*model.scale[2*model.np:][None,:]
            root=np.sqrt(w);matrix*=root[:,None];offset=(d@model.ac)*root
            add(matrix,offset,np.radians(angular_rate_change_degps)**2/timescale_s,'angular_acceleration','rad^2/s^3')
        self.matrix=np.concatenate(matrices) if matrices else np.empty((0,n))
        self.offset=np.concatenate(offsets) if offsets else np.empty(0)
        self.gram=2*self.matrix.T@self.matrix
        self.settings=dict(acceleration_change_mps2=acceleration_change_mps2,angular_rate_change_degps=angular_rate_change_degps,
            timescale_s=timescale_s,interpretation='Per-axis one-standard-deviation acceleration/rate increment over the stated duration; provisional physical prior, not independently calibrated',
            source='https://arxiv.org/abs/1809.06518',basis_scope='Integrated derivative penalty in the existing finite cubic-spline basis')

    def evaluate(self,q):
        residual=self.offset+self.matrix@q
        return float(residual@residual),2*self.matrix.T@residual

    def diagnostics(self,q):
        output=dict(self.settings);output['components']=[]
        for d in self.components:
            residual=d['offset']+d['matrix']@q;integral=float(residual@residual)
            output['components'].append(dict(kind=d['kind'],density=d['density'],density_units=d['density_units'],integrated_squared_derivative=integral,negative_log_prior=integral/(2*d['density'])))
        output['negative_log_prior']=self.evaluate(q)[0];return output
