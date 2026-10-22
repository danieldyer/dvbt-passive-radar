"""Short-arc finite-side hypothesis for joint multiband range/rate/curvature.

One physical rectangle, centre trajectory, heading and turn rate predict all
bands. The PO aperture supplies frequency-dependent group delay and phase;
these are not independent fitted per-band corrections. This diagnostic uses
the same summary covariance as the point control and needs a subsequent raw
complex-data check before any geometry can be accepted.
"""
import numpy as np
from scipy.optimize import least_squares

from flash_geometry import observables
from motion import C, enu_basis
from specular_facet import response
from facet_kinematics import aperture


def facet_observables(position, velocity, acceleration, transmitters, receiver,
                      frequency_hz, heading_rad, turn_rate_rad_s, length_m, height_m,
                      *, step_s=.01, order=(64,8)):
    times=np.arange(-2,3)*step_s
    position,velocity,acceleration=map(np.asarray,(position,velocity,acceleration))
    centres=position+times[:,None]*velocity+.5*times[:,None]**2*acceleration
    heading=heading_rad+turn_rate_rad_s*times
    along=np.column_stack([np.sin(heading),np.cos(heading),np.zeros(5)])
    physical=observables(position,velocity,acceleration,transmitters,receiver)
    extra=[];shapes=[]
    for i,(tx,f) in enumerate(zip(transmitters,frequency_hz)):
        aperture=response(centres,tx,receiver,along,[0.,0.,1.],length_m,height_m,[f],order=order)
        shape=aperture['relative_field'][:,0]
        # Log ratios remove the unobservable multiple of 2pi. A small temporal
        # step is independently halved in the convergence control.
        phase=np.angle(shape*np.conj(shape[2]))
        k=2*np.pi*f/C
        extra.append([aperture['path_moment_m'][2,0].real,
            -(phase[0]-8*phase[1]+8*phase[3]-phase[4])/(12*step_s*k),
            -(-phase[0]+16*phase[1]+16*phase[3]-phase[4])/(12*step_s**2*k)])
        shapes.append(shape[2])
    return physical+np.array(extra),np.array(shapes),physical


class FacetFlash:
    def __init__(self,base,frequency_hz,*,length_m,height_m,reference_heading_deg=331.1):
        self.base,self.frequency=base,np.array(frequency_hz)
        self.length,self.height=length_m,height_m
        self.heading=np.radians(reference_heading_deg)
        self.scale=np.radians([1.,.1])
        state=base.track.sample([base.utc],max_gap_s=180)
        self.frame=base.rotation@enu_basis(state['latitude'][0],state['longitude'][0])
        self.count=base.count+2
        self.lower=np.r_[base.lower,-3.,-3.]
        self.upper=np.r_[base.upper,3.,3.]
        # Keep this declared vertical surface above approximate local sea level.
        # A 2 m margin covers the local-frame/geoid difference in this aperture.
        self.lower[base.nantenna+2]=(height_m/2+2.)/base.nuisance_sigma[2]

    def seed(self,base_parameters,heading_offset_deg,turn_rate_deg_s):
        parameters=np.r_[base_parameters,0.,0.]
        parameters=np.clip(parameters,self.lower+1e-7,self.upper-1e-7)
        position,_,_,transmitters,receiver=self.base.state(parameters[:self.base.count])
        towards_tx=transmitters-position
        towards_tx/=np.linalg.norm(towards_tx,axis=1)[:,None]
        towards_rx=receiver-position;towards_rx/=np.linalg.norm(towards_rx)
        bisector=(towards_tx.mean(axis=0)+towards_rx)@self.frame
        heading=np.arctan2(bisector[0],bisector[1])-np.pi/2+np.radians(heading_offset_deg)
        parameters[-2]=np.angle(np.exp(1j*(heading-self.heading)))/self.scale[0]
        parameters[-1]=np.radians(turn_rate_deg_s)/self.scale[1]
        return np.clip(parameters,self.lower+1e-7,self.upper-1e-7)

    def prediction(self,parameters):
        state=self.base.state(parameters[:self.base.count])
        position,velocity,acceleration,transmitters,receiver=state
        physical=observables(*state)
        shapes,corrections=aperture(position,velocity,acceleration,transmitters,receiver,self.frequency,
            self.heading+parameters[-2]*self.scale[0],parameters[-1]*self.scale[1],
            self.length,self.height,basis=self.frame)
        return physical+corrections,shapes,physical

    def residual(self,parameters):
        observed=self.prediction(parameters)[0]-self.base.measured
        data=np.concatenate([w@e[self.base.columns] for w,e in zip(self.base.whiten,observed)])
        return np.r_[data,parameters]

    def parameter_jacobian(self,p,step=2e-5):
        # Absolute parameter steps remain meaningful when an optimized
        # coordinate is close to zero. Time derivatives are analytical.
        steps=step*np.maximum(1.,abs(p))
        return np.column_stack([(self.residual(p+d)-self.residual(p-d))/(2*h)
            for d,h in zip(np.diag(steps),steps)])

    def fit(self,seeds):
        fits=[]
        for seed in seeds:
            initial=np.clip(seed,self.lower+1e-7,self.upper-1e-7)
            fit=least_squares(self.residual,initial,jac=self.parameter_jacobian,
                bounds=(self.lower,self.upper),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-7)
            fits.append(fit)
        return min(fits,key=lambda f:float(f.fun@f.fun)),fits

    def describe(self,fit):
        predicted,shape,physical=self.prediction(fit.x)
        ndata=len(self.base.data_residual(fit.x[:self.base.count]))
        active=((fit.x<self.lower+1e-6)&(fit.grad>0))|((fit.x>self.upper-1e-6)&(fit.grad<0))
        free=fit.jac[:,~active]
        step=-np.linalg.lstsq(free,fit.fun,rcond=None)[0]
        proposed=free@step
        decrease=-2*fit.fun@proposed-proposed@proposed
        check=self.parameter_jacobian(fit.x,step=1e-5)
        return {'parameters':fit.x.tolist(),'predicted_observables':predicted.tolist(),
            'centre_observables':physical.tolist(),'facet_observable_corrections':(predicted-physical).tolist(),
            'shape_amplitude':abs(shape).tolist(),'shape_phase_deg':np.degrees(np.angle(shape)).tolist(),
            'heading_deg':float(np.degrees(self.heading+fit.x[-2]*self.scale[0])),
            'turn_rate_deg_s':float(np.degrees(fit.x[-1]*self.scale[1])),
            'data_squared_residual':float(fit.fun[:ndata]@fit.fun[:ndata]),
            'prior_squared_residual':float(fit.fun[ndata:]@fit.fun[ndata:]),
            'penalized_squared_residual':float(fit.fun@fit.fun),
            'derivatives':'Exact second-order time derivatives of path, spreading and HH current; central parameter Jacobian with absolute steps.',
            'converged':bool(fit.success),'termination':fit.message,'first_order_optimality':float(fit.optimality),
            'evaluations':int(fit.nfev),'bound_parameter_indices':np.flatnonzero((fit.x<self.lower+1e-4)|(fit.x>self.upper-1e-4)).tolist(),
            'estimated_remaining_gauss_newton_squared_cost_decrease':float(decrease),
            'jacobian_half_step_relative_difference':float(np.linalg.norm(check-fit.jac)/np.linalg.norm(fit.jac)),
            'antenna_fit':self.base.antennas.describe(fit.x[:18]) if self.base.free_antennas else None,
            'accepted_calibration':False,'model':'One finite vertical uniform-current PEC side; common position, motion, heading and turn rate. No sea interaction or nuisance per-band phase rate.'}
