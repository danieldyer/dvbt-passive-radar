"""Physical planar point motion with positive brightness and explicit AIS priors.

The original-row amplitude operator is reused from PhysicalAmplitudeArc. One
constant phase per band remains; no time-dependent complex phase is fitted.
Position/velocity priors are declared separately from radar information.
"""
import copy
import numpy as np

from flash_geometry import observables
from physical_amplitude_arc import PhysicalAmplitudeArc,PhysicalAmplitudeFit
from physical_amplitude_arc import state_derivatives
from coordinated_turn import trajectory as turn_trajectory


class PlanarGeometry:
    count=6

    def __init__(self,receiver,rotation,transmitters,coarse_state,*,sigma=(200.,200.,.2,.2,.01,.01)):
        self.receiver=np.asarray(receiver);self.rotation=np.asarray(rotation)
        self.local_transmitters=(np.asarray(transmitters)-self.receiver)@self.rotation.T
        self.coarse=np.asarray(coarse_state).copy();self.coarse[2]=0.
        self.sigma=np.asarray(sigma)
        if self.sigma.shape!=(6,) or np.any(self.sigma<=0):raise ValueError('Six positive prior standard deviations are required')

    def state(self,parameters):
        state=self.coarse.astype(np.result_type(parameters,float))
        state[:,:2]+=(np.asarray(parameters)[:6]*self.sigma).reshape(3,2)
        return *state,self.local_transmitters,np.zeros(3)

    def prediction(self,parameters):
        return observables(*self.state(parameters))

    def bounds(self):
        limit=np.array([500.,500.,3.,3.,.1,.1])/self.sigma
        return -limit,limit


class RadialGeometry(PlanarGeometry):
    """Condition weak transverse motion on AIS; infer only radial corrections.

    All bands still use exact two-leg Cartesian propagation. This reduces
    the kinematic subspace, not the phase bookkeeping or antenna geometry.
    The result is explicitly conditional on the chosen transverse prior.
    """
    count=3

    def __init__(self,receiver,rotation,transmitters,coarse_state,*,position_bound_m=900.):
        super().__init__(receiver,rotation,transmitters,coarse_state)
        p=self.coarse[0]
        g=sum((p-a)/np.linalg.norm(p-a) for a in (self.local_transmitters[2],np.zeros(3)))
        self.direction=np.r_[g[:2]/np.dot(g[:2],g[:2]),0.]
        self.sigma=np.array([200.,.5,.05])
        if not np.isfinite(position_bound_m) or position_bound_m<=0:raise ValueError('A positive radial position bound is required')
        self.position_bound_m=position_bound_m

    def state(self,parameters):
        state=self.coarse.astype(np.result_type(parameters,float))
        state+=(np.asarray(parameters)[:3]*self.sigma)[:,None]*self.direction
        return *state,self.local_transmitters,np.zeros(3)

    def bounds(self):
        limit=np.array([self.position_bound_m,3.,.15])/self.sigma
        return -limit,limit


class TrainingMetric:
    def __init__(self,field,training,metric):
        self.shape=field.y.shape;self.training=training;self.metric=metric

    def apply(self,values):
        if self.metric is None:return values
        shape=np.shape(values);value=np.asarray(values).reshape(*self.shape,-1).copy()
        selected=value[self.training]
        value[self.training]=self.metric.apply(selected.reshape(-1,value.shape[-1])).reshape(selected.shape)
        return value.reshape(shape)


class FadingPointArc(PhysicalAmplitudeArc):
    def __init__(self,field,training,knots,geometry,channel,midpoint,stationary,metric=None):
        self.fixed_metric=TrainingMetric(field,training,metric)
        self.original_field=field;self.original_stationary=stationary
        transformed=copy.copy(field);transformed.y=self.fixed_metric.apply(field.y.ravel()).reshape(field.y.shape)
        super().__init__(transformed,training,knots,geometry,channel,midpoint,
            stationary_design=self.fixed_metric.apply(stationary))

    def reduced(self,raw):
        return self.fixed_metric.apply(super().reduced(raw))

    def phase_seed(self,parameters):
        raw,_=self.response(parameters)
        h=self.reduced(raw[:,:,None])[:,0]
        h=self.nuisance.remove(h[self.selected])
        return float(np.angle(np.vdot(h,self.clean)))

    def original_prediction(self,parameters):
        _,_,design,amplitude,_,_=self.evaluate(parameters,True)
        beta=self.nuisance.solve(self.field.y.ravel()[self.selected]-(design@amplitude)[self.selected])
        raw,_=self.response(parameters)
        original=PhysicalAmplitudeArc.reduced(self,raw[:,:,None]*self.amplitude_basis[:,None,:])@amplitude
        return original+self.original_stationary@beta,amplitude,beta


class FadingPointFit(PhysicalAmplitudeFit):
    def objective(self,parameters):
        terms=[b.evaluate(parameters) for b in self.bands];n=self.geometry.count
        # Proper-complex NLL = RSS; real Gaussian prior NLL = theta^2 / 2.
        value=sum(b.energy*t[0] for b,t in zip(self.bands,terms))+.5*np.dot(parameters[:n],parameters[:n])
        gradient=sum(b.energy*t[1] for b,t in zip(self.bands,terms));gradient[:n]+=parameters[:n]
        return value/self.energy,gradient/self.energy


class TurningGeometry(PlanarGeometry):
    def bounds(self):
        limit=np.array([500.,500.,3.,3.,.04,.04])/self.sigma
        return -limit,limit


class TurningPointArc(FadingPointArc):
    """The same positive-brightness likelihood with an integrated turn path."""
    def response(self,parameters,*,range_bias_m=0.,envelope_derivative=False):
        g,c=self.geometry,self.channel;f=self.field
        (p,v,a,tx,rx),_,length,dlength=state_derivatives(g,parameters[:g.count])
        dt=f.data['row_time']-self.midpoint
        local,local_velocity,_=turn_trajectory(dt,p,v,a)
        dp=np.empty((len(dt),3,g.count));dv=np.empty_like(dp)
        for j in range(g.count):
            perturbed=np.asarray(parameters[:g.count],complex);perturbed[j]+=1e-20j
            pp,vv,aa,_,_=g.state(perturbed)
            lp,lv,_=turn_trajectory(dt,pp,vv,aa)
            dp[:,:,j]=lp.imag/1e-20;dv[:,:,j]=lv.imag/1e-20
        offset=(local-self.reference_local)@g.rotation
        position=f.data['centre_ecef_m']+offset;velocity=local_velocity@g.rotation
        raw,derivative=f.raw_trajectory(position,velocity,position_jacobian=np.einsum('nij,ik->nkj',dp,g.rotation),
            velocity_jacobian=np.einsum('nij,ik->nkj',dv,g.rotation),transmitter=g.receiver+tx[c]@g.rotation,
            receiver=g.receiver+rx@g.rotation,position_offset=offset,range_bias_m=range_bias_m,
            range_bias_derivative=envelope_derivative)
        envelope=derivative[:,:,-1] if envelope_derivative else None
        derivative=derivative[:,:,:g.count]+1j*f.k*raw[:,:,None]*dlength[c]
        rotation=np.exp(1j*(f.k*(length[c]-self.reference_path)+parameters[g.count+c]))
        raw,derivative=raw*rotation,derivative*rotation
        phases=np.zeros(raw.shape+(5,),complex);phases[:,:,c]=1j*raw
        changes=[derivative,phases]
        if envelope_derivative:changes.append((envelope*rotation)[:,:,None])
        return raw,np.concatenate(changes,axis=2)
