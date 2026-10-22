"""Shared physical short-arc motion with independent constant band phases.

The geometry is Flash's AIS-anchored Cartesian position/velocity/acceleration.
Positive brightness knots and static clutter are nuisance coefficients, fitted
with AmplitudeArc's original-row likelihood. Antenna displacement affects both
propagation legs and the direct-path reference. One constant phase gauge per
band removes midpoint phase only; it cannot absorb rate or curvature errors.
"""
import numpy as np
from scipy.optimize import minimize

from amplitude_arc import AmplitudeArc
from field import average


def state_derivatives(geometry, parameters):
    values = geometry.state(parameters)
    derivative = [np.empty(np.shape(v)+(len(parameters),)) for v in values]
    length = geometry.prediction(parameters)[:,0]
    length_derivative = np.empty((5,len(parameters)))
    for j in range(len(parameters)):
        perturbed = np.asarray(parameters,complex);perturbed[j] += 1e-20j
        for target,value in zip(derivative,geometry.state(perturbed)):
            target[...,j] = value.imag/1e-20
        length_derivative[:,j] = geometry.prediction(perturbed)[:,0].imag/1e-20
    return values,derivative,length,length_derivative


class PhysicalAmplitudeArc(AmplitudeArc):
    def __init__(self,field,training,knot_times,geometry,channel,midpoint_s,*,stationary_paths=None,stationary_design=None):
        super().__init__(field,training,knot_times,stationary_paths=stationary_paths,stationary_design=stationary_design)
        self.geometry,self.channel,self.midpoint = geometry,channel,midpoint_s
        self.reference_local = (field.data['centre_ecef_m']-geometry.receiver)@geometry.rotation.T
        self.reference_path = np.interp(midpoint_s,field.data['row_time'],field.data['centre_path_m'])

    def response(self,parameters,*,range_bias_m=0.,envelope_derivative=False):
        g,c = self.geometry,self.channel
        state,jac,length,dlength = state_derivatives(g,parameters[:g.count])
        p,v,a,tx,rx = state;dp,dv,da,dtx,drx = jac
        dt = self.field.data['row_time']-self.midpoint
        local = p+dt[:,None]*v+.5*dt[:,None]**2*a
        offset = (local-self.reference_local)@g.rotation
        position = self.field.data['centre_ecef_m']+offset
        velocity = (v+dt[:,None]*a)@g.rotation
        dx = np.einsum('nij,ik->nkj',dp[None]+dt[:,None,None]*dv+.5*dt[:,None,None]**2*da,g.rotation)
        dvel = np.einsum('nij,ik->nkj',dv[None]+dt[:,None,None]*da,g.rotation)
        raw,derivative = self.field.raw_trajectory(position,velocity,position_jacobian=dx,velocity_jacobian=dvel,
            transmitter=g.receiver+tx[c]@g.rotation,receiver=g.receiver+rx@g.rotation,
            antenna_derivatives=True,position_offset=offset,range_bias_m=range_bias_m,
            range_bias_derivative=envelope_derivative)
        envelope=derivative[:,:,-1] if envelope_derivative else None
        antenna = np.r_[g.rotation.T@dtx[c],g.rotation.T@drx]
        derivative = derivative[:,:,:g.count]+derivative[:,:,g.count:g.count+6]@antenna
        derivative += 1j*self.field.k*raw[:,:,None]*dlength[c]
        rotation = np.exp(1j*(self.field.k*(length[c]-self.reference_path)+parameters[g.count+c]))
        raw,derivative = raw*rotation,derivative*rotation
        phase = np.zeros(raw.shape+(5,),complex);phase[:,:,c] = 1j*raw
        changes=[derivative,phase]
        if envelope_derivative:changes.append((envelope*rotation)[:,:,None])
        return raw,np.concatenate(changes,axis=2)

    def phase_seed(self,parameters):
        raw,_ = self.response(parameters)
        h = self.field.apply(average(raw,self.field.data['row_information'],self.field.metadata['block_rows'])[0]).ravel()
        clean = self.nuisance.remove(h[self.selected])
        return float(np.angle(np.vdot(clean,self.clean)))


class PhysicalAmplitudeFit:
    phase_count = 5

    def __init__(self,bands):
        self.bands = bands;self.geometry = bands[0].geometry
        self.energy = sum(b.energy for b in bands)

    def mapping(self,parameters):
        return np.asarray(parameters),np.eye(len(parameters))

    def metric(self,parameters,transform=None):
        """Mean-response metric after static and active real amplitudes."""
        if transform is None:transform=np.eye(len(parameters))
        rows=[]
        for band in self.bands:
            _,_,_,amplitude,design,derivative=band.evaluate(parameters,True)
            a=np.r_[design.real,design.imag][:,amplitude>0]
            derivative=np.r_[derivative.real,derivative.imag]@transform
            derivative-=a@np.linalg.lstsq(a,derivative,rcond=1e-10)[0]
            rows.append(derivative)
        j=np.concatenate(rows);n=self.geometry.count
        prior=np.diag(np.r_[np.ones(n),np.zeros(transform.shape[1]-n)])
        return j.T@j+prior,j

    def objective(self,parameters):
        terms = [band.evaluate(parameters) for band in self.bands]
        n = self.geometry.count
        value = sum(b.energy*t[0] for b,t in zip(self.bands,terms))+parameters[:n]@parameters[:n]
        gradient = sum(b.energy*t[1] for b,t in zip(self.bands,terms))
        gradient[:n] += 2*parameters[:n]
        return value/self.energy,gradient/self.energy

    def fit(self,seeds):
        g = self.geometry
        lower,upper = g.bounds()
        bounds = [*zip(lower,upper),*[(None,None)]*5]
        results = []
        for seed in seeds:
            p = np.r_[seed,np.zeros(5)]
            p[g.count:] = [b.phase_seed(p) for b in self.bands]
            results.append(minimize(self.objective,p,jac=True,method='L-BFGS-B',bounds=bounds,
                options={'gtol':2e-8,'ftol':2e-13,'maxiter':300,'maxcor':25}))
        return min(results,key=lambda r:r.fun),results

    def prediction(self,parameters,testing):
        result=[]
        for b,test in zip(self.bands,testing):
            _,_,design,amplitude,_,_=b.evaluate(parameters,True)
            y=b.field.y.ravel();pred=design@amplitude
            beta=b.nuisance.solve(y[b.selected]-pred[b.selected])
            clean=y-b.design@beta
            use=np.repeat(test,len(b.field.offsets))
            baseline=b.nuisance.solve(y[b.selected])
            error=float(np.sum(abs(clean[use]-pred[use])**2))
            baseline_error=float(np.sum(abs(y[use]-b.design[use]@baseline)**2))
            raw,_=b.response(parameters)
            unit=b.field.apply(average(raw,b.field.data['row_information'],b.field.metadata['block_rows'])[0])
            row_product=np.sum(unit.conj()*clean.reshape(b.field.y.shape),axis=1)
            result.append({'test_error':error,'baseline_error':baseline_error,'test_complex_prediction_fraction':1-error/baseline_error,
                'time_s':b.field.time.tolist(),'testing':test.tolist(),'phase_residual_deg':np.degrees(np.angle(row_product)).tolist(),
                'amplitude_coefficients':amplitude.tolist()})
        return result
