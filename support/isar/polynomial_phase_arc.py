"""Short phase-equivalent paths with explicit nonnegative brightness.

Fit original complex range-cell samples, not unwrapped or averaged phase.
Each band has an unknown constant scattering phase. A polynomial residual
path may be common across bands or independent. Real brightness is profiled
at each training epoch, then interpolated for unused-block prediction.

This tests local phase compatibility. It does not identify a material point:
unresolved echoes and a changing complex range response can violate the model.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class Band:
    time: np.ndarray
    value: np.ndarray
    weight: np.ndarray
    known_path: np.ndarray
    k: float
    training: np.ndarray
    row_time: np.ndarray | None = None
    row_weight: np.ndarray | None = None
    row_known_path: np.ndarray | None = None
    block_rows: int = 16


class PolynomialPhaseArc:
    def __init__(self,bands,first,last,degree=2):
        if not bands or degree not in (1,2,3) or not last>first:
            raise ValueError('Nonempty bands, ordered bounds and degree 1–3 required')
        self.bands=bands;self.epoch=(first+last)/2;self.scale=(last-first)/2;self.degree=degree
        self.count=len(bands)+degree;self.last=None
        self.basis=[];self.row_basis=[]
        for b in bands:
            if (b.time.ndim!=1 or any(x.shape!=b.time.shape for x in (b.value,b.weight,b.known_path,b.training))
                    or np.any(np.diff(b.time)<=0) or np.any(b.weight<0) or b.k<=0
                    or np.count_nonzero(b.training)<degree+3):
                raise ValueError('Matching ordered band arrays and enough training samples required')
            u=(b.time-self.epoch)/self.scale
            self.basis.append(np.column_stack([u**j for j in range(1,degree+1)]))
            if b.row_time is None:self.row_basis.append(None)
            else:
                if (b.row_weight is None or b.row_known_path is None or
                    b.row_time.shape!=b.row_weight.shape or b.row_time.shape!=b.row_known_path.shape or
                    (len(b.row_time)+b.block_rows-1)//b.block_rows!=len(b.time)):
                    raise ValueError('Original rows must match complete retained-bin support')
                u=(b.row_time-self.epoch)/self.scale
                self.row_basis.append(np.column_stack([u**j for j in range(1,degree+1)]))
        self.energy=sum(float(np.sum(b.weight[b.training]*abs(b.value[b.training])**2)) for b in bands)
        if not self.energy>0:raise ValueError('Positive training energy required')
        self.lower=np.r_[np.full(len(bands),-np.inf),[-3*self.scale,-self.scale**2,-self.scale**3/6][:degree]]
        self.upper=-self.lower

    def phase(self,parameters,index):
        b=self.bands[index]
        return parameters[index]-b.k*(b.known_path+self.basis[index]@parameters[len(self.bands):])

    def response(self,parameters,index):
        """Exact saved information-weighted original-row phase averaging."""
        b=self.bands[index];basis=self.row_basis[index]
        if basis is None:
            h=np.exp(1j*self.phase(parameters,index))
            design=np.zeros((len(h),self.count));design[:,index]=1
            design[:,len(self.bands):]=-b.k*self.basis[index]
            return h,1j*h[:,None]*design
        phase=parameters[index]-b.k*(b.row_known_path+basis@parameters[len(self.bands):])
        raw=np.exp(1j*phase);starts=np.arange(0,len(raw),b.block_rows)
        weight=np.add.reduceat(b.row_weight,starts)
        design=np.zeros((len(raw),self.count));design[:,index]=1
        design[:,len(self.bands):]=-b.k*basis
        h=np.divide(np.add.reduceat(raw*b.row_weight,starts),weight,
            out=np.zeros(len(starts),complex),where=weight>0)
        derivative=np.divide(np.add.reduceat((1j*raw*b.row_weight)[:,None]*design,starts,axis=0),weight[:,None],
            out=np.zeros((len(starts),self.count),complex),where=weight[:,None]>0)
        return h,derivative

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];derivatives=[]
        for c,b in enumerate(self.bands):
            use=b.training&(b.weight>0);h,dh=self.response(parameters,c);h,dh=h[use],dh[use]
            norm=np.maximum(abs(h)**2,1e-300);cross=(h.conj()*b.value[use]).real
            amplitude=np.maximum(0.,cross)/norm
            dcross=(dh.conj()*b.value[use,None]).real;dnorm=2*(h.conj()[:,None]*dh).real
            da=(dcross-amplitude[:,None]*dnorm)/norm[:,None];da[cross<=0]=0
            residual=(b.value[use]-amplitude*h)*np.sqrt(b.weight[use]/self.energy)
            jacobian=-(amplitude[:,None]*dh+h[:,None]*da)*np.sqrt(b.weight[use,None]/self.energy)
            residuals.extend([residual.real,residual.imag]);derivatives.extend([jacobian.real,jacobian.imag])
        residual=np.concatenate(residuals);jacobian=np.concatenate(derivatives)
        self.last=(np.asarray(parameters).copy(),residual,jacobian)
        return residual,jacobian

    def seeds(self,extra=()):
        rates=np.linspace(-3,3,121);score=np.zeros(len(rates));phases=[]
        for b in self.bands:
            use=b.training&(b.weight>0);t=b.time[use]-self.epoch;w=b.weight[use]
            product=(w*b.value[use]*np.exp(1j*b.k*b.known_path[use]))@np.exp(1j*b.k*t[:,None]*rates)
            phases.append(np.angle(product));score+=abs(product)**2/w.sum()
        peaks=np.r_[0,np.flatnonzero((score[1:-1]>=score[:-2])&(score[1:-1]>=score[2:]))+1,len(rates)-1]
        order=peaks[np.argsort(-score[peaks])[:5]]
        seeds=[np.r_[[p[i] for p in phases],rates[i]*self.scale,np.zeros(self.degree-1)] for i in order]
        # The unwrap is an initial guess only. All optimization and prediction
        # below use the original complex likelihood without integer decisions.
        estimates=[];strength=[]
        for c,b in enumerate(self.bands):
            use=b.training&(b.weight>0);phase=np.unwrap(np.angle(b.value[use]*np.exp(1j*b.k*b.known_path[use])))
            w=np.sqrt(b.weight[use])*abs(b.value[use]);design=np.column_stack([np.ones(use.sum()),-b.k*self.basis[c][use]])
            q=np.linalg.lstsq(w[:,None]*design,w*phase,rcond=None)[0]
            estimates.append(q[1:]);strength.append(np.sum(w**2))
        common=np.average(estimates,axis=0,weights=strength)
        for coefficients in (np.zeros(self.degree),common,*extra):
            coefficients=np.clip(np.asarray(coefficients),self.lower[len(self.bands):],self.upper[len(self.bands):])
            phase=[]
            for c,b in enumerate(self.bands):
                use=b.training&(b.weight>0)
                phase.append(np.angle(np.sum(b.weight[use]*b.value[use]*np.exp(1j*b.k*(b.known_path[use]+self.basis[c][use]@coefficients)))))
            seeds.append(np.r_[phase,coefficients])
        return seeds

    def fit(self,extra=(),max_evaluations=180):
        trials=[]
        for initial in self.seeds(extra):
            fit=least_squares(lambda p:self.evaluate(p)[0],initial,jac=lambda p:self.evaluate(p)[1],
                bounds=(self.lower,self.upper),ftol=2e-12,xtol=2e-12,gtol=2e-10,max_nfev=max_evaluations)
            r,j=self.evaluate(fit.x);gradient=2*j.T@r
            active=((fit.x<self.lower+1e-5)&(gradient>0))|((fit.x>self.upper-1e-5)&(gradient<0))
            projected=gradient.copy();projected[active]=0
            trials.append(dict(parameters=fit.x.tolist(),cost=float(r@r*self.energy),
                maximum_projected_gradient=float(abs(projected).max()),evaluations=fit.nfev,
                termination=str(fit.message),bound_indices=np.flatnonzero(active).tolist()))
        selected=int(np.argmin([r['cost'] for r in trials]));result=trials[selected].copy()
        result.update(selected_start=selected,trials=trials,epoch_s=self.epoch,scale_s=self.scale,
            degree=self.degree,data_energy=self.energy,prediction=self.prediction(np.asarray(result['parameters'])))
        return result

    def prediction(self,parameters):
        records=[]
        for c,b in enumerate(self.bands):
            h,_=self.response(parameters,c);training=b.training&(b.weight>0);testing=~b.training&(b.weight>0)
            amplitude=np.maximum(0.,(b.value[training]*h[training].conj()).real)/np.maximum(abs(h[training])**2,1e-300)
            predicted_amplitude=np.interp(b.time,b.time[training],amplitude,left=np.nan,right=np.nan)
            testing&=np.isfinite(predicted_amplitude)
            predicted=predicted_amplitude*h
            error=float(np.sum(b.weight[testing]*abs(b.value[testing]-predicted[testing])**2))
            baseline=float(np.sum(b.weight[testing]*abs(b.value[testing])**2))
            delta=np.angle(b.value*h.conj());weights=b.weight[testing]*abs(predicted[testing])**2
            phase_rms=float(np.degrees(np.sqrt(np.average(delta[testing]**2,weights=weights)))) if weights.sum()>0 else None
            cross=np.sum(b.weight[testing]*predicted[testing].conj()*b.value[testing])
            norm=float(np.sum(b.weight[testing]*abs(predicted[testing])**2))
            coherence=float(abs(cross)**2/(norm*baseline)) if norm*baseline>0 else None
            records.append(dict(test_error=error,test_energy=baseline,explained_fraction=1-error/baseline if baseline>0 else None,
                test_phase_rms_deg=phase_rms,test_coherence_squared=coherence,testing=testing.tolist(),
                time_s=b.time.tolist(),phase_residual_deg=np.degrees(delta).tolist(),
                predicted_amplitude=predicted_amplitude.tolist(),training=b.training.tolist()))
        return records

    def trajectory(self,parameters,time,derivative=0):
        coefficients=np.r_[0.,np.asarray(parameters)[len(self.bands):]]
        polynomial=np.polynomial.polynomial.polyder(coefficients,derivative)
        return np.polynomial.polynomial.polyval((np.asarray(time)-self.epoch)/self.scale,polynomial)/self.scale**derivative
