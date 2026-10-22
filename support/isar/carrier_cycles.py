"""Carrier-cycle proposals from the complete local paired-echo likelihood.

The quadratic includes all fitted motion, echo and prior correlations. Its
integer search is only an initializer: nonlinear projection enforces the
original bounds, and the original complex-data fit must judge every proposal.
No local Gauss--Newton metric is promoted to a calibration confidence region.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares,LinearConstraint,NonlinearConstraint,minimize

from bounded_quadratic import bounded_step
from physical_amplitude_arc import state_derivatives


def joint_mapping(common,independent,parameters):
    """Exact common-to-independent coordinate change, sharing antennas once."""
    mapped=np.zeros(independent.count);transform=np.zeros((independent.count,common.count))
    for arc,source,target in zip(common.arcs,common.indices,independent.indices):
        local,jacobian=arc.mapping(parameters[source])
        mapped[target]=local;transform[np.ix_(target,source)]=jacobian
    return mapped,transform


def cycle_constraints(common,independent,parameters):
    """Relative physical primary phases in cycles; common phases are integers."""
    values=[];rows=[];phase_indices=[]
    for arc,index in zip(common.arcs,independent.indices):
        local=parameters[index];phase=arc.phase_map;reference=phase.reference_band
        _,_,length,derivative=state_derivatives(arc.geometry,local[:28])
        physical=local[28:33]+phase.k*(length-phase.reference)
        for channel in range(5):
            if channel==reference:continue
            values.append((physical[channel]-physical[reference])/(2*np.pi))
            row=np.zeros(independent.count)
            row[index[:28]]=(phase.k[channel]*derivative[channel]-phase.k[reference]*derivative[reference])/(2*np.pi)
            row[index[28+channel]]=1/(2*np.pi);row[index[28+reference]]=-1/(2*np.pi)
            rows.append(row);phase_indices.append(int(index[28+channel]))
    return np.asarray(values),np.asarray(rows),np.asarray(phase_indices)


class CycleSurrogate:
    def __init__(self,independent,common,parameters):
        if len(independent.arcs)!=len(common.arcs):raise ValueError('Interval counts differ')
        if any(a.primary_count!=33 for a in independent.arcs) or any(a.primary_count!=29 for a in common.arcs):
            raise ValueError('Expected independent and common primary phases')
        self.independent,self.common=independent,common;self.origin=np.asarray(parameters).copy()
        residual,jacobian=independent.evaluate(self.origin)
        residual=residual*np.sqrt(independent.energy);jacobian=jacobian*np.sqrt(independent.energy)
        q,self.r=np.linalg.qr(jacobian,mode='reduced')
        singular=np.linalg.svd(self.r,compute_uv=False)
        if singular[-1]<=np.finfo(float).eps*max(jacobian.shape)*singular[0]:
            raise ValueError('Independent local likelihood is rank deficient; integer metric is not justified')
        self.condition=float(singular[0]/singular[-1])
        self.mean_step=solve_triangular(self.r,-q.T@residual)
        remaining=residual+jacobian@self.mean_step
        self.minimum_cost=float(remaining@remaining)
        self.value,self.derivative,self.phase_indices=cycle_constraints(common,independent,self.origin)
        b=solve_triangular(self.r.T,self.derivative.T,lower=True).T
        self.metric=b@b.T
        self.float_cycles=self.value+self.derivative@self.mean_step
        self.lift=solve_triangular(self.r,b.T)

    def displacement(self,integer):
        return self.mean_step+self.lift@np.linalg.solve(self.metric,np.asarray(integer)-self.float_cycles)

    def _common_seed(self,independent):
        result=np.zeros(self.common.count)
        for arc,source,target in zip(self.common.arcs,self.independent.indices,self.common.indices):
            local=independent[source]
            result[target]=np.r_[local[:28],local[28+arc.phase_map.reference_band],local[33:]]
        return np.clip(result,self.common.lower,self.common.upper)

    def project(self,integer,*,max_evaluations=150,bounded_initializer=False):
        """Bounded nonlinear geometry projection into the local quadratic.

        Integer labels specify an unwrapped phase branch only for this seed.
        The subsequent complex waveform fit is allowed to change branches.
        """
        integer=np.asarray(integer);step=self.displacement(integer);linear_audit=None
        if bounded_initializer:
            feasible=np.zeros(self.independent.count)
            feasible[self.phase_indices]=2*np.pi*(integer-self.value)
            step,linear_audit=bounded_step(self.r,self.derivative,integer-self.value,self.mean_step,
                self.independent.lower-self.origin,self.independent.upper-self.origin,feasible)
            if linear_audit['equality_error']>1e-7 or linear_audit['bound_error']>1e-7:
                raise ValueError('Bounded quadratic initializer failed feasibility')
        seed=self._common_seed(self.origin+step)
        shift=np.zeros(self.independent.count);shift[self.phase_indices]=2*np.pi*integer
        centre=self.origin+self.mean_step
        cached=None
        def evaluate(parameters):
            nonlocal cached
            if cached is not None and np.array_equal(cached[0],parameters):return cached[1:]
            mapped,transform=joint_mapping(self.common,self.independent,parameters)
            residual=self.r@(mapped+shift-centre);jacobian=self.r@transform
            cached=(parameters.copy(),residual,jacobian)
            return residual,jacobian
        fit=least_squares(lambda p:evaluate(p)[0],seed,jac=lambda p:evaluate(p)[1],
            bounds=(self.common.lower,self.common.upper),x_scale='jac',max_nfev=max_evaluations,
            ftol=1e-12,xtol=1e-12,gtol=1e-9)
        difference=integer-self.float_cycles
        cost=float(difference@np.linalg.solve(self.metric,difference))
        return dict(integer=integer.tolist(),parameters=fit.x.tolist(),linear_penalty=cost,
            bounded_linear_initializer=linear_audit,
            projected_surrogate_cost=self.minimum_cost+float(fit.fun@fit.fun),
            projection_optimality=float(fit.optimality),projection_evaluations=int(fit.nfev),
            projection_status=int(fit.status),initial_linear_step_norm=float(np.linalg.norm(step)))

    def project_whitened(self,integer,*,max_evaluations=200):
        """The same bounded nonlinear projection in unit-quadratic coordinates.

        Independent phases remain explicit variables, with eight exact phase
        closure constraints. This avoids the poor conditioning introduced by
        embedding all phase closure inside the Cartesian parameter map.
        """
        integer=np.asarray(integer);feasible=np.zeros(self.independent.count)
        feasible[self.phase_indices]=2*np.pi*(integer-self.value)
        lower=self.independent.lower-self.origin;upper=self.independent.upper-self.origin
        step,linear=bounded_step(self.r,self.derivative,integer-self.value,self.mean_step,lower,upper,feasible)
        if linear['equality_error']>1e-7 or linear['bound_error']>1e-7:
            raise ValueError('Bounded quadratic initializer failed feasibility')
        inverse=solve_triangular(self.r,np.eye(self.independent.count))
        mean=self.r@self.mean_step;initial=self.r@step
        scale=max(1.,float(np.linalg.norm(initial-mean)),float(np.linalg.norm(mean)))
        transform=inverse*scale;mean=mean/scale
        finite=np.isfinite(lower)|np.isfinite(upper)
        bound=transform[finite];norm=np.linalg.norm(bound,axis=1)
        bounds=LinearConstraint(bound/norm[:,None],lower[finite]/norm,upper[finite]/norm)
        phase_scale=np.linalg.norm(self.derivative@transform,axis=1)
        cached=None
        def closure(z):
            nonlocal cached
            if cached is not None and np.array_equal(cached[0],z):return cached[1:]
            value,derivative,_=cycle_constraints(self.common,self.independent,self.origin+transform@z)
            residual=(value-integer)/phase_scale;jacobian=(derivative@transform)/phase_scale[:,None]
            cached=(z.copy(),residual,jacobian)
            return residual,jacobian
        def objective(z):
            residual=z-mean
            return .5*float(residual@residual),residual
        constraint=NonlinearConstraint(lambda z:closure(z)[0],0.,0.,jac=lambda z:closure(z)[1])
        fit=minimize(objective,initial/scale,jac=True,method='SLSQP',constraints=[bounds,constraint],
            options={'maxiter':max_evaluations,'ftol':1e-13})
        independent=self.origin+transform@fit.x
        equality_error=float(np.max(abs(closure(fit.x)[0]*phase_scale)))
        bound_error=float(max(0.,np.max(self.independent.lower-independent),np.max(independent-self.independent.upper)))
        parameters=self._common_seed(independent)
        mapped,change=joint_mapping(self.common,self.independent,parameters)
        mapped[self.phase_indices]+=2*np.pi*integer
        residual=self.r@(mapped-self.origin-self.mean_step)
        gradient=change.T@(self.r.T@residual)
        active=((parameters<self.common.lower+1e-6)&(gradient>0))|((parameters>self.common.upper-1e-6)&(gradient<0))
        gradient[active]=0
        delta=integer-self.float_cycles
        return dict(integer=integer.tolist(),parameters=parameters.tolist(),
            linear_penalty=float(delta@np.linalg.solve(self.metric,delta)),bounded_linear_initializer=linear,
            projection_method='whitened equality constraints',projection_success=bool(fit.success),
            projection_status=1 if fit.success else (0 if fit.status==9 else -int(fit.status)),
            projection_message=str(fit.message),projection_evaluations=int(fit.nit),
            projection_equality_error=equality_error,projection_bound_error=bound_error,
            projection_optimality=float(abs(gradient).max()),
            projected_surrogate_cost=self.minimum_cost+float(residual@residual))
