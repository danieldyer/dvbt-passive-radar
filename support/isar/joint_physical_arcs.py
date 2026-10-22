"""Shared antenna coordinates, separate physical target states and phases.

Each interval keeps the exact original-row profiled likelihood. The antenna
prior is counted once. No band phase-rate adjustments or per-interval antenna
positions are introduced. These offline fits do not alter captured data.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from profiled_arc_solver import ProfiledArcSolver
from stationarity import polish


class JointPhysicalArcs:
    def __init__(self,arcs):
        if not arcs or any(a.geometry.count!=28 or not a.geometry.free_antennas for a in arcs):
            raise ValueError('Each arc must expose the same 18 antenna and 10 motion parameters')
        self.arcs=arcs;self.solvers=[ProfiledArcSolver(a) for a in arcs]
        self.count=18;self.indices=[]
        for arc in arcs:
            size=10+arc.phase_count
            self.indices.append(np.r_[np.arange(18),self.count+np.arange(size)])
            self.count+=size
        self.energy=sum(a.energy for a in arcs)
        self.prior=np.zeros(self.count,bool);self.prior[:18]=True
        for index in self.indices:self.prior[index[18:28]]=True
        self.lower=np.full(self.count,-np.inf);self.upper=-self.lower
        for arc,index in zip(arcs,self.indices):
            lo,hi=arc.geometry.bounds()
            self.lower[index[:28]]=np.maximum(self.lower[index[:28]],lo)
            self.upper[index[:28]]=np.minimum(self.upper[index[:28]],hi)
        self.last=None

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last[0]):return self.last[1:]
        residuals=[];jacobians=[]
        for solver,index in zip(self.solvers,self.indices):
            r,j=solver.evaluate(parameters[index])
            scale=np.sqrt(solver.model.energy/self.energy)
            residuals.append(r[:-28]*scale)
            derivative=np.zeros((len(r)-28,self.count));derivative[:,index]=j[:-28]*scale
            jacobians.append(derivative)
        residuals.append(parameters[self.prior]/np.sqrt(self.energy))
        jacobians.append(np.eye(self.count)[self.prior]/np.sqrt(self.energy))
        value=np.concatenate(residuals);derivative=np.concatenate(jacobians)
        self.last=(parameters.copy(),value,derivative)
        return value,derivative

    def objective(self,parameters):
        residual,jacobian=self.evaluate(parameters)
        return float(residual@residual),2*jacobian.T@residual

    def information(self,parameters):
        """Conditional mean-response information, without coordinate priors."""
        rows=[]
        for arc,index in zip(self.arcs,self.indices):
            _,local=arc.metric(parameters[index])
            derivative=np.zeros((len(local),self.count));derivative[:,index]=local*np.sqrt(2.)
            rows.append(derivative)
        jacobian=np.concatenate(rows)
        def span(columns):
            scale=np.linalg.norm(columns,axis=0)
            columns=columns[:,scale>0]/scale[scale>0]
            if not columns.shape[1]:return np.empty((len(jacobian),0))
            u,s,_=np.linalg.svd(columns,full_matrices=False)
            tolerance=np.finfo(float).eps*max(columns.shape)*s[0]
            return u[:,s>tolerance]
        # Remove the explicit common phases first. The physical and midpoint
        # phase charts differ by precisely these columns; treating tiny motion
        # singular values relative to the unprojected RF phase is not invariant
        # to that harmless coordinate change.
        phase=span(jacobian[:,np.concatenate([index[28:] for index in self.indices])])
        jacobian-=phase@(phase.T@jacobian)
        antennas=jacobian[:,np.r_[np.arange(15),17]]
        # At one epoch, the AIS time shift changes only p/v/a. Those nine
        # Cartesian coordinates already span it exactly, so do not retain a
        # numerically redundant tenth motion column as extra information.
        motion=np.concatenate([index[18:27] for index in self.indices])
        q=span(jacobian[:,motion])
        projected=antennas-q@(q.T@antennas)
        singular=np.linalg.svd(projected,compute_uv=False)
        return dict(matrix=(projected.T@projected).tolist(),eigenvalues=(singular**2).tolist(),
            nuisance_rank=q.shape[1]+phase.shape[1],antenna_coordinates=16,
            scope='Conditional thermal mean-response information after stationary, active real amplitudes and all target states/phases; no location priors and no use of active bounds as information.')

    def fit(self,seed,*,max_evaluations=400):
        seed=np.clip(seed,self.lower,self.upper)
        fit=least_squares(lambda p:self.evaluate(p)[0],seed,jac=lambda p:self.evaluate(p)[1],
            bounds=(self.lower,self.upper),x_scale='jac',max_nfev=max_evaluations,
            ftol=2e-13,xtol=2e-13,gtol=1e-10)
        p=fit.x;_,jac=self.evaluate(p)
        root=np.linalg.cholesky(jac.T@jac)
        transform=solve_triangular(root.T,np.eye(self.count),lower=False)
        finite=np.isfinite(self.lower)&np.isfinite(self.upper)
        def objective(z):
            value,gradient=self.objective(p+transform@z)
            return value,transform.T@gradient
        z,audit=polish(objective,np.zeros(self.count),np.eye(self.count),transform[finite],
            self.lower[finite]-p[finite],self.upper[finite]-p[finite],
            np.full(self.count,-np.inf),np.full(self.count,np.inf))
        final=p+transform@z
        return final,dict(evaluations=fit.nfev,termination=str(fit.message),optimality=float(fit.optimality),
            objective=self.objective(final)[0],polishing=audit,
            converged=bool(audit.get('final_kkt_residual',np.inf)<1e-6 and min(audit.get('curvature_eigenvalues',[-1]))>0),
            bound_indices=np.flatnonzero((final<self.lower+1e-4)|(final>self.upper-1e-4)).tolist())
