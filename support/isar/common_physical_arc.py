"""Raw-data common-scattering-phase control, with explicit carrier branches.

This is an optimistic hypothesis: relative instrumental phase is known, and
the selected point has a common scattering phase at all five RF frequencies.
It must be checked against independently phased band responses and transfer.
The midpoint phases used to condition PhysicalAmplitudeArc are undone, with
their geometry derivatives, before imposing the common physical phase.
"""
import numpy as np
from scipy.optimize import LinearConstraint, least_squares, minimize, nnls
from scipy.linalg import solve_triangular

from physical_amplitude_arc import state_derivatives
from stationarity import polish


class CommonPhysicalFit:
    phase_count = 1

    def __init__(self, independent):
        self.independent = independent
        self.geometry = independent.geometry
        self.bands,self.energy=independent.bands,independent.energy
        self.k = np.array([b.field.k for b in independent.bands])
        self.reference = np.array([b.reference_path for b in independent.bands])

    def mapping(self, parameters):
        n = self.geometry.count
        _, _, length, derivative = state_derivatives(self.geometry, parameters[:n])
        phases = parameters[-1]-self.k*(length-self.reference)
        transform = np.zeros((n+5,n+1));transform[:n,:n] = np.eye(n)
        transform[n:,:n] = -self.k[:,None]*derivative;transform[n:,-1] = 1.
        return np.r_[parameters[:n],phases],transform

    def objective(self, parameters):
        mapped,transform = self.mapping(parameters)
        value,gradient = self.independent.objective(mapped)
        return value,transform.T@gradient

    def prediction(self, parameters, testing):
        return self.independent.prediction(self.mapping(parameters)[0],testing)

    def phase_branches(self, independent_parameters, *, adjacent=True,free_coordinates=None):
        """Local cycle hypotheses; not an exhaustive search of location priors.

        Fit small antenna corrections and a common phase to the independently
        fitted midpoint phases. For a frozen calibration, only the target
        position can move. Velocity and acceleration stay frozen here.
        Final ranking always uses the full original-sample training likelihood.
        """
        g,n = self.geometry,self.geometry.count
        seed = np.asarray(independent_parameters[:n])
        initial_length = g.prediction(seed)[:,0]
        phase = np.angle(np.exp(1j*(independent_parameters[n:]+self.k*(initial_length-self.reference))))
        breaks = np.sort(np.angle(np.exp(1j*(phase+np.pi))))
        edges = np.unique(np.r_[-np.pi,breaks,np.pi])
        origins = (edges[1:]+edges[:-1])/2
        patterns = [(origin,np.rint((origin-phase)/(2*np.pi)).astype(int)) for origin in origins]
        if adjacent:
            origin,base = patterns[len(patterns)//2]
            patterns += [(origin,base+sign*np.eye(5,dtype=int)[ch]) for ch in range(1,5) for sign in (-1,1)]
        active = (np.r_[np.arange(15),17] if g.free_antennas else np.arange(3)) if free_coordinates is None else np.asarray(free_coordinates,int)
        if active.ndim!=1 or not len(active) or len(np.unique(active))!=len(active) or np.any((active<0)|(active>=n)):
            raise ValueError('Distinct valid phase-lifting geometry coordinates required')
        lower,upper = g.bounds()
        def expand(x):
            q = seed.astype(np.result_type(x,seed));q[active] = x[:-1]
            return q
        records=[]
        for origin,cycles in patterns:
            def residual(x):
                q = expand(x)
                difference = g.prediction(q)[:,0]-initial_length
                # 0.1 rad merely conditions the phase-lifting initializer.
                # It is not a weight or covariance used in the raw-data fit.
                error = (x[-1]-self.k*difference-phase-2*np.pi*cycles)/.1
                return np.r_[error,x[:-1]-seed[active]]
            fit = least_squares(residual,np.r_[seed[active],origin],jac='cs',
                bounds=(np.r_[lower[active],-np.inf],np.r_[upper[active],np.inf]),
                ftol=1e-12,xtol=1e-12,gtol=1e-10,max_nfev=100,x_scale='jac')
            records.append({'parameters':np.r_[expand(fit.x),fit.x[-1]],'carrier_cycles':cycles.tolist(),
                'phase_lift_cost':float(fit.fun@fit.fun),'phase_lift_gradient':float(fit.optimality)})
        return records

    def metric(self, parameters):
        """Mean-response metric after static and positive-amplitude nuisance."""
        mapped,transform = self.mapping(parameters)
        return self.independent.metric(mapped,transform)

    def fit(self, seed):
        n = self.geometry.count
        normal,_ = self.metric(seed)
        root = np.linalg.cholesky(normal)
        transform = solve_triangular(root.T,np.eye(len(seed)),lower=False)*np.sqrt(self.independent.energy)
        lower,upper = self.geometry.bounds()
        constraint = LinearConstraint(transform[:n],lower-seed[:n],upper-seed[:n])
        def objective(x):
            value,gradient = self.objective(seed+transform@x)
            return value,transform.T@gradient
        fit = minimize(objective,np.zeros(len(seed)),jac=True,method='SLSQP',constraints=[constraint],
            options={'ftol':2e-13,'maxiter':250})
        parameters = seed+transform@fit.x
        value,gradient = self.objective(parameters)
        active = np.r_[np.eye(len(seed))[:n][parameters[:n]>upper-1e-6],
                       -np.eye(len(seed))[:n][parameters[:n]<lower+1e-6]]
        multiplier = nnls((active@transform).T,-transform.T@gradient)[0] if len(active) else np.empty(0)
        residual = transform.T@gradient+(active@transform).T@multiplier
        violation = float(max(0.,np.max(lower-parameters[:n]),np.max(parameters[:n]-upper)))
        return {'parameters':parameters,'objective':value,'iterations':fit.nit,'termination':str(fit.message),
            'maximum_scaled_kkt_residual':float(abs(residual).max()),'constraint_violation':violation,
            'converged':bool(fit.success and abs(residual).max()<1e-6 and violation<1e-7)}

    def polish(self, parameters):
        """Audit/correct a local solution on its current constraint face."""
        n=self.geometry.count
        normal,_=self.metric(parameters)
        root=np.linalg.cholesky(normal)
        transform=solve_triangular(root.T,np.eye(len(parameters)),lower=False)*np.sqrt(self.independent.energy)
        lower,upper=self.geometry.bounds()
        def objective(x):
            value,gradient=self.objective(parameters+transform@x)
            return value,transform.T@gradient
        z,report=polish(objective,np.zeros(len(parameters)),np.eye(len(parameters)),transform[:n],
            lower-parameters[:n],upper-parameters[:n],np.full(len(parameters),-np.inf),np.full(len(parameters),np.inf))
        return parameters+transform@z,report
