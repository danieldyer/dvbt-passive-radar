"""Two physical local echoes, with no freely varying phase per time bin.

Each echo uses Cluster's delay/phase/moment operator. Two nonnegative real
strengths per retained bin are profiled exactly. Delay and carrier migration
remain coupled within each echo. This tests unresolved-target interference;
it is not a prior assertion of two particular physical ship scatterers.
"""
import numpy as np
from scipy.optimize import minimize

from cluster_phase import Cluster, Fit


def strengths(templates, observed):
    """Exact two-column real nonnegative least squares, independently per bin."""
    if templates.shape[:2] != observed.shape or templates.shape[-1] != 2:
        raise ValueError('Two templates must have the same time/range support as the observations')
    gram = np.einsum('nrj,nrk->njk', templates.conj(), templates).real
    cross = np.einsum('nrj,nr->nj', templates.conj(), observed).real
    a, b, d = gram[:, 0, 0], gram[:, 0, 1], gram[:, 1, 1]
    determinant = a*d-b*b
    candidates = np.zeros((len(observed), 4, 2))
    candidates[:, 1, 0] = np.maximum(cross[:, 0], 0)/np.maximum(a, 1e-300)
    candidates[:, 2, 1] = np.maximum(cross[:, 1], 0)/np.maximum(d, 1e-300)
    usable = determinant > 1e-13*np.maximum(a*d, 1e-300)
    both = np.column_stack([d*cross[:, 0]-b*cross[:, 1], a*cross[:, 1]-b*cross[:, 0]])
    both = np.divide(both, determinant[:, None], out=np.zeros_like(both), where=usable[:, None])
    candidates[:, 3] = np.where((usable & np.all(both >= 0, axis=1))[:, None], both, 0.)
    improvement = 2*np.einsum('nkj,nj->nk', candidates, cross)-np.einsum('nki,nij,nkj->nk', candidates, gram, candidates)
    return candidates[np.arange(len(observed)), improvement.argmax(axis=1)]


class Pair:
    def __init__(self, field, training, degree=2):
        self.single = Cluster(field, training, degree)
        self.field, self.training = field, self.single.training
        self.size = degree+2
        self.energy = self.single.energy

    def evaluate(self, parameters, full=False):
        pieces = [self.single.template(p, True) for p in np.asarray(parameters).reshape(2, self.size)]
        h = np.stack([p[0] for p in pieces], axis=-1)
        amplitude = strengths(h, self.field.y)
        residual = self.field.y-np.einsum('nrj,nj->nr', h, amplitude)
        use = self.training
        value = float(np.sum(abs(residual[use])**2)/self.energy)
        gradient = np.concatenate([-2*np.real(np.einsum('nr,nrp,n->p', residual[use].conj(),
            p[1][use], amplitude[use, j]))/self.energy for j, p in enumerate(pieces)])
        if full:
            return value, gradient, h, pieces, amplitude, residual
        return value, gradient

    def initial(self, single_parameters, origin_bounds, rate_bound, count=5):
        """Find the second echo in training residuals only; preserve observations."""
        seed = np.asarray(single_parameters)
        h = self.single.template(seed)
        product = np.real(np.sum(h.conj()*self.field.y, axis=1))
        amplitude = np.maximum(product, 0)/np.maximum(np.sum(abs(h)**2, axis=1), 1e-300)
        original = self.field.y
        try:
            self.field.y = original-amplitude[:, None]*h
            secondary = self.single.initial(origin_bounds, rate_bound, count)
        finally:
            self.field.y = original
        return [np.r_[seed, other] for other in secondary]

    def fit(self, seeds, origin_bounds, rate_bound=2.):
        bound = rate_bound*self.single.scale
        if len(origin_bounds) != 2:
            raise ValueError('One delay-origin interval per component is required')
        bounds = [b for origin in origin_bounds for b in [origin,(None,None),*([(-bound,bound)]*(self.size-2))]]
        trials = []
        for seed in seeds:
            fit = minimize(self.evaluate, seed, jac=True, method='L-BFGS-B', bounds=bounds,
                options={'gtol': 1e-8, 'ftol': 2e-13, 'maxiter': 400, 'maxcor': 20})
            trials.append(fit)
        best = min(trials, key=lambda x:x.fun)
        _, gradient, h, pieces, amplitude, _ = self.evaluate(best.x, True)
        use = self.training
        h = h[use]
        derivative = np.concatenate([p[1][use]*amplitude[use, j, None, None] for j,p in enumerate(pieces)], axis=2)
        # Conditional Fisher information profiles only active real strengths.
        h = h*(amplitude[use] > 0)[:, None, :]
        gram = np.einsum('nrj,nrk->njk', h.conj(), h).real
        cross = np.einsum('nrj,nrp->njp', h.conj(), derivative).real
        projection = np.linalg.pinv(gram, rcond=1e-12)@cross
        derivative -= np.einsum('nrj,njp->nrp', h, projection)
        information = 2*np.einsum('nrj,nrk->jk', derivative.conj(), derivative).real
        eigenvalues = np.linalg.eigvalsh(information)
        rank = int(np.count_nonzero(eigenvalues > eigenvalues[-1]*1e-10))
        def describe(t):
            low = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
            high = np.array([np.inf if b[1] is None else b[1] for b in bounds])
            active = ((t.x <= low+1e-7)&(t.jac>0))|((t.x >= high-1e-7)&(t.jac<0))
            return {'objective':float(t.fun), 'parameters':t.x.tolist(), 'termination':str(t.message),
                'converged':bool(t.success), 'iterations':int(t.nit),
                'bound_parameter_indices':np.flatnonzero((t.x<=low+1e-7)|(t.x>=high-1e-7)).tolist(),
                'maximum_projected_gradient':float(np.max(abs(np.where(active,0.,t.jac))))}
        return Fit(best.x, float(best.fun), float(abs(gradient).max()), [],
            [describe(t) for t in trials],
            np.linalg.pinv(information, rcond=1e-10), rank)

    def prediction(self, parameters):
        _, _, h, _, amplitude, _ = self.evaluate(parameters, True)
        # No test-bin strength or phase is used in these predictions.
        predicted = np.column_stack([np.interp(self.field.time, self.field.time[self.training],
            amplitude[self.training, j]) for j in range(2)])
        return np.einsum('nrj,nj->nr', h, predicted)

    def diagnostics(self, fit):
        prediction = self.prediction(fit.parameters)
        test = ~self.training
        error = float(np.sum(abs(self.field.y[test]-prediction[test])**2))
        baseline = float(np.sum(abs(self.field.y[test])**2))
        _, _, h, _, amplitude, residual = self.evaluate(fit.parameters, True)
        component_energy = np.sum(abs(h)**2, axis=1)*amplitude**2
        return {'test_squared_error':error, 'test_complex_prediction_fraction':1-error/baseline,
            'test_error_per_complex_cell':error/self.field.y[test].size,
            'training_residual_per_complex_cell':float(np.mean(abs(residual[self.training])**2)),
            'time_s':self.field.time.tolist(), 'training':self.training.tolist(),
            'amplitude':amplitude.tolist(), 'component_matched_energy':component_energy.tolist(),
            'scope':'Frozen delays and phase trajectories; withheld strengths interpolated from training bins only. Conditional thermal covariance; component labels are exchangeable.'}
