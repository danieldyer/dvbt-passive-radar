"""Physical marginal likelihood with an updated local search curvature.

The augmented-residual Gauss-Newton matrix is a trust-region search model,
not a Fisher matrix or a claimed exact marginal-likelihood Hessian. Acceptance
uses the actual marginal likelihood, and stationarity uses its exact gradient.
"""
import time

import numpy as np
from scipy.optimize import minimize

from marginal_scene import profile
from marginal_score import ScoreMotion


class CurvatureMotion(ScoreMotion):
    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        curvature = np.zeros((len(parameters), len(parameters)))
        for band, scale, core, strength in zip(self.bands, self.scales, self.cores, self.strengths):
            local = profile(band, motion, scale, core, strength, derivatives=True)
            value += local['value']
            gradient += local['gradient'][self.active]
            curvature += local['gauss_newton'][np.ix_(self.active, self.active)]
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value,
            'gradient': gradient, 'gauss_newton': curvature, 'evaluations': self.evaluations}
        return self.last


def fit(joint, initial, transform, *, iterations, method='BFGS', progress=None):
    started, trace = time.monotonic(), []
    def objective(z):
        value, gradient = joint.objective(initial+transform@z)
        return value, transform.T@gradient
    def hessian(z):
        matrix = transform.T@joint.evaluate(initial+transform@z)['gauss_newton']@transform
        return (matrix+matrix.T)/2
    def callback(z):
        value, gradient = objective(z)
        record = {'iteration': len(trace)+1, 'objective': value,
            'score_norm': float(np.linalg.norm(gradient)), 'evaluations': joint.evaluations,
            'seconds': time.monotonic()-started}
        trace.append(record)
        if progress:
            progress(record)
    options = {'gtol': 1e-3, 'maxiter': iterations}
    extra = {}
    if method == 'trust-exact':
        extra['hess'] = hessian
        options.update(initial_trust_radius=1., max_trust_radius=1e6)
    fitted = minimize(objective, np.zeros(len(initial)), jac=True, method=method,
        callback=callback, options=options, **extra)
    parameters = initial+transform@fitted.x
    value, gradient = joint.objective(parameters)
    return parameters, {'objective': value, 'information_normalized_score_norm': float(np.linalg.norm(transform.T@gradient)),
        'scipy_success': bool(fitted.success), 'termination': str(fitted.message), 'method': method,
        'iterations': int(fitted.nit), 'evaluations': int(fitted.nfev), 'trace': trace,
        'curvature_scope': __doc__ if method == 'trust-exact' else 'BFGS updates, initialized with the fixed Fisher coordinate scaling.'}
