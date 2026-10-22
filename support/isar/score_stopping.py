"""Stop a local motion solve using its information-normalized score.

The caller still independently checks the final original-row gradient. The
criterion controls optimization error in fixed Fisher coordinates and does
not change the objective, observations, trials, initialization or final gate.
"""
import time

import numpy as np
from scipy.optimize import minimize


def fit(joint, initial, transform, *, iterations, method='BFGS', progress=None, score_norm=.005):
    if not 0 < score_norm < .01:
        raise ValueError('The inner score threshold must be positive and below the .01 final gate')
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
        if record['score_norm'] <= score_norm:
            raise StopIteration
    start_value, start_gradient = objective(np.zeros(len(initial)))
    if np.linalg.norm(start_gradient) <= score_norm:
        parameters = initial.copy()
        status = {'scipy_success': None, 'termination': 'Initial point satisfies the declared score gate',
                  'iterations': 0, 'evaluations': 1}
    else:
        options = {'gtol': score_norm/np.sqrt(len(initial)), 'maxiter': iterations}
        extra = {}
        if method == 'trust-exact':
            extra['hess'] = hessian
            options.update(initial_trust_radius=1., max_trust_radius=1e6)
        fitted = minimize(objective, np.zeros(len(initial)), jac=True, method=method,
            callback=callback, options=options, **extra)
        parameters = initial+transform@fitted.x
        status = {'scipy_success': bool(fitted.success), 'termination': str(fitted.message),
            'iterations': int(fitted.nit), 'evaluations': int(fitted.nfev)}
    value, gradient = joint.objective(parameters)
    final = float(np.linalg.norm(transform.T@gradient))
    return parameters, {**status, 'objective': value, 'information_normalized_score_norm': final,
        'score_threshold': score_norm, 'score_converged': final <= score_norm,
        'method': method, 'trace': trace, 'stopping_scope': __doc__,
        'curvature_scope': 'Updated augmented-residual Gauss-Newton search curvature; not an exact marginal Hessian or Fisher-information claim.'}
