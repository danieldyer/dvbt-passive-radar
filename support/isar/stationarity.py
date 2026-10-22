"""Local constrained Newton correction after an objective line search stalls.

This does not loosen the convergence gate. It solves the analytic gradient on
the current constraint face, shortens a step that reaches another face, then
rebuilds its tangent. Curvature, feasibility and the complete KKT residual are
checked; a correction is rejected unless that residual improves.
"""
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import nnls


def check(parameters, gradient, matrix, lo, hi, lower, upper):
    free = lower < upper
    v = matrix@parameters
    active = np.r_[matrix[v >= hi-1e-6], -matrix[v <= lo+1e-6],
        np.eye(len(parameters))[(parameters >= upper-1e-6)&free],
        -np.eye(len(parameters))[(parameters <= lower+1e-6)&free]][:, free]
    multiplier = nnls(active.T, -gradient[free])[0] if len(active) else np.empty(0)
    residual = gradient[free]+(active.T@multiplier if len(active) else 0)
    violation = max(0., np.max(np.r_[lo-v, v-hi, lower-parameters, parameters-upper]))
    return float(abs(residual).max()), float(violation)


def polish(objective, parameters, transform, matrix, lo, hi, lower, upper, *, curvature_gradient=None):
    parameters = np.asarray(parameters).copy()
    value, gradient = objective(parameters)
    initial_residual, violation = check(parameters, gradient, matrix, lo, hi, lower, upper)
    report = {'initial_kkt_residual': initial_residual, 'accepted': False, 'steps': [],
        'curvature_source': 'supplied search approximation' if curvature_gradient else 'exact objective gradient'}
    search_gradient = curvature_gradient or (lambda p: objective(p)[1])
    if violation > 1e-7:
        report['reason'] = 'Initial point is not feasible'
        return parameters, report
    free = lower < upper
    best = initial_residual
    for iteration in range(3):
        v = matrix@parameters
        active = np.r_[matrix[(v >= hi-1e-7)|(v <= lo+1e-7)],
                      np.eye(len(parameters))[((parameters >= upper-1e-7)|(parameters <= lower+1e-7))&free]]
        tangent = transform@null_space(active@transform, rcond=1e-10)
        if tangent.shape[1] == 0:
            report['reason'] = 'No free tangent direction'
            break
        # Differencing gradients avoids sub-ulp objective differences. These
        # information-scaled probes never commit an infeasible physical fit.
        step = 1e-5
        columns = []
        for direction in tangent.T:
            plus = search_gradient(parameters+step*direction)
            minus = search_gradient(parameters-step*direction)
            columns.append(tangent.T@(plus-minus)/(2*step))
        hessian = np.column_stack(columns)
        eigen, vectors = np.linalg.eigh((hessian+hessian.T)/2)
        report.update(tangent_dimensions=tangent.shape[1], curvature_eigenvalues=eigen.tolist(), gradient_difference_step=step)
        if eigen.min() <= eigen.max()*1e-8:
            report['reason'] = 'Tangent Hessian is not reliably positive definite'
            break
        inverse = (vectors/eigen)@vectors.T
        delta = -tangent@(inverse@(tangent.T@gradient))
        # Active faces are already preserved by the tangent. Only inactive
        # faces can shorten this step; ignoring round-off in a zero normal
        # component prevents spurious zero-length corrections at active faces.
        change = matrix@delta
        slack = np.r_[hi-v, v-lo, upper-parameters, parameters-lower]
        toward = np.r_[change, -change, delta, -delta]
        approach = (slack > 1e-7)&(toward > 0)
        fraction = min(1., float(np.min(slack[approach]/toward[approach]))) if approach.any() else 1.
        proposal = parameters+fraction*delta
        candidate, derivative = objective(proposal)
        residual, violation = check(proposal, derivative, matrix, lo, hi, lower, upper)
        accepted = residual < best and violation <= 1e-7 and candidate <= value+1e-10
        report['steps'].append({'objective': candidate, 'kkt_residual': residual, 'violation': violation,
                                'maximum_parameter_step': float(abs(fraction*delta).max()),
                                'feasible_step_fraction': fraction, 'accepted': bool(accepted)})
        if not accepted:
            break
        parameters, value, gradient, best = proposal, candidate, derivative, residual
        report['accepted'] = True
        if best < 1e-8:
            break
    report['final_kkt_residual'] = best
    return parameters, report
