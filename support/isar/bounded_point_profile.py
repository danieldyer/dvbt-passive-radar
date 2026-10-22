"""Exact profiled point residuals with native physical-coordinate bounds.

This keeps the established physical forward model and variable-projection
derivative. The trust-region reflective solver never evaluates an out-of-box
trial. Independent physical-coordinate KKT checks govern acceptance.
"""
import time

import numpy as np
from scipy.optimize import least_squares

from point_profile import PointProfile, profile_points
from stationarity import check


class BoundedPointProfile(PointProfile):
    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        outputs, residuals, derivatives = [], [], []
        points = self.coordinates(parameters)
        for band, origin in zip(self.bands, self.origins):
            result = profile_points(band, points, origin)
            derivative = result.pop('jacobian')
            residuals.append(np.r_[result['residual'].real, result['residual'].imag])
            derivatives.append(np.r_[derivative.real, derivative.imag])
            outputs.append(result)
        residual = np.concatenate(residuals)/np.sqrt(self.energy)
        jacobian = np.concatenate(derivatives)/np.sqrt(self.energy)
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': float(residual@residual),
            'gradient': 2*jacobian.T@residual, 'residual': residual, 'jacobian': jacobian,
            'scene_conditions': [result['condition'] for result in outputs], 'bands': outputs}
        return self.last


def solve_points(problem, lower, upper, *, evaluations=180, checkpoint=None):
    lower, upper = np.broadcast_arrays(lower, upper)
    if lower.shape != problem.initial.shape or np.any(lower >= upper):
        raise ValueError('Use a finite-width physical-coordinate box')
    n = len(lower); empty = np.empty((0, n)); rhs = np.empty(0)
    started = time.monotonic(); trace = []
    first = problem.evaluate(problem.initial)
    initial_kkt, violation = check(problem.initial, first['gradient'], empty, rhs, rhs, lower, upper)
    if violation > 0:
        raise ValueError('Initial point lies outside physical bounds')
    def residual(parameters):
        # This is an assertion, not clipping or an out-of-domain penalty.
        if np.any(parameters < lower) or np.any(parameters > upper):
            raise ValueError('The bounded solver proposed an out-of-box evaluation')
        return problem.evaluate(parameters)['residual']
    def callback(intermediate_result):
        value = problem.evaluate(intermediate_result.x)
        kkt, violation = check(intermediate_result.x, value['gradient'], empty, rhs, rhs, lower, upper)
        trace.append({'iteration': len(trace)+1, 'objective': value['value'],
            'kkt_residual': kkt, 'constraint_violation': violation,
            'evaluations': problem.evaluations, 'seconds': time.monotonic()-started})
        if checkpoint:
            checkpoint(intermediate_result.x, trace)
        if kkt <= 1e-7 and violation <= 1e-7 and value['value'] <= first['value']:
            raise StopIteration
    result = least_squares(residual, problem.initial,
        jac=lambda p: problem.evaluate(p)['jacobian'], bounds=(lower, upper),
        method='trf', x_scale='jac', loss='linear', ftol=1e-13, xtol=1e-13, gtol=1e-13,
        max_nfev=evaluations, callback=callback)
    final = problem.evaluate(result.x)
    kkt, violation = check(result.x, final['gradient'], empty, rhs, rhs, lower, upper)
    fit = {'converged': bool(kkt <= 1e-7 and violation <= 1e-7),
        'termination': str(result.message), 'method': 'trust-region reflective, native physical bounds',
        'initial_objective': first['value'], 'objective': final['value'],
        'initial_kkt_residual': initial_kkt, 'maximum_kkt_residual': kkt,
        'maximum_constraint_violation': violation, 'iterations': len(trace),
        'evaluations': problem.evaluations, 'trace': trace,
        'scene_conditions': final['scene_conditions'], 'seconds': time.monotonic()-started}
    return result.x, fit
