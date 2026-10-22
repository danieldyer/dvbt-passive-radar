"""Profile a complex scene while fitting physical range origin and path rate.

The rate changes the envelope, carrier phase and symbol Doppler at the original
row epochs. A constant carrier phase is absorbed into the complex scene, so it
is not added as a redundant calibration variable. These effective parameters
are not independently measured transmitter clock or antenna-location errors.
"""
import numpy as np
from scipy.optimize import minimize

from stationary import Stationary


class ProfiledAlignment:
    scale = np.array([20., .02])

    def __init__(self, band, phase_epoch_s, ranges=None):
        self.band, self.epoch = band, phase_epoch_s
        field = band.field
        ranges = np.arange(field.y.shape[1]) if ranges is None else np.asarray(ranges)
        mask = np.zeros(field.y.shape, bool)
        mask[np.ix_(band.selected_time, ranges)] = True
        self.selected = mask.ravel()
        self.nuisance = Stationary(band.nuisance.design, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)
        self.last = None

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        origin, rate = np.asarray(parameters)*self.scale
        target, derivatives = self.band.design(origin, derivatives=True, path_rate_bias_mps=rate,
            phase_epoch_s=self.epoch)
        design = self.nuisance.remove(target[self.selected])
        u, singular, vh = np.linalg.svd(design, full_matrices=False)
        if singular[-1] <= singular[0]*1e-10:
            raise ValueError('Range/rate control needs a full-rank complex point dictionary')
        coefficient = vh.conj().T@((u.conj().T@self.clean)/singular)
        residual = self.clean-design@coefficient
        derivative = derivatives[self.selected]*self.scale
        fixed = self.nuisance.remove(np.einsum('npq,p->nq', derivative, coefficient))
        cross = np.einsum('npq,n->pq', derivative.conj(), residual)
        jacobian = -fixed+u@(u.conj().T@fixed)-u@((vh@cross)/singular[:, None])
        self.last = {'parameters': np.asarray(parameters).copy(), 'target': target, 'coefficient': coefficient,
            'value': float(np.vdot(residual, residual).real/self.energy), 'residual': residual,
            'jacobian': jacobian, 'gradient': 2*np.real(jacobian.conj().T@residual)/self.energy,
            'condition': float(singular[0]/singular[-1])}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']

    def fit(self, initial_origin):
        lower, upper = np.array([-75., -.1])/self.scale, np.array([75., .1])/self.scale
        trials = []
        for rate in (-.08, -.04, 0., .04, .08):
            initial = np.array([initial_origin, rate])/self.scale
            fit = minimize(self.objective, initial, jac=True, method='L-BFGS-B', bounds=list(zip(lower, upper)),
                options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 120, 'maxls': 30})
            gradient = fit.jac.copy()
            gradient[((fit.x <= lower+1e-8)&(gradient > 0))|((fit.x >= upper-1e-8)&(gradient < 0))] = 0
            trials.append({'initial_rate_mps': rate, 'parameters': fit.x.tolist(), 'objective': float(fit.fun),
                'maximum_projected_gradient': float(abs(gradient).max()), 'iterations': fit.nit,
                'termination': str(fit.message)})
        best = min(trials, key=lambda t: t['objective'])
        parameters = np.array(best['parameters'])
        final = self.evaluate(parameters)
        step = 1e-5
        hessian = np.column_stack([(self.objective(parameters+step*d)[1]-
            self.objective(parameters-step*d)[1])/(2*step) for d in np.eye(2)])
        origin, rate = parameters*self.scale
        return final, {'converged': best['maximum_projected_gradient'] < 1e-7,
            'maximum_projected_gradient': best['maximum_projected_gradient'],
            'objective': final['value'], 'range_origin_m': float(origin), 'path_rate_mps': float(rate),
            'phase_epoch_s': self.epoch, 'trials': trials, 'scene_condition': final['condition'],
            'curvature_eigenvalues': np.linalg.eigvalsh((hessian+hessian.T)/2).tolist(),
            'bound_hit': bool(np.any(parameters <= lower+1e-7) or np.any(parameters >= upper-1e-7)),
            'scope': __doc__}
