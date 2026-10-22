"""Constant complex band gains with a shared regularized spatial scene.

The reference gain is one. Scene coefficients are eliminated with augmented
QR, not normal equations. Regularization and gain bounds are fixed by the
caller; they are not selected using prediction samples.
"""
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize


class GainRegistration:
    def __init__(self, designs, data, strength, *, reference=3,
                 orthogonal_energy=0., maximum_gain_ratio=10.):
        if not (np.isfinite(strength) and strength > 0 and
                np.isfinite(maximum_gain_ratio) and maximum_gain_ratio > 1):
            raise ValueError('Need positive regularization and a gain bound greater than one')
        if not 0 <= reference < len(designs) or len(designs) != len(data):
            raise ValueError('Invalid reference channel or observations')
        columns = designs[0].shape[1]
        if any(a.shape != (len(y), columns) for a, y in zip(designs, data)):
            raise ValueError('All channels must share the same spatial coefficients')
        self.designs, self.data = designs, data
        self.other = np.array([c for c in range(len(designs)) if c != reference])
        self.strength, self.limit = strength, np.log(maximum_gain_ratio)
        self.orthogonal_energy = orthogonal_energy
        self.energy = sum(np.vdot(y, y).real for y in data)+orthogonal_energy
        if self.energy <= 0 or orthogonal_energy < 0:
            raise ValueError('Need a nonzero observation and nonnegative discarded energy')
        self.penalty = np.sqrt(strength)*np.eye(columns)
        self.target = np.r_[np.concatenate(data), np.zeros(columns, complex)]

    def project(self, parameters):
        n = len(self.other)
        gains = np.ones(len(self.designs), complex)
        gains[self.other] = np.exp(parameters[:n]+1j*parameters[n:])
        design = np.concatenate([g*a for g, a in zip(gains, self.designs)]+[self.penalty])
        coefficient = lstsq(design, self.target, lapack_driver='gelsy')[0]
        return gains, coefficient

    def objective(self, parameters):
        gains, coefficient = self.project(parameters)
        value = self.orthogonal_energy+self.strength*np.vdot(coefficient, coefficient).real
        gradients = []
        for c, (a, y, g) in enumerate(zip(self.designs, self.data, gains)):
            predicted = g*(a@coefficient)
            residual = predicted-y
            value += np.vdot(residual, residual).real
            if c in self.other:
                product = np.vdot(residual, predicted)
                gradients.append([2*product.real, -2*product.imag])
        gradient = np.asarray(gradients).T.ravel()
        return float(value/self.energy), gradient/self.energy

    def fit(self, *, phase_start=None, random_starts=4):
        n = len(self.other)
        phases = np.zeros(n) if phase_start is None else np.angle(np.asarray(phase_start)[self.other])
        starts = [np.r_[np.zeros(n), phases]]
        starts += [np.r_[np.zeros(n), p] for p in
                   np.random.default_rng(7821).uniform(-np.pi, np.pi, (random_starts, n))]
        lower = np.r_[np.full(n, -self.limit), np.full(n, -np.inf)]
        upper = np.r_[np.full(n, self.limit), np.full(n, np.inf)]
        trials = [minimize(self.objective, s, jac=True, method='L-BFGS-B',
            bounds=list(zip(lower, upper)), options={'gtol': 1e-10, 'ftol': 1e-14, 'maxiter': 500})
            for s in starts]
        best = min(trials, key=lambda r: r.fun)
        gains, coefficient = self.project(best.x)
        projected = best.x-np.clip(best.x-best.jac, lower, upper)
        return gains, coefficient, {'objective': float(best.fun),
            'converged': bool(abs(projected).max() < 1e-7),
            'maximum_projected_gradient': float(abs(projected).max()),
            'gain_bound_hit': bool(np.any(abs(best.x[:n]) > self.limit-1e-7)),
            'gain_increment_real_imag': np.c_[gains.real, gains.imag].tolist(),
            'regularization_strength': self.strength,
            'trials': [{'objective': float(t.fun), 'termination': str(t.message)} for t in trials],
            'scope': __doc__}
