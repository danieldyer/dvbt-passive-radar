"""Type-II Gaussian evidence for physical scatterer relevance and band gains.

The common model has one complex coefficient per physical point, independent
Gaussian point variances, and constant band gains. The comparison has separate
coefficients and variances per band. Neither model changes observations or
their noise weights. Sparsity and numerical stationarity do not calibrate a
physical scene: transfer must be tested separately.
"""
import numpy as np
from scipy.optimize import minimize

from phase_quadratic import solve as solve_phases
from stationarity import polish


def gaussian_posterior(design, data):
    """Unit complex Gaussian prior/noise, evaluated without normal equations."""
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    inverse = 1/np.hypot(1., s)
    z = u.conj().T@data
    latent = vh.conj().T@((s*inverse)*inverse*z)
    response = (s*inverse)**2
    fitted = u@(response*z)
    residual = data-fitted
    quadratic = float(np.vdot(residual, residual).real+np.vdot(latent, latent).real)
    logdet = float(-2*np.log(inverse).sum())
    return {'latent': latent, 'prediction': fitted, 'residual': residual,
        'quadratic': quadratic, 'logdet': logdet,
        'leverage': abs(vh.T)**2@response, 'u': u, 'response': response}


class RelevanceScene:
    def __init__(self, records, *, shared, reference=3):
        self.records, self.shared, self.reference = records, shared, reference
        self.channels, self.points = len(records), records[0]['design'].shape[1]
        if not 0 <= reference < self.channels or any(r['design'].shape[1] != self.points for r in records):
            raise ValueError('Use one common physical support and a valid reference channel')
        self.scale = np.sqrt(sum(np.linalg.norm(r['design'])**2 for r in records)/(self.channels*self.points))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('The physical dictionary must have finite nonzero energy')
        self.designs = [r['design']/self.scale for r in records]
        self.data = [r['data'] for r in records]
        self.other = [c for c in range(self.channels) if c != reference] if shared else []
        self.variance_count = self.points if shared else self.channels*self.points
        self.count = self.variance_count+2*len(self.other)
        self.energy = sum(np.vdot(y, y).real+r['outside'] for y, r in zip(self.data, records))
        self.last = None

    def unpack(self, parameters):
        p = np.asarray(parameters)
        if p.shape != (self.count,) or not np.isfinite(p).all():
            raise ValueError('Invalid relevance or band-response parameters')
        scale = np.exp(p[:self.variance_count]).reshape((-1, self.points))
        gain = np.ones(self.channels, complex)
        n = len(self.other)
        gain[self.other] = np.exp(p[self.variance_count:self.variance_count+n]+1j*p[self.variance_count+n:])
        return scale, gain

    def evaluate(self, parameters):
        p = np.asarray(parameters)
        if self.last is not None and np.array_equal(p, self.last['parameters']):
            return self.last
        standard, gain = self.unpack(p)
        groups = [list(range(self.channels))] if self.shared else [[c] for c in range(self.channels)]
        gradient = np.zeros(self.count)
        coefficients = [None]*self.channels
        quadratic, logdet = 0., 0.
        for group_index, channels in enumerate(groups):
            root = standard[group_index]
            designs = [self.designs[c]*root*gain[c] for c in channels]
            posterior = gaussian_posterior(np.concatenate(designs), np.concatenate([self.data[c] for c in channels]))
            first = group_index*self.points
            gradient[first:first+self.points] = 2*(posterior['leverage']-abs(posterior['latent'])**2)
            start = 0
            for c in channels:
                stop = start+len(self.data[c])
                fitted, residual = posterior['prediction'][start:stop], posterior['residual'][start:stop]
                if c in self.other:
                    index = self.other.index(c)
                    volume = np.sum(abs(posterior['u'][start:stop])**2@posterior['response'])
                    cross = np.vdot(residual, fitted)
                    gradient[self.variance_count+index] = 2*(volume-cross.real)
                    gradient[self.variance_count+len(self.other)+index] = 2*cross.imag
                coefficients[c] = gain[c]*root*posterior['latent']/self.scale
                start = stop
            quadratic += posterior['quadratic']+sum(self.records[c]['outside'] for c in channels)
            logdet += posterior['logdet']
        self.last = {'parameters': p.copy(), 'value': float(quadratic+logdet-self.energy),
            'quadratic': float(quadratic), 'logdet': float(logdet), 'gradient': gradient,
            'coefficients': coefficients, 'point_standard_deviation': standard/self.scale, 'band_gains': gain}
        return self.last

    def objective(self, parameters):
        value = self.evaluate(parameters)
        return value['value'], value['gradient']

    def phase_matrix(self, parameters):
        if not self.shared:
            raise ValueError('Independent scenes do not identify relative band phase')
        standard, gain = self.unpack(parameters)
        design = np.concatenate([a*standard[0]*abs(g) for a, g in zip(self.designs, gain)])
        u, s, _ = np.linalg.svd(design, full_matrices=False)
        split = np.cumsum([len(y) for y in self.data])[:-1]
        cross = np.column_stack([v.conj().T@y for v, y in zip(np.split(u, split), self.data)])
        weighted = cross*(s/np.hypot(1., s))[:, None]
        matrix = weighted.conj().T@weighted
        return (matrix+matrix.conj().T)/2

    def fit(self, *, initial_phases=None, initial_parameters=None, iterations=1000):
        # Bounds protect numerical exponentiation. Upper hits disqualify a
        # claimed evidence optimum; tiny lower variances approximate zero.
        n = len(self.other)
        lower = np.r_[np.full(self.variance_count, -20.), np.full(n, -8.), np.full(n, -np.inf)]
        upper = -lower
        logs = np.array([.5*np.log(max(np.vdot(y, y).real-len(y), 1.)/np.linalg.norm(a)**2)
                         for a, y in zip(self.designs, self.data)])
        if self.shared:
            initial = np.r_[np.full(self.points, logs[self.reference]),
                (logs-logs[self.reference])[self.other], np.zeros(n)]
        else:
            initial = np.repeat(logs, self.points)
        if initial_parameters is not None:
            initial = np.asarray(initial_parameters)
        initial = np.clip(initial, lower, upper)
        starts = [initial]
        if self.shared and n:
            phase, _ = solve_phases(self.phase_matrix(initial), reference=self.reference)
            second = initial.copy(); second[self.variance_count+n:] = -np.angle(phase)[self.other]
            starts.append(second)
            if initial_phases is not None:
                third = initial.copy()
                third[self.variance_count+n:] = (np.asarray(initial_phases)-initial_phases[self.reference])[self.other]
                starts.append(third)
        trials = []
        for start in starts:
            result = minimize(self.objective, start, jac=True, method='L-BFGS-B', bounds=list(zip(lower, upper)),
                options={'maxiter': iterations, 'gtol': 1e-7, 'ftol': 1e-14, 'maxls': 50, 'maxcor': 30})
            p = result.x
            def kkt(p):
                g = self.objective(p)[1].copy()
                g[(p <= lower+1e-8)&(g > 0)] = 0.
                g[(p >= upper-1e-8)&(g < 0)] = 0.
                return float(abs(g).max())
            refinement = None
            if kkt(p) > 1e-5:
                p, refinement = polish(self.objective, p, np.eye(len(p)),
                    np.zeros((0, len(p))), np.array([]), np.array([]), lower, upper)
            trials.append({'parameters': p.tolist(), 'value': self.objective(p)[0],
                'maximum_kkt_residual': kkt(p), 'stationary': kkt(p) <= 1e-5,
                'variance_upper_hits': int(np.sum(p[:self.variance_count] >= 20-1e-7)),
                'gain_bound_hits': int(np.sum(abs(p[self.variance_count:self.variance_count+n]) >= 8-1e-7)),
                'iterations': result.nit, 'termination': str(result.message), 'polish': refinement})
        best = min(trials, key=lambda r: r['value'])
        p = np.asarray(best['parameters']); value = self.evaluate(p)
        certificate = None
        if self.shared and n:
            matrix = self.phase_matrix(p)
            _, certificate = solve_phases(matrix, reference=self.reference)
            z = np.ones(self.channels, complex); z[self.other] = np.exp(-1j*p[self.variance_count+n:])
            captured = float(np.vdot(z, matrix@z).real)
            certificate = {**certificate, 'retained_solution_gap': max(0., certificate['upper_bound']-captured),
                'scope': 'Conditional phase-subproblem bound at fixed learned point variances and band powers; not a global evidence or physical calibration certificate.'}
        return value['coefficients'], {'shared_response': self.shared, 'scale': self.scale, 'selected': best,
            'trials': trials, 'phase_certificate': certificate, 'quadratic': value['quadratic'], 'logdet': value['logdet'],
            'point_standard_deviation': value['point_standard_deviation'].tolist(),
            'band_gains_real_imag': np.c_[value['band_gains'].real, value['band_gains'].imag].tolist()}
