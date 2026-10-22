"""Small phase-transfer diagnostic, independent of the ship autofocus fit.

The local model is a fixed complex range pattern rotating with one path
history, plus a fixed stationary pattern. Both patterns are fitted on the
first half of an interval and frozen on the second half. A common scalar
path-rate correction is searched on a bounded grid and locally refined.
This tests temporal phase transfer, not absolute interband calibration or
the physical identity of a scatterer. Range migration within an interval
and changing scattering structure are explicitly outside this local model.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class Patch:
    time: np.ndarray
    value: np.ndarray
    weight: np.ndarray
    path: np.ndarray
    k: float
    epoch: float
    training: np.ndarray
    background: np.ndarray | None = None

    def static(self):
        return np.ones(len(self.time), complex) if self.background is None else self.background

    def tone(self, rate):
        return np.exp(-1j*self.k*(self.path+rate*(self.time-self.epoch)))

    def coefficients(self, rate):
        use = self.training
        h, w, y = self.tone(rate)[use], self.weight[use], self.value[use]
        g = self.static()[use]
        norm_g = np.sum(w*abs(g)**2)
        mean_h = np.sum(w*g.conj()*h)/norm_g
        mean_y = np.sum((w*g.conj())[:, None]*y, axis=0)/norm_g
        centred = h-g*mean_h
        norm = np.sum(w*abs(centred)**2)
        # Identically stationary tones do not identify two complex patterns.
        if norm < 1e-12*np.sum(w):
            return np.zeros_like(mean_y), mean_y, 0., 0.
        a = np.sum((w*centred.conj())[:, None]*y, axis=0)/norm
        b = mean_y-mean_h*a
        score = float(norm*np.vdot(a, a).real)
        return a, b, score, float(norm/np.sum(w))

    def scores(self, rates):
        use = self.training
        w, y = self.weight[use], self.value[use]
        h = np.exp(-1j*self.k*(self.path[use, None]+np.asarray(rates)[None, :]*
                             (self.time[use, None]-self.epoch)))
        g = self.static()[use]
        h -= g[:, None]*(np.sum((w*g.conj())[:, None]*h, axis=0)/np.sum(w*abs(g)**2))[None, :]
        norm = np.sum(w[:, None]*abs(h)**2, axis=0)
        cross = h.conj().T@(w[:, None]*y)
        return np.divide(np.sum(abs(cross)**2, axis=1), norm,
                         out=np.zeros_like(norm), where=norm > 1e-12*np.sum(w))

    def brightness(self):
        """Training energy after a constant mean; no fitted phase involved."""
        use = self.training
        w, y = self.weight[use], self.value[use]
        g = self.static()[use]
        y = y-g[:, None]*np.sum((w*g.conj())[:, None]*y, axis=0)/np.sum(w*abs(g)**2)
        return float(np.sum(w[:, None]*abs(y)**2)/((len(w)-1)*y.shape[1]))

    def evaluate(self, rate):
        a, b, score, separation = self.coefficients(rate)
        h = self.tone(rate)
        g = self.static()
        predicted = h[:, None]*a[None, :]+g[:, None]*b[None, :]
        test = ~self.training
        w = self.weight[test]
        error = float(np.sum(w[:, None]*abs(self.value[test]-predicted[test])**2))
        train = self.training
        static_mean = np.sum((self.weight[train]*g[train].conj())[:, None]*self.value[train], axis=0)/np.sum(self.weight[train]*abs(g[train])**2)
        baseline_error = float(np.sum(w[:, None]*abs(self.value[test]-g[test, None]*static_mean)**2))
        norm = np.vdot(a, a).real
        q = ((self.value-g[:, None]*b)@a.conj())/(norm*h) if norm > 0 else np.zeros_like(h)
        # These phases are relative to training coefficients, not refitted on test.
        phase = np.angle(q[test])
        coherence = abs(np.sum(w*q[test]))**2/(np.sum(w)*np.sum(w*abs(q[test])**2))
        return {'rate_mps': float(rate), 'training_score': score,
                'tone_static_separation': separation, 'test_error': error,
                'static_only_test_error': baseline_error,
                'test_predicted_energy_fraction': 1-error/baseline_error,
                'test_phase_bias_deg': float(np.degrees(np.angle(np.sum(w*q[test])))),
                'test_phase_rms_deg': float(np.degrees(np.sqrt(np.average(phase**2, weights=w)))),
                'test_temporal_coherence_squared': float(coherence),
                'time_s': self.time.tolist(), 'training': self.training.tolist(),
                'phase_residual_deg': np.degrees(np.angle(q)).tolist(),
                'projected_amplitude': abs(q).tolist()}


def fit_rate(patches, bound=3., count=121):
    """All grid local maxima are refined; no arbitrary initialisation."""
    grid = np.linspace(-bound, bound, count)
    score = sum(p.scores(grid) for p in patches)
    candidates = [(float(score[0]), float(grid[0])), (float(score[-1]), float(grid[-1]))]
    local = np.flatnonzero((score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:]))+1
    for i in local:
        solution = minimize_scalar(lambda rate: -sum(p.coefficients(rate)[2] for p in patches),
                                   bounds=(grid[i-1], grid[i+1]), method='bounded',
                                   options={'xatol': 1e-9})
        candidates.append((-float(solution.fun), float(solution.x)))
    return max(candidates)[1]
