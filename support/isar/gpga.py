"""Iterative phase-gradient estimation from backprojected point histories.

Scatterer histories are isolated with a Gaussian Doppler window, expressed at
their actual epochs as a normalized convolution. Only training samples enter
that convolution. The input observations are never filtered or overwritten.
Constant and linear phase are fixed gauges; this operator alone does not fit
physical translation, heading, or cross-channel calibration.

References: Evers and Jackson, TCI 2019, doi:10.1109/TCI.2019.2899453;
Garry and Smith, IET RSN 2019, doi:10.1049/iet-rsn.2018.5233.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


def remove_affine(time, phase, weights):
    x = np.column_stack([np.ones(len(time)), time-np.average(time, weights=weights)])
    root = np.sqrt(weights)
    return phase-x@np.linalg.lstsq(x*root[:, None], phase*root, rcond=None)[0]


class Isolation:
    """Gaussian amplitude response, half amplitude at the specified bandwidth.

    The derivative includes the normalization derivative. Both matrices use
    only selected observations, so held-out complex data cannot leak into a fit.
    Five standard deviations bound the finite convolution support.
    """
    def __init__(self, time, weights, selected, bandwidth_hz):
        time, weights = np.asarray(time), np.asarray(weights)
        selected = np.asarray(selected, bool)
        if bandwidth_hz <= 0 or np.any(np.diff(time) <= 0):
            raise ValueError('Need positive bandwidth and strictly increasing epochs')
        self.selected = selected & (weights > 0)
        source = time[self.selected]
        sigma = np.sqrt(2*np.log(2))/(2*np.pi*bandwidth_hz)
        first = np.searchsorted(source, time-5*sigma)
        last = np.searchsorted(source, time+5*sigma, side='right')
        counts = last-first
        if np.any(counts == 0):
            raise ValueError('Training gaps exceed the isolation window support')
        offsets = np.r_[0, np.cumsum(counts)]
        row = np.repeat(np.arange(len(time)), counts)
        column = np.repeat(first-offsets[:-1], counts)+np.arange(offsets[-1])
        delta = time[row]-source[column]
        value = np.exp(-.5*(delta/sigma)**2)*weights[self.selected][column]
        total = np.bincount(row, value, minlength=len(time))
        gradient = -delta/sigma**2
        mean_gradient = np.bincount(row, value*gradient, minlength=len(time))/total
        value /= total[row]
        shape = (len(time), len(source))
        self.kernel = csr_matrix((value, column, offsets), shape=shape)
        self.derivative = csr_matrix((value*(gradient-mean_gradient[row]), column, offsets), shape=shape)

    def apply(self, values):
        samples = values[self.selected]
        return self.kernel@samples, self.derivative@samples


def phase_metrics(time, values, weights, selected, reference):
    """Measured residuals versus training amplitudes; includes noise and glint.

    A phase correction's own RMS is not a residual-error measurement. These
    diagnostics evaluate the corrected complex observations independently.
    No positive-SNR selection is used to hide weak or decorrelated histories.
    """
    z, w = values[selected], weights[selected]
    angle = np.angle(z*reference.conj())
    unit = np.divide(z, abs(z), out=np.zeros_like(z), where=abs(z) > 0)
    resultant = abs(np.sum(w[:, None]*unit, axis=0))/w.sum()
    coherent = abs(np.sum(w[:, None]*z, axis=0))**2/w.sum()
    power = np.sum(w[:, None]*abs(z)**2, axis=0)
    return {'phase_rms_degrees_by_point': np.degrees(np.sqrt(np.sum(w[:, None]*angle**2, axis=0)/w.sum())).tolist(),
            # A unit-vector resultant is at most one; roundoff at a perfectly
            # coherent bin must not turn its zero circular variance into NaN.
            'circular_std_degrees_by_point': np.degrees(np.sqrt(-2*np.log(np.clip(resultant, 1e-15, 1.)))).tolist(),
            'coherent_fraction_by_point': (coherent/np.maximum(power, 1e-300)).tolist(),
            'coherent_energy_by_point': coherent.tolist(),
            'prediction_error': float(np.sum(w[:, None]*abs(z-reference)**2)),
            'samples': int(selected.sum()),
            'scope': 'Measured phase includes noise, unresolved scattering and amplitude variation; not an isolated motion-error RMS or CRLB.'}


class SplineLikelihood:
    """Concentrated complex likelihood for a common smooth phase screen.

    Cubic splines permit local motion that a low-order aperture-wide polynomial
    cannot represent. Only constant and linear gauges are projected out. The
    analytic gradient uses the adjoint of that projection, not the projection
    itself (they differ when sample information is nonuniform).
    """
    def __init__(self, time, values, information, selected, knot_seconds, *, gauge_information=None):
        self.time = np.asarray(time)
        self.weight = np.asarray(information)*np.asarray(selected, bool)
        self.gauge_weight = self.weight if gauge_information is None else np.asarray(gauge_information)*np.asarray(selected, bool)
        if self.gauge_weight.shape != self.weight.shape or not np.isfinite(self.gauge_weight).all() or np.any(self.gauge_weight < 0) or not np.any(self.gauge_weight > 0):
            raise ValueError('Need finite nonnegative information for the phase gauge')
        if knot_seconds <= 0 or not np.any(self.weight > 0):
            raise ValueError('Need positive knot spacing and training information')
        self.values = np.asarray(values, complex)
        self.energy = np.sum(self.weight[:, None]*abs(self.values)**2)
        if not self.energy > 0:
            raise ValueError('No training signal energy')
        interior = np.arange(time[0]+knot_seconds, time[-1], knot_seconds)
        knots = np.r_[np.repeat(time[0], 4), interior, np.repeat(time[-1], 4)]
        self.basis = BSpline.design_matrix(time, knots, 3)
        u = (time-np.average(time, weights=self.gauge_weight))/(time[-1]-time[0])
        self.affine = np.column_stack([np.ones(len(time)), u])
        self.normal = self.affine.T@(self.gauge_weight[:, None]*self.affine)

    def project(self, phase):
        return phase-self.affine@np.linalg.solve(self.normal, self.affine.T@(self.gauge_weight*phase))

    def phase(self, coefficients):
        return self.project(self.basis@coefficients)

    def phase_adjoint(self, gradient):
        projected = gradient-self.gauge_weight*(self.affine@np.linalg.solve(self.normal, self.affine.T@gradient))
        return self.basis.T@projected

    def objective(self, coefficients):
        corrected = self.values*np.exp(-1j*self.phase(coefficients)[:, None])
        amplitude = np.sum(self.weight[:, None]*corrected, axis=0)/self.weight.sum()
        value = -self.weight.sum()*np.sum(abs(amplitude)**2)/self.energy
        gradient = -2*self.weight*np.imag(corrected@amplitude.conj())/self.energy
        return float(value), self.phase_adjoint(gradient)

    def seed(self, phase):
        return lsqr(self.basis, phase, atol=1e-12, btol=1e-12)[0]


def refine(time, values, information, selected, initial_phase, *, knot_seconds=1., iterations=600, gauge_information=None):
    likelihood = SplineLikelihood(time, values, information, selected, knot_seconds, gauge_information=gauge_information)
    trials = []
    for initial in (np.zeros(likelihood.basis.shape[1]), likelihood.seed(initial_phase)):
        trace = []
        def callback(parameters):
            value, gradient = likelihood.objective(parameters)
            trace.append({'objective': value, 'maximum_gradient': float(abs(gradient).max())})
        result = minimize(likelihood.objective, initial, jac=True, method='L-BFGS-B', callback=callback,
                          options={'maxiter': iterations, 'gtol': 1e-8, 'ftol': 1e-14, 'maxcor': 30})
        trials.append((result, trace))
    solution, trace = min(trials, key=lambda trial: trial[0].fun)
    phase = likelihood.phase(solution.x)
    diagnostics = {'knot_seconds': knot_seconds, 'coefficients': solution.x.tolist(),
        'maximum_gradient': float(abs(solution.jac).max()), 'objective': float(solution.fun),
        'iterations': solution.nit, 'evaluations': solution.nfev, 'termination': str(solution.message),
        'converged': bool(solution.success and abs(solution.jac).max() <= 1e-7),
        'trace': trace, 'starting_trials': [{'objective': float(r.fun), 'maximum_gradient': float(abs(r.jac).max()),
            'iterations': r.nit, 'success': bool(r.success)} for r, _ in trials],
        'scope': 'Smooth phase-screen maximum likelihood conditional on selected fixed histories. Numerical convergence does not establish physical motion or heading.'}
    return phase, diagnostics


def fit(time, values, information, selected, *, bandwidths_hz=(2., 1., .5, .25, .125),
        iterations_per_bandwidth=12, tolerance_degrees=.25):
    time, values, information = np.asarray(time), np.asarray(values), np.asarray(information)
    selected = np.asarray(selected, bool) & (information > 0)
    if (values.ndim != 2 or values.shape[0] != len(time) or information.shape != time.shape or
            not np.isfinite(values).all() or not np.isfinite(information).all() or
            np.any(information < 0) or selected.sum() < 8 or (~selected).sum() < 4):
        raise ValueError('Invalid training/validation histories or information')
    weight = information/np.mean(information[selected])
    training_weight = weight*selected
    phase = np.zeros(len(time))
    history = []
    scale = np.sqrt(np.average(np.sum(abs(values)**2, axis=1), weights=training_weight))
    if not scale > 0:
        raise ValueError('No training signal energy')
    values = values/scale

    def energy(p):
        rotated = values*np.exp(-1j*p[:, None])
        return float(np.sum(abs(np.sum(training_weight[:, None]*rotated, axis=0))**2)/training_weight.sum())

    current = energy(phase)
    final_converged = False
    for bandwidth in bandwidths_hz:
        isolation = Isolation(time, weight, selected, bandwidth)
        final_converged = False
        for iteration in range(iterations_per_bandwidth):
            rotated = values*np.exp(-1j*phase[:, None])
            z, derivative = isolation.apply(rotated)
            power = np.sum(abs(z)**2, axis=1)
            gradient = np.divide(np.imag(np.sum(z.conj()*derivative, axis=1)), power,
                                 out=np.zeros(len(time)), where=power > np.finfo(float).tiny)
            increment = remove_affine(time, cumulative_trapezoid(gradient, time, initial=0), training_weight)
            rms = float(np.sqrt(np.average(increment**2, weights=training_weight)))
            final_converged = rms <= np.radians(tolerance_degrees)
            step, score = 1., energy(phase+increment)
            while score < current and step > 1/256:
                step /= 2
                score = energy(phase+step*increment)
            accepted = score >= current
            history.append({'bandwidth_hz': float(bandwidth), 'iteration': iteration,
                            'increment_rms_degrees': float(np.degrees(rms)),
                            'line_search_step': step, 'accepted': bool(accepted),
                            'training_energy_before': current, 'training_energy_after': score,
                            'fixed_point_tolerance_met': bool(final_converged)})
            if accepted:
                phase += step*increment
                current = score
            if final_converged or not accepted:
                break
    test = (~selected) & (weight > 0)
    corrected = values*np.exp(-1j*phase[:, None])
    before_reference = np.sum(training_weight[:, None]*values, axis=0)/training_weight.sum()
    after_reference = np.sum(training_weight[:, None]*corrected, axis=0)/training_weight.sum()
    before = phase_metrics(time, values, weight, test, before_reference)
    after = phase_metrics(time, corrected, weight, test, after_reference)
    before_energy = sum(before['coherent_energy_by_point'])
    after_energy = sum(after['coherent_energy_by_point'])
    return {'phase_radians': phase, 'iterations': history, 'converged': bool(final_converged),
            'before': before, 'after': after,
            'heldout_coherent_power_gain_db': float(10*np.log10(after_energy/before_energy)),
            'heldout_prediction_error_change_percent': float(100*(after['prediction_error']/before['prediction_error']-1)),
            'gauge': 'Information-weighted constant and linear phase fixed to zero',
            'convergence_definition': 'Undamped residual GPGA increment RMS below the stated tolerance at final isolation bandwidth; line-search stall is not convergence.',
            'tolerance_degrees': tolerance_degrees,
            'scope': 'Temporal holdout of fixed BP histories. Original peak selection used full prior images. No physical heading, calibrated multiband phase or image re-selection is inferred by this diagnostic.'}
