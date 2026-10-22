"""Sufficient statistic for a common residual phase against a known waveform.

Inputs are whitened complex range vectors. Dividing the matched product by
template information gives unit expectation for an exact template, with inverse
information as its noise variance. This is likelihood weighting, not local
image normalization; original observations remain unchanged.
"""
import numpy as np
from scipy.optimize import minimize

from gpga import SplineLikelihood
from stationary import Stationary
from stationarity import polish


def history(data, template, ranges=None):
    data, template = np.asarray(data), np.asarray(template)
    if data.shape != template.shape or data.ndim != 2:
        raise ValueError('Need matching time/range matrices')
    if ranges is not None:
        data, template = data[:, ranges], template[:, ranges]
    information = np.sum(abs(template)**2, axis=1)
    product = np.sum(template.conj()*data, axis=1)
    value = np.divide(product, information, out=np.zeros_like(product), where=information > 0)
    return value[:, None], information


def prediction_error(data, template, phase, selected_time, ranges=None):
    if ranges is not None:
        data, template = data[:, ranges], template[:, ranges]
    return float(np.sum(abs(data[selected_time]-template[selected_time]*np.exp(1j*phase[selected_time, None]))**2))


class ProfiledPhase:
    """Fit a phase screen while eliminating stationary nuisance at every trial.

    Profiling the background couples times, so the per-time scalar phase
    statistic alone is no longer sufficient. This evaluates the full complex
    residual and its exact gradient; the moving template is held fixed.
    """
    def __init__(self, time, data, template, background, selected, knot_seconds, gauge_information):
        self.data, self.template = np.asarray(data), np.asarray(template)
        self.selected = np.asarray(selected, bool)
        if self.data.shape != self.template.shape or self.selected.shape != self.data.shape:
            raise ValueError('Data, template and training mask must have matching time/range dimensions')
        information = np.sum(abs(self.template)**2*self.selected, axis=1)
        self.curve = SplineLikelihood(np.asarray(time), np.ones((len(time), 1), complex),
            information, np.any(self.selected, axis=1), knot_seconds, gauge_information=gauge_information)
        self.nuisance = Stationary(background, self.selected.ravel())
        self.y = self.nuisance.remove(self.data[self.selected])
        self.energy = float(np.vdot(self.y, self.y).real)
        if not self.energy > 0:
            raise ValueError('No retained phase information')

    def objective(self, coefficients):
        model = self.template*np.exp(1j*self.curve.phase(coefficients)[:, None])
        residual = self.y-self.nuisance.remove(model[self.selected])
        full = np.zeros(self.template.shape, complex)
        full[self.selected] = residual
        gradient = 2*np.imag(np.sum(full.conj()*model, axis=1))/self.energy
        return float(np.vdot(residual, residual).real/self.energy), self.curve.phase_adjoint(gradient)


def refine_profiled(time, data, template, background, selected, initial_phase, *, knot_seconds, gauge_information):
    likelihood = ProfiledPhase(time, data, template, background, selected, knot_seconds, gauge_information)
    trials = []
    for initial in (np.zeros(likelihood.curve.basis.shape[1]), likelihood.curve.seed(initial_phase)):
        result = minimize(likelihood.objective, initial, jac=True, method='L-BFGS-B',
            options={'maxiter': 600, 'gtol': 1e-9, 'ftol': 1e-14, 'maxcor': 30})
        trials.append(result)
    best = min(trials, key=lambda r:r.fun)
    parameters = best.x
    correction = None
    if abs(best.jac).max() > 1e-9:
        n = len(parameters)
        phase_basis = np.column_stack([likelihood.curve.phase(e) for e in np.eye(n)])
        _, singular, vh = np.linalg.svd(phase_basis, full_matrices=False)
        transform = vh[singular > singular[0]*1e-10].T
        parameters, correction = polish(likelihood.objective, parameters, transform,
            np.empty((0,n)), np.empty(0), np.empty(0), np.full(n,-np.inf), np.full(n,np.inf))
    value, gradient = likelihood.objective(parameters)
    return likelihood.curve.phase(parameters), {
        'converged': bool(abs(gradient).max() < 1e-7),
        'maximum_gradient': float(abs(gradient).max()), 'objective': float(value),
        'iterations': best.nit, 'termination': str(best.message), 'knot_seconds': knot_seconds,
        'stationarity_polish':correction,
        'stationary_rank': likelihood.nuisance.rank,
        'scope': 'Full complex-range residual; stationary nuisance reprofiled at every phase trial; fixed target template. Constant and linear phase gauges use the common full-template information.'}
