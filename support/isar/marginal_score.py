"""Single-SVD Gaussian likelihood score for repeated motion experiments.

This algebraically equivalent path avoids constructing the profiled-residual
Jacobian when a solver needs only the likelihood and gradient. The independent
augmented-SVD implementation in marginal_scene remains the numerical check.
There is no observation-sized covariance, normal equation or rank truncation.
"""
import numpy as np

from marginal_scene import Covariance
from mixed_scene import MixedMotion
from ridge_motion import point_derivatives
from spline_motion import SplineMotion


def linear(design, data, core, strength, *, variance=1.):
    if not np.isfinite(variance) or variance <= 0 or not 0 < core <= design.shape[1]:
        raise ValueError('Declare a nonempty core and a finite positive thermal variance')
    covariance = None if strength is None else Covariance(design[:, core:], strength)
    def whiten(value):
        return value if covariance is None else covariance.apply_inverse(value, square_root=True)
    mean = whiten(design[:, :core])
    observed = whiten(data)
    u, singular, vh = np.linalg.svd(mean, full_matrices=False)
    if singular[-1] <= singular[0]*1e-12:
        raise ValueError('The deterministic core is numerically rank deficient')
    inner = u.conj().T@observed
    alpha = vh.conj().T@(inner/singular)
    residual = observed-u@inner
    quadratic = float(np.vdot(residual, residual).real/variance)
    weighted = whiten(residual)
    beta = np.empty(0, complex)
    if covariance is not None:
        factor = covariance.s/(strength*np.sqrt(1+covariance.ratio))
        beta = covariance.vh.conj().T@(factor*(covariance.u.conj().T@residual))
    elif design.shape[1] != core:
        raise ValueError('Remove diffuse columns for the core-only limit')
    return {'coefficient': np.r_[alpha, beta], 'weighted_residual': weighted,
        'quadratic': quadratic, 'logdet': 0. if covariance is None else covariance.logdet,
        'covariance': covariance, 'core_condition': float(singular[0]/singular[-1])}


def profile(band, parameters, scale, core, strength, *, variance=1., derivatives=True):
    target, _ = band.design(parameters)
    design = band.nuisance.remove(target[band.selected])*scale
    fit = linear(design, band.clean, core, strength, variance=variance)
    fit['physical_coefficient'] = fit['coefficient']*scale
    fit['value'] = fit['quadratic']+fit['logdet']
    fit['target'] = target
    if derivatives:
        gradient = np.zeros(len(parameters))
        for j, derivative in enumerate(point_derivatives(band, parameters)):
            influence = -fit['weighted_residual']*fit['physical_coefficient'][j].conjugate()/variance
            if j >= core:
                influence += fit['covariance'].design_gradient[:, j-core]*scale[j]
            observed = np.zeros(band.field.y.size, complex)
            observed[band.selected] = band.nuisance.remove(influence)
            adjoint = band.field.raw_adjoint(observed.reshape(band.field.y.shape))
            local = np.einsum('nr,nrq->nq', adjoint.conj(), derivative)
            gradient += 2*np.real(np.concatenate([
                b.T@local[:, k]+db.T@local[:, band.components+k]
                for k, (b, db) in enumerate(zip(band.bases, band.rate_bases))]))
        fit['gradient'] = gradient
    return fit


class ScoreMotion(MixedMotion):
    """Yaw-free exact value/gradient; no optimizer curvature is called Fisher."""
    def __init__(self, bands, initial_motion, weights, cores, strengths):
        super().__init__(bands, initial_motion, weights, cores, strengths)
        curve = SplineMotion({**bands[0].model, 'parameters': initial_motion})
        if not bands[0].model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
            raise ValueError('This experiment keeps yaw exactly zero')
        self.active = self.active[(self.active < curve.sections[1].start)|(self.active >= curve.sections[1].stop)]
        self.nmotion = len(self.active)
        self.initial = self.full_initial[self.active].copy()

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        for band, scale, core, strength in zip(self.bands, self.scales, self.cores, self.strengths):
            fit = profile(band, motion, scale, core, strength)
            value += fit['value']
            gradient += fit['gradient'][self.active]
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value,
            'gradient': gradient, 'evaluations': self.evaluations}
        return self.last
