"""Unpenalized reflector responses plus a regularized distributed scene.

min ||y - A0 a - A1 b||^2 + lambda ||b||^2.

The infinite-penalty limit is exactly the original A0 model, not an empty
target. The motion derivative includes refitting both sets of coefficients.
SVD solves and original-row derivatives preserve the complex phase convention.
This is an experimental conditional model, not a data-dependent normalization.
"""
import numpy as np

from blocked_ridge import RidgePath
from distributed_fixed_frame import prediction_metrics
from physical_profile import PhysicalProfile, observe_partials
from ridge_motion import point_derivatives
from stationary import Stationary


class MixedPath:
    """Solve many diffuse penalties after eliminating the fixed-size core."""
    def __init__(self, design, data, core):
        if not 0 < core <= design.shape[1]:
            raise ValueError('Need an explicit nonempty core')
        self.design, self.data, self.core = design, data, core
        self.u, self.s, self.vh = np.linalg.svd(design[:, :core], full_matrices=False)
        if self.s[-1] <= self.s[0]*1e-10:
            raise ValueError('The original reflector core is numerically rank deficient')
        self.projected = self.u.conj().T@data
        self.coupling = self.u.conj().T@design[:, core:]
        residual = data-self.u@self.projected
        diffuse = design[:, core:]-self.u@self.coupling
        self.path = RidgePath(diffuse, residual) if diffuse.shape[1] else None

    def coefficients(self, strengths):
        """One column per finite strength, or None for the exact core limit."""
        finite = [v for v in strengths if v is not None]
        if finite and self.path is None:
            raise ValueError('A finite diffuse penalty needs diffuse columns')
        b = np.zeros((self.design.shape[1]-self.core, len(strengths)), complex)
        if finite:
            select = np.array([v is not None for v in strengths])
            b[:, select] = self.path.coefficients(finite)
        a = self.vh.conj().T@((self.projected[:, None]-self.coupling@b)/self.s[:, None])
        return np.r_[a, b]


def mixed_fit(design, data, core, strength):
    """Augmented SVD reference used for every physical-motion trial."""
    if strength is None:
        if design.shape[1] != core:
            raise ValueError('Remove diffuse columns for the exact core-only motion fit')
        augmented, observations = design, data
    else:
        if not np.isfinite(strength) or strength <= 0 or not 0 < core < design.shape[1]:
            raise ValueError('Finite positive diffuse penalty and a nonempty core are required')
        penalty = np.c_[np.zeros((design.shape[1]-core, core)),
            np.sqrt(strength)*np.eye(design.shape[1]-core)]
        augmented = np.r_[design, penalty]
        observations = np.r_[data, np.zeros(len(penalty))]
    u, s, vh = np.linalg.svd(augmented, full_matrices=False)
    if len(s) != design.shape[1] or s[-1] <= s[0]*1e-12:
        raise ValueError('The declared augmented scene is numerically rank deficient; no silent rank change')
    projected = u.conj().T@observations
    coefficient = vh.conj().T@(projected/s)
    residual = observations-u@projected
    return {'basis': u, 'singular': s, 'right': vh, 'coefficient': coefficient,
        'residual': residual[:len(data)], 'augmented_residual': residual,
        'value': float(np.vdot(residual, residual).real), 'condition': float(s[0]/s[-1]),
        'effective_rank': float(np.sum(abs(u[:len(data)])**2))}


def residual_derivatives(fit, fixed, cross):
    """Full variable-projection Jacobian of observation and penalty residuals."""
    u, s, vh = fit['basis'], fit['singular'], fit['right']
    augmented = np.r_[fixed, np.zeros((len(u)-len(fixed), fixed.shape[1]), complex)]
    jacobian = -augmented+u@(u.conj().T@augmented)-u@((vh@cross)/s[:, None])
    gradient = -2*np.real(fixed.conj().T@fit['residual'])
    return jacobian, gradient, 2*np.real(jacobian.conj().T@jacobian)


def profile(band, parameters, scale, core, strength, *, derivatives=False):
    target, _ = band.design(parameters)
    design = band.nuisance.remove(target[band.selected])*scale
    result = mixed_fit(design, band.clean, core, strength)
    result['target'] = target
    result['physical_coefficient'] = result['coefficient']*scale
    if not derivatives:
        return result
    full_residual = np.zeros(band.field.y.size, complex)
    full_residual[band.selected] = result['residual']
    adjoint = band.field.raw_adjoint(full_residual.reshape(band.field.y.shape))
    raw_sum, cross = None, []
    for weight, coefficient, derivative in zip(scale, result['physical_coefficient'], point_derivatives(band, parameters)):
        if raw_sum is None:
            raw_sum = np.zeros_like(derivative)
        raw_sum += coefficient*derivative
        local = weight*np.einsum('nr,nrq->nq', adjoint.conj(), derivative)
        column = np.concatenate([b.T@local[:, j]+db.T@local[:, band.components+j]
            for j, (b, db) in enumerate(zip(band.bases, band.rate_bases))])
        cross.append(column.conj())
    fixed = observe_partials(band, raw_sum, False)
    _, gradient, curvature = residual_derivatives(result, fixed, np.array(cross))
    result.update(gradient=gradient, gauss_newton=curvature)
    return result


class MixedMotion(PhysicalProfile):
    def __init__(self, bands, initial_motion, weights, cores, strengths):
        super().__init__(bands, initial_motion)
        self.scales, self.cores, self.strengths = [np.sqrt(w) for w in weights], list(cores), list(strengths)
        if len(bands) != len(self.scales) or len(bands) != len(cores) or len(bands) != len(strengths):
            raise ValueError('Every channel needs its declared scene and penalty')

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        curvature = np.zeros((len(parameters), len(parameters)))
        conditions, ranks = [], []
        for band, scale, core, strength in zip(self.bands, self.scales, self.cores, self.strengths):
            fitted = profile(band, motion, scale, core, strength, derivatives=True)
            value += fitted['value']
            gradient += fitted['gradient'][self.active]
            curvature += fitted['gauss_newton'][np.ix_(self.active, self.active)]
            conditions.append(fitted['condition']); ranks.append(fitted['effective_rank'])
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': curvature/self.energy,
            'scene_conditions': conditions, 'effective_scene_ranks': ranks, 'evaluations': self.evaluations}
        return self.last

    def validation(self, parameters):
        result = []
        for band, scale, core, strength in zip(self.bands, self.scales, self.cores, self.strengths):
            fitted = profile(band, self.expand(parameters), scale, core, strength)
            prediction = fitted['target']@fitted['physical_coefficient']
            nuisance = Stationary(band.nuisance.design, ~band.selected)
            rows = (~band.selected_time).sum()
            y = nuisance.remove(band.field.y.ravel()[~band.selected]).reshape(rows, -1)
            prediction = nuisance.remove(prediction[~band.selected]).reshape(rows, -1)
            result.append(prediction_metrics(y, prediction, band.field.time[~band.selected_time]))
        return result
