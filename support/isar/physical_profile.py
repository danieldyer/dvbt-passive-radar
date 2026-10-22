"""Exact variable-projection derivatives for physical path/yaw refinement.

The envelope, carrier phase and within-symbol Doppler response all move
together through Band's original-row operator. Every trial re-solves complex
reflector amplitudes and stationary nuisance. The residual Jacobian includes
the amplitude-refitting term, not just projection of a fixed scene derivative.
"""
import numpy as np

from field import average
from spline_motion import SplineMotion
from validate_motion import validate


def observe_partials(band, raw, include_range, include_antennas=False):
    def observed(value):
        binned, _ = average(value, band.field.data['row_information'], band.field.metadata['block_rows'])
        return band.field.apply(binned).ravel()[band.selected]
    columns = []
    n = band.components
    for j, (basis, rate) in enumerate(zip(band.bases, band.rate_bases)):
        for k in range(basis.shape[1]):
            columns.append(observed(raw[:, :, j]*basis[:, k, None]+raw[:, :, n+j]*rate[:, k, None]))
    if include_antennas:
        columns.extend(observed(raw[:, :, 2*n+i]) for i in range(6))
    if include_range:
        columns.append(observed(raw[:, :, -1]))
    return band.nuisance.remove(np.column_stack(columns))


def profile(band, parameters, *, range_origin=None, derivatives=False, antenna_offsets=None):
    target, partials = band.design(parameters, derivatives=derivatives, range_bias_m=range_origin,
        range_derivative=range_origin is not None, antenna_offsets=antenna_offsets)
    design = band.nuisance.remove(target[band.selected])
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError('Physical scene is numerically rank deficient; do not change its rank silently')
    coefficient = vh.conj().T@((u.conj().T@band.clean)/singular)
    residual = band.clean-design@coefficient
    result = {'target': target, 'coefficient': coefficient, 'residual': residual,
        'value': float(np.vdot(residual, residual).real), 'condition': float(singular[0]/singular[-1])}
    if not derivatives:
        return result
    full_residual = np.zeros(band.field.y.size, complex)
    full_residual[band.selected] = residual
    adjoint = band.field.raw_adjoint(full_residual.reshape(band.field.y.shape))
    raw_sum = np.zeros_like(partials[0])
    cross = []
    n = band.components
    for amplitude, derivative in zip(coefficient, partials):
        raw_sum += amplitude*derivative
        # residual^H dA on original rows, then the real spline chain rule.
        local = np.einsum('nr,nrq->nq', adjoint.conj(), derivative)
        column = np.concatenate([b.T@local[:, j]+db.T@local[:, n+j]
            for j, (b, db) in enumerate(zip(band.bases, band.rate_bases))])
        if antenna_offsets is not None:
            column = np.r_[column, local[:, 2*n:2*n+6].sum(axis=0)]
        if range_origin is not None:
            column = np.r_[column, local[:, -1].sum()]
        cross.append(column.conj())
    del partials
    fixed = observe_partials(band, raw_sum, range_origin is not None, antenna_offsets is not None)
    # dr = -(I-AA+) dA a - (A+)^H dA^H r, for real nonlinear parameters.
    jacobian = -fixed+u@(u.conj().T@fixed)-u@((vh@np.array(cross))/singular[:, None])
    result.update(jacobian=jacobian, gradient=2*np.real(jacobian.conj().T@residual),
        gauss_newton=2*np.real(jacobian.conj().T@jacobian))
    return result


class PhysicalProfile:
    def __init__(self, bands, initial_motion, *, fit_ranges=False):
        self.bands = bands
        curve = SplineMotion({**bands[0].model, 'parameters': initial_motion})
        self.full_initial, self.active = curve.gauge_fixed_parameters()
        self.nmotion, self.fit_ranges = len(self.active), fit_ranges
        self.initial = np.r_[self.full_initial[self.active], [b.range_bias for b in bands] if fit_ranges else []]
        self.energy = sum(b.energy for b in bands)
        self.last = None
        self.evaluations = 0

    def expand(self, parameters):
        full = np.zeros_like(self.full_initial)
        full[self.active] = parameters[:self.nmotion]
        return full

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        hessian = np.zeros((len(parameters), len(parameters)))
        conditions = []
        for c, band in enumerate(self.bands):
            local = profile(band, motion, range_origin=parameters[self.nmotion+c] if self.fit_ranges else None,
                derivatives=True)
            axes = np.r_[self.active, len(motion)] if self.fit_ranges else self.active
            indices = np.r_[np.arange(self.nmotion), self.nmotion+c] if self.fit_ranges else np.arange(self.nmotion)
            value += local['value']
            gradient[indices] += local['gradient'][axes]
            hessian[np.ix_(indices, indices)] += local['gauss_newton'][np.ix_(axes, axes)]
            conditions.append(local['condition'])
            del local
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': hessian/self.energy,
            'scene_conditions': conditions, 'evaluations': self.evaluations}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']

    def validation(self, parameters):
        motion = self.expand(parameters)
        checks = []
        for c, band in enumerate(self.bands):
            local = profile(band, motion, range_origin=parameters[self.nmotion+c] if self.fit_ranges else None)
            checks.append(validate(band.field, local['target'], band.nuisance, band.selected_time,
                band.model['interval_s'], coefficients=local['coefficient'], stationary_mode='project'))
        return checks
