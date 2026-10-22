"""Regularized variable projection for a distributed physical target.

The scene penalty is selected on separate inner training blocks and then held
fixed. This is a declared regularized least-squares estimator, not an unbiased
CRLB claim. Motion changes delay, carrier and symbol Doppler together. Linear
coefficients are re-solved exactly using SVD at every trial. Derivatives are
streamed over points, avoiding a points x rows x motion-parameters allocation.
"""
import numpy as np

from field import average
from physical_profile import PhysicalProfile, observe_partials
from spline_motion import SplineMotion
from stationary import Stationary
from distributed_fixed_frame import prediction_metrics


def ridge(design, data, strength):
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('A fixed positive scene penalty is required')
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    if len(s) != design.shape[1]:
        raise ValueError('Need at least as many observations as scene points')
    projected = u.conj().T@data
    denominator = s**2+strength
    coefficient = vh.conj().T@(s/denominator*projected)
    outside = data-u@projected
    residual = outside+u@(strength/denominator*projected)
    return {'basis': u, 'singular': s, 'right': vh, 'strength': strength,
        'coefficient': coefficient, 'residual': residual,
        'value': float(np.vdot(outside, outside).real+np.sum(strength/denominator*abs(projected)**2)),
        'effective_rank': float(np.sum(s**2/denominator)),
        'condition': float(s[0]/max(s[-1], 1e-300))}


def residual_derivatives(fit, fixed, cross):
    """Exact augmented residual Jacobian, including the coefficient refit.

    fixed = dA coefficient; cross = dA^H residual for real motion parameters.
    No normal equations or numerical truncation of scene singular modes.
    """
    u, s, vh, strength = (fit[k] for k in ('basis', 'singular', 'right', 'strength'))
    denominator = s**2+strength
    inner = u.conj().T@fixed
    rhs = vh@cross
    top = -fixed+u@((s**2/denominator)[:, None]*inner-(s/denominator)[:, None]*rhs)
    bottom = vh.conj().T@((np.sqrt(strength)/denominator)[:, None]*(s[:, None]*inner-rhs))
    gradient = -2*np.real(fixed.conj().T@fit['residual'])
    curvature = 2*np.real(top.conj().T@top+bottom.conj().T@bottom)
    return top, bottom, gradient, curvature


def point_derivatives(band, parameters):
    """Original-row point partials in Band's component/rate convention."""
    state = SplineMotion({**band.model, 'parameters': parameters}).apply(
        band.time, band.prior, band.reference_transmitter, band.receiver, derivatives=True)
    frame, rate = state['body_to_ecef'], state['frame_velocity']
    offset = band.prior['position_ecef_m']-band.field.data['centre_ecef_m']+state['translation_offset_m']
    for point in band.points:
        directions = np.stack([state['translation_direction'],
            *(part@point for part in state['attitude_partials'])], axis=-1)
        rates = np.stack([state['translation_direction_rate'],
            *(part@point for part in state['attitude_rate_partials'])], axis=-1)
        if band.model.get('transverse_spacing_seconds') is not None:
            directions = np.concatenate([directions, state['transverse_direction'][:, :, None]], axis=-1)
            rates = np.concatenate([rates, state['transverse_direction_rate'][:, :, None]], axis=-1)
        _, derivative = band.field.raw_trajectory(state['position_ecef_m']+frame@point,
            state['velocity_ecef_mps']+rate@point, position_offset=offset+frame@point,
            position_jacobian=np.concatenate([directions, np.zeros_like(directions)], axis=-1),
            velocity_jacobian=np.concatenate([rates, directions], axis=-1), range_bias_m=band.range_bias)
        yield derivative


def profile(band, parameters, scale, strength, *, derivatives=False):
    target, _ = band.design(parameters)
    design = band.nuisance.remove(target[band.selected])*scale
    result = ridge(design, band.clean, strength)
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
    _, _, gradient, curvature = residual_derivatives(result, fixed, np.array(cross))
    result.update(gradient=gradient, gauss_newton=curvature)
    return result


class RidgeMotion(PhysicalProfile):
    def __init__(self, bands, initial_motion, weights, strengths):
        super().__init__(bands, initial_motion)
        self.scales = [np.sqrt(w) for w in weights]
        self.strengths = list(strengths)
        if len(bands) != len(self.scales) or len(bands) != len(strengths):
            raise ValueError('Each band needs its declared quadrature and penalty')

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        motion = self.expand(parameters)
        value, gradient = 0., np.zeros(len(parameters))
        curvature = np.zeros((len(parameters), len(parameters)))
        conditions, ranks = [], []
        for band, scale, strength in zip(self.bands, self.scales, self.strengths):
            if strength is None:
                # The zero-prediction candidate won inner validation. It has
                # constant training error and supplies no motion information.
                value += band.energy
                conditions.append(None); ranks.append(0.)
                continue
            local = profile(band, motion, scale, strength, derivatives=True)
            value += local['value']
            gradient += local['gradient'][self.active]
            curvature += local['gauss_newton'][np.ix_(self.active, self.active)]
            conditions.append(local['condition'])
            ranks.append(local['effective_rank'])
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': curvature/self.energy,
            'scene_conditions': conditions, 'effective_scene_ranks': ranks, 'evaluations': self.evaluations}
        return self.last

    def validation(self, parameters):
        result = []
        for band, scale, strength in zip(self.bands, self.scales, self.strengths):
            if strength is None:
                prediction = np.zeros(band.field.y.size, complex)
            else:
                fitted = profile(band, self.expand(parameters), scale, strength)
                prediction = fitted['target']@fitted['physical_coefficient']
            nuisance = Stationary(band.nuisance.design, ~band.selected)
            rows = (~band.selected_time).sum()
            y = nuisance.remove(band.field.y.ravel()[~band.selected]).reshape(rows, -1)
            prediction = nuisance.remove(prediction[~band.selected]).reshape(rows, -1)
            result.append(prediction_metrics(y, prediction, band.field.time[~band.selected_time]))
        return result
