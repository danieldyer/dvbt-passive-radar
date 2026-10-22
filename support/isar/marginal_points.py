"""Refit physical reflector positions at fixed translation and orientation.

The point coordinates change range, carrier phase and symbol Doppler together.
Each band retains its own complex scattering coefficients; a training-only
Gaussian scene variance controls cancellation between near-dependent columns.
No channel calibration or physical-heading inference is implied.
"""
import numpy as np

from gaussian_scene import marginal, motion_information
from spline_motion import SplineMotion
from validate_motion import validate


class MarginalPoints:
    def __init__(self, band, parameters):
        self.band = band
        self.points = band.points.copy()
        self.initial = np.zeros(2*len(self.points))
        self.energy = band.energy
        self.state = SplineMotion({**band.model, 'parameters': parameters}).apply(
            band.time, band.prior, band.reference_transmitter, band.receiver)
        self.base_offset = (band.prior['position_ecef_m']-band.field.data['centre_ecef_m']+
            self.state['translation_offset_m'])
        self.last, self.evaluations = None, 0

    def coordinates(self, parameters):
        value = self.points.copy()
        value[:, :2] += np.asarray(parameters).reshape(-1, 2)
        return value

    def design(self, parameters, *, derivatives=False):
        state, field = self.state, self.band.field
        frame, frame_rate = state['body_to_ecef'], state['frame_velocity']
        values, partials = [], []
        for point in self.coordinates(parameters):
            response = field.trajectory(state['position_ecef_m']+frame@point,
                state['velocity_ecef_mps']+frame_rate@point,
                position_offset=self.base_offset+frame@point, range_bias_m=self.band.range_bias,
                position_jacobian=frame[:, :, :2] if derivatives else None,
                velocity_jacobian=frame_rate[:, :, :2] if derivatives else None)
            if derivatives:
                response, derivative = response
                partials.append(derivative.reshape(-1, 2))
            values.append(response.ravel())
        return np.column_stack(values), partials

    def response(self, parameters, *, derivatives=False):
        band = self.band
        target, partials = self.design(parameters, derivatives=derivatives)
        design = band.nuisance.remove(target[band.selected])
        result = marginal(design, band.clean)
        result['target'] = target
        if derivatives:
            derivative = np.zeros((*design.shape, len(parameters)), complex)
            gradient = []
            for point, partial in enumerate(partials):
                observed = band.nuisance.remove(partial[band.selected])
                derivative[:, point, 2*point:2*point+2] = observed
                gradient.extend(2*np.real(result['design_gradient'][:, point].conj()@observed))
            result['gradient'] = np.array(gradient)
            result['information'] = motion_information(result, derivative)
        return result

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        result = self.response(parameters, derivatives=True)
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': result['value']/self.energy,
            'gradient': result['gradient']/self.energy, 'gauss_newton': result['information']/self.energy,
            'scene_conditions': [result['condition']], 'scene_variance': result['variance'],
            'effective_scene_rank': result['effective_scene_rank'], 'evaluations': self.evaluations}
        return self.last

    def objective(self, parameters):
        value = self.evaluate(parameters)
        return value['value'], value['gradient']

    def validation(self, parameters):
        band = self.band
        result = self.response(parameters)
        return validate(band.field, result['target'], band.nuisance, band.selected_time,
            band.model['interval_s'], coefficients=result['coefficient'], stationary_mode='project')
