"""Refine physical reflector coordinates with independent complex band responses.

Motion, antenna locations, envelope origins and heights are fixed. Coordinates
move the envelope, RF phase and symbol Doppler together before row averaging.
Every trial profiles complex reflector amplitudes and stationary nuisance.
"""
import numpy as np

from audit_residual_phase import score
from phase_coupling import PhaseCoupling


def profile_points(band, points, range_origin):
    values, partials = [], []
    for point in points:
        value, derivative = band.field.point(point, motion=band.state, derivatives=True, range_bias_m=range_origin)
        values.append(value.ravel())
        partials.append(derivative[:, :, :2].reshape(-1, 2))
    target = np.column_stack(values)
    design = band.nuisance.remove(target[band.selected])
    u, singular, vh = np.linalg.svd(design, full_matrices=False)
    if singular[-1] <= singular[0]*1e-10:
        raise ValueError('Point refinement made the physical scene numerically rank deficient')
    coefficient = vh.conj().T@((u.conj().T@band.clean)/singular)
    residual = band.clean-design@coefficient
    derivative = band.nuisance.remove(np.column_stack(partials)[band.selected])
    fixed = derivative*np.repeat(coefficient, 2)
    cross = np.zeros((len(points), 2*len(points)), complex)
    cross[np.repeat(np.arange(len(points)), 2), np.arange(2*len(points))] = derivative.conj().T@residual
    jacobian = -fixed+u@(u.conj().T@fixed)-u@((vh@cross)/singular[:, None])
    return {'target': target, 'coefficient': coefficient, 'residual': residual, 'jacobian': jacobian,
        'value': float(np.vdot(residual, residual).real), 'condition': float(singular[0]/singular[-1])}


class PointProfile:
    def __init__(self, bands, points, range_origins):
        self.bands, self.points, self.origins = bands, np.asarray(points), range_origins
        self.initial = np.zeros(2*len(points))
        self.energy = sum(b.energy for b in bands)
        self.last, self.evaluations = None, 0

    def coordinates(self, parameters):
        points = self.points.copy()
        points[:, :2] += np.asarray(parameters).reshape(-1, 2)
        return points

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        value, gradient = 0., np.zeros(len(parameters))
        expected = np.zeros((len(parameters), len(parameters)))
        outputs = []
        for band, origin in zip(self.bands, self.origins):
            result = profile_points(band, self.coordinates(parameters), origin)
            jacobian = result.pop('jacobian')
            value += result['value']
            gradient += 2*np.real(jacobian.conj().T@result['residual'])
            expected += 2*np.real(jacobian.conj().T@jacobian)
            outputs.append(result)
        self.evaluations += 1
        self.last = {'parameters': np.asarray(parameters).copy(), 'value': value/self.energy,
            'gradient': gradient/self.energy, 'gauss_newton': expected/self.energy,
            'scene_conditions': [o['condition'] for o in outputs], 'bands': outputs}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']

    def validation(self, parameters, *, coupled=False, reference=3):
        result = self.evaluate(parameters)
        coefficients = [b['coefficient'] for b in result['bands']]
        calibration = None
        if coupled:
            phase = PhaseCoupling([b.nuisance.remove(o['target'][b.selected]) for b, o in zip(self.bands, result['bands'])],
                [b.clean for b in self.bands], reference=reference)
            coefficients, calibration = phase.fit()
        channels = [score(b.field, b.nuisance.design, (o['target']@a).reshape(b.field.y.shape),
            np.zeros(len(b.field.time)), b.selected_time, np.arange(b.field.y.shape[1]))
            for b, o, a in zip(self.bands, result['bands'], coefficients)]
        return {'channels': channels, 'phase_coupling': calibration,
            'prediction_error': sum(c['prediction_error'] for c in channels),
            'matched_energy': sum(c['matched_energy'] for c in channels)}
