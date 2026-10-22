import unittest

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation

from imaging import geometry_from_state
from motion import C, body_basis, ecef, enu_basis, excess_path, path_difference
from spline_motion import SplineMotion, basis, canonical_model
from test_heading_seed import fixture


class SplineMotionTests(unittest.TestCase):
    def setUp(self):
        lat, lon = np.radians([-37.87, 144.93])
        self.lat, self.lon = lat, lon
        self.origin, self.local = ecef(lat, lon, 5), enu_basis(lat, lon)
        self.tx = self.origin+self.local@np.array([35000., 15000., 700.])
        self.rx = self.origin+self.local@np.array([1000., 8000., 50.])
        npath = basis([6.], [0, 12], 4).shape[1]
        nyaw = basis([6.], [0, 12], 6).shape[1]
        self.model = SplineMotion({'interval_s': [0, 12], 'path_spacing_seconds': 4,
            'heading_spacing_seconds': 6, 'parameters': np.r_[np.linspace(-2, 3, npath),
                np.linspace(-7, 2, nyaw)].tolist()})

    def state(self, t):
        v = self.local@np.array([.4, 4., 0.])
        frame = body_basis(np.full(len(t), self.lat), np.full(len(t), self.lon), -.1-.003*t)
        generator = np.array([[0, -.003, 0], [.003, 0, 0], [0, 0, 0]])
        return {'position_ecef_m': self.origin+t[:, None]*v,
                'velocity_ecef_mps': np.broadcast_to(v, (len(t), 3)),
                'body_to_ecef': frame, 'frame_velocity': frame@generator}

    def test_direct_scalar_evaluation_preserves_all_coefficient_bases(self):
        for model in [self.model,SplineMotion({**self.model.description,'anchor_path':False})]:
            time=np.r_[0.,.01,np.linspace(.1,11.9,71),11.99,12.]
            for derivative in (0,1,2):
                reference=np.column_stack([b@model.parameters[s] for b,s in zip(model.coefficient_bases(time,derivative),model.sections)])
                np.testing.assert_allclose(model.components(time,derivative),reference,rtol=2e-13,atol=2e-13)

    def test_physical_velocity_matches_displaced_and_rotated_point_derivative(self):
        t = np.linspace(.1, 11.9, 99)
        state = self.model.apply(t, self.state(t), self.tx, self.rx)
        point = np.array([142., -19., 30.])
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        step = .001
        def positions(t):
            s = self.model.apply(t, self.state(t), self.tx, self.rx)
            return s['position_ecef_m']+s['body_to_ecef']@point
        measured = (positions(t+step)-positions(t-step))/(2*step)
        np.testing.assert_allclose(velocity, measured, atol=1e-6)
        # The same point's phase derivative and packed moment Doppler agree.
        geom, prior = geometry_from_state(state, self.tx, self.rx, 219.5e6, 320, np.ones(len(t)))
        path_plus = excess_path(positions(t+step), self.tx, self.rx)[0]
        path_minus = excess_path(positions(t-step), self.tx, self.rx)[0]
        point_position = positions(t)
        _, gradient = excess_path(point_position, self.tx, self.rx)
        np.testing.assert_allclose(np.sum(gradient*velocity, axis=1), (path_plus-path_minus)/(2*step), atol=1e-6)
        np.testing.assert_allclose(geom[:, 9, 0], prior['path_m']*8e6/C-320, atol=4e-6)

    def test_phase_seed_and_full_motion_agree_at_every_point(self):
        b = fixture()
        moved = self.model.apply(b.time, self.state(b.time), self.tx, self.rx)
        original = b.base
        points = moved['position_ecef_m'][:, None, :]+np.einsum('nij,pj->npi', moved['body_to_ecef'], b.pixels)
        delta = path_difference(original, points-original, self.tx, self.rx)
        np.testing.assert_allclose(-b.k*delta, b.phase(self.model.parameters), atol=1e-7)

    def test_no_silent_motion_extrapolation(self):
        with self.assertRaises(ValueError):
            self.model.curves(np.array([-1.]))

    def test_fractional_epoch_does_not_duplicate_a_clamped_endpoint(self):
        interval = [223.27204431250001, 268.27204431250004]
        t = np.linspace(*interval, 362)
        b = basis(t, interval, 3.)
        self.assertEqual(b.shape[1], 18)
        np.testing.assert_allclose(b.sum(axis=1), 1., atol=1e-15)
        np.testing.assert_allclose(basis(t, interval, 3., derivative=1).sum(axis=1), 0., atol=1e-15)
        self.assertEqual(np.linalg.matrix_rank(b), b.shape[1])
        self.assertEqual(b[-1, -1], 1.)
        # Saved earlier fits have one extra, unused endpoint coefficient. Read
        # them without changing their interior physical trajectory or source.
        old_knots = np.r_[np.repeat(interval[0], 4), np.arange(interval[0]+3., interval[1], 3.),
                          np.repeat(interval[1], 4)]
        p = np.arange(len(old_knots)-4, dtype=float)
        model = {'interval_s': interval, 'path_spacing_seconds': 3., 'heading_spacing_seconds': 8.,
                 'anchor_path': False, 'parameters': np.r_[p, np.zeros(basis([240.], interval, 8.).shape[1])]}
        migrated = SplineMotion(canonical_model(model))
        expected = BSpline(old_knots, p, 3)(t[:-1])
        np.testing.assert_allclose(migrated.curves(t[:-1])[0], expected, atol=1e-12)
        self.assertEqual(len(model['parameters']), 28)

    def test_transverse_motion_velocity_and_phase_follow_physical_position(self):
        description = {**self.model.description, 'transverse_spacing_seconds': 5.,
            'horizontal_translation': True, 'anchor_transverse': False}
        n = basis([6.], [0., 12.], 5.).shape[1]
        description['parameters'] = np.r_[self.model.parameters, np.linspace(-150., 170., n)]
        model = SplineMotion(description)
        t = np.linspace(.1, 11.9, 99)
        state = model.apply(t, self.state(t), self.tx, self.rx)
        point = np.array([142., -19., 30.])
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        def positions(time):
            s = model.apply(time, self.state(time), self.tx, self.rx)
            return s['position_ecef_m']+s['body_to_ecef']@point
        # Richardson differences avoid both Earth-coordinate round-off at
        # microscopic steps and cubic-spline truncation at millisecond steps.
        dt = .003
        def differentiate(function):
            coarse = (function(t+dt)-function(t-dt))/(2*dt)
            fine = (function(t+dt/2)-function(t-dt/2))/dt
            return (4*fine-coarse)/3
        np.testing.assert_allclose(differentiate(positions), velocity, atol=1e-6)
        _, gradient = excess_path(positions(t), self.tx, self.rx)
        path_rate = differentiate(lambda time: excess_path(positions(time), self.tx, self.rx)[0])
        np.testing.assert_allclose(np.sum(gradient*velocity, axis=1), path_rate, atol=1e-6)
        prior = self.state(t)
        _, g = excess_path(prior['position_ecef_m'], self.tx, self.rx)
        np.testing.assert_allclose(np.sum(g*state['transverse_direction'], axis=1), 0., atol=1e-15)
        np.testing.assert_allclose(np.sum(g*state['translation_direction'], axis=1), 1., atol=1e-15)
        displacement = state['position_ecef_m']-prior['position_ecef_m']
        np.testing.assert_allclose(np.sum(displacement*prior['body_to_ecef'][:, :, 2], axis=1), 0., atol=1e-9)

    def test_zero_transverse_extension_preserves_original_motion(self):
        n = basis([6.], [0., 12.], 5.).shape[1]
        model = SplineMotion({**self.model.description, 'transverse_spacing_seconds': 5.,
            'parameters': np.r_[self.model.parameters, np.zeros(n)]})
        t = np.linspace(.1, 11.9, 99)
        old, new = [m.apply(t, self.state(t), self.tx, self.rx) for m in (self.model, model)]
        for key in ('position_ecef_m', 'velocity_ecef_mps', 'body_to_ecef', 'frame_velocity'):
            np.testing.assert_array_equal(old[key], new[key])

    def test_absolute_yaw_gauge_requires_a_body_shape_or_heading_reference(self):
        t = np.linspace(.1, 11.9, 99)
        points = np.array([[-110., -12., 10.], [70., 17., 25.]])
        state = self.model.apply(t, self.state(t), self.tx, self.rx)
        changed = dict(self.model.description)
        coefficients = self.model.parameters.copy()
        coefficients[-len(self.model.yaw_coefficients):] += 8.
        changed['parameters'] = coefficients
        new_state = SplineMotion(changed).apply(t, self.state(t), self.tx, self.rx)
        rotated_points = points@Rotation.from_euler('z', -8., degrees=True).as_matrix().T
        original = state['position_ecef_m'][:, None, :]+np.einsum('nij,pj->npi', state['body_to_ecef'], points)
        equivalent = new_state['position_ecef_m'][:, None, :]+np.einsum('nij,pj->npi', new_state['body_to_ecef'], rotated_points)
        np.testing.assert_allclose(original, equivalent, atol=1e-9)
        old_velocity = state['velocity_ecef_mps'][:, None, :]+np.einsum('nij,pj->npi', state['frame_velocity'], points)
        new_velocity = new_state['velocity_ecef_mps'][:, None, :]+np.einsum('nij,pj->npi', new_state['frame_velocity'], rotated_points)
        np.testing.assert_allclose(old_velocity, new_velocity, atol=1e-13)


if __name__ == '__main__':
    unittest.main()
