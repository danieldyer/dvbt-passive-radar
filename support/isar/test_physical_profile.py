import unittest

import numpy as np

from full_motion import Band
from physical_profile import profile, PhysicalProfile
from fit_physical_profile import solve
from spline_motion import basis
from test_rigid_motion import fixture


class PhysicalProfileTests(unittest.TestCase):
    def fixture(self, noise=.02):
        field = fixture()
        d = field.data
        interval = [0., d['row_time'][-1]]
        model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.}
        count = sum(basis([8.], interval, s).shape[1] for s in (6., 8.))
        rng = np.random.default_rng(1491)
        truth = rng.normal(size=count)*.02
        prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'], 'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        points = [[-70., 4., 20.], [90., -7., 20.], [15., 8., 20.]]
        block = np.floor(field.time).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        def band():
            return Band(field, prior, d['transmitter_ecef_m'], d['receiver_ecef_m'], points, model, selected, range_bias_m=1.3)
        initial = band()
        a = initial.design(truth)[0]
        y = a@np.array([.8+.3j, -.5+.4j, .3-.5j])+initial.nuisance.design@np.full(initial.nuisance.design.shape[1], .2+.1j)
        y += noise*(rng.normal(size=len(y))+1j*rng.normal(size=len(y)))
        field.y = y.reshape(field.y.shape)
        return band(), truth

    def test_exact_profile_residual_jacobian_with_envelope_phase_doppler_and_clutter(self):
        band, truth = self.fixture()
        parameters = truth*.4
        result = profile(band, parameters, range_origin=1.1, derivatives=True)
        for j in range(len(parameters)+1):
            step = 2e-4
            if j == len(parameters):
                plus = profile(band, parameters, range_origin=1.1+step)['residual']
                minus = profile(band, parameters, range_origin=1.1-step)['residual']
            else:
                delta = np.eye(len(parameters))[j]*step
                plus = profile(band, parameters+delta, range_origin=1.1)['residual']
                minus = profile(band, parameters-delta, range_origin=1.1)['residual']
            numeric = (plus-minus)/(2*step)
            relative = np.linalg.norm(numeric-result['jacobian'][:, j])/np.linalg.norm(numeric)
            self.assertLess(relative, 8e-5)
        value, gradient = band.evaluate(parameters)
        fixed = profile(band, parameters, derivatives=True)
        self.assertAlmostEqual(value/fixed['value'], 1., places=10)
        np.testing.assert_allclose(fixed['gradient'], gradient, rtol=1e-10, atol=1e-8)

    def test_joint_mapping_and_range_derivatives(self):
        band, truth = self.fixture()
        joint = PhysicalProfile([band], truth*.3, fit_ranges=True)
        x = joint.initial.copy()
        result = joint.evaluate(x)
        for j in (0, joint.nmotion-1, joint.nmotion):
            delta = np.eye(len(x))[j]*2e-4
            derivative = (joint.objective(x+delta)[0]-joint.objective(x-delta)[0])/.0004
            self.assertAlmostEqual(derivative, result['gradient'][j], delta=2e-6)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gauss_newton']).min(), -1e-9)
        checks = joint.validation(x)
        self.assertEqual(checks[0]['stationary_validation_mode'], 'project')

    def test_small_motion_steps_do_not_round_through_earth_sized_coordinates(self):
        band, truth = self.fixture()
        parameters = truth*.4
        result = profile(band, parameters, derivatives=True)
        # A nanometre-scale change is meaningful to the derivative check even
        # though it is far below the physical resolution of the recording.
        for j in (0, 3, len(parameters)-1):
            step = 1e-8
            delta = np.eye(len(parameters))[j]*step
            numeric = (profile(band, parameters+delta)['residual']-
                profile(band, parameters-delta)['residual'])/(2*step)
            relative = np.linalg.norm(numeric-result['jacobian'][:, j])/np.linalg.norm(numeric)
            self.assertLess(relative, 5e-6)

    def test_known_physical_motion_recovers_with_interior_block_holdouts(self):
        band, truth = self.fixture(noise=0.)
        joint = PhysicalProfile([band], truth*.3, fit_ranges=True)
        n = len(joint.initial)
        parameters, result = solve(joint, np.empty((0, n)), np.empty(0), np.empty(0),
            np.full(n, -5.), np.full(n, 5.), iterations=80)
        self.assertTrue(result['converged'], result)
        self.assertLess(result['objective'], 1e-13)
        self.assertLess(joint.validation(parameters)[0]['conditional_efficiency_loss_db'], 1e-6)


if __name__ == '__main__':
    unittest.main()
