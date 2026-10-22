import unittest

import numpy as np

from full_road_likelihood import original_row_field, stationary_support
from integrated_moving_road import IntegratedMovingField
from profiled_road_field import ProfiledRoadField
from road_gate_information import full_field
from test_integrated_moving_road import fixture


class FullRoadLikelihoodTests(unittest.TestCase):
    def test_original_row_static_profile_matches_joint_complex_least_squares(self):
        integrated, _, _ = fixture()
        raw, bins = full_field(integrated.original), integrated.bins
        delays = np.array([985., 1000., 1015.])
        static = bins.stationary(raw, delays)
        projected = ProfiledRoadField(raw, delays, static)
        model = original_row_field(projected, bins)
        path = model.original.seed_path+4.+1.7*(model.time-.5)
        u = model.original.template(path, range_bias=2.)[0]
        rng = np.random.default_rng(623015)
        noise = .02*(rng.normal(size=len(u))+1j*rng.normal(size=len(u)))
        y = (.7-.2j)*u+projected.stationary@np.array([3+2j, -2+1j, 4-3j])+noise
        raw.y = y
        model = original_row_field(ProfiledRoadField(raw, delays, static), bins)
        v = model.template(path, range_bias=2.)[0]
        alpha = np.vdot(v, model.y)/np.vdot(v, v).real
        residual = model.y-alpha*v
        matrix = np.column_stack((u, model.stationary))
        coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
        direct = y-matrix@coefficients
        np.testing.assert_allclose(alpha, coefficients[0], rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(residual, direct, rtol=2e-9, atol=2e-11)

    def test_original_row_path_and_origin_derivatives_after_static_projection(self):
        integrated, _, _ = fixture()
        raw, bins = full_field(integrated.original), integrated.bins
        delays = np.array([985., 1000., 1015.])
        model = original_row_field(ProfiledRoadField(raw, delays,
            bins.stationary(raw, delays)), bins)
        tau = model.time-.5
        path = model.original.seed_path+4.+1.7*tau
        basis = np.column_stack((np.ones(len(tau)), tau, tau*tau))
        _, derivative = model.template(path, basis, range_bias=2., bias_derivative=True)
        for j in range(4):
            step = 1e-4
            delta = basis[:, j]*step if j < 3 else np.zeros(len(tau))
            offset = step if j == 3 else 0.
            a = model.template(path+delta, range_bias=2.+offset)[0]
            b = model.template(path-delta, range_bias=2.-offset)[0]
            np.testing.assert_allclose(derivative[:, j], (a-b)/(2*step),
                rtol=3e-6, atol=3e-9)

    def test_required_static_support_covers_every_recorded_epoch(self):
        integrated, _, _ = fixture()
        raw, bins = integrated.original, integrated.bins
        delays, bound = stationary_support(raw, bins)
        from field import C, FS
        seed = IntegratedMovingField(raw, bins).seed_path
        lags = raw.offsets[None, :, None]-(delays[None, None, :]-seed[:, None, None])*FS/C
        self.assertAlmostEqual(bound, np.max(abs(lags)), places=12)


if __name__ == '__main__':
    unittest.main()
