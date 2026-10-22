import unittest

import numpy as np

from profiled_road_field import ProfiledRoadField
from road_gate_information import full_field
from test_integrated_moving_road import fixture


class ProfiledRoadFieldTests(unittest.TestCase):
    def test_unknown_static_response_profiles_without_corrupting_moving_gain(self):
        integrated, _, _ = fixture()
        raw = full_field(integrated.original)
        delays = np.array([985., 1000., 1015.])
        binned = integrated.bins.stationary(integrated.original, delays)
        model = ProfiledRoadField(raw, delays, binned)
        path = raw.seed_path+3.+1.2*(raw.time-.5)
        u = raw.template(path, range_bias=2.)[0]
        amplitude = .6-.3j
        raw.y = amplitude*u+model.stationary@np.array([3+2j, -2+1j, 4-3j])
        model = ProfiledRoadField(raw, delays, binned)
        projected = model.template(path, range_bias=2.)[0]
        np.testing.assert_allclose(model.y, amplitude*projected, atol=2e-11)
        fit = np.vdot(projected, model.y)/np.vdot(projected, projected).real
        self.assertLess(abs(fit-amplitude), 1e-11)
        reconstructed = amplitude*u+model.stationary@model.coefficients(raw.y-amplitude*u)
        np.testing.assert_allclose(reconstructed, raw.y, atol=2e-11)

    def test_origin_derivative_passes_through_fixed_stationary_projection(self):
        integrated, _, _ = fixture(); raw = full_field(integrated.original)
        delays = np.array([985., 1000., 1015.])
        field = ProfiledRoadField(raw, delays, integrated.bins.stationary(integrated.original, delays))
        path = raw.seed_path+3.+1.2*(raw.time-.5)
        _, derivative = field.template(path, np.ones((len(path), 1)), range_bias=2., bias_derivative=True)
        step = 1e-4
        a = field.template(path, range_bias=2.+step)[0]
        b = field.template(path, range_bias=2.-step)[0]
        np.testing.assert_allclose(derivative[:, -1], (a-b)/(2*step), atol=2e-9, rtol=2e-6)


if __name__ == '__main__':
    unittest.main()
