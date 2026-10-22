import copy
import unittest

import numpy as np
from scipy.optimize import least_squares

from geometry_profile import GeometryProfile
from scattered_road import ScatteredRoad
from scattered_origin_geometry import ScatteredOriginRoad, ScatteredOriginGeometry, FrozenScatteredRoad
from shared_origin_geometry import SharedOriginGeometry
from shared_target_geometry import SharedGeometry, complex_gain
from test_scattered_road import fixture
import test_physical_profile as ship_tests


class ScatteredOriginTests(unittest.TestCase):
    def test_common_origins_preserve_physical_offsets_and_have_correct_derivatives(self):
        road, truth = fixture()
        physical = ScatteredRoad(road, [2., -1.], reference=1)
        model = ScatteredOriginRoad(physical)
        motion = np.r_[truth*.3, [1.2, -.4, .7]]
        antenna = np.arange(road.antennas.count)*.0004
        origins = np.array([2., -1.])
        parameters = np.r_[motion, antenna, origins]
        for c, field in enumerate(model.training):
            value, derivative = model.template(motion, antenna, origins, c, field)
            original, du = physical.template(motion, antenna, c, field)
            np.testing.assert_allclose(value, original, atol=1e-12)
            np.testing.assert_allclose(derivative[:, :-2], du, atol=1e-12)
            np.testing.assert_array_equal(derivative[:, -2+(1-c)], 0.)
            for j in (0, physical.base_count, physical.base_count+2, model.nmotion,
                      len(parameters)-3, len(parameters)-2+c):
                # A step sweep resolves the finite-difference cancellation
                # from subtracting 30 km paths in weak vertical directions.
                step = 1e-4; delta = np.eye(len(parameters))[j]*step
                values = [model.template(p[:model.nmotion], p[model.nmotion:-2], p[-2:], c, field)[0]
                    for p in (parameters+delta, parameters-delta)]
                np.testing.assert_allclose((values[0]-values[1])/(2*step), derivative[:, j],
                    atol=5e-7, rtol=5e-5, err_msg=f'channel {c}, coordinate {j}')

    def test_joint_expansion_preserves_original_likelihood_bounds_and_common_parameters(self):
        road, _ = fixture(); band, truth = ship_tests.PhysicalProfileTests().fixture()
        ship = GeometryProfile([band, copy.deepcopy(band)], truth*.3, road.antennas)
        original = SharedOriginGeometry(SharedGeometry(ship, [road]))
        original.initial[original.range_slice] = [1.1, -.7]
        model = ScatteredOriginGeometry(original)
        before, after = original.evaluate(original.initial), model.evaluate(model.initial)
        self.assertAlmostEqual(before['value'], after['value'], places=12)
        np.testing.assert_allclose(before['gradient'], after['gradient'][model.base_columns], atol=1e-12)
        n = len(original.initial)
        matrix = np.arange(2*n).reshape(2, n)*.001
        bounds = (matrix, [-10., -20.], [10., 20.], np.full(n, -100.), np.full(n, 100.))
        expanded = model.expand_bounds(bounds)
        np.testing.assert_allclose(expanded[0]@model.initial, matrix@original.initial)
        np.testing.assert_array_equal(expanded[3][model.base_columns], bounds[3])
        np.testing.assert_array_equal(expanded[4][-3:], [3., 2., 2.])
        p = model.initial.copy(); p[-3:] = [1.2, -.4, .7]
        result = model.evaluate(p)
        independent = (ship.evaluate(p[:model.ship_count])['value']*ship.energy+
            model.roads[0].evaluate(p[model.road_slices[0]], p[model.antenna_slice],
                p[model.range_slice])['value'])/model.energy
        self.assertAlmostEqual(independent, result['value'], places=12)
        for j in (model.range_slice.start, model.antenna_slice.start, len(p)-3, len(p)-1):
            delta = np.eye(len(p))[j]*2e-5
            numeric = (model.objective(p+delta)[0]-model.objective(p-delta)[0])/4e-5
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=3e-6+abs(numeric)*3e-5)

    def test_known_origins_recover_without_absorbing_reflector_offsets_into_timing(self):
        road, truth = fixture()
        physical = ScatteredRoad(road, [0., 0.], reference=1)
        model = ScatteredOriginRoad(physical)
        motion = np.r_[truth, [1.2, -.4, .7]]; antenna = np.zeros(road.antennas.count)
        origins = np.array([2.37, -1.42])
        for c, field in enumerate(model.training):
            field.y = (.7-.4j)*model.template(motion, antenna, origins, c, field)[0]
        def residual(value, jacobian=False):
            pieces = []
            for c, field in enumerate(model.training):
                u, du = model.template(motion, antenna, value, c, field)
                r, j, _, _ = complex_gain(field.y, u, du[:, -2:])
                pieces.append(np.r_[j.real, j.imag] if jacobian else np.r_[r.real, r.imag])
            return np.concatenate(pieces)
        fit = least_squares(residual, np.zeros(2), jac=lambda p: residual(p, True),
            gtol=1e-12, ftol=1e-12, xtol=1e-12)
        np.testing.assert_allclose(fit.x, origins, atol=1e-7)
        frozen = FrozenScatteredRoad(road, origins)
        origins[:] = 100.
        np.testing.assert_array_equal(frozen.origins, [2.37, -1.42])
        self.assertAlmostEqual(frozen.evaluate(motion, antenna)['value'],
            model.evaluate(motion, antenna, [2.37, -1.42])['value'], places=12)


if __name__ == '__main__':
    unittest.main()
