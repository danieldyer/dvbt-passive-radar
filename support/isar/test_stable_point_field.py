import unittest

import numpy as np

from stable_point_field import StablePointField
from test_rigid_motion import fixture


class StablePointFieldTests(unittest.TestCase):
    def test_same_physical_response_without_earth_coordinate_rounding(self):
        field = fixture(); stable = StablePointField(field)
        point = np.array([82.7, -35.8, 21.3])
        expected, old_derivative = field.point(point, derivatives=True, range_bias_m=17.2)
        actual, derivative = stable.point(point, derivatives=True, range_bias_m=17.2)
        self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(actual), 1e-8)
        np.testing.assert_allclose(derivative, old_derivative, rtol=1e-7, atol=1e-7)
        for axis in range(3):
            delta = np.eye(3)[axis]*1e-7
            plus, minus = [stable.point(p, range_bias_m=17.2) for p in (point+delta, point-delta)]
            numeric = (plus-minus)/2e-7
            self.assertLess(np.linalg.norm(numeric-derivative[:, :, axis])/
                            np.linalg.norm(derivative[:, :, axis]), 1e-5)


if __name__ == '__main__':
    unittest.main()
