import unittest

import numpy as np

from marginal_points import MarginalPoints
import test_physical_profile as physical_tests


class MarginalPointsTests(unittest.TestCase):
    def test_point_operator_equals_existing_physical_operator(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = MarginalPoints(band, truth)
        points = model.initial.copy()
        points[::2] = [.01, -.02, .03]
        design, _ = model.design(points)
        reference, _ = band.design(truth, points=model.coordinates(points))
        np.testing.assert_allclose(design, reference, rtol=1e-12, atol=1e-12)

    def test_marginal_coordinate_gradient_and_tiny_step_precision(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = MarginalPoints(band, truth*.4)
        at = np.linspace(-.05, .06, len(model.initial))
        result = model.evaluate(at)
        _, derivatives = model.design(at, derivatives=True)
        for j in range(len(at)):
            delta = np.eye(len(at))[j]*1e-4
            numeric = (model.objective(at+delta)[0]-model.objective(at-delta)[0])/2e-4
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=3e-6)
            delta *= 1e-4
            plus = model.design(at+delta)[0][:, j//2]
            minus = model.design(at-delta)[0][:, j//2]
            numeric = (plus-minus)/2e-8
            self.assertLess(np.linalg.norm(numeric-derivatives[j//2][:, j%2])/
                np.linalg.norm(numeric), 1e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gauss_newton']).min(), -1e-8)
        self.assertEqual(model.validation(at)['stationary_validation_mode'], 'project')


if __name__ == '__main__':
    unittest.main()
