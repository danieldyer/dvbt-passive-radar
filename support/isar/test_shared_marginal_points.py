import copy
import unittest

import numpy as np

from shared_marginal_points import SharedMarginalPoints
import test_physical_profile as physical_tests


class SharedMarginalPointsTests(unittest.TestCase):
    def test_joint_shared_position_gradient_with_independent_scene_variances(self):
        bands = []
        for gain in (1., .35*np.exp(1.2j), 2.1*np.exp(-.7j)):
            band, truth = physical_tests.PhysicalProfileTests().fixture(noise=.2)
            band.field = copy.copy(band.field); band.field.y = gain*band.field.y
            band.clean = band.nuisance.remove(band.field.y.ravel()[band.selected])
            band.energy = float(np.vdot(band.clean, band.clean).real)
            bands.append(band)
        problem = SharedMarginalPoints(bands, truth, bands[0].points)
        point = np.array([.06, -.03, .1, -.08, -.04, .05])
        result = problem.evaluate(point)
        for axis in range(len(point)):
            delta = np.eye(len(point))[axis]*1e-4
            plus, minus = [problem.objective(p)[0] for p in (point+delta, point-delta)]
            self.assertAlmostEqual((plus-minus)/2e-4, result['gradient'][axis], delta=2e-7)
        self.assertGreater(max(result['scene_variances']), 10*min(result['scene_variances']))
        self.assertEqual(len(problem.validation(point)), len(bands))


if __name__ == '__main__':
    unittest.main()
