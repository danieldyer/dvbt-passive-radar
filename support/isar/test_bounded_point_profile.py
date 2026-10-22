import unittest

import numpy as np

from bounded_point_profile import BoundedPointProfile, solve_points
from point_profile import PointProfile
from test_channel_alignment import synthetic


class BoundedPointProfileTests(unittest.TestCase):
    def test_exact_residual_and_gradient_reproduce_original_objective(self):
        alignment, origins, _ = synthetic([.25, -.15, -.2, .1])
        points = alignment.bands[0].points
        original = PointProfile(alignment.bands, points, origins)
        bounded = BoundedPointProfile(alignment.bands, points, origins)
        p = np.array([.02, -.01, .04, -.03, -.02, .05])
        a, b = original.evaluate(p), bounded.evaluate(p)
        self.assertAlmostEqual(a['value'], b['value'], places=14)
        np.testing.assert_allclose(a['gradient'], b['gradient'], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(a['gauss_newton'], 2*b['jacobian'].T@b['jacobian'], rtol=1e-12, atol=1e-12)

    def test_recovers_independent_known_scene_with_native_bounds(self):
        alignment, origins, _ = synthetic([.25, -.15, -.2, .1])
        model = BoundedPointProfile(alignment.bands, alignment.bands[0].points, origins)
        n = len(model.initial); visited = []
        def checkpoint(p, trace):
            visited.append(p.copy())
        parameters, fit = solve_points(model, np.full(n, -1.), np.full(n, 1.), checkpoint=checkpoint)
        self.assertTrue(fit['converged'], fit)
        self.assertTrue(all(np.max(abs(p)) <= 1. for p in visited))
        np.testing.assert_allclose(parameters, [0., 0., .25, -.15, -.2, .1], atol=.004)
        self.assertLess(fit['objective'], 1e-7)


if __name__ == '__main__':
    unittest.main()
