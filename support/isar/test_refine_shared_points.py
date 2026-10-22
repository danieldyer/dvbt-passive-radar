import unittest

import numpy as np

from bounded_point_profile import BoundedPointProfile
from refine_shared_ne_points import refine
from test_channel_alignment import synthetic


class SharedPointRefinementTests(unittest.TestCase):
    def test_native_quasi_newton_recovers_independent_synthesized_positions(self):
        alignment, origins, _ = synthetic([.25, -.15, -.2, .1])
        problem = BoundedPointProfile(alignment.bands, alignment.bands[0].points, origins)
        count = len(problem.initial)
        parameters, fit = refine(problem, np.full(count, -1.), np.full(count, 1.))
        self.assertTrue(fit['converged'], fit)
        self.assertEqual(fit['maximum_constraint_violation'], 0.)
        np.testing.assert_allclose(parameters, [0., 0., .25, -.15, -.2, .1], atol=.004)
        self.assertLess(fit['objective'], 1e-7)


if __name__ == '__main__':
    unittest.main()
