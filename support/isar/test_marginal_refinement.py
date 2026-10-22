import unittest

import numpy as np

from marginal_refinement import CurvatureMotion, fit
from marginal_score import ScoreMotion
import test_physical_profile as physical_tests


class MarginalRefinementTests(unittest.TestCase):
    def test_search_curvature_preserves_exact_value_score_and_zero_yaw(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        # The shared fixture contains fitted yaw; this adapter must explicitly
        # reject it rather than silently turn rotation into a zero-yaw test.
        with self.assertRaises(ValueError):
            CurvatureMotion([band], truth, [np.ones(3)], [2], [500.])
        band.model['fixed_ne_frame'] = True
        truth[band.npath:] = 0.
        args = ([band], truth, [np.ones(3)], [2], [500.])
        exact, curved = ScoreMotion(*args), CurvatureMotion(*args)
        a, b = exact.evaluate(exact.initial), curved.evaluate(curved.initial)
        self.assertAlmostEqual(a['value'], b['value'], delta=1e-8)
        np.testing.assert_allclose(a['gradient'], b['gradient'], rtol=2e-8, atol=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(b['gauss_newton']).min(), -1e-7)

    def test_trust_refinement_uses_likelihood_gradient_not_curvature_as_score(self):
        # A non-quadratic physical likelihood is exercised in the integration
        # experiment; this independent analytic objective detects accidentally
        # returning the search curvature as the statistical information/score.
        class Objective:
            evaluations = 0
            def evaluate(self, x):
                self.evaluations += 1
                residual = np.expm1(x)
                return {'value': float(residual@residual),
                    'gradient': 2*residual*np.exp(x), 'gauss_newton': np.diag(2*np.exp(2*x))}
            def objective(self,x):
                r=self.evaluate(x)
                return r['value'],r['gradient']
        p,r=fit(Objective(),np.array([.2,-.3]),np.eye(2),iterations=30,method='trust-exact')
        self.assertLess(np.linalg.norm(p),1e-3)
        self.assertLess(r['information_normalized_score_norm'],1e-3)
        self.assertTrue(r['trace'])


if __name__ == '__main__':
    unittest.main()
