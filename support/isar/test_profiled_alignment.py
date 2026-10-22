import unittest

import numpy as np

from profiled_alignment import ProfiledAlignment
from test_channel_alignment import synthetic


class ProfiledAlignmentTests(unittest.TestCase):
    def test_exact_residual_derivatives_with_refitted_scene_and_stationary_clutter(self):
        alignment, _, _ = synthetic(rate_offsets=[.012, 0., -.008])
        model = ProfiledAlignment(alignment.bands[0], 8., ranges=np.arange(0, 17, 2))
        x = np.array([.5, .3])
        result = model.evaluate(x)
        for j in range(2):
            delta = np.eye(2)[j]*1e-5
            finite = (model.evaluate(x+delta)['residual']-model.evaluate(x-delta)['residual'])/2e-5
            np.testing.assert_allclose(finite, result['jacobian'][:, j], rtol=3e-6, atol=1e-7)
            numeric = (model.objective(x+delta)[0]-model.objective(x-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=1e-8)

    def test_known_range_and_rate_recovered_with_independent_complex_scattering(self):
        rates = [.012, 0., -.008]
        alignment, biases, _ = synthetic(rate_offsets=rates)
        model = ProfiledAlignment(alignment.bands[0], 8.)
        result, report = model.fit(0.)
        self.assertTrue(report['converged'], report)
        self.assertFalse(report['bound_hit'])
        self.assertLess(report['objective'], 1e-7)
        self.assertAlmostEqual(report['range_origin_m'], biases[0], delta=.003)
        self.assertAlmostEqual(report['path_rate_mps'], rates[0], delta=1e-6)
        self.assertGreater(min(report['curvature_eigenvalues']), 0.)


if __name__ == '__main__':
    unittest.main()
