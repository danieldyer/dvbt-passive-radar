import unittest

import numpy as np

from fit_physical_profile import solve
from point_profile import PointProfile, profile_points
from test_channel_alignment import synthetic


class PointProfileTests(unittest.TestCase):
    def test_residual_derivatives_include_refitted_complex_scattering(self):
        alignment, biases, _ = synthetic([.25, -.15, -.2, .1])
        band = alignment.bands[0]
        points = band.points.copy()
        points[:, :2] += [[.05, -.03], [.1, -.06], [-.08, .04]]
        result = profile_points(band, points, biases[0])
        for j in range(2*len(points)):
            delta = np.zeros_like(points)
            delta[j//2, j%2] = .0002
            plus, minus = [profile_points(band, p, biases[0]) for p in (points+delta, points-delta)]
            finite = (plus['residual']-minus['residual'])/.0004
            self.assertLess(np.linalg.norm(finite-result['jacobian'][:, j])/np.linalg.norm(finite), 3e-5)

    def test_known_subpixel_scene_recovers_with_independent_band_amplitudes(self):
        shift = [.25, -.15, -.2, .1]
        alignment, biases, _ = synthetic(shift)
        model = PointProfile(alignment.bands, alignment.bands[0].points, biases)
        count = len(model.initial)
        p, result = solve(model, np.empty((0, count)), np.empty(0), np.empty(0),
            np.full(count, -1.), np.full(count, 1.), iterations=100)
        self.assertTrue(result['converged'], result)
        np.testing.assert_allclose(p, [0., 0., *shift], atol=.004)
        self.assertLess(result['objective'], 1e-7)
        before, after = model.validation(model.initial), model.validation(p)
        self.assertLess(after['prediction_error']/before['prediction_error'], 1e-4)


if __name__ == '__main__':
    unittest.main()
