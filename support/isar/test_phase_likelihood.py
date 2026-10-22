import unittest

import numpy as np

from phase_likelihood import fit


class FullPhaseTests(unittest.TestCase):
    def test_known_common_phase_with_independent_range_responses_and_clutter(self):
        t = np.linspace(0, 30, 301)
        u = 2*t/30-1
        basis = np.polynomial.legendre.legvander(u, 3)[:, 1:]
        truth = np.array([.7, -.3, .2])
        spatial = np.array([[1, .2j], [.3j, 1], [.4, -.3]])
        a = (np.exp(1j*t[:, None, None]*np.array([.8, -1.3]))*spatial[None]).reshape(-1, 2)
        stationary = np.tile(np.array([[1., .2j], [.3j, 1], [-.1, .2]]), (len(t), 1))
        amplitude = np.array([.7+.3j, -.4+.8j])
        y = a@amplitude*np.exp(-1j*np.repeat(basis@truth, 3))+stationary@np.array([15+7j, -6j])
        train = (t//2)%2 == 0
        result = fit(t, a, stationary, y.reshape(-1, 3), train, degree=3)
        np.testing.assert_allclose(result['coefficients_radians'], truth, atol=1e-6)
        self.assertLess(result['fitted']['prediction_error'], 1e-10)


if __name__ == '__main__':
    unittest.main()
