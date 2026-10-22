import unittest

import numpy as np

from common_road_phase import phase_seeds, positive_gain


class CommonPhaseTests(unittest.TestCase):
    def test_profiled_positive_amplitude_jacobian_and_inactive_branch(self):
        rng = np.random.default_rng(1023)
        u = rng.normal(size=31)+1j*rng.normal(size=31)
        derivative = rng.normal(size=(31, 5))+1j*rng.normal(size=(31, 5))
        y = 1.3*u+.1*(rng.normal(size=31)+1j*rng.normal(size=31))
        residual, jacobian, amplitude, mean = positive_gain(y, u, derivative)
        for j in range(5):
            step = derivative[:, j]*1e-6
            finite = (positive_gain(y, u+step, derivative)[0]-positive_gain(y, u-step, derivative)[0])/2e-6
            np.testing.assert_allclose(jacobian[:, j], finite, atol=2e-9)
        self.assertGreater(amplitude, 1.)
        np.testing.assert_allclose(np.real(u.conj()@mean), 0., atol=2e-14)
        r, j, a, m = positive_gain(-u, u, derivative)
        np.testing.assert_array_equal(r, u)
        np.testing.assert_array_equal(j, np.zeros_like(derivative))
        self.assertEqual(a, 0.)

    def test_common_phase_gauge_and_archived_carrier_origins(self):
        rng = np.random.default_rng(864)
        beta = rng.uniform(-np.pi, np.pi, 5); beta -= beta[0]
        phi = rng.uniform(-np.pi, np.pi, 9)
        frequency = np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6
        k = 2*np.pi*frequency/299792458.
        # Different empirical polynomial constants are allowed for every
        # target and band. Omitting them destroys the shared phase relation.
        origins = rng.uniform(400., 4000., (9, 5))
        amplitude = rng.uniform(.5, 2., (9, 5))
        archived = amplitude*np.exp(1j*(phi[:, None]+beta-k*origins))
        restored = archived*np.exp(1j*k*origins)
        score, fit = phase_seeds(restored)[0]
        fitted = fit[4:, None]+np.r_[0., fit[:4]]
        np.testing.assert_allclose(np.angle(np.exp(1j*(fitted-phi[:, None]-beta))), 0., atol=3e-11)
        self.assertAlmostEqual(score, np.sum(amplitude**2), places=10)
        self.assertLess(phase_seeds(archived)[0][0], .8*np.sum(amplitude**2))


if __name__ == '__main__':
    unittest.main()
