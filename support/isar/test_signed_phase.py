import unittest

import numpy as np

from phase_coupling import PhaseCoupling
from signed_phase import maximum


class SignedPhaseTests(unittest.TestCase):
    def test_exact_conditional_update_and_unresolved_two_ray_sign_change(self):
        rng = np.random.default_rng(41375)
        grid = np.linspace(-np.pi/2, np.pi/2, 10001)
        for _ in range(8):
            values = rng.normal(size=5)+1j*rng.normal(size=5)
            phase, strength = maximum(values)
            dense = np.sum(np.real(np.exp(-1j*grid[:, None])*values)**2, axis=1).max()
            self.assertGreaterEqual(strength@strength+1e-12, dense)
            self.assertLess(strength@strength-dense, 1e-6)
        frequency = np.array([177.5, 184.5, 191.625, 219.5, 226.5])*1e6
        k = 2*np.pi*frequency/299792458.
        # Independent equal-strength paths straddle the reference by 0.4 m.
        # Their unresolved field is real after removing the centre phase and
        # changes sign within the observed band group, without calibration error.
        centre = 14700.
        echo = np.exp(-1j*k*(centre-.4))+np.exp(-1j*k*(centre+.4))
        response = echo*np.exp(1j*k*centre)
        self.assertLess(abs(response.imag).max(), 2e-11)
        self.assertTrue(response.real[0] > 0 and response.real[-1] < 0)
        phase, strength = maximum(response)
        np.testing.assert_allclose(np.exp(1j*phase)*strength, response, atol=2e-11)
        phase, strength = maximum(np.zeros(5, complex))
        self.assertEqual(phase, 0.)
        np.testing.assert_array_equal(strength, 0.)

    def test_gradient_recovery_and_exact_band_pi_ambiguity(self):
        rng = np.random.default_rng(3156)
        designs = [rng.normal(size=(80, 5))+1j*rng.normal(size=(80, 5)) for _ in range(3)]
        point_phase = rng.uniform(-1., 1., 5)
        band_phase = np.array([.35, -.55, 0.])
        strength = rng.uniform(.5, 1.5, (3, 5))*np.array([[1,-1,1,-1,1],[-1,1,1,-1,-1],[1,1,-1,1,1]])
        truth = strength*np.exp(1j*(point_phase+band_phase[:, None]))
        data = [a@c for a, c in zip(designs, truth)]
        model = PhaseCoupling(designs, data, reference=2, nonnegative=False)
        x = np.r_[point_phase, band_phase[:2]]+rng.normal(0, .2, 7)
        value = model.evaluate(x)
        for j in range(len(x)):
            step = np.eye(len(x))[j]*1e-6
            numeric = (model.objective(x+step)[0]-model.objective(x-step)[0])/2e-6
            self.assertAlmostEqual(value['gradient'][j], numeric, delta=1e-8)
        shifted = x.copy(); shifted[-1] += np.pi
        np.testing.assert_allclose(model.evaluate(shifted)['coefficients'], value['coefficients'], atol=1e-13)
        coefficient, fitted = model.fit(coordinate_sweeps=8)
        self.assertTrue(fitted['converged'])
        self.assertLess(fitted['objective'], 1e-15)
        np.testing.assert_allclose(coefficient, truth, atol=2e-7)
        np.testing.assert_allclose(np.exp(2j*np.array(fitted['channel_phase_radians'])),
                                   np.exp(2j*band_phase), atol=2e-7)


if __name__ == '__main__':
    unittest.main()
