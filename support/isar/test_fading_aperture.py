import unittest
import numpy as np
from audit_fading_aperture import coherent_fraction, conditioned_information, width


class FadingApertureTests(unittest.TestCase):
    def test_constant_brightness_has_no_flat_filter_loss(self):
        rng = np.random.default_rng(891)
        h = rng.normal(size=(17, 9))+1j*rng.normal(size=(17, 9))
        self.assertAlmostEqual(coherent_fraction(h, (2-3j)*h), 1., places=14)
        self.assertAlmostEqual(coherent_fraction(h, h*np.arange(17)[:, None]),
            coherent_fraction(h*9j, h*np.arange(17)[:, None]*.2), places=14)

    def test_information_eliminates_constant_complex_gain(self):
        rng = np.random.default_rng(162)
        h = rng.normal(size=(20, 7))+1j*rng.normal(size=(20, 7))
        d = rng.normal(size=(20, 7, 2))+1j*rng.normal(size=(20, 7, 2))
        expected = conditioned_information([h], [d])
        actual = conditioned_information([h], [d+h[:, :, None]*np.array([2j, 4-3j])])
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-12)
        self.assertGreater(np.linalg.eigvalsh(actual).min(), 0)

    def test_width_does_not_claim_resolution_outside_the_cut(self):
        x = np.linspace(-3, 3, 601)
        self.assertAlmostEqual(width(x, np.exp(-x*x)), 2*np.sqrt(np.log(2)), places=4)
        self.assertIsNone(width(x, np.ones_like(x)))


if __name__ == '__main__':
    unittest.main()
