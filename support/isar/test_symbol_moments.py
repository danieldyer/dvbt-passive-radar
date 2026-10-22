import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss

from symbol_moments import moments


class SymbolMomentTests(unittest.TestCase):
    def test_moments_and_derivatives_against_independent_quadrature(self):
        omega = np.r_[-120., -20., -1., -.1, -.099, -1e-8, 0., 1e-8, .099, .1, 1., 20., 120.]
        time, weight = leggauss(256)
        time, weight = time/2, weight/2
        tone = np.exp(1j*omega[:, None]*time)
        reference = (tone@weight, (tone*time)@weight,
                     (tone*(1j*time))@weight, (tone*(1j*time**2))@weight)
        for actual, expected in zip(moments(omega), reference):
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(complex(moments(0.)[3]), 1j/12)


if __name__ == '__main__':
    unittest.main()
