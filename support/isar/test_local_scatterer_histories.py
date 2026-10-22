import unittest
import numpy as np
from audit_local_scatterer_histories import deblend


class TestHistories(unittest.TestCase):
    def test_complex_deblending_and_covariance(self):
        rng = np.random.default_rng(45)
        h = rng.normal(size=(15, 33, 3)) + 1j*rng.normal(size=(15, 33, 3))
        h[:, :, 1] += .7*h[:, :, 0]
        coefficient = np.array([.3+1j, -.2-.3j, .8-.1j])
        y = np.einsum('trp,p->tr', h, coefficient)
        value, covariance = deblend(h, y)
        np.testing.assert_allclose(value, coefficient, atol=2e-15)
        a = h.reshape(-1, 3)
        np.testing.assert_allclose(covariance @ (a.conj().T@a), np.eye(3), atol=1e-14)
        # A deterministic independent noise matrix verifies the GLS map.
        noise = rng.normal(size=y.shape) + 1j*rng.normal(size=y.shape)
        noisy, _ = deblend(h, y+noise)
        np.testing.assert_allclose(noisy-value, covariance @ (a.conj().T@noise.ravel()), atol=2e-15)

    def test_unresolved_columns_are_flagged(self):
        h = np.ones((5, 8, 2), complex)
        with self.assertRaises(ValueError):
            deblend(h, np.ones((5, 8)))


if __name__ == '__main__':
    unittest.main()
