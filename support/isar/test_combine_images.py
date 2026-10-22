import unittest

import numpy as np

from combine_images import combine


class CoherentImageTests(unittest.TestCase):
    def test_matches_concatenated_complex_least_squares(self):
        rng = np.random.default_rng(719)
        data = rng.normal(size=(5, 13, 4))+1j*rng.normal(size=(5, 13, 4))
        weights = rng.uniform(.5, 2, (5, 13))
        gains = rng.normal(size=5)+1j*rng.normal(size=5)
        information = np.broadcast_to(weights.sum(axis=1)[:, None], (5, 4))
        images = (weights[:, :, None]*data).sum(axis=1)/information
        actual, total, efficiency = combine(images, information, gains)
        design = (gains[:, None]*np.sqrt(weights)).reshape(-1, 1)
        target = (data*np.sqrt(weights[:, :, None])).reshape(-1, 4)
        expected = np.linalg.lstsq(design, target, rcond=None)[0][0]
        np.testing.assert_allclose(actual, expected, atol=2e-15)
        np.testing.assert_allclose(total, np.sum(abs(design)**2))
        self.assertTrue(np.all((efficiency >= 0)&(efficiency <= 1+1e-14)))

    def test_opposite_phases_cancel_without_magnitude_blending(self):
        image, _, efficiency = combine(np.array([[1+2j], [-1-2j]]), np.ones((2, 1)))
        np.testing.assert_array_equal(image, 0)
        np.testing.assert_array_equal(efficiency, 0)


if __name__ == '__main__':
    unittest.main()
