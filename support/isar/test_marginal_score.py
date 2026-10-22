import unittest

import numpy as np

from marginal_scene import profile as reference
from marginal_score import linear, profile
from mixed_scene import mixed_fit
import test_physical_profile as physical_tests


class MarginalScoreTests(unittest.TestCase):
    def test_gls_core_and_diffuse_posterior_equal_augmented_svd(self):
        rng = np.random.default_rng(7842)
        a = rng.normal(size=(70, 12))+1j*rng.normal(size=(70, 12))
        a[:, 4] = a[:, 0]
        y = rng.normal(size=70)+1j*rng.normal(size=70)
        for strength in (1e-5, .7, 1e5, None):
            design = a if strength is not None else a[:, :3]
            actual = linear(design, y, 3, strength)
            expected = mixed_fit(design, y, 3, strength)
            np.testing.assert_allclose(actual['coefficient'], expected['coefficient'], rtol=2e-7, atol=2e-9)
            np.testing.assert_allclose(actual['weighted_residual'], expected['residual'], atol=5e-12)
            self.assertAlmostEqual(actual['quadratic'], expected['value'], delta=1e-11)

    def test_combined_original_row_score_equals_augmented_reference(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        scale = np.array([1., 1., 2.])
        for strength in (500., None):
            core = 3 if strength is None else 2
            actual = profile(band, truth*.4, scale, core, strength, variance=.3)
            expected = reference(band, truth*.4, scale, core, strength, variance=.3, derivatives=True)
            self.assertAlmostEqual(actual['value'], expected['value'], delta=1e-8)
            np.testing.assert_allclose(actual['gradient']/band.energy, expected['gradient']/band.energy, rtol=1e-8, atol=1e-11)
            np.testing.assert_allclose(actual['coefficient'], expected['coefficient'], rtol=1e-8, atol=1e-11)


if __name__ == '__main__':
    unittest.main()
