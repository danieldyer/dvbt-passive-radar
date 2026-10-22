import unittest

import numpy as np

from profile_information import from_jacobian, profile


class ProfileInformationTests(unittest.TestCase):
    def test_matches_direct_projection_with_real_parameter_scales_and_nulls(self):
        rng = np.random.default_rng(9021)
        nuisance = rng.normal(size=(100, 5))
        nuisance[:, 4] = 0.
        nuisance *= np.array([.01, 1., 20., 3., 1.])
        interest = rng.normal(size=(100, 3))+nuisance[:, :3]@rng.normal(size=(3, 3))
        jacobian = np.c_[nuisance, interest]
        expected = interest-nuisance@np.linalg.lstsq(nuisance, interest, rcond=1e-12)[0]
        actual, details = profile(2*jacobian.T@jacobian, np.arange(5, 8), np.arange(5), cutoff=1e-12)
        np.testing.assert_allclose(actual, 2*expected.T@expected, rtol=1e-11, atol=1e-10)
        self.assertEqual(details['nuisance_rank'], 4)

    def test_exact_gain_scene_gauge_has_zero_information(self):
        rng = np.random.default_rng(441)
        nuisance = rng.normal(size=(40, 4))
        interest = nuisance@rng.normal(size=(4, 2))
        j = np.c_[nuisance, interest]
        fitted, _ = profile(2*j.T@j, [4, 5], [0, 1, 2, 3], cutoff=1e-12)
        self.assertLess(np.linalg.norm(fitted), 1e-10)

    def test_square_root_projection_retains_small_information_beside_large_nuisance(self):
        rng = np.random.default_rng(2304)
        axes, _ = np.linalg.qr(rng.normal(size=(100, 3)))
        u, v, w = axes.T
        j = np.c_[1e9*u, 1e8*u+3*v, 3e7*u-2*v+w, np.zeros(100)]
        information, detail = from_jacobian(np.sqrt(2)*j, [1, 2], [0, 3], cutoff=1e-12)
        np.testing.assert_allclose(information, [[18., -12.], [-12., 10.]], atol=2e-7)
        self.assertEqual(detail['nuisance_rank'], 1)
        self.assertGreater(np.linalg.eigvalsh(information).min(), 0.)


if __name__ == '__main__':
    unittest.main()
