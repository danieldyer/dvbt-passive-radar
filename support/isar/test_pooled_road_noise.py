import unittest

import numpy as np

from pooled_road_noise import residual_scale


class PooledRoadNoiseTests(unittest.TestCase):
    def test_unknown_static_and_moving_range_vectors_do_not_change_noise_estimate(self):
        rng = np.random.default_rng(883161)
        t = np.sort(rng.uniform(0, .02, 19))
        temporal = np.column_stack((np.ones(len(t)), np.exp(2j*np.pi*18*t)))
        variance = rng.uniform(.4, 1.3, len(t))
        noise = np.sqrt(variance[:, None]/2)*(rng.normal(size=(len(t), 13))+1j*rng.normal(size=(len(t), 13)))
        unknown = 20*(rng.normal(size=(2, 13))+1j*rng.normal(size=(2, 13)))
        clean, degrees, rank = residual_scale(noise, variance, temporal)
        mixed, _, _ = residual_scale(noise+temporal@unknown, variance, temporal)
        self.assertEqual(degrees, 17*13); self.assertEqual(rank, 2)
        self.assertAlmostEqual(clean, mixed, places=13)

    def test_exact_gaussian_moments_and_mean_innovation_independence(self):
        rng = np.random.default_rng(916213)
        n, ranges, trials = 18, 9, 1600
        t = np.arange(n)/1000
        temporal = np.column_stack((np.ones(n), np.exp(2j*np.pi*21*t)))
        variance = np.linspace(.6, 1.4, n)
        q = np.linalg.qr(temporal/np.sqrt(variance[:, None]))[0]
        scales, amplitudes = [], []
        for _ in range(trials):
            noise = np.sqrt(variance[:, None]*1.7/2)*(rng.normal(size=(n, ranges))+1j*rng.normal(size=(n, ranges)))
            scale, degrees, rank = residual_scale(noise, variance, temporal)
            scales.append(scale)
            amplitudes.append(np.vdot(q[:, 0], noise[:, 0]/np.sqrt(variance)).real)
        self.assertEqual(rank, 2)
        self.assertLess(abs(np.mean(scales)-1.7), 4*1.7/np.sqrt(degrees*trials))
        self.assertLess(abs(np.var(scales, ddof=1)/(1.7**2/degrees)-1), .13)
        self.assertLess(abs(np.corrcoef(scales, amplitudes)[0, 1]), .09)

    def test_coincident_temporal_modes_use_the_actual_rank(self):
        rng = np.random.default_rng(71643)
        data = rng.normal(size=(18, 7))+1j*rng.normal(size=(18, 7))
        _, degrees, rank = residual_scale(data, np.ones(18), np.ones((18, 2), complex))
        self.assertEqual(rank, 1); self.assertEqual(degrees, 17*7)


if __name__ == '__main__':
    unittest.main()
