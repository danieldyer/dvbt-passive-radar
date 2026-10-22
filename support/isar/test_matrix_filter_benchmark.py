import unittest

import numpy as np

from matrix_filter_benchmark import fold_keys, frequency_labels, score, signal_block
from spectral_gls import orthogonal_nuisance


class MatrixFilterBenchmarkTests(unittest.TestCase):
    def test_nested_splits_exclude_outer_range_and_test_times(self):
        train_tiles = [0, 1, 2, 7, 8, 9]
        records = {(g, t): None for g in [0, 1, 2, 4, 5, 6] for t in train_tiles}
        for outer in [0, 1, 2, 4, 5, 6]:
            for heldout in np.array_split(train_tiles, 3):
                keys = fold_keys(records, outer, heldout)
                self.assertEqual(len(keys), 5*4)
                self.assertTrue(all(g != outer and t not in heldout and t not in [3, 4, 5, 6]
                    for g, t in keys))

    def test_complex_injection_retains_amplitude_and_phase_without_mutating_input(self):
        rng = np.random.default_rng(91052)
        times = np.arange(100)*.1
        points = rng.normal(size=(100, 3, 2))+1j*rng.normal(size=(100, 3, 2))
        a = signal_block(times, points, times[20:84])
        q = orthogonal_nuisance(rng.normal(size=(64, 3, 4))+1j*rng.normal(size=(64, 3, 4)))
        y = rng.normal(size=(64, 3))+1j*rng.normal(size=(64, 3))
        original = y.copy()
        labels = frequency_labels(len(y), .1)
        covariance = np.tile(np.diag([1., 3., 7.]), (4, 1, 1))
        baseline = score(a, y, q, labels, covariance)
        alpha = -3.2+7.1j
        for j in range(a.shape[-1]):
            injected = score(a, y+alpha*a[:, :, j]+q@np.array([20, 30j, 40, -50j]), q, labels, covariance)
            self.assertAlmostEqual(abs(injected['amplitude'][j]-baseline['amplitude'][j]-alpha), 0., delta=1e-12)
        np.testing.assert_array_equal(y, original)
        self.assertLess(baseline['response_error'], 1e-13)
        self.assertLess(baseline['nuisance_error'], 1e-13)


if __name__ == '__main__':
    unittest.main()
