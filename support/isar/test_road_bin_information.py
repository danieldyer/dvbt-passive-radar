import unittest

import numpy as np

from road_bin_information import AveragingInformation, point_fisher


class RoadBinInformationTests(unittest.TestCase):
    def test_constant_bin_means_preserve_all_gaussian_information(self):
        rng = np.random.default_rng(472161)
        bins = np.repeat(np.arange(7), [2, 4, 3, 7, 5, 4, 3])
        weight = rng.uniform(.3, 2., len(bins))
        selected = np.array([True, True, False, True, False, True, True])
        mean = rng.normal(size=(7, 3, 4))+1j*rng.normal(size=(7, 3, 4))
        model = AveragingInformation(7, 3, 3, selected)
        for begin in range(0, len(bins), 5):
            take = slice(begin, begin+5); values = mean[bins[take]]
            model.add(bins[take], weight[take], .7/weight[take], values[:, :, 0], values[:, :, 1:])
        result = model.finish()
        self.assertAlmostEqual(result['matched_snr_retention'], 1., places=13)
        for check in result['local_information']:
            np.testing.assert_allclose(check['retention_eigenvalues'], 1., atol=2e-13)

    def test_irregular_phase_and_variance_match_dense_gaussian_operator(self):
        rng = np.random.default_rng(716811)
        bins = np.repeat(np.arange(4), [3, 4, 3, 2]); n = len(bins)
        weight, variance = rng.uniform(.2, 2, (2, n))
        mean = rng.normal(size=(n, 2, 4))+1j*rng.normal(size=(n, 2, 4))
        model = AveragingInformation(4, 2, 3, np.ones(4, bool))
        model.add(bins, weight, variance, mean[:, :, 0], mean[:, :, 1:])
        result = model.finish(np.array([.3, .4, .1, .6]))
        averaging = np.zeros((4, n)); averaging[bins, np.arange(n)] = weight
        averaging /= averaging.sum(axis=1)[:, None]
        transform = np.kron(averaging, np.eye(2))
        full = mean.reshape(2*n, 4)
        cov = np.diag(np.repeat(variance, 2))
        raw_gram = full.conj().T@np.linalg.solve(cov, full)
        observed = transform@full; reduced_cov = transform@cov@transform.T
        compressed = observed.conj().T@np.linalg.solve(reduced_cov, observed)
        np.testing.assert_allclose(result['raw_information'], point_fisher(raw_gram), atol=1e-12)
        np.testing.assert_allclose(result['binned_information'], point_fisher(compressed), atol=1e-12)
        self.assertAlmostEqual(result['matched_snr_retention'], compressed[0, 0].real/raw_gram[0, 0].real, places=13)
        empirical = np.diag(np.repeat(1/np.array([.3, .4, .1, .6]), 2))
        u = observed[:, 0]; template = empirical@u
        snr = abs(np.vdot(template, u))**2/np.vdot(template, reduced_cov@template).real
        self.assertAlmostEqual(result['fixed_empirical_metric_snr_loss_db'],
            -10*np.log10(snr/compressed[0, 0].real), places=12)

    def test_gain_gauge_subtraction_does_not_change_efficient_information(self):
        rng = np.random.default_rng(817661)
        a = rng.normal(size=(100, 5))+1j*rng.normal(size=(100, 5))
        b = a.copy(); b[:, 1:] -= a[:, :1]*(rng.normal(size=4)+1j*rng.normal(size=4))*10
        np.testing.assert_allclose(point_fisher(a.conj().T@a), point_fisher(b.conj().T@b), rtol=2e-12, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
