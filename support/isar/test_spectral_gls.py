import unittest

import numpy as np
from scipy.linalg import null_space

from spectral_gls import (fit_variance, matched_filter, orthogonal_nuisance,
    restricted_likelihood, sufficient_statistics)


class SpectralGlsTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(5341)
        n, r, k = 24, 3, 4
        q = orthogonal_nuisance(rng.normal(size=(n, r, k))+1j*rng.normal(size=(n, r, k)))
        labels = np.arange(n)%3
        logvar = np.log([.7, 2., 9.])
        target = rng.normal(size=(n, r))+1j*rng.normal(size=(n, r))
        y = rng.normal(size=(n, r))+1j*rng.normal(size=(n, r))
        return rng, q, labels, logvar, target, y

    def test_reml_is_likelihood_of_full_error_contrasts(self):
        _, q, labels, logvar, _, y = self.fixture()
        statistics = sufficient_statistics(y, q, labels)
        value, gradient = restricted_likelihood(logvar, statistics)
        contrast = null_space(q.reshape(y.size, -1).conj().T)
        variance = np.repeat(np.exp(logvar[labels]), y.shape[1])
        covariance = contrast.conj().T@(variance[:, None]*contrast)
        z = contrast.conj().T@y.ravel()
        direct = np.log(np.linalg.eigvalsh(covariance)).sum()+np.vdot(z, np.linalg.solve(covariance, z)).real
        self.assertAlmostEqual(value, direct, delta=1e-11)
        for i in range(len(logvar)):
            delta = np.eye(len(logvar))[i]*1e-5
            numeric = (restricted_likelihood(logvar+delta, statistics)[0]-
                restricted_likelihood(logvar-delta, statistics)[0])/2e-5
            self.assertAlmostEqual(numeric, gradient[i], delta=2e-8)

    def test_gls_unit_complex_response_nuisance_null_and_crlb(self):
        rng, q, labels, logvar, target, _ = self.fixture()
        h, information = matched_filter(target, q, labels, logvar)
        np.testing.assert_allclose(np.vdot(h, target), 1., atol=2e-14)
        np.testing.assert_allclose(q.reshape(target.size, -1).conj().T@h.ravel(), 0., atol=1e-14)
        variance = np.exp(logvar[labels])[:, None]*np.ones(target.shape)
        self.assertAlmostEqual(float(np.sum(abs(h)**2*variance)), 1/information, delta=1e-15)
        count = 8000
        noise = ((rng.normal(size=(count, *target.shape))+1j*rng.normal(size=(count, *target.shape)))/
            np.sqrt(2)*np.sqrt(variance))
        error = np.einsum('ij,nij->n', h.conj(), noise)
        self.assertAlmostEqual(float(np.mean(abs(error)**2))*information, 1., delta=.04)
        amplitude = np.exp(.71j)
        nuisance = q@np.array([2+3j, -1j, 7., -4])
        self.assertAlmostEqual(abs(np.vdot(h, amplitude*target+nuisance)-amplitude), 0., delta=1e-13)

    def test_secondary_variance_fit_recovers_known_colored_noise(self):
        rng, q, labels, logvar, _, _ = self.fixture()
        records = []
        for _ in range(400):
            noise = ((rng.normal(size=q.shape[:2])+1j*rng.normal(size=q.shape[:2]))/
                np.sqrt(2)*np.exp(logvar[labels, None]/2))
            y = noise+q@np.array([3+2j, 4j, -6., 9j])
            records.append(sufficient_statistics(y, q, labels))
        estimate, fit = fit_variance(records)
        self.assertTrue(fit['converged'])
        np.testing.assert_allclose(np.exp(estimate-logvar), 1., atol=.04)


if __name__ == '__main__':
    unittest.main()
