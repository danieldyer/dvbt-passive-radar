import unittest

import numpy as np
from scipy.linalg import block_diag, null_space

from matrix_spectral_gls import expectation, fit_covariance, matched_filter, matched_filters, statistics
from spectral_gls import orthogonal_nuisance


class MatrixSpectralGlsTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(911084)
        n, r, k = 24, 3, 4
        labels = np.arange(n)%2
        q = orthogonal_nuisance(rng.normal(size=(n, r, k))+1j*rng.normal(size=(n, r, k)))
        a = rng.normal(size=(2, r, r))+1j*rng.normal(size=(2, r, r))
        covariance = np.eye(r)*1.5+a@a.conj().transpose(0, 2, 1)
        y = rng.normal(size=(n, r))+1j*rng.normal(size=(n, r))
        signal = rng.normal(size=(n, r))+1j*rng.normal(size=(n, r))
        return rng, labels, q, covariance, y, signal

    def test_restricted_likelihood_and_matrix_gradient_against_full_contrasts(self):
        rng, labels, q, covariance, y, _ = self.fixture()
        record = statistics(y, q, labels)
        value, complete, gradient = expectation(covariance, [record])
        contrast = null_space(q.reshape(y.size, -1).conj().T)
        full = block_diag(*covariance[labels])
        reduced = contrast.conj().T@full@contrast
        z = contrast.conj().T@y.ravel()
        exact = np.log(np.linalg.eigvalsh(reduced)).sum()+np.vdot(z, np.linalg.solve(reduced, z)).real
        self.assertAlmostEqual(value, exact, delta=1e-11)
        direction = rng.normal(size=covariance.shape)+1j*rng.normal(size=covariance.shape)
        direction += direction.conj().transpose(0, 2, 1)
        step = 1e-5
        numeric = (expectation(covariance+step*direction, [record])[0]-
            expectation(covariance-step*direction, [record])[0])/(2*step)
        self.assertAlmostEqual(numeric, float(np.einsum('bij,bji->', gradient, direction).real), delta=2e-8)
        shifted = statistics(y+q@np.array([2e5+3e5j, -9e5j, 4e5, -2e5]), q, labels)
        self.assertAlmostEqual(expectation(covariance, [shifted])[0], value, delta=2e-8)

    def test_unit_response_nuisance_rejection_and_exact_conditional_bound(self):
        rng, labels, q, covariance, _, signal = self.fixture()
        h, information = matched_filter(signal, q, labels, covariance)
        full = block_diag(*covariance[labels])
        self.assertAlmostEqual(abs(np.vdot(h, signal)-1), 0., delta=2e-14)
        np.testing.assert_allclose(q.reshape(signal.size, -1).conj().T@h.ravel(), 0., atol=2e-14)
        self.assertAlmostEqual(float(np.vdot(h, full@h.ravel()).real)*information, 1., delta=1e-13)
        # Independent dense constrained solution verifies the block implementation.
        design = np.c_[signal.ravel(), q.reshape(signal.size, -1)]
        weighted = np.linalg.solve(full, design)
        direct = weighted@np.linalg.solve(design.conj().T@weighted, np.eye(design.shape[1])[:, 0])
        np.testing.assert_allclose(h.ravel(), direct, atol=2e-14)
        noise = (rng.normal(size=(6000, signal.size))+1j*rng.normal(size=(6000, signal.size)))/np.sqrt(2)
        error = noise@(np.linalg.cholesky(full).T@h.ravel().conj())
        self.assertAlmostEqual(float(np.mean(abs(error)**2))*information, 1., delta=.04)

    def test_em_recovers_covariance_and_keeps_the_declared_floor(self):
        rng, labels, q, covariance, _, _ = self.fixture()
        root = np.linalg.cholesky(covariance)
        records = []
        for _ in range(200):
            noise = (rng.normal(size=q.shape[:2])+1j*rng.normal(size=q.shape[:2]))/np.sqrt(2)
            y = np.einsum('nij,nj->ni', root[labels], noise)+q@np.array([30+40j, -90j, 60., 80])
            records.append(statistics(y, q, labels))
        fitted, diagnostic = fit_covariance(records)
        self.assertTrue(diagnostic['converged'], diagnostic)
        self.assertLess(np.linalg.norm(fitted-covariance)/np.linalg.norm(covariance), .045)
        self.assertGreaterEqual(np.linalg.eigvalsh(fitted).min(), 1.-1e-12)
        values = np.array([v['value_per_dof'] for v in diagnostic['trace']])
        self.assertTrue(np.all(np.diff(values) <= 1e-10))

    def test_batched_hypotheses_match_independent_dense_constrained_solutions(self):
        rng, labels, q, covariance, _, signal = self.fixture()
        signals = np.stack([signal, signal[::-1]*np.exp(.37j)], axis=-1)
        h, information = matched_filters(signals, q, labels, covariance)
        full = block_diag(*covariance[labels])
        for j in range(signals.shape[-1]):
            design = np.c_[signals[:, :, j].ravel(), q.reshape(signal.size, -1)]
            weighted = np.linalg.solve(full, design)
            exact = weighted@np.linalg.solve(design.conj().T@weighted, np.eye(design.shape[1])[:, 0])
            np.testing.assert_allclose(h[:, :, j].ravel(), exact, atol=2e-14)
            self.assertAlmostEqual(float(np.vdot(exact, full@exact).real)*information[j], 1., delta=1e-13)

    def test_active_covariance_floor_satisfies_cone_kkt_conditions(self):
        # Independent analytic optimum: first range variance is pinned at one,
        # while the second has variance 24/6. The nuisance consumes one first-
        # range observation; it must not reduce the second range's denominator.
        y = np.zeros((6, 2), complex)
        y[0, 0], y[1, 1] = np.sqrt(1.2), np.sqrt(24.)
        q = np.zeros((6, 2, 1), complex)
        q[-1, 0, 0] = 1.
        labels = np.zeros(6, int)
        records = [statistics(y+q[:, :, 0]*100j, q, labels)]
        fitted, diagnostic = fit_covariance(records)
        self.assertTrue(diagnostic['converged'])
        np.testing.assert_allclose(fitted[0], np.diag([1., 4.]), atol=1e-14)
        _, _, gradient = expectation(fitted, records)
        self.assertGreaterEqual(np.linalg.eigvalsh(gradient[0]).min(), -1e-13)
        np.testing.assert_allclose(gradient@(fitted-np.eye(2)), 0., atol=1e-13)


if __name__ == '__main__':
    unittest.main()
