import unittest

import numpy as np

from marginal_information import information


class MarginalInformationTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(481153)
        a = rng.normal(size=(31, 7))+1j*rng.normal(size=(31, 7))
        da = rng.normal(size=(31, 7, 3))+1j*rng.normal(size=(31, 7, 3))
        alpha = rng.normal(size=7)+1j*rng.normal(size=7)
        return rng, a, da, alpha

    def test_full_information_matches_independent_dense_schur_complement(self):
        _, a, da, alpha = self.fixture()
        core, strength, variance = 2, 1.7, .3
        b = a[:, core:]
        covariance = variance*(np.eye(len(a))+b@b.conj().T/strength)
        inverse = np.linalg.inv(covariance)
        mean = np.einsum('niq,i->nq', da[:, :core], alpha[:core])
        derivatives = np.c_[mean, a[:, :core], 1j*a[:, :core]]
        full = 2*np.real(derivatives.conj().T@inverse@derivatives)
        dcs = [variance*(da[:, core:, j]@b.conj().T+b@da[:, core:, j].conj().T)/strength
            for j in range(3)]
        expected_covariance = np.array([[np.trace(inverse@d@inverse@e).real for e in dcs] for d in dcs])
        full[:3, :3] += expected_covariance
        schur = full[:3, :3]-full[:3, 3:]@np.linalg.solve(full[3:, 3:], full[3:, :3])
        result = information(a, alpha, da, core, strength, variance=variance)
        np.testing.assert_allclose(result['total'], schur, rtol=2e-13, atol=2e-10)
        np.testing.assert_allclose(result['covariance'], expected_covariance, rtol=2e-13, atol=2e-10)

    def test_unknown_complex_amplitude_and_common_diffuse_phase_have_zero_information(self):
        _, a, _, alpha = self.fixture()
        # All columns receive the same constant phase. Core coefficients absorb
        # it, and a circular diffuse prior is exactly phase invariant.
        da = (1j*a)[:, :, None]
        result = information(a, alpha, da, 2, .7)
        self.assertLess(float(result['total'][0, 0]), 1e-24)

    def test_information_is_positive_and_basis_invariant(self):
        rng, a, da, alpha = self.fixture()
        q, _ = np.linalg.qr(rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5)))
        first = information(a, alpha, da, 2, .7)
        a[:, 2:] = a[:, 2:]@q
        da[:, 2:] = np.einsum('niq,ij->njq', da[:, 2:], q)
        second = information(a, alpha, da, 2, .7)
        self.assertGreater(np.linalg.eigvalsh(first['total']).min(), 0.)
        np.testing.assert_allclose(first['total'], second['total'], rtol=2e-13, atol=2e-10)

    def test_core_only_information_matches_direct_nuisance_projection(self):
        _, a, da, alpha = self.fixture()
        a, da, alpha = a[:, :2], da[:, :2], alpha[:2]
        q, _ = np.linalg.qr(a)
        derivative = np.einsum('niq,i->nq', da, alpha)
        efficient = derivative-q@(q.conj().T@derivative)
        result = information(a, alpha, da, 2, None, variance=3.)
        np.testing.assert_allclose(result['mean'], 2*np.real(efficient.conj().T@efficient)/3., atol=1e-12)
        np.testing.assert_array_equal(result['covariance'], np.zeros((3, 3)))


if __name__ == '__main__':
    unittest.main()
