import unittest
from types import SimpleNamespace

import numpy as np

from compression_information import point_information, range_crop_information, relative_information, temporal_information, scene_information


class CompressionInformationTests(unittest.TestCase):
    def test_joint_scene_modes_expose_destructive_interference(self):
        # Each column loses under 0.01 dB, but their difference loses half its
        # information. A pointwise test must not certify every scene mixture.
        original = np.array([[100., 100.], [1., -1.], [1., -1.]])
        compressed = original[:2]
        result = scene_information(original.T@original, compressed.T@compressed)
        np.testing.assert_allclose(result['relative_information_eigenvalues'], [.5, 1.], atol=1e-12)
        self.assertEqual(result['unresolved_modes'], 0)

    def test_unknown_complex_amplitude_matches_real_nuisance_schur_complement(self):
        rng = np.random.default_rng(2093)
        signal = rng.normal(size=40)+1j*rng.normal(size=40)
        derivative = rng.normal(size=(40, 3))+1j*rng.normal(size=(40, 3))
        design = np.c_[signal, 1j*signal, derivative]
        full = 2*(design.conj().T@design).real
        expected = full[2:, 2:]-full[2:, :2]@np.linalg.solve(full[:2, :2], full[:2, 2:])
        energy, actual = point_information(signal, derivative)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(relative_information(actual, actual), 1., atol=1e-13)
        self.assertAlmostEqual(energy, np.sum(abs(signal)**2), places=12)

    def test_integer_matched_sample_is_sufficient_for_known_target_amplitude(self):
        w = np.linspace(-.8*np.pi, .8*np.pi, 511)
        q = 1+.4*np.sin(w)
        result = range_crop_information(w, q, np.arange(-8, 9), 3.)
        self.assertAlmostEqual(result['matched_snr_fraction'], 1., places=12)
        self.assertLess(result['delay_information_fraction'], 1.)
        self.assertGreater(result['delay_information_fraction'], .999)

    def test_temporal_compression_matches_dense_linear_gaussian_reference(self):
        rng = np.random.default_rng(5531)
        weights = rng.uniform(.5, 2, 12)
        weights[3] = 0.
        omega = np.linspace(-.3, .4, 12)
        covariance = np.array([[1., .2+.1j], [.2-.1j, 1.]])
        field = SimpleNamespace(data={'row_information': weights, 'centre_omega': omega},
            metadata={'block_rows': 3}, whiten=np.linalg.inv(np.linalg.cholesky(covariance)))
        signal = rng.normal(size=(12, 2))+1j*rng.normal(size=(12, 2))
        derivative = rng.normal(size=(12, 2, 3))+1j*rng.normal(size=(12, 2, 3))
        measured = temporal_information(field, signal, derivative)
        keep = np.repeat(weights > 0, 2)
        bins = np.zeros((4, 12))
        for i in range(4):
            bins[i, 3*i:3*i+3] = weights[3*i:3*i+3]/weights[3*i:3*i+3].sum()
        operator = np.kron(bins, np.eye(2))[:, keep]
        variance = np.repeat((1+omega**2/12)/(8192*np.where(weights > 0, weights, 1)), 2)
        dense = np.kron(np.eye(12), covariance)*variance[:, None]
        dense = dense[np.ix_(keep, keep)]
        lower = np.linalg.cholesky(dense)
        original = point_information(np.linalg.solve(lower, signal.ravel()[keep]),
            np.linalg.solve(lower, derivative.reshape(-1, 3)[keep]))[1]
        lower = np.linalg.cholesky(operator@dense@operator.conj().T)
        compressed = point_information(np.linalg.solve(lower, operator@signal.ravel()[keep]),
            np.linalg.solve(lower, operator@derivative.reshape(-1, 3)[keep]))[1]
        np.testing.assert_allclose(measured['original_information'], original, rtol=1e-12)
        np.testing.assert_allclose(measured['compressed_information'], compressed, rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
