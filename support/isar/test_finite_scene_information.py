import unittest

import numpy as np
from scipy.linalg import solve_triangular

from finite_scene_information import (amplitude_risk, append_root, probe_groups,
    metric_risk, projected_statistics, retained_information, whiten_block)


class FiniteSceneInformationTests(unittest.TestCase):
    def test_streaming_qr_and_correlated_compression_match_observation_space(self):
        rng = np.random.default_rng(71391)
        signal = rng.normal(size=(70, 5))+1j*rng.normal(size=(70, 5))
        operator = rng.normal(size=(30, 70))+1j*rng.normal(size=(30, 70))
        covariance = operator@operator.conj().T
        root = np.zeros((0, 5), complex)
        for block in np.array_split(signal, 7):
            root = append_root(root, block)
        np.testing.assert_allclose(root.conj().T@root, signal.conj().T@signal, rtol=2e-14, atol=1e-12)
        white = solve_triangular(np.linalg.cholesky(covariance), operator@signal, lower=True)
        result = retained_information(root, white, 1e-12)
        projector = operator.conj().T@np.linalg.solve(covariance, operator)
        u, _, _ = np.linalg.svd(signal, full_matrices=False)
        expected = np.linalg.eigvalsh(u.conj().T@projector@u)
        np.testing.assert_allclose(result['relative_information_eigenvalues'], expected, atol=2e-14)

    def test_scene_mixtures_expose_loss_that_pointwise_energy_hides(self):
        # Both columns are nearly collinear, so each loses almost no energy.
        # Their difference is exactly the direction the compressor removes.
        signal = np.array([[1., 1.], [1e-3, -1e-3], [0., 0.]], complex)
        compressed = signal[:1]
        root = append_root(np.zeros((0, 2), complex), signal)
        result = retained_information(root, compressed, 1e-8)
        self.assertEqual(result['resolved_modes'], 2)
        self.assertGreater(result['isotropic_amplitude_average_retention'], .99999)
        self.assertLess(result['relative_information_eigenvalues'][0], 1e-20)
        projected, rank = projected_statistics(np.array([[1.], [0.], [0.]], complex), signal)
        np.testing.assert_allclose(projected.conj().T@projected, compressed.conj().T@compressed)
        self.assertEqual(rank['resolved_model_modes'], 1)

    def test_mean_model_bias_is_not_hidden_by_a_lossless_template_span(self):
        signal = np.eye(4, dtype=complex)
        template = signal*(1.1+.2j)
        projected, _ = projected_statistics(template, signal)
        np.testing.assert_allclose(retained_information(signal, projected, 1e-12)[
            'relative_information_eigenvalues'], 1., atol=1e-14)
        risk = amplitude_risk(signal, signal, template)
        inverse = 1/(1.1+.2j)
        self.assertAlmostEqual(risk['noise_risk_over_raw_crlb'], abs(inverse)**2)
        self.assertAlmostEqual(risk['average_squared_bias_over_raw_crlb_per_snr'], abs(inverse-1)**2/4)
        self.assertAlmostEqual(risk['mse_over_raw_crlb_at_integrated_snr_db']['30'],
            abs(inverse)**2+1000*abs(inverse-1)**2/4)

    def test_weighted_deramped_block_matches_full_observation_operator(self):
        from scipy.linalg import block_diag
        rng = np.random.default_rng(45713)
        operators = rng.normal(size=(4, 3, 7))+1j*rng.normal(size=(4, 3, 7))
        signal = rng.normal(size=(4, 7, 2))+1j*rng.normal(size=(4, 7, 2))
        variance = rng.uniform(.5, 2., size=(4, 7))
        noise = np.einsum('nrk,nk,nsk->nrs', operators, variance, operators.conj())
        exact = operators@signal
        weights = np.array([.7, 1., 1.4, .9]); deramp = np.exp(1j*np.array([1., 2., 3., 4.]))
        obtained, _ = whiten_block(exact, exact, noise, weights, deramp)
        combined = np.concatenate([a*w*p/weights.sum() for a, w, p in zip(operators, weights, deramp)], axis=1)
        dense_noise = combined@block_diag(*[np.diag(v) for v in variance])@combined.conj().T
        expected = solve_triangular(np.linalg.cholesky(dense_noise),
            combined@signal.reshape(-1, 2), lower=True)
        np.testing.assert_allclose(obtained, expected, rtol=2e-14, atol=2e-14)

    def test_probe_block_rejects_missing_or_cross_gap_rows(self):
        rows = [dict(probe_interval=0, probe_bin=b, probe_offset=i, segment=1,
                     time_s=b+i*.001088, guard=16) for b in range(3) for i in range(4)]
        rows[6]['time_s'] += .001088
        rows[-1]['segment'] = 2
        accepted, rejected = probe_groups(rows, 4)
        self.assertEqual([r['probe_bin'] for r in accepted[0]], [0]*4)
        self.assertEqual(len(accepted), 1)
        self.assertEqual([r['bin'] for r in rejected], [1, 2])

    def test_mismatched_metric_risk_propagates_actual_noise(self):
        rng = np.random.default_rng(352)
        design = rng.normal(size=(14, 3))+1j*rng.normal(size=(14, 3))
        template = design+.02*(rng.normal(size=design.shape)+1j*rng.normal(size=design.shape))
        weights = np.exp(rng.normal(size=14))
        metric = np.diag(np.sqrt(weights))
        root = append_root(np.zeros((0, 3)), design)
        value = metric_risk(root, design, template, metric)
        inverse = np.linalg.solve(template.conj().T@(weights[:, None]*template), template.conj().T*weights)
        information = design.conj().T@design
        expected = np.trace(information@inverse@inverse.conj().T).real/3
        bias = inverse@design-np.eye(3)
        expected_bias = np.trace(information@bias@bias.conj().T).real/(3*np.linalg.norm(root)**2)
        self.assertAlmostEqual(value['noise_risk_over_raw_crlb'], expected, places=12)
        self.assertAlmostEqual(value['average_squared_bias_over_raw_crlb_per_snr'], expected_bias, places=12)


if __name__ == '__main__':
    unittest.main()
