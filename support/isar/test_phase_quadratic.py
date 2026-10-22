import unittest

import numpy as np

from phase_quadratic import phase_matrix, solve


class PhaseQuadraticTests(unittest.TestCase):
    def test_qr_likelihood_equals_original_joint_complex_fit(self):
        rng = np.random.default_rng(3209)
        designs = [rng.normal(size=(n, 4))+1j*rng.normal(size=(n, 4)) for n in (11, 13, 17)]
        data = [rng.normal(size=len(a))+1j*rng.normal(size=len(a)) for a in designs]
        matrix, r, projected = phase_matrix(designs, data)
        phase = np.exp(1j*np.array([.3, -1.4, 2.7]))
        coefficient = np.linalg.solve(r, projected@phase)
        direct = np.linalg.lstsq(np.concatenate(designs), np.concatenate(
            [z*y for z, y in zip(phase, data)]), rcond=None)[0]
        np.testing.assert_allclose(coefficient, direct, rtol=1e-13, atol=1e-13)
        captured = np.linalg.norm(np.concatenate(designs)@direct)**2
        self.assertAlmostEqual(captured, np.vdot(phase, matrix@phase).real, places=12)

    def test_noiseless_phase_recovery_has_global_certificate_and_correct_sign(self):
        rng = np.random.default_rng(7132)
        designs = [rng.normal(size=(20, 3))+1j*rng.normal(size=(20, 3)) for _ in range(5)]
        truth = np.exp(1j*np.array([.3, -2., 1.2, 0., 2.5]))
        scene = np.array([.4+.8j, -.2+.3j, .7-.4j])
        data = [g*(a@scene) for g, a in zip(truth, designs)]
        matrix, r, projected = phase_matrix(designs, data)
        phase, result = solve(matrix, reference=3)
        np.testing.assert_allclose(phase.conj(), truth, atol=1e-7)
        self.assertTrue(result['globally_certified'])
        self.assertLess(result['absolute_gap'], 1e-5)
        self.assertGreaterEqual(result['dual_minimum_slack_eigenvalue'], 0.)
        np.testing.assert_allclose(np.linalg.solve(r, projected@phase), scene, atol=1e-8)

    def test_rank_one_analytic_bound_and_diagonal_invariance(self):
        vector = np.array([1., .4+.3j, -1.3+.6j, .2-1.5j])
        matrix = np.outer(vector, vector.conj())
        phase, result = solve(matrix)
        expected = np.sum(abs(vector))**2
        self.assertAlmostEqual(result['captured_energy'], expected, places=10)
        self.assertGreaterEqual(result['upper_bound']+1e-11, expected)
        changed, detail = solve(matrix+np.diag([100., -2., 10., 7.]))
        np.testing.assert_allclose(phase, changed, atol=1e-8)
        self.assertAlmostEqual(detail['captured_energy']-result['captured_energy'], 115., places=10)

    def test_dual_bound_covers_exhaustive_three_channel_search(self):
        rng = np.random.default_rng(551)
        matrix = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3))
        matrix = (matrix+matrix.conj().T)/2
        phase, result = solve(matrix)
        angle = np.arange(0., 2*np.pi, .03)
        p, q = np.meshgrid(angle, angle)
        grid = np.column_stack([np.ones(p.size), np.exp(1j*p.ravel()), np.exp(1j*q.ravel())])
        energy = np.einsum('ni,ij,nj->n', grid.conj(), matrix, grid).real.max()
        self.assertGreaterEqual(result['upper_bound']+1e-10, energy)
        self.assertGreaterEqual(result['captured_energy']+1e-10, energy)

    def test_no_phase_information_is_not_reported_as_identifiable(self):
        phase, result = solve(np.diag([1., 4., 2.]))
        self.assertFalse(result['phase_identifiable'])
        self.assertEqual(result['absolute_gap'], 0.)


if __name__ == '__main__':
    unittest.main()
