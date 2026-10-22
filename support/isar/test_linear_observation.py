import unittest

import numpy as np

from dense_scatterers import gcv
from linear_observation import LinearObservation
from phase_registration import PhaseRegistration


class LinearObservationTests(unittest.TestCase):
    def test_complex_cost_and_gcv_preserve_discarded_energy_and_dimensions(self):
        rng = np.random.default_rng(942)
        a = rng.normal(size=(240, 25))+1j*rng.normal(size=(240, 25))
        a[:, -1] = a[:, -2]+1e-8*a[:, -1]
        y = rng.normal(size=240)+1j*rng.normal(size=240)
        nuisance = np.linalg.qr(rng.normal(size=(240, 7))+1j*rng.normal(size=(240, 7)))[0]
        a -= nuisance@(nuisance.conj().T@a)
        y -= nuisance@(nuisance.conj().T@y)
        compact = LinearObservation.compress(a, y, 7)
        coefficient = rng.normal(size=25)+1j*rng.normal(size=25)
        for phase in [1., np.exp(.7j)]:
            direct = np.sum(abs(phase*(a@coefficient)-y)**2)
            self.assertAlmostEqual(compact.error(coefficient, phase)/direct, 1., places=13)
        self.assertGreater(compact.orthogonal_energy, .5*np.vdot(y, y).real)
        expected, original = gcv(a, y, 7)
        actual, reduced = gcv(compact.design, compact.data, 0,
            orthogonal_energy=compact.orthogonal_energy, observations=compact.observations)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
        self.assertAlmostEqual(reduced['regularization_strength']/original['regularization_strength'], 1., places=12)
        self.assertEqual(reduced['effective_observation_count'], 233)

    def test_wide_and_rank_deficient_observations_preserve_error(self):
        rng = np.random.default_rng(943)
        for rows, columns in [(7, 12), (12, 12), (120, 12)]:
            a = rng.normal(size=(rows, columns))+1j*rng.normal(size=(rows, columns))
            a[:, -2:] = a[:, :2]
            y = rng.normal(size=rows)+1j*rng.normal(size=rows)
            c = rng.normal(size=columns)+1j*rng.normal(size=columns)
            compact = LinearObservation.compress(a, y)
            direct = np.sum(abs(a@c-y)**2)
            self.assertAlmostEqual(compact.error(c)/direct, 1., places=13)

    def test_phase_objective_gradient_and_solution_match_uncompressed(self):
        rng = np.random.default_rng(944)
        a = [rng.normal(size=(120, 12))+1j*rng.normal(size=(120, 12)) for _ in range(5)]
        c = rng.normal(size=12)+1j*rng.normal(size=12)
        phases = np.exp(1j*np.array([.3, -.7, 1.1, 0., -.4]))
        y = [g*b@c+.01*(rng.normal(size=120)+1j*rng.normal(size=120)) for g, b in zip(phases, a)]
        original = PhaseRegistration(a, y, .02)
        compact = [LinearObservation.compress(b, d) for b, d in zip(a, y)]
        reduced = PhaseRegistration([b.design for b in compact], [b.data for b in compact], .02,
            orthogonal_energy=sum(b.orthogonal_energy for b in compact))
        probe = np.array([.2, -.6, .8, -.5])
        self.assertAlmostEqual(original.objective(probe)[0], reduced.objective(probe)[0], places=13)
        np.testing.assert_allclose(original.objective(probe)[1], reduced.objective(probe)[1], atol=1e-14)
        before, _, _ = original.fit()
        after, _, detail = reduced.fit()
        self.assertTrue(detail['converged'])
        np.testing.assert_allclose(before, after, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
