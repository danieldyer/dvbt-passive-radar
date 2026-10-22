import unittest

import numpy as np

from phase_branches import BranchScreen
from coherent_fit import OTHER


class BranchScreenTest(unittest.TestCase):
    def test_screen_is_complex_amplitude_profile_likelihood(self):
        rng = np.random.default_rng(1770)
        z = rng.normal(size=(3, 5))+1j*rng.normal(size=(3, 5))
        slopes = rng.normal(size=(3, 5))
        shifts = np.tile(np.linspace(-4, 5, 61), (3, 1))
        beta = rng.normal(size=(7, 4))
        screen = BranchScreen(z, slopes, shifts)
        actual = screen.scores(beta)
        for target in range(3):
            for trial in range(7):
                phase = np.zeros(5);phase[OTHER] = beta[trial]
                for k in (0, 20, 60):
                    u = abs(z[target])*np.exp(1j*(phase-slopes[target]*shifts[target, k]))
                    expected = abs(np.vdot(u, z[target]))**2/np.vdot(u, u).real
                    self.assertAlmostEqual(actual[target, trial, k], expected, places=12)
        np.testing.assert_allclose(screen.objective(beta.T),
                                   screen.energy.sum()-actual.max(axis=-1).sum(axis=0))

    def test_all_path_branches_and_shared_target_phase_are_eliminated(self):
        rng = np.random.default_rng(929)
        grid = np.tile(np.linspace(-25, 25, 501), (4, 1))
        slopes = np.tile(np.array([-.15, 0, .154, .74, .9]), (4, 1))
        shifts = np.array([-17.3, 19.8, 8.1, -2.2])
        beta = np.array([1.8, -.7, 2.9, -.4])
        phases = np.zeros(5);phases[OTHER] = beta
        target_phase = rng.uniform(-np.pi, np.pi, 4)
        z = rng.uniform(.2, 3., (4, 5))*np.exp(1j*(target_phase[:, None]+phases-slopes*shifts[:, None]))
        screen = BranchScreen(z, slopes, grid)
        self.assertAlmostEqual(screen.objective(beta), 0., places=12)
        np.testing.assert_allclose(screen.path_shifts(beta), shifts, atol=1e-12)
        rotated = BranchScreen(z*np.exp(.9j), slopes, grid)
        np.testing.assert_allclose(rotated.scores(beta), screen.scores(beta), atol=2e-14)


if __name__ == '__main__':
    unittest.main()
