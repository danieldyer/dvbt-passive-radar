from itertools import combinations
import unittest

import numpy as np

from audit_pair_phase import closures
from phase_coupling import PhaseCoupling


class PairPhaseTests(unittest.TestCase):
    def test_known_band_phases_recover_and_close_despite_independent_strengths(self):
        rng = np.random.default_rng(20261015)
        design = [rng.normal(size=(70, 5))+1j*rng.normal(size=(70, 5)) for _ in range(4)]
        point_phase = rng.uniform(-np.pi, np.pi, 5)
        bands = np.array([1.9, -.7, 2.8, -2.7])
        coefficients = rng.uniform(.2, 2., (4, 5))*np.exp(1j*(point_phase+bands[:, None]))
        data = [a@c for a, c in zip(design, coefficients)]
        phases = np.zeros((4, 4))
        for i, j in combinations(range(4), 2):
            model = PhaseCoupling([design[i], design[j]], [data[i], data[j]], reference=0)
            _, fit = model.fit()
            self.assertTrue(fit['converged'])
            phases[i, j] = fit['channel_phase_radians'][1]
            phases[j, i] = -phases[i, j]
            self.assertLess(abs(np.angle(np.exp(1j*(phases[i, j]-bands[j]+bands[i])))), 1e-7)
        self.assertLess(max(abs(c['phase_radians']) for c in closures(phases)), 1e-7)
        # A single inconsistent pair must remain detectable across angle wraps.
        phases[0, 2] += .42
        phases[2, 0] -= .42
        errors = np.array([abs(c['phase_radians']) for c in closures(phases)])
        self.assertEqual(np.sum(errors > .4), 2)


if __name__ == '__main__':
    unittest.main()
