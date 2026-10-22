import unittest

import numpy as np

from partial_phase import common_visibility, fit


class PartialPhaseTests(unittest.TestCase):
    def test_recovers_phases_despite_strong_unshared_scattering(self):
        rng = np.random.default_rng(5189)
        anchors = np.array([1, 4, 7, 10])
        designs = [rng.normal(size=(100, 12))+1j*rng.normal(size=(100, 12)) for _ in range(3)]
        coefficient = 30*(rng.normal(size=(3, 12))+1j*rng.normal(size=(3, 12)))
        band_phase = np.array([.5, -.4, 0.])
        point_phase = rng.uniform(-np.pi, np.pi, 4)
        coefficient[:, anchors] = rng.uniform(.5, 2., (3, 4))*np.exp(1j*(band_phase[:, None]+point_phase))
        data = [a@c for a, c in zip(designs, coefficient)]
        recovered, result = fit(designs, data, anchors, reference=2, coordinate_sweeps=8)
        self.assertTrue(result['converged'], result)
        np.testing.assert_allclose(recovered, coefficient, atol=2e-7)
        np.testing.assert_allclose(np.exp(1j*np.array(result['channel_phase_radians'])), np.exp(1j*band_phase), atol=2e-7)

    def test_visibility_is_exact_leave_one_column_energy_reduction(self):
        rng = np.random.default_rng(2535)
        designs = [rng.normal(size=(45, 6))+1j*rng.normal(size=(45, 6)) for _ in range(3)]
        data = [rng.normal(size=45)+1j*rng.normal(size=45) for _ in designs]
        ranking, energy = common_visibility(designs, data)
        for c, (a, y) in enumerate(zip(designs, data)):
            full = np.linalg.norm(y-a@np.linalg.lstsq(a, y, rcond=None)[0])**2
            for j in range(6):
                reduced = np.delete(a, j, axis=1)
                missing = np.linalg.norm(y-reduced@np.linalg.lstsq(reduced, y, rcond=None)[0])**2
                self.assertAlmostEqual(missing-full, energy[c, j], places=11)
        np.testing.assert_array_equal(ranking, np.argsort(-energy.min(axis=0)))


if __name__ == '__main__':
    unittest.main()
