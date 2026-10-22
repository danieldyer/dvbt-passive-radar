import unittest

import numpy as np

from phase_coupling import PhaseCoupling
from positive_phase import maximum


class PositivePhaseTests(unittest.TestCase):
    def test_exact_maximum_dominates_dense_phase_grid(self):
        rng = np.random.default_rng(17105)
        grid = np.linspace(-np.pi, np.pi, 100001)
        for count in (1, 2, 5, 11):
            for _ in range(10):
                b = rng.normal(size=count)+1j*rng.normal(size=count)
                phase, magnitude = maximum(b)
                actual = np.sum(magnitude**2)
                dense = np.sum(np.maximum(0., np.real(np.exp(-1j*grid[:, None])*b))**2, axis=1).max()
                self.assertGreaterEqual(actual+1e-12, dense)
                self.assertLess(actual-dense, 1e-7*max(actual, 1.))
                np.testing.assert_allclose(magnitude, np.maximum(0., np.real(np.exp(-1j*phase)*b)))
        phase, magnitude = maximum(np.zeros(3, complex))
        self.assertTrue(np.isfinite(phase))
        np.testing.assert_array_equal(magnitude, 0.)

    def test_coordinate_update_escapes_a_zero_amplitude_phase_plateau(self):
        rng = np.random.default_rng(941)
        designs = [np.eye(4, dtype=complex) for _ in range(3)]
        phase = np.array([.2, -.8, 1.4, -2.])
        truth = rng.uniform(.3, 2., (3, 4))*np.exp(1j*phase)
        model = PhaseCoupling(designs, list(truth), reference=2)
        initial = np.r_[phase+np.pi, 0., 0.]
        before = model.evaluate(initial)
        self.assertAlmostEqual(before['value'], 1.)
        np.testing.assert_array_equal(before['coefficients'], 0.)
        np.testing.assert_array_equal(before['gradient'], 0.)
        parameters, trace = model.coordinate_refine(initial, sweeps=10)
        after = model.evaluate(parameters)
        self.assertLess(after['value'], 1e-28)
        np.testing.assert_allclose(after['coefficients'], truth, atol=1e-14)
        self.assertTrue(np.all(np.diff(trace['objective']) <= 1e-13))

    def test_correlated_dictionary_and_spatial_phase_updates_are_monotone(self):
        rng = np.random.default_rng(2517)
        points = rng.uniform(-80., 80., (6, 2))
        designs = [rng.normal(size=(40, 6))+1j*rng.normal(size=(40, 6)) for _ in range(3)]
        slopes = np.array([[.003, -.002], [-.001, .004], [0., 0.]])
        phase = rng.uniform(-np.pi, np.pi, 6)
        band = np.array([.3, -.4, 0.])
        truth = rng.uniform(.4, 1.7, (3, 6))*np.exp(1j*(phase+band[:, None]+slopes@points.T))
        data = [a@c for a, c in zip(designs, truth)]
        model = PhaseCoupling(designs, data, reference=2, point_coordinates_m=points)
        initial = np.r_[phase+rng.normal(size=6), band[:2]+[.5, -.5], (100*slopes[:2]).ravel()]
        parameters, trace = model.coordinate_refine(initial, sweeps=100)
        self.assertTrue(np.all(np.diff(trace['objective']) <= 1e-11))
        self.assertLess(model.evaluate(parameters)['value'], 1e-11)
        np.testing.assert_array_equal(parameters[-4:], initial[-4:])
        _, baseline = model.fit()
        _, final = model.fit(coordinate_sweeps=50)
        self.assertTrue(final['converged'])
        self.assertLess(final['objective'], 1e-14)
        self.assertLessEqual(final['objective'], baseline['objective']+1e-14)
        self.assertEqual(len(final['trials']), 2*len(designs))


if __name__ == '__main__':
    unittest.main()
