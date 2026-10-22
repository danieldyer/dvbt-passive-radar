"""Independent numerical checks for the retained-aperture diagnostics."""
import unittest
import numpy as np

from audit_best_ship_focus import coherence, statistics
from audit_ship_motion_sensitivity import path_change


class FocusDiagnostics(unittest.TestCase):
    def test_complex_response_bound_and_scale(self):
        rng = np.random.default_rng(771)
        w = np.array([1., 2., 7., 11., .2])
        gain = rng.normal(size=5)+1j*rng.normal(size=5)
        source = rng.normal(size=9)+1j*rng.normal(size=9)
        score = w[:, None]*gain[:, None]*source
        coherent, independent = coherence(score, w, gain)
        np.testing.assert_allclose(coherent, independent, rtol=2e-15)
        scaled, _ = coherence(score, w, gain*(3-2j))
        np.testing.assert_allclose(scaled, coherent, rtol=2e-15)
        wrong, _ = coherence(score, w, np.ones(5))
        self.assertTrue(np.all(wrong <= independent*(1+1e-14)))

    def test_original_epoch_integration(self):
        histories = [dict(time=np.array([1., 2., 3.]), information=np.array([1., 2., 3.]),
                          value=np.array([[2+1j], [4-2j], [8+4j]])),
                     dict(time=np.array([1.2, 2.2, 3.2, 4.2]), information=np.ones(4),
                          value=np.array([[3j], [4j], [5j], [6j]]))]
        score, weight = statistics(histories, 2., 3.1)
        np.testing.assert_array_equal(weight, [5., 1.])
        np.testing.assert_array_equal(score[:, 0], [32+8j, 4j])

    def test_stable_path_difference_and_derivative(self):
        rng = np.random.default_rng(81)
        ray = rng.normal(size=(100, 3))*40e3
        d = rng.normal(size=(100, 3))*100
        np.testing.assert_allclose(path_change(ray, d),
            np.linalg.norm(ray+d, axis=1)-np.linalg.norm(ray, axis=1), rtol=0, atol=4e-11)
        direction = rng.normal(size=(100, 3))
        step = 1e-4
        numeric = (path_change(ray, d+step*direction)-path_change(ray, d-step*direction))/(2*step)
        exact = np.sum((ray+d)*direction, axis=1)/np.linalg.norm(ray+d, axis=1)
        np.testing.assert_allclose(numeric, exact, rtol=0, atol=2e-9)


if __name__ == '__main__': unittest.main()
