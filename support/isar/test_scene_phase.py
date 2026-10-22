import unittest

import numpy as np

from scene_phase import ScenePhase, phase_basis


class ScenePhaseTests(unittest.TestCase):
    def fixture(self, phase_scale=1.):
        rng = np.random.default_rng(4563)
        time = np.linspace(.01, 11.99, 240)
        selected = np.floor(time).astype(int)%2 == 0
        basis = phase_basis(time, [0., 12.], selected, 3.)
        parameters = rng.normal(size=basis.shape[1])*.025
        parameters *= .2*phase_scale/np.max(abs(basis@parameters))
        phase = basis@parameters
        design = rng.normal(size=(720, 7))+1j*rng.normal(size=(720, 7))
        background = rng.normal(size=(720, 4))+1j*rng.normal(size=(720, 4))
        coefficient = rng.normal(size=7)+1j*rng.normal(size=7)
        data = (np.repeat(np.exp(1j*phase), 3)*(design@coefficient)+background@np.array([10+5j, -6j, 1., -3.])).reshape(-1, 3)
        model = ScenePhase(design, data, background, time, selected, np.arange(3), [0., 12.])
        return model, parameters, phase, coefficient

    def test_profile_gradient_includes_refitted_complex_amplitudes(self):
        model, truth, _, _ = self.fixture()
        value, gradient = model.objective(truth*.3)
        for j in range(len(truth)):
            delta = np.eye(len(truth))[j]*1e-5
            numeric = (model.objective(truth*.3+delta)[0]-model.objective(truth*.3-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, gradient[j], places=8)
        self.assertLess(model.objective(truth)[0], 1e-25)
        self.assertGreater(value, 1e-5)

    def test_injected_phase_and_complex_scene_are_recovered_across_time_gaps(self):
        model, _, truth, amplitude = self.fixture()
        phase, coefficient, detail = model.fit()
        self.assertTrue(detail['converged'])
        self.assertFalse(detail['phase_bound_hit'])
        np.testing.assert_allclose(phase, truth, atol=2e-6)
        np.testing.assert_allclose(coefficient, amplitude, atol=2e-6)
        selected = model.selected
        self.assertLess(abs(np.mean(phase[selected])), 1e-12)
        self.assertLess(abs(np.dot(model.time[selected]-6., phase[selected])), 1e-10)

    def test_phase_recovery_beyond_previous_search_limit(self):
        model, _, phase, amplitude = self.fixture(phase_scale=8.)
        # Keep the original known complex scene and stationary nuisance. The
        # injected 1.6-radian phase must be recovered, including withheld times.
        fitted, coefficients, detail = model.fit(maximum_phase_radians=np.pi)
        self.assertTrue(detail['converged'])
        self.assertFalse(detail['phase_bound_hit'])
        np.testing.assert_allclose(fitted, phase, atol=2e-6)
        np.testing.assert_allclose(coefficients, amplitude, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
