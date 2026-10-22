import unittest

import numpy as np

from phase_coupling import PhaseCoupling


class PhaseCouplingTests(unittest.TestCase):
    def fixture(self, noise=0.):
        rng = np.random.default_rng(841)
        designs = [rng.normal(size=(80, 4))+1j*rng.normal(size=(80, 4)) for _ in range(3)]
        point_phase = np.array([.4, -.2, 1.2, -.7])
        band_phase = np.array([-.45, .72, 0.])
        coefficients = rng.uniform(.5, 1.5, size=(3, 4))*np.exp(1j*(point_phase+band_phase[:, None]))
        observations = [a@c+noise*(rng.normal(size=80)+1j*rng.normal(size=80)) for a, c in zip(designs, coefficients)]
        return PhaseCoupling(designs, observations, reference=2), designs, observations, coefficients, band_phase

    def test_profile_gradient_and_exact_qr_likelihood(self):
        model, designs, observations, _, _ = self.fixture(noise=.05)
        x = model.initial(0)+np.linspace(-.15, .12, model.points+len(model.other))
        result = model.evaluate(x)
        raw = sum(np.sum(abs(y-a@c)**2) for a, y, c in zip(designs, observations, result['coefficients']))
        self.assertAlmostEqual(result['value'], raw/model.energy, places=12)
        for j in range(len(x)):
            delta = np.eye(len(x))[j]*1e-6
            derivative = (model.objective(x+delta)[0]-model.objective(x-delta)[0])/2e-6
            self.assertAlmostEqual(derivative, result['gradient'][j], delta=1e-8)

    def test_recovers_band_phases_with_independent_positive_magnitudes(self):
        model, _, _, truth, phases = self.fixture()
        coefficient, result = model.fit()
        self.assertTrue(result['converged'], result)
        self.assertLess(result['objective'], 1e-15)
        np.testing.assert_allclose(coefficient, truth, atol=1e-7)
        np.testing.assert_allclose(np.exp(1j*np.array(result['channel_phase_radians'])), np.exp(1j*phases), atol=1e-7)
        self.assertGreater(min(result['curvature_eigenvalues']), 0.)

    def test_incompatible_point_phases_are_not_explained_by_band_phase(self):
        model, designs, _, coefficients, _ = self.fixture()
        coefficients[0, 0] *= np.exp(1.2j)
        observations = [a@c for a, c in zip(designs, coefficients)]
        incompatible = PhaseCoupling(designs, observations, reference=2)
        _, result = incompatible.fit()
        self.assertLess(result['independent_objective'], 1e-25)
        self.assertGreater(result['objective'], .015)

    def test_strength_changes_do_not_create_independent_block_phases(self):
        rng = np.random.default_rng(7721)
        indices = np.tile(np.arange(4), 3)
        designs = [rng.normal(size=(90, 12))+1j*rng.normal(size=(90, 12)) for _ in range(3)]
        phase = np.array([.2, -.6, 1.1, -.9])
        bands = np.array([-.35, .45, 0.])
        coefficients = rng.uniform(.1, 2., (3, 12))*np.exp(1j*(phase[indices]+bands[:, None]))
        observations = [a@c for a, c in zip(designs, coefficients)]
        joint = PhaseCoupling(designs, observations, reference=2, phase_indices=indices)
        x = joint.initial(0)+np.linspace(-.1, .1, 6)
        gradient = joint.objective(x)[1]
        for j in range(len(x)):
            delta = np.eye(len(x))[j]*1e-6
            finite = (joint.objective(x+delta)[0]-joint.objective(x-delta)[0])/2e-6
            self.assertAlmostEqual(finite, gradient[j], delta=1e-8)
        recovered, result = joint.fit()
        self.assertTrue(result['converged'])
        self.assertLess(result['objective'], 1e-15)
        np.testing.assert_allclose(recovered, coefficients, atol=1e-7)

    def test_spatial_phase_profile_gradient_and_known_recovery(self):
        rng = np.random.default_rng(4492)
        points = rng.uniform(-100., 100., (9, 2))
        designs = [rng.normal(size=(110, 9))+1j*rng.normal(size=(110, 9)) for _ in range(3)]
        phase = rng.uniform(-1., 1., 9)
        bands = np.array([.3, -.4, 0.])
        slopes = np.array([[.003, -.002], [-.001, .004], [0., 0.]])
        truth = rng.uniform(.5, 1.5, (3, 9))*np.exp(1j*(phase+bands[:, None]+slopes@points.T))
        observations = [a@c for a, c in zip(designs, truth)]
        model = PhaseCoupling(designs, observations, reference=2, point_coordinates_m=points)
        parameters = model.initial(0)+np.linspace(-.1, .1, 15)
        gradient = model.objective(parameters)[1]
        for j in range(len(parameters)):
            delta = np.eye(len(parameters))[j]*1e-6
            finite = (model.objective(parameters+delta)[0]-model.objective(parameters-delta)[0])/2e-6
            self.assertAlmostEqual(finite, gradient[j], delta=1e-8)
        coefficients, result = model.fit()
        self.assertTrue(result['converged'], result)
        self.assertLess(result['objective'], 1e-15)
        np.testing.assert_allclose(coefficients, truth, atol=2e-7)
        np.testing.assert_allclose(result['spatial_phase_gradient_radians_per_m'], slopes, atol=1e-8)
        np.testing.assert_allclose(np.exp(1j*np.array(result['channel_phase_radians'])), np.exp(1j*bands), atol=1e-7)
        self.assertGreater(min(result['curvature_eigenvalues']), 0.)


if __name__ == '__main__':
    unittest.main()
