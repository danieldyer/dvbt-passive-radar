import json
import unittest

import numpy as np

from gaussian_scene import marginal, motion_information


class GaussianSceneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(2034)
        a = (rng.normal(size=(40, 5))+1j*rng.normal(size=(40, 5)))/np.sqrt(2)
        coefficient = (rng.normal(size=5)+1j*rng.normal(size=5))*.8
        y = a@coefficient+(rng.normal(size=40)+1j*rng.normal(size=40))/np.sqrt(2)
        return rng, a, y

    def test_complex_covariance_likelihood_posterior_and_derivative(self):
        rng, a, y = self.fixture()
        result = marginal(a, y, variance=.7)
        covariance = np.eye(len(y))+.7*a@a.conj().T
        determinant = 2*np.log(np.diag(np.linalg.cholesky(covariance)).real).sum()
        expected = np.vdot(y, np.linalg.solve(covariance, y)).real+determinant
        self.assertAlmostEqual(result['value'], expected, places=10)
        np.testing.assert_allclose(result['coefficient'], .7*a.conj().T@np.linalg.solve(covariance, y), atol=1e-12)
        posterior = np.linalg.inv(np.eye(a.shape[1])/.7+a.conj().T@a)
        np.testing.assert_allclose(result['covariance'], posterior, atol=1e-12)
        direction = rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
        numeric = (marginal(a+1e-5*direction, y, variance=.7)['value']-
                   marginal(a-1e-5*direction, y, variance=.7)['value'])/2e-5
        analytic = 2*np.vdot(result['design_gradient'], direction).real
        self.assertAlmostEqual(numeric, analytic, delta=1e-7)

    def test_profiled_variance_gradient_and_scale_invariance(self):
        rng, a, y = self.fixture()
        result = marginal(a, y)
        direction = rng.normal(size=a.shape)+1j*rng.normal(size=a.shape)
        numeric = (marginal(a+1e-5*direction, y)['value']-marginal(a-1e-5*direction, y)['value'])/2e-5
        self.assertAlmostEqual(numeric, 2*np.vdot(result['design_gradient'], direction).real, delta=1e-7)
        self.assertLess(abs(result['variance_fit']['variance_stationarity']), 1e-9)
        self.assertFalse(json.loads(json.dumps(result['variance_fit']))['zero_scene'])
        scaled = marginal(a*37, y)
        self.assertAlmostEqual(scaled['value'], result['value'], places=10)
        np.testing.assert_allclose(a@result['coefficient'], 37*a@scaled['coefficient'], atol=1e-10)
        self.assertAlmostEqual(scaled['variance']*37**2, result['variance'], places=10)

    def test_duplicate_templates_and_zero_signal_are_well_defined(self):
        _, a, y = self.fixture()
        repeated = np.repeat(a[:, :1], 5, axis=1)
        result, single = marginal(repeated, y), marginal(a[:, :1], y)
        self.assertAlmostEqual(result['value'], single['value'], places=10)
        np.testing.assert_allclose(repeated@result['coefficient'], a[:, :1]@single['coefficient'], atol=1e-11)
        empty = marginal(a, np.zeros(len(a), complex))
        self.assertEqual(empty['variance'], 0.)
        self.assertEqual(empty['value'], 0.)
        self.assertEqual(np.linalg.norm(empty['design_gradient']), 0.)

    def test_large_cancelling_coefficients_preserve_inverse_covariance_action(self):
        # The covariance is analytically diagonal even though its two scene
        # columns nearly coincide. Forming y-A@coefficient loses precision.
        for epsilon in (1e-6, 1e-8, 1e-10):
            a = np.zeros((20, 2), complex)
            a[0], a[1] = [1., 1j], [epsilon, -1j*epsilon]
            y = np.ones(20, complex)
            y[1] = 100j
            variance = 1/epsilon**2
            diagonal = np.ones(20)
            diagonal[:2] += variance*np.array([2., 2*epsilon**2])
            result = marginal(a, y, variance=variance)
            np.testing.assert_allclose(result['residual'], y/diagonal, rtol=1e-12, atol=1e-10)
            expected = np.sum(abs(y)**2/diagonal)+np.log(diagonal).sum()
            self.assertAlmostEqual(result['value'], expected, delta=1e-10)

    def test_posterior_prediction_attains_conditional_gaussian_mmse(self):
        rng, a, _ = self.fixture()
        trials, variance = 6000, .7
        truth = np.sqrt(variance/2)*(rng.normal(size=(5, trials))+1j*rng.normal(size=(5, trials)))
        noise = (rng.normal(size=(40, trials))+1j*rng.normal(size=(40, trials)))/np.sqrt(2)
        observed = a@truth+noise
        fit = marginal(a, observed[:, 0], variance=variance)
        estimate = fit['covariance']@a.conj().T@observed
        error = np.mean(abs(estimate-truth)**2, axis=1)
        ratio = error/fit['covariance'].diagonal().real
        np.testing.assert_allclose(ratio, 1., atol=.04)

    def test_motion_information_matches_full_covariance_and_profiles_variance(self):
        rng, a, y = self.fixture()
        variance = .7
        result = marginal(a, y, variance=variance)
        derivatives = rng.normal(size=(*a.shape, 3))+1j*rng.normal(size=(*a.shape, 3))
        covariance = np.eye(len(y))+variance*a@a.conj().T
        dc = [variance*(d@a.conj().T+a@d.conj().T) for d in np.moveaxis(derivatives, 2, 0)]
        dc.append(variance*a@a.conj().T)
        whitened = [np.linalg.solve(covariance, d) for d in dc]
        full = np.array([[np.trace(x@y).real for y in whitened] for x in whitened])
        np.testing.assert_allclose(motion_information(result, derivatives, profile_variance=False),
            full[:3, :3], rtol=1e-12, atol=1e-10)
        expected = full[:3, :3]-np.outer(full[:3, 3], full[3, :3])/full[3, 3]
        np.testing.assert_allclose(motion_information(result, derivatives), expected, rtol=1e-12, atol=1e-10)
        # Constant scene phase and scene scale are exact nuisance gauges.
        gauges = np.stack([1j*a, a], axis=-1)
        self.assertLess(np.max(abs(motion_information(result, gauges))), 1e-22)


if __name__ == '__main__':
    unittest.main()
