import unittest

import numpy as np

from mixed_scene import MixedPath, MixedMotion, mixed_fit, profile, residual_derivatives
from mixed_scene_cv import choose_penalty
from physical_profile import PhysicalProfile, profile as unpenalized_profile
from full_motion import Band
import test_physical_profile as physical_tests


class MixedSceneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(45071)
        design = rng.normal(size=(70, 8))+1j*rng.normal(size=(70, 8))
        design[:, 3] = design[:, 1]  # Exact overlap between core and diffuse basis.
        data = rng.normal(size=70)+1j*rng.normal(size=70)
        return rng, design, data

    def test_elimination_path_equals_augmented_svd_including_core_overlap(self):
        _, design, data = self.fixture()
        strengths = [1e-5, .01, 1., 1e5, None]
        coefficients = MixedPath(design, data, 3).coefficients(strengths)
        for j, strength in enumerate(strengths):
            local = design if strength is not None else design[:, :3]
            fitted = mixed_fit(local, data, 3, strength)
            np.testing.assert_allclose(design@coefficients[:, j], local@fitted['coefficient'], rtol=1e-9, atol=2e-10)
        np.testing.assert_allclose(coefficients[:3, -1], np.linalg.lstsq(design[:, :3], data, rcond=None)[0], atol=2e-14)
        np.testing.assert_array_equal(coefficients[3:, -1], 0.)

    def test_exact_augmented_residual_derivative_for_core_and_diffuse_motion(self):
        rng, design, data = self.fixture()
        derivative = rng.normal(size=(*design.shape, 3))+1j*rng.normal(size=(*design.shape, 3))
        fit = mixed_fit(design, data, 3, .3)
        fixed = np.einsum('npq,p->nq', derivative, fit['coefficient'])
        cross = np.einsum('npq,n->pq', derivative.conj(), fit['residual'])
        jacobian, gradient, curvature = residual_derivatives(fit, fixed, cross)
        for j in range(3):
            h = 1e-5
            plus, minus = [mixed_fit(design+sign*h*derivative[:, :, j], data, 3, .3) for sign in (1,-1)]
            np.testing.assert_allclose((plus['augmented_residual']-minus['augmented_residual'])/(2*h),
                jacobian[:, j], atol=2e-7, rtol=2e-6)
            self.assertAlmostEqual((plus['value']-minus['value'])/(2*h), gradient[j], delta=2e-6)
        self.assertGreater(np.linalg.eigvalsh(curvature).min(), 0.)

    def test_core_prediction_and_objective_preserved_at_infinite_penalty(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        fitted = profile(band, truth*.3, np.ones(len(band.points)), len(band.points), None, derivatives=True)
        control = unpenalized_profile(band, truth*.3, derivatives=True)
        for key in ('residual','coefficient','gradient','gauss_newton'):
            np.testing.assert_allclose(fitted[key], control[key], rtol=1e-10, atol=1e-8)

    def test_original_row_gradient_for_mixed_physical_scene(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        parameters = truth*.4
        scale = np.array([1., 1., 2.])
        result = profile(band, parameters, scale, 2, 500., derivatives=True)
        for j in range(len(parameters)):
            delta = np.eye(len(parameters))[j]*1e-5
            numeric = (profile(band, parameters+delta, scale, 2, 500.)['value']-
                profile(band, parameters-delta, scale, 2, 500.)['value'])/2e-5
            self.assertAlmostEqual(numeric/band.energy, result['gradient'][j]/band.energy, delta=3e-9)

    def test_complex_mean_and_posterior_match_independent_covariance_formulation(self):
        _, design, data = self.fixture()
        core, strength = 3, .7
        a, b = design[:, :core], design[:, core:]
        covariance = np.eye(len(data))+b@b.conj().T/strength
        wa, wy = np.linalg.solve(covariance, a), np.linalg.solve(covariance, data)
        alpha = np.linalg.solve(a.conj().T@wa, a.conj().T@wy)
        beta = b.conj().T@np.linalg.solve(covariance, data-a@alpha)/strength
        result = mixed_fit(design, data, core, strength)
        np.testing.assert_allclose(result['coefficient'], np.r_[alpha,beta], atol=1e-13)

    def test_core_only_joint_motion_retains_all_original_information(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        control = PhysicalProfile([band], truth*.4)
        mixed = MixedMotion([band], truth*.4, [np.ones(len(band.points))], [len(band.points)], [None])
        a, b = control.evaluate(control.initial), mixed.evaluate(mixed.initial)
        for key in ('value','gradient','gauss_newton'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12)

    def test_inner_selection_is_invariant_to_arbitrary_outer_test_observations(self):
        original, truth = physical_tests.PhysicalProfileTests().fixture()
        selected = np.floor(original.field.time/.5).astype(int)%2 == 0
        def band():
            return Band(original.field, original.prior, original.reference_transmitter,
                original.receiver, original.points, original.model, selected, range_bias_m=original.range_bias)
        first = band()
        target, _ = first.design(truth*.4)
        coefficient, tuning = choose_penalty(first, target, np.ones(len(first.points)), 2, .5)
        original.field.y[~selected] = 1e12*(1+2j)
        second, changed = choose_penalty(band(), target, np.ones(len(first.points)), 2, .5)
        np.testing.assert_array_equal(coefficient, second)
        np.testing.assert_array_equal(tuning['inner_prediction_errors'], changed['inner_prediction_errors'])


if __name__ == '__main__':
    unittest.main()
