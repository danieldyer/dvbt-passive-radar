import unittest

import numpy as np

from ridge_motion import RidgeMotion, point_derivatives, profile, residual_derivatives, ridge
from full_motion import Band
import test_physical_profile as physical_tests


class RidgeMotionTests(unittest.TestCase):
    def test_augmented_linear_solution_and_full_residual_derivative(self):
        rng = np.random.default_rng(491052)
        a = rng.normal(size=(80, 5))+1j*rng.normal(size=(80, 5))
        a[:, -1] = a[:, 0]+a[:, -1]*1e-7
        y = rng.normal(size=80)+1j*rng.normal(size=80)
        derivative = rng.normal(size=(80, 5, 3))+1j*rng.normal(size=(80, 5, 3))
        strength = .03
        result = ridge(a, y, strength)
        augmented = np.r_[a, np.sqrt(strength)*np.eye(5)]
        direct = np.linalg.lstsq(augmented, np.r_[y, np.zeros(5)], rcond=None)[0]
        np.testing.assert_allclose(result['coefficient'], direct, rtol=1e-11, atol=1e-13)
        fixed = np.einsum('npk,p->nk', derivative, result['coefficient'])
        cross = np.einsum('npk,n->pk', derivative.conj(), result['residual'])
        top, bottom, gradient, curvature = residual_derivatives(result, fixed, cross)
        for j in range(3):
            h = 1e-5
            plus, minus = [ridge(a+sign*h*derivative[:, :, j], y, strength) for sign in (1, -1)]
            for key, expected in [('residual', top[:, j]), ('coefficient', -bottom[:, j]/np.sqrt(strength))]:
                np.testing.assert_allclose((plus[key]-minus[key])/(2*h), expected, rtol=2e-5, atol=2e-6)
            self.assertAlmostEqual((plus['value']-minus['value'])/(2*h), gradient[j], delta=1e-5)
        self.assertGreaterEqual(np.linalg.eigvalsh(curvature).min(), -1e-10)

    def test_streamed_partials_equal_band_original_row_partials(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        _, expected = band.design(truth*.3, derivatives=True)
        for observed, reference in zip(point_derivatives(band, truth*.3), expected):
            np.testing.assert_allclose(observed, reference, rtol=0., atol=1e-13)

    def test_physical_profile_gradient_includes_regularized_scene_refit(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        p = truth*.3
        weights = np.linspace(.8, 1.3, len(band.points))
        value = profile(band, p, weights, .7, derivatives=True)
        for j in range(len(p)):
            delta = np.eye(len(p))[j]*1e-4
            numeric = (profile(band, p+delta, weights, .7)['value']-
                profile(band, p-delta, weights, .7)['value'])/2e-4
            # The physical fixture has energy 3.2e6. Check the same normalized
            # objective used by motion fitting; its central-difference error
            # decreases quadratically from h=1e-3 to h=1e-4.
            self.assertAlmostEqual(numeric/band.energy, value['gradient'][j]/band.energy, delta=2e-9)

    def test_zero_prediction_limit_supplies_no_fictitious_motion_information(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = RidgeMotion([band], truth*.3, [np.ones(len(band.points))], [None])
        value = model.evaluate(model.initial)
        self.assertEqual(value['value'], 1.)
        np.testing.assert_array_equal(value['gradient'], 0.)
        np.testing.assert_array_equal(value['gauss_newton'], 0.)
        result = model.validation(model.initial)[0]
        self.assertEqual(result['heldout_matched_energy'], 0.)
        self.assertEqual(result['heldout_explained_fraction'], 0.)


if __name__ == '__main__':
    unittest.main()
