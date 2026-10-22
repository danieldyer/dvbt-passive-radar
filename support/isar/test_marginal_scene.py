import unittest

import numpy as np

from marginal_scene import Covariance, MarginalMotion, profile
from fit_physical_profile import solve
from mixed_scene import MixedMotion
import test_physical_profile as physical_tests


class MarginalSceneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(881216)
        b = rng.normal(size=(31, 7))+1j*rng.normal(size=(31, 7))
        b[:, -1] = b[:, 0]+b[:, 2]  # No full-column-rank assumption for covariance.
        db = rng.normal(size=b.shape)+1j*rng.normal(size=b.shape)
        return rng, b, db

    def test_volume_inverse_and_derivative_against_dense_covariance(self):
        rng, b, db = self.fixture()
        strength = .7
        c = np.eye(len(b))+b@b.conj().T/strength
        factor = Covariance(b, strength)
        self.assertAlmostEqual(factor.logdet, 2*np.log(np.diag(np.linalg.cholesky(c)).real).sum(), places=12)
        y = rng.normal(size=(len(b), 3))+1j*rng.normal(size=(len(b), 3))
        np.testing.assert_allclose(factor.apply_inverse(y), np.linalg.solve(c, y), atol=8e-14)
        whitened = factor.apply_inverse(y, square_root=True)
        np.testing.assert_allclose(whitened.conj().T@whitened, y.conj().T@np.linalg.solve(c, y), atol=2e-12)
        analytic = 2*np.vdot(factor.design_gradient, db).real
        h = 1e-5
        numeric = (Covariance(b+h*db, strength).logdet-Covariance(b-h*db, strength).logdet)/(2*h)
        self.assertAlmostEqual(analytic, numeric, delta=2e-8)

    def test_diffuse_basis_unitary_change_preserves_likelihood(self):
        rng, b, _ = self.fixture()
        q, _ = np.linalg.qr(rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7)))
        a, c = Covariance(b, 2.), Covariance(b@q, 2.)
        self.assertAlmostEqual(a.logdet, c.logdet, places=12)
        np.testing.assert_allclose(a.apply_inverse(b), c.apply_inverse(b), atol=2e-14)

    def test_original_row_physical_likelihood_gradient(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        parameters, scale = truth*.4, np.array([1., 1., 2.])
        fitted = profile(band, parameters, scale, 2, 500., variance=.3, derivatives=True)
        for j in range(len(parameters)):
            delta = np.eye(len(parameters))[j]*1e-4
            plus, minus = [profile(band, parameters+sign*delta, scale, 2, 500., variance=.3)
                for sign in (1, -1)]
            self.assertAlmostEqual((plus['value']-minus['value'])/2e-4/band.energy,
                fitted['gradient'][j]/band.energy, delta=3e-8)
            self.assertAlmostEqual((plus['logdet']-minus['logdet'])/2e-4,
                fitted['volume_gradient'][j], delta=2e-8)

    def test_core_only_joint_limit_preserves_quadratic_control(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        args = ([band], truth*.4, [np.ones(len(band.points))], [len(band.points)], [None])
        a, b = MixedMotion(*args), MarginalMotion(*args)
        first, second = a.evaluate(a.initial), b.evaluate(b.initial)
        for key in ('value', 'gradient', 'gauss_newton'):
            np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-12)
        self.assertEqual(second['channels'][0]['logdet'], 0.)

    def test_stationary_contrast_likelihood_matches_dense_gaussian_elimination(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        scale, core, strength, variance = np.array([1., 1., 2.]), 2, 500., .3
        fitted = profile(band, truth*.4, scale, core, strength, variance=variance)
        # An explicit orthonormal basis for the retained contrast subspace is
        # independent of the low-rank implementation under test.
        q, _ = np.linalg.qr(band.nuisance.q, mode='complete')
        keep = q[:, band.nuisance.rank:]
        design = keep.conj().T@(fitted['target'][band.selected]*scale)
        y = keep.conj().T@band.clean
        a, b = design[:, :core], design[:, core:]
        c = np.eye(len(y))+b@b.conj().T/strength
        wa, wy = np.linalg.solve(c, a), np.linalg.solve(c, y)
        alpha = np.linalg.solve(a.conj().T@wa, a.conj().T@wy)
        residual = y-a@alpha
        logdet = 2*np.log(np.diag(np.linalg.cholesky(c)).real).sum()
        expected = np.vdot(residual, np.linalg.solve(c, residual)).real/variance+logdet
        self.assertAlmostEqual(fitted['value'], expected, delta=2e-8)

    def test_shared_solver_accepts_marginal_model_and_reports_scene_conditions(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture(noise=0.)
        joint = MarginalMotion([band], truth*.9, [np.ones(3)], [3], [None])
        n = len(joint.initial)
        _, result = solve(joint, np.empty((0, n)), np.empty(0), np.empty(0),
            np.full(n, -5.), np.full(n, 5.), iterations=80)
        self.assertTrue(result['converged'], result)
        self.assertEqual(len(result['scene_conditions']), 1)


if __name__ == '__main__':
    unittest.main()
