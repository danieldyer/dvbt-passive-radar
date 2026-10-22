import unittest

import numpy as np

from assess_dense_scene import local_grid, rectangular_grid
from dense_scatterers import gcv


class DenseSceneTests(unittest.TestCase):
    def test_complete_support_has_no_peak_selection_gaps(self):
        points = rectangular_grid([-80., 160., -20., 60.], 5., 20.)
        self.assertEqual(len(points), 49*17)
        np.testing.assert_array_equal(np.unique(points[:, 0]), np.arange(-80, 161, 5.))
        np.testing.assert_array_equal(np.unique(points[:, 1]), np.arange(-20, 61, 5.))
        with self.assertRaises(ValueError):
            rectangular_grid([0., -1., 0., 1.], 5., 20.)

    def test_spatial_support_is_fixed_and_contains_original_points(self):
        points = [[0., 0., 20.], [5., 5., 20.]]
        grid = local_grid(points, 5., 5.)
        self.assertEqual(len(grid), 14)
        for point in points:
            self.assertTrue(np.any(np.all(grid == point, axis=1)))

    def test_gcv_matches_explicit_regularized_solve_and_predicts_unseen_rows(self):
        rng = np.random.default_rng(881)
        a = rng.normal(size=(1200, 15))+1j*rng.normal(size=(1200, 15))
        a[:, -1] = a[:, -2]+1e-7*a[:, -1]
        coefficient = rng.normal(size=15)+1j*rng.normal(size=15)
        y = a@coefficient+.05*(rng.normal(size=1200)+1j*rng.normal(size=1200))
        fitted, detail = gcv(a[::2], y[::2], 0)
        strength = detail['regularization_strength']
        expected = np.linalg.lstsq(np.r_[a[::2], np.sqrt(strength)*np.eye(15)],
                                 np.r_[y[::2], np.zeros(15)], rcond=None)[0]
        np.testing.assert_allclose(fitted, expected, atol=2e-7)
        self.assertLess(np.mean(abs(a[1::2]@fitted-y[1::2])**2), .007)
        self.assertLess(detail['effective_degrees_of_freedom'], 14.01)

    def test_gcv_noiseless_residual_does_not_subtract_nearly_equal_energies(self):
        rng = np.random.default_rng(882)
        a = rng.normal(size=(100,8))+1j*rng.normal(size=(100,8))
        coefficient = rng.normal(size=8)+1j*rng.normal(size=8)
        y = a@coefficient
        fitted, detail = gcv(a,y,0)
        self.assertEqual(detail['regularization_relative_to_max_gram'],1e-12)
        self.assertGreater(detail['training_residual_energy'],0.)
        self.assertLess(detail['training_residual_energy'],1e-19)
        self.assertAlmostEqual(detail['training_residual_energy']/np.sum(abs(a@fitted-y)**2),1.,places=2)


if __name__ == '__main__':
    unittest.main()
