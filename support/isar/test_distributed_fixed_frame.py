import unittest

import numpy as np

from distributed_fixed_frame import prediction_metrics, quadrature
from distributed_scene_cv import inner_masks
from gaussian_scene import marginal


class DistributedFixedFrameTests(unittest.TestCase):
    def test_spatial_covariance_quadrature_converges_to_continuous_rectangle(self):
        # The exact covariance for two plane-wave observations of a white
        # reflectivity rectangle is its Fourier transform, a product of sincs.
        bounds = [-20., 40., -30., 50.]
        difference = np.array([.075, -.09])
        width, centre = np.array([60., 80.]), np.array([10., 10.])
        exact = np.prod(width*np.sinc(difference*width/(2*np.pi)))*np.exp(1j*difference@centre)
        errors = []
        for spacing in [10., 5., 2.5]:
            points, weights = quadrature(bounds, spacing)
            approximate = weights@np.exp(1j*(points[:, :2]@difference))
            errors.append(abs(approximate-exact))
            self.assertEqual(float(weights.sum()), 4800.)
        self.assertGreater(errors[0]/errors[1], 3.8)
        self.assertGreater(errors[1]/errors[2], 3.8)

    def test_refining_identical_spatial_cell_preserves_prior_and_posterior(self):
        rng = np.random.default_rng(39091)
        a = rng.normal(size=(50, 4))+1j*rng.normal(size=(50, 4))
        y = rng.normal(size=50)+1j*rng.normal(size=50)
        original = marginal(a, y, variance=2.)
        refined = np.repeat(a, 4, axis=1)/2
        result = marginal(refined, y, variance=2.)
        self.assertAlmostEqual(original['value'], result['value'], delta=1e-12)
        np.testing.assert_allclose(a@original['coefficient'], refined@result['coefficient'], atol=1e-13)

    def test_prediction_metrics_do_not_refit_test_phase(self):
        y = np.ones((20, 3), complex)
        wrong = np.ones_like(y)*1j
        metrics = prediction_metrics(y, wrong, np.arange(20)*.1)
        self.assertAlmostEqual(metrics['heldout_explained_fraction'], -1.)
        self.assertAlmostEqual(metrics['prediction_error'], 120.)
        self.assertAlmostEqual(metrics['phase_rms_degrees'], 90.)

    def test_nested_temporal_selection_never_reads_outer_test_blocks(self):
        time = 100.03+np.arange(1700)*.017408
        outer = np.floor((time-100.)/.5).astype(int)%2 == 0
        heldouts = []
        for fold in range(3):
            train, test = inner_masks(time, [100., 130.], .5, fold)
            self.assertFalse(np.any(train & test))
            np.testing.assert_array_equal(train | test, outer)
            self.assertFalse(np.any((train | test) & ~outer))
            heldouts.append(test)
        np.testing.assert_array_equal(np.sum(heldouts, axis=0), outer.astype(int))


if __name__ == '__main__':
    unittest.main()
