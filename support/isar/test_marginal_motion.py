import unittest

import numpy as np

from marginal_motion import MarginalMotion
from fit_joint_motion import information_scaling
import test_physical_profile as physical_tests


class MarginalMotionTests(unittest.TestCase):
    def test_zero_scene_scaling_does_not_invert_absent_information(self):
        transform, diagnostic = information_scaling(np.zeros((3, 3)), 1., np.array([True, False, True]))
        np.testing.assert_array_equal(transform, [[1., 0.], [0., 0.], [0., 1.]])
        self.assertEqual(diagnostic['eigenvalue_floor'], 0.)

    def test_physical_motion_derivative_includes_fitted_variance_and_log_determinant(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = MarginalMotion([band], truth*.3)
        p = model.initial.copy()
        value = model.evaluate(p)
        for j in range(len(p)):
            delta = np.eye(len(p))[j]*.0001
            numeric = (model.objective(p+delta)[0]-model.objective(p-delta)[0])/.0002
            self.assertAlmostEqual(numeric, value['gradient'][j], delta=3e-6)
        self.assertGreaterEqual(np.linalg.eigvalsh(value['gauss_newton']).min(), -1e-8)
        self.assertEqual(model.validation(p)[0]['stationary_validation_mode'], 'project')


if __name__ == '__main__':
    unittest.main()
