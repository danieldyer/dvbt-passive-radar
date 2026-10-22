import unittest

import numpy as np

from audit_anchor_geometry import example, excess


class AnchorGeometryTests(unittest.TestCase):
    def test_bistatic_origin_jacobian_matches_finite_differences(self):
        record = example()
        targets, receiver, parameters = [np.asarray(record[key]) for key in
            ('targets_m', 'receiver_m', 'true_transmitter_and_origin_m')]
        _, derivative = excess(targets, receiver, parameters)
        for i in range(4):
            delta = np.eye(4)[i]*1e-4
            numerical = (excess(targets, receiver, parameters+delta)[0]-
                excess(targets, receiver, parameters-delta)[0])/2e-4
            np.testing.assert_allclose(numerical, derivative[:, i], atol=2e-9, rtol=2e-8)

    def test_planar_local_full_rank_coexists_with_exact_global_mirror(self):
        result = example()
        points = np.asarray(result['targets_m'])
        self.assertEqual(np.linalg.matrix_rank(points-points.mean(axis=0)), 2)
        self.assertGreater(result['local_jacobian_singular_values'][-1], .1)
        self.assertEqual(result['maximum_mirror_observation_difference_m'], 0.)
        for fit in result['fits']:
            self.assertTrue(fit['success'])
            self.assertLess(fit['maximum_residual_m'], 1e-9)
            self.assertLess(fit['maximum_parameter_error_from_corresponding_truth_m'], 1e-7)


if __name__ == '__main__':
    unittest.main()
