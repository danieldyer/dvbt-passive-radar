import unittest

import numpy as np

from antenna_motion import Antennas, exact_spatial_phase, spatial_phase_mapping


class AntennaPhaseTests(unittest.TestCase):
    def test_local_phase_slope_matches_finite_antenna_offsets(self):
        position = np.array([10000., -18000., 30.])
        tx = np.array([[32000., -3000., 650.], [32300., -2800., 620.], [32000., -3000., 660.]])
        rx, frame, k = np.array([0., 0., 50.]), np.eye(3), np.array([3.7, 4., 4.6])
        prior = Antennas(np.eye(3), [0, 1, 0])
        jacobian = spatial_phase_mapping(position, frame, tx, rx, k, prior, 2)
        for j in range(prior.count):
            delta = np.eye(prior.count)[j]*1e-4
            plus = exact_spatial_phase(position, frame, tx, rx, k, prior, delta, 2)
            minus = exact_spatial_phase(position, frame, tx, rx, k, prior, -delta, 2)
            np.testing.assert_allclose((plus-minus)/2e-4, jacobian[:, :, j], rtol=2e-6, atol=1e-10)
        np.testing.assert_array_equal(jacobian[2], 0.)


if __name__ == '__main__':
    unittest.main()
