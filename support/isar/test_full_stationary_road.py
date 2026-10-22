import unittest

import numpy as np

from antenna_motion import Antennas
from full_stationary_road import FullField
from test_shared_target_geometry import road_fixture


class FullStationaryTests(unittest.TestCase):
    def test_whitening_retains_every_selected_sample_and_the_full_colored_noise_likelihood(self):
        road, _ = road_fixture(Antennas(np.eye(3), [0]))
        field = road.training[0]
        rng = np.random.default_rng(75281)
        data = rng.normal(size=field.data['value'].shape)+1j*rng.normal(size=field.data['value'].shape)
        binned = rng.normal(size=data.shape+(2,))
        full = FullField(field, binned)
        selected = data[field.selected].ravel()
        variance = field.data['variance'][field.selected, len(field.offsets)//2]
        lag = field.offsets[:, None]-field.offsets[None, :]
        range_covariance = field.pulse.samples[lag+64]/field.pulse.samples[64].real
        covariance = np.kron(np.diag(variance), range_covariance)
        factor = np.linalg.cholesky(covariance)
        white = np.linalg.solve(factor, selected)
        np.testing.assert_allclose(full.apply(data), white, atol=2e-14)
        self.assertEqual(len(full.y), len(selected))
        np.testing.assert_allclose(factor@full.apply(data), selected, atol=2e-15)


if __name__ == '__main__':
    unittest.main()
