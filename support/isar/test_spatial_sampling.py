import unittest

import numpy as np

from spatial_sampling import bounds
from motion import C, excess_path


class SpatialSamplingTests(unittest.TestCase):
    def test_local_gradient_bound_contains_exact_near_field_gradients(self):
        rng = np.random.default_rng(498)
        p = np.array([[12000., 5000., 30.], [12100., 5200., 30.]])
        tx, rx = np.array([[-24000., 4000., 300.], [-24500., 4100., 310.]]), np.zeros(3)
        frame = np.broadcast_to(np.eye(3), (2, 3, 3))
        frequencies, bandwidth, radius = [180e6, 226e6], 7e6, 200.
        result = bounds(p, frame, tx, rx, frequencies, bandwidth, radius)
        centre, displaced = [], []
        for transmitter, carrier in zip(tx, frequencies):
            k = 2*np.pi*(carrier+np.array([-bandwidth/2, bandwidth/2]))/C
            centre.extend((k[:, None, None]*excess_path(p, transmitter, rx)[1]).reshape(-1, 3))
            for _ in range(100):
                offset = rng.normal(size=3)
                offset *= radius/np.linalg.norm(offset)
                displaced.extend((k[:, None, None]*excess_path(p+offset, transmitter, rx)[1]).reshape(-1, 3))
        expected = np.ptp(centre, axis=0)[:2]
        np.testing.assert_allclose(result['wavevector_span_rad_per_m'], expected)
        actual = np.ptp(displaced, axis=0)[:2]
        self.assertTrue(np.all(actual < expected+result['conservative_nearfield_span_margin_rad_per_m']))
        with self.assertRaises(ValueError):
            bounds(p, frame, tx, rx, frequencies, bandwidth, 1e6)


if __name__ == '__main__':
    unittest.main()
