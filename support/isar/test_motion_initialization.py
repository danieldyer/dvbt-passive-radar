import unittest

import numpy as np

from refine_full_capture_motion import constraints
from spline_motion import SplineMotion, basis


class MotionInitializationTests(unittest.TestCase):
    def test_yaw_rate_seed_is_physical_and_inside_existing_kinematic_bounds(self):
        interval = [143.2720443125, 188.2720443125]
        spacing = [3., 8., 20.]
        count = sum(basis([np.mean(interval)], interval, s).shape[1] for s in spacing)
        time = np.linspace(*interval, 361)
        for rate in (-.3, -.15, 0., .15, .3):
            model = {'interval_s': interval, 'path_spacing_seconds': spacing[0],
                'heading_spacing_seconds': spacing[1], 'transverse_spacing_seconds': spacing[2],
                'anchor_path': False, 'anchor_transverse': True, 'parameters': np.zeros(count),
                'initial_yaw_rate_deg_ps': rate}
            p, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
            curve = SplineMotion({**model, 'parameters': p})
            np.testing.assert_allclose(curve.components(time)[:, 1], np.radians(rate*(time-np.mean(interval))), atol=2e-12)
            np.testing.assert_allclose(curve.components(time, 1)[:, 1], np.radians(rate), atol=2e-12)
            parameters, active = curve.gauge_fixed_parameters()
            x = parameters[active]
            self.assertLessEqual(np.max(matrix@x-hi), 1e-10)
            self.assertLessEqual(np.max(lo-matrix@x), 1e-10)
            self.assertTrue(np.all((lower <= x)&(x <= upper)))


if __name__ == '__main__':
    unittest.main()
