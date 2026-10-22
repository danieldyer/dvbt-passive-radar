import unittest

import numpy as np

from ais import AisTrack
from imaging import physical_state
from kinematics import constraints
from motion import ecef, enu_basis, excess_path
from spline_motion import SplineMotion, basis


class KinematicTests(unittest.TestCase):
    def test_actual_motion_constraints_and_heading_wrap(self):
        seconds = np.array([0., 30., 60.])
        lat, lon = np.radians([-37.87, 144.93])
        missing = np.full(3, np.nan)
        track = AisTrack(seconds, lat+seconds*3/6359500, np.full(3, lon), np.full(3, 3.),
            np.radians([359., 1., 4.]), missing.copy(), missing.copy(), missing.copy(), missing.copy(), 563052200, 5.)
        origin, local = ecef(lat, lon, 5.), enu_basis(lat, lon)
        tx, rx = origin+local@np.array([35000., 15000., 700.]), origin+local@np.array([1000., 8000., 50.])
        model = {'interval_s': [20., 40.], 'path_spacing_seconds': 4., 'heading_spacing_seconds': 5.,
                 'attitude_prior': 'reported-course-prior'}
        npath = basis([30.], model['interval_s'], 4.).shape[1]
        nyaw = basis([30.], model['interval_s'], 5.).shape[1]
        full = np.linspace(-.02, .03, npath+nyaw)
        full[npath-1] = 0
        active = np.r_[np.arange(npath-1), np.arange(npath, len(full))]
        matrix, lower, upper, description = constraints(track, 0., model, tx, rx, active, len(active))
        t = np.array(description['constraint_epochs_s'])
        # Avoid knot crossings in the finite difference reference.
        keep = (t > 20.1)&(t < 39.9)
        for knot in (24, 28, 32, 36, 25, 30, 35):
            keep &= abs(t-knot) > .1
        t = t[keep]
        motion = SplineMotion({**model, 'parameters': full})
        def evaluate(offset):
            prior = physical_state(track, t+offset, model['attitude_prior'])
            moved = motion.apply(t+offset, prior, tx, rx)
            path = excess_path(moved['position_ecef_m'], tx, rx)[0]
            forward = np.einsum('nji,nj->ni', enu_basis(prior['latitude'], prior['longitude']), moved['body_to_ecef'][:, :, 0])
            return path, np.unwrap(np.arctan2(forward[:, 0], forward[:, 1]))
        h = .05
        before, middle, after = evaluate(-h), evaluate(0), evaluate(h)
        prediction = (matrix@full[active]).reshape(3, -1)
        expected_acceleration = prediction[0]+description['prior_reference_path_acceleration_mps2']
        np.testing.assert_allclose(expected_acceleration[keep], (after[0]-2*middle[0]+before[0])/h**2, atol=5e-6)
        expected_rate = prediction[1]+description['prior_heading_rate_degrees_ps']
        np.testing.assert_allclose(expected_rate[keep], np.degrees(after[1]-before[1])/(2*h), atol=2e-6)
        self.assertLess(max(abs(np.array(description['prior_heading_rate_degrees_ps']))), .2)
        self.assertTrue(np.all(upper > lower))


if __name__ == '__main__':
    unittest.main()
