import unittest

import numpy as np

from ais import AisTrack
from imaging import attitude_basis, physical_state, motion_state
from motion import enu_basis
from test_attitude_motion import setup
from test_rigid_motion import fixture
from spline_motion import SplineMotion


class AttitudeTests(unittest.TestCase):
    def test_linear_motion_reference_needs_no_ais_trajectory(self):
        field = fixture()
        reference = {'utc_seconds': 1000., 'position_ecef_m': field.data['centre_ecef_m'][0],
                     'velocity_ecef_mps': field.data['centre_velocity_ecef_mps'][0],
                     'body_to_ecef': field.data['body_to_ecef'][0]}
        state = motion_state(None, np.array([990., 1000., 1015.]), {'linear_reference': reference})
        np.testing.assert_allclose(state['position_ecef_m'][2]-state['position_ecef_m'][0],
                                   25*reference['velocity_ecef_mps'], atol=1e-9)
        np.testing.assert_array_equal(state['body_to_ecef'][0], state['body_to_ecef'][-1])
        np.testing.assert_array_equal(state['frame_velocity'], 0.)

    def test_fixed_reference_heading_does_not_inherit_ais_course_dynamics(self):
        seconds = np.array([0., 60., 120.])
        missing = np.full(3, np.nan)
        track = AisTrack(seconds, np.radians([-37.8, -37.799, -37.798]),
            np.radians([144.9, 144.9002, 144.9004]), np.full(3, 2.), np.radians([25., 5., -15.]),
            missing.copy(), missing.copy(), missing.copy(), missing.copy(), 563052200, 5.)
        t = np.array([10., 30., 59., 61., 90., 110.])
        state = physical_state(track, t.tolist(), 'course-prior', reference_heading_degrees=37.)
        local = np.einsum('nji,nj->ni', enu_basis(state['latitude'], state['longitude']), state['body_to_ecef'][:, :, 0])
        np.testing.assert_allclose(np.degrees(np.arctan2(local[:, 0], local[:, 1])), 37., atol=1e-12)
        point = np.array([140., -20., 20.])
        def position(time):
            z = physical_state(track, time, 'course-prior', reference_heading_degrees=37.)
            return z['position_ecef_m']+z['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        np.testing.assert_allclose((position(t+.01)-position(t-.01))/.02, velocity, atol=1e-7)
        self.assertTrue(np.isnan(state['true_heading_radians']).all())

    def test_explicit_course_hypothesis_keeps_missing_true_heading_missing(self):
        seconds = np.array([0., 60., 120.])
        latitude, longitude = np.radians(np.array([-37.8]*3)), np.radians(np.array([144.9]*3))
        missing = np.full(3, np.nan)
        track = AisTrack(seconds, latitude, longitude, np.ones(3)*5,
                         np.radians([359., 1., 4.]), missing.copy(), missing.copy(),
                         missing.copy(), missing.copy(), 563052200, 5.)
        state = track.sample(np.array([30., 60., 90.]), max_gap_s=61)
        frame = attitude_basis(track, state, 'reported-course-prior')
        np.testing.assert_allclose(np.einsum('nji,njk->nik', frame, frame),
                                   np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-14)
        self.assertTrue(np.isnan(state['true_heading_radians']).all())
        self.assertTrue(np.isnan(track.heading).all())
        # No 180-degree excursion through the 359/1 degree wrap.
        self.assertGreater(np.sum(frame[0, :, 0]*frame[-1, :, 0]), .99)

    def test_body_origin_and_heading_gauge_preserve_exact_complex_observations(self):
        field = fixture()
        model, prior = setup(field)
        d = field.data
        state = SplineMotion(model).apply(d['row_time'], prior, d['transmitter_ecef_m'], d['receiver_ecef_m'])
        # Merely relabel the body origin/axes. This is not additional physical
        # motion: attitude and every reflector coordinate must transform together.
        origin = np.array([83., -4., 7.])
        yaw = np.radians(27.)
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.],
            [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])
        shifted = {**state,
            'position_ecef_m': state['position_ecef_m']+state['body_to_ecef']@origin,
            'velocity_ecef_mps': state['velocity_ecef_mps']+state['frame_velocity']@origin,
            'body_to_ecef': state['body_to_ecef']@rotation,
            'frame_velocity': state['frame_velocity']@rotation}
        for point in (np.array([-120., -18., 8.]), np.array([24., 15., 35.]), np.array([130., -6., 41.])):
            new_point = rotation.T@(point-origin)
            np.testing.assert_allclose(shifted['position_ecef_m']+shifted['body_to_ecef']@new_point,
                state['position_ecef_m']+state['body_to_ecef']@point, rtol=0., atol=2e-9)
            np.testing.assert_allclose(shifted['velocity_ecef_mps']+shifted['frame_velocity']@new_point,
                state['velocity_ecef_mps']+state['frame_velocity']@point, rtol=0., atol=2e-14)
            before, after = field.point(point, motion=state), field.point(new_point, motion=shifted)
            self.assertLess(np.linalg.norm(after-before)/np.linalg.norm(before), 2e-8)


if __name__ == '__main__':
    unittest.main()
