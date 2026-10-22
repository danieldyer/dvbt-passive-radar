import unittest
import tempfile
from pathlib import Path

import numpy as np

from full_capture_motion import constrained_curve
from geographic_movie import ImageRing, body_corners, channel_geometry, coherent_window_plan, frame_transform, project_power, BLOCK
from imaging import geometry_from_state
from motion import C
from spline_motion import SplineMotion, basis


class GeographicMovieTests(unittest.TestCase):
    def test_fitted_rays_preserve_motion_and_pack_physical_range_phase_and_velocity(self):
        centres = np.array([[2000., 4000., 50.], [2002., 4003., 50.], [2004., 4006., 50.]])
        velocity = np.broadcast_to([2., 3., 0.], centres.shape).copy()
        state = {'position_ecef_m': centres.copy(), 'velocity_ecef_mps': velocity,
            'body_to_ecef': np.broadcast_to(np.eye(3), (3, 3, 3)), 'frame_velocity': np.zeros((3, 3, 3))}
        tx, rx = np.array([[30000., -1500., 600.], [30200., -1700., 500.]]), np.zeros(3)
        angle = .4
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        model = {'range_bias_m': [1.7, -2.3], 'channel_gains_real_imag': [[.8, .3], [1.1, -.2]],
            'antenna_fit': {'ecef_to_enu': rotation.tolist(),
                'transmitter_offsets_enu_m': [[20., -30., 12.], [-10., 25., -7.]],
                'receiver_offset_enu_m': [2., -4., 1.]}}
        channel, rf, first = 1, 191.625e6, 3.5
        actual = channel_geometry(model, state, tx, rx, channel, rf, first, np.array([1., 2., 3.]))
        transmitter = tx[channel]+np.array(model['antenna_fit']['transmitter_offsets_enu_m'][channel])@rotation
        receiver = rx+np.array(model['antenna_fit']['receiver_offset_enu_m'])@rotation
        u, v = centres-transmitter, centres-receiver
        path = np.linalg.norm(u, axis=1)+np.linalg.norm(v, axis=1)-np.linalg.norm(transmitter-receiver)
        expected_range = (path+model['range_bias_m'][channel])*8e6/C-first
        gain = complex(*model['channel_gains_real_imag'][channel])
        expected_phase = np.exp(2j*np.pi*rf/C*path)*gain.conjugate()
        np.testing.assert_allclose(actual[:, 9, 0], expected_range, atol=3e-5, rtol=0)
        np.testing.assert_allclose(actual[:, 9, 1]+1j*actual[:, 9, 2], expected_phase, atol=1e-7, rtol=0)
        # Recover the physical centre path rate from the packed rays. An
        # antenna change must not rotate/translate the fitted target itself.
        units = actual[:, 0, :3]/actual[:, 0, 3, None]+actual[:, 1, :3]/actual[:, 1, 3, None]
        expected_rate = np.sum((u/np.linalg.norm(u, axis=1)[:, None]+v/np.linalg.norm(v, axis=1)[:, None])*velocity, axis=1)
        np.testing.assert_allclose(np.sum(units*actual[:, 5, :3], axis=1), expected_rate, atol=5e-7, rtol=0)
        np.testing.assert_array_equal(state['position_ecef_m'], centres)
        np.testing.assert_array_equal(actual[:, 5, :3], velocity)

    def test_nominal_site_geometry_remains_bit_identical(self):
        centre = np.array([[1500., 2400., 30.]])
        state = {'position_ecef_m': centre, 'velocity_ecef_mps': np.array([[4., -2., 0.]]),
            'body_to_ecef': np.eye(3)[None], 'frame_velocity': np.zeros((1, 3, 3))}
        tx, rx = np.array([[35000., -1200., 700.]]), np.zeros(3)
        model = {'range_bias_m': [-4.2], 'channel_gains_real_imag': [[.43, -.27]]}
        rf = 177.5e6
        before, _ = geometry_from_state(state, tx[0], rx, rf, 20., np.array([1.7]))
        before[:, 9, 0] += model['range_bias_m'][0]*8e6/C
        gain = complex(*model['channel_gains_real_imag'][0])
        phase = (before[:, 9, 1].astype(float)+1j*before[:, 9, 2])*gain.conjugate()
        before[:, 9, 1], before[:, 9, 2] = phase.real, phase.imag
        before[:, 5, 3], before[:, 8, 3] = 2*np.pi*rf/C, abs(gain)**2
        np.testing.assert_array_equal(channel_geometry(model, state, tx, rx, 0, rf, 20., np.array([1.7])), before)

    def test_short_and_long_coherent_windows_have_constant_complete_support(self):
        period = .001088/5
        epochs = np.arange(170000)*period+10.
        centres = np.arange(26., 27., 1/30)
        for seconds in (3., 10., 30.):
            blocks, chosen = coherent_window_plan(epochs, centres, period, seconds)
            first, last = (chosen-blocks+1)*BLOCK, (chosen+1)*BLOCK-1
            np.testing.assert_allclose(epochs[last]-epochs[first], (blocks*BLOCK-1)*period, atol=1e-14)
            self.assertLess(np.max(abs((epochs[first]+epochs[last])/2-centres)), period*BLOCK)
            self.assertLess(np.max(abs(epochs[last]-epochs[first]-seconds)), period*BLOCK)
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            coherent_window_plan(epochs[:1000], centres, period, 3.)
        with self.assertRaisesRegex(ValueError, 'complete coherent support'):
            coherent_window_plan(epochs, np.array([10.]), period, 3.)

    def test_window_ring_is_lossless_and_does_not_alias_slots(self):
        rng = np.random.default_rng(42)
        expected = (rng.normal(size=(3, 731))+1j*rng.normal(size=(3, 731))).astype('complex64')
        with tempfile.TemporaryDirectory() as directory:
            ring = ImageRing(Path(directory)/'ring', 3, 731)
            try:
                for i in range(3):
                    ring.write(i, expected[i])
                for i in (2, 0, 1):
                    np.testing.assert_array_equal(ring.read(i), expected[i])
                ring.write(1, expected[0])
                np.testing.assert_array_equal(ring.read(1), expected[0])
                np.testing.assert_array_equal(ring.read(2), expected[2])
            finally:
                ring.close()

    def test_north_up_camera_preserves_ground_coordinate_and_scale(self):
        angle = .7
        body = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        centre, origin = np.array([123., 456., 2.]), np.zeros(3)
        matrix, shift = frame_transform(centre, body, origin, np.eye(3), 20.)
        bounds = [100., 140., 430., 480.]
        q = body_corners(bounds, matrix, shift)
        expected = np.array([[e, n] for e in bounds[:2] for n in bounds[2:]])
        np.testing.assert_allclose(q@matrix.T+shift, expected, atol=1e-12)

    def test_display_affine_reprojection_of_linear_power(self):
        x, y = np.arange(-30., 31.), np.arange(-30., 31.)
        xx, yy = np.meshgrid(x, y)
        angle = .4
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        shift = np.array([4., -3.])
        power = 100+2*(matrix[0, 0]*xx+matrix[0, 1]*yy)+3*(matrix[1, 0]*xx+matrix[1, 1]*yy)
        bounds, size = [-10, 10, -9, 11], (20, 20)
        actual = project_power(power, x, y, bounds, matrix, shift, size)
        east, north = np.meshgrid(np.arange(-9.5, 10), np.arange(10.5, -9, -1))
        np.testing.assert_allclose(actual, 100+2*(east-shift[0])+3*(north-shift[1]), atol=1e-5)

    def test_explicit_path_reference_changes_only_constant_curve(self):
        model = {'interval_s': [0., 60.], 'path_spacing_seconds': 3., 'heading_spacing_seconds': 8.}
        count = sum(basis([30.], model['interval_s'], model[k]).shape[1]
                    for k in ('path_spacing_seconds', 'heading_spacing_seconds'))
        model['parameters'] = np.random.default_rng(701).normal(size=count).tolist()
        a, b = SplineMotion(model), SplineMotion({**model, 'path_reference_time_s': 17.})
        t = np.linspace(0, 60, 101)
        self.assertAlmostEqual(b.curves([17.])[0][0], 0.)
        np.testing.assert_allclose(b.curves(t)[0], a.curves(t)[0]-a.curves([17.])[0][0], atol=1e-14)
        for derivative in (1, 2):
            np.testing.assert_allclose(b.components(t, derivative), a.components(t, derivative), atol=0.)

    def test_information_scaled_convex_join_is_exact_when_feasible(self):
        rng = np.random.default_rng(21)
        a = rng.normal(size=(300, 8))*np.logspace(-3, 2, 8)
        truth = rng.uniform(-.5, .5, 8)
        result = constrained_curve(a, a@truth, np.ones(300), np.eye(8), 1.)
        np.testing.assert_allclose(result, truth, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
