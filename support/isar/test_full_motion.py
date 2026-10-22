import unittest

import numpy as np

from field import average, FS
from motion import C
from antenna_motion import Antennas
from full_motion import Band
from fit_joint_motion import Joint, information_scaling
from spline_motion import basis
from test_rigid_motion import fixture


class FullMotionTests(unittest.TestCase):
    def test_fixed_locations_are_eliminated_from_preconditioned_search(self):
        rng = np.random.default_rng(157)
        a = rng.normal(size=(12, 12))
        expected = a.T@a+np.eye(12)
        free = np.arange(12) < 6
        transform, _ = information_scaling(expected, 5., free)
        self.assertEqual(transform.shape, (12, 6))
        np.testing.assert_array_equal(transform[~free], 0.)
        np.testing.assert_allclose(transform.T@expected@transform/5., np.eye(6), atol=1e-13)

    def test_antenna_derivatives_include_direct_baseline_and_doppler(self):
        field = fixture()
        d = field.data
        point = np.array([120., -12., 20.])
        position = d['centre_ecef_m']+d['body_to_ecef']@point
        velocity = d['centre_velocity_ecef_mps']+np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)@point
        tx = d['transmitter_ecef_m']+np.array([3., -1., 2.])
        rx = d['receiver_ecef_m']+np.array([-.5, 2., 1.])
        _, derivative = field.raw_trajectory(position, velocity, transmitter=tx, receiver=rx, antenna_derivatives=True)
        for axis in range(6):
            delta = np.eye(6)[axis].reshape(2, 3)*.0005
            plus = field.raw_trajectory(position, velocity, transmitter=tx+delta[0], receiver=rx+delta[1])
            minus = field.raw_trajectory(position, velocity, transmitter=tx-delta[0], receiver=rx-delta[1])
            numeric = (plus-minus)/.001
            self.assertLess(np.linalg.norm(numeric-derivative[:, :, axis])/np.linalg.norm(numeric), 1e-5)

    def test_correlated_antenna_coordinates_and_mapping(self):
        angle = .32
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        model = Antennas(rotation, [0, 0, 0, 0, 2])
        rng = np.random.default_rng(36)
        p = rng.normal(size=model.count)
        enu = (model.factor@p).reshape(6, 3)
        for channel in range(5):
            np.testing.assert_allclose(model.offsets(p, channel), enu[[channel, 5]]@rotation, atol=1e-12)
        covariance = model.factor@model.factor.T
        self.assertAlmostEqual(covariance[0, 3], 30.**2)
        self.assertEqual(covariance[0, 12], 0.)

    def test_exact_original_row_adjoint_with_erasures_and_complex_covariance(self):
        field = fixture()
        rng = np.random.default_rng(394)
        raw = rng.normal(size=(len(field.data['row_time']), field.y.shape[1]))+1j*rng.normal(size=(len(field.data['row_time']), field.y.shape[1]))
        residual = rng.normal(size=field.y.shape)+1j*rng.normal(size=field.y.shape)
        observed = field.apply(average(raw, field.data['row_information'], field.metadata['block_rows'])[0])
        np.testing.assert_allclose(np.vdot(residual, observed), np.vdot(field.raw_adjoint(residual), raw), rtol=1e-12)

    def test_profiled_gradient_matches_full_resampling_and_phase_finite_difference(self):
        field = fixture()
        d = field.data
        interval = [0, d['row_time'][-1]]
        model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.}
        npath = basis([8.], interval, 6.).shape[1]
        nyaw = basis([8.], interval, 8.).shape[1]
        prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
                 'body_to_ecef': d['body_to_ecef'], 'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        points = np.array([[-70., 4., 14.], [90., -7., 35.]])
        rng = np.random.default_rng(8532)
        true_parameters = rng.normal(size=npath+nyaw)*.05
        band = Band(field, prior, d['transmitter_ecef_m'], d['receiver_ecef_m'], points, model, np.arange(len(field.time))%2 == 0)
        target, _ = band.design(true_parameters)
        stationary_path = round((d['centre_path_m'].mean()+80)*FS/C)*C/FS
        stationary = field.stationary(stationary_path).ravel()
        field.y = (target@np.array([.8+.3j, -.5+.4j])+(15+8j)*stationary).reshape(field.y.shape)
        band = Band(field, prior, d['transmitter_ecef_m'], d['receiver_ecef_m'], points, model, band.selected_time)
        p = true_parameters*.3
        value, gradient = band.evaluate(p)
        for parameter in range(len(p)):
            step = np.eye(len(p))[parameter]*.0002
            numeric = (band.evaluate(p+step)[0]-band.evaluate(p-step)[0])/.0004
            self.assertLess(abs(numeric-gradient[parameter])/max(abs(numeric), 1.), 8e-5)
        self.assertLess(band.evaluate(true_parameters)[0], value*1e-10)
        # Joint location/motion derivatives must include profiling the complex
        # amplitudes. Constant antenna-induced phase is not calibration data.
        antennas = Antennas(np.eye(3), [0])
        joint = Joint([band], antennas, p)
        probe = joint.initial.copy()
        probe[-6:] = [.03, -.02, .01, -.015, .025, -.01]
        _, derivative = joint.objective(probe)
        for parameter in range(len(probe)-6, len(probe)):
            step = np.eye(len(probe))[parameter]*.00005
            numeric = (joint.objective(probe+step)[0]-joint.objective(probe-step)[0])/.0001
            self.assertLess(abs(numeric-derivative[parameter]), 2e-6)


if __name__ == '__main__':
    unittest.main()
