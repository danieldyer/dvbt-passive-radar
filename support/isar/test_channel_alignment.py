"""Recover independently synthesized delay/phase offsets, without image fitting."""
import unittest

import numpy as np

from channel_alignment import Alignment, AlignmentBand
from coherent_scene import CoherentScene
from test_attitude_motion import independent_echo
from test_rigid_motion import fixture


def synthetic(position_offsets=None, rate_offsets=None):
    points = np.array([[-105., -12., 8.], [15., 14., 35.], [115., -6., 45.]])
    actual_points = points.copy()
    if position_offsets is not None:
        actual_points[1:, :2] += np.asarray(position_offsets).reshape(-1, 2)
    amplitude = np.array([.8+.3j, -.5+.4j, .4-.7j])
    biases = np.array([12., 0., -7.5])
    gains = np.array([.8*np.exp(.7j), 1., .7*np.exp(-1.2j)])
    bands = []
    for i, channel in enumerate((0, 3, 4)):
        field = fixture(channel)
        d = field.data
        state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'],
            'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        field.y = gains[i]*independent_echo(field, state, actual_points, amplitude, range_bias_m=biases[i],
            path_rate_bias_mps=0 if rate_offsets is None else rate_offsets[i], phase_epoch_s=8.)
        # Include strong stationary clutter. Its coefficients must be learned on
        # the same training split, not subtracted using the validation data.
        path = round(d['centre_path_m'].mean()*8e6/299792458)*299792458/8e6
        field.y += (3+2j)*field.stationary(path)
        selected = (np.floor(field.time/2).astype(int)%2) == 0
        bands.append(AlignmentBand(field, state, points, selected))
    return Alignment(bands, reference=1), biases, gains


class ChannelAlignmentTests(unittest.TestCase):
    def test_alternative_antennas_keep_the_original_data_phase_origin(self):
        original, _, _ = synthetic()
        b = original.bands[0]
        tx = b.field.data['transmitter_ecef_m']+np.array([110., -180., 40.])
        rx = b.field.data['receiver_ecef_m']+np.array([8., -4., 2.])
        alternate = AlignmentBand(b.field, b.state, b.points, b.selected_time, transmitter=tx, receiver=rx)
        amplitude = np.array([.8+.3j, -.5+.4j, .4-.7j])
        predicted = alternate.design(4., path_rate_bias_mps=.012, phase_epoch_s=8.)[0]@amplitude
        expected = independent_echo(b.field, b.state, b.points, amplitude, transmitter=tx, receiver=rx,
            range_bias_m=4., path_rate_bias_mps=.012, phase_epoch_s=8.).ravel()
        self.assertLess(np.linalg.norm(predicted-expected)/np.linalg.norm(expected), 5e-5)

    def test_path_rate_updates_range_phase_and_symbol_doppler_together(self):
        rates = [.012, 0., -.008]
        original, biases, gains = synthetic(rate_offsets=rates)
        alignment = Alignment(original.bands, reference=1, path_rate_scale_mps=.01, phase_epoch_s=8.)
        parameters = np.r_[[.4, -.2, -.1, -.25, .25, -.7], [.3, -.2]]
        _, gradient = alignment.objective(parameters)
        for i in range(len(parameters)):
            delta = np.eye(len(parameters))[i]*1e-5
            numeric = (alignment.objective(parameters+delta)[0]-alignment.objective(parameters-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, gradient[i], delta=1e-8)
        parameters, fit = alignment.fit()
        self.assertTrue(fit['converged'], fit)
        np.testing.assert_allclose(alignment.rates(parameters), rates, atol=1e-6)
        actual_bias, actual_gain = alignment.unpack(parameters)
        np.testing.assert_allclose(actual_bias, biases, atol=.003)
        self.assertLess(np.max(abs(np.degrees(np.angle(actual_gain/gains)))), .005)
        check = alignment.validation(parameters, [0, 16])
        self.assertLess(check['prediction_error']/check['stationary_only_error'], 2e-8)

    def test_shared_subpixel_recovery_on_independent_known_scene(self):
        shift = np.array([.25, -.15, -.2, .1])
        alignment, biases, gains = synthetic(shift)
        scene = CoherentScene(alignment)
        seed, _ = alignment.fit()
        parameters, result = scene.fit(seed, iterations=250)
        actual_points, actual_bias, actual_gain = scene.split(parameters)
        check = scene.validation(parameters, [0, 16])
        self.assertTrue(result['converged'], result)
        np.testing.assert_allclose(parameters[scene.nalignment:], shift, atol=.003)
        np.testing.assert_allclose(actual_bias, biases, atol=.003)
        self.assertLess(np.max(abs(np.degrees(np.angle(actual_gain/gains)))), .005)
        self.assertLess(check['prediction_error']/check['stationary_only_error'], 2e-8)

    def test_joint_subpixel_scene_and_calibration_gradient(self):
        alignment, _, _ = synthetic()
        scene = CoherentScene(alignment)
        p = np.r_[[.4, -.2, -.1, -.25, .25, -.7], [.06, -.03, -.05, .04]]
        _, gradient = scene.objective(p)
        for i in range(len(p)):
            delta = np.eye(len(p))[i]*.0002
            numeric = (scene.objective(p+delta)[0]-scene.objective(p-delta)[0])/.0004
            self.assertLess(abs(numeric-gradient[i]), 3e-7)

    def test_all_profiled_delay_gain_and_phase_derivatives(self):
        fit, _, _ = synthetic()
        p = np.array([.4, -.2, -.1, -.25, .25, -.7])
        _, gradient = fit.objective(p)
        for i in range(len(p)):
            step = np.eye(len(p))[i]*1e-5
            numeric = (fit.objective(p+step)[0]-fit.objective(p-step)[0])/2e-5
            self.assertAlmostEqual(numeric, gradient[i], delta=1e-8)

    def test_recovers_known_offsets_and_predicts_untouched_time_blocks(self):
        fit, biases, gains = synthetic()
        zero = np.zeros(3*fit.count)
        before = fit.validation(zero, [0, 16])
        parameters, result = fit.fit(iterations=180)
        actual_bias, actual_gain = fit.unpack(parameters)
        after = fit.validation(parameters, [0, 16])
        self.assertTrue(result['converged'], result)
        self.assertFalse(result['range_bound_hit'])
        np.testing.assert_allclose(actual_bias, biases, atol=.003)
        np.testing.assert_allclose(abs(actual_gain), abs(gains), atol=1e-4)
        self.assertLess(np.max(abs(np.degrees(np.angle(actual_gain/gains)))), .005)
        self.assertLess(after['prediction_error']/after['stationary_only_error'], 2e-8)
        self.assertLess(after['prediction_error']/before['prediction_error'], 1e-7)
        self.assertLess(after['conditional_coherent_loss_db'], 1e-6)


if __name__ == '__main__':
    unittest.main()
