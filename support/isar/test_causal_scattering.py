import unittest

import numpy as np

from causal_scattering import response, uniform_delays
from field import average
from spline_motion import SplineMotion
import test_physical_profile as physical_tests


class CausalScatteringTests(unittest.TestCase):
    def test_delay_moves_envelope_and_rf_phase_through_the_original_row_operator(self):
        band, motion = physical_tests.PhysicalProfileTests().fixture()
        point = band.points[0]
        state = SplineMotion({**band.model, 'parameters': motion}).apply(
            band.time, band.prior, band.reference_transmitter, band.receiver)
        position = state['position_ecef_m']+state['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        offset = (band.prior['position_ecef_m']-band.field.data['centre_ecef_m']+
                  state['translation_offset_m']+state['body_to_ecef']@point)
        delta, omega, _ = band.field.path_kinematics(position, velocity, position_offset=offset)
        delays = [0., 6., 12.]
        actual = response(band, motion, [point], delays)
        expected = []
        for delay in delays:
            raw = band.field.path_response(delta+delay, omega, range_bias_m=band.range_bias)
            binned, _ = average(raw, band.field.data['row_information'], band.field.metadata['block_rows'])
            expected.append(band.field.apply(binned).ravel()/np.sqrt(len(delays)))
        np.testing.assert_allclose(actual, np.column_stack(expected), rtol=2e-12, atol=2e-12)
        unshifted = band.design(motion, points=[point])[0]
        np.testing.assert_allclose(response(band, motion, [point], [0.]), unshifted, atol=1e-13)
        for invalid in ([-1.], [1., 0.], [0., 0.], [float('nan')], []):
            with self.assertRaises(ValueError):response(band, motion, [point], invalid)

    def test_uniform_causal_covariance_matches_its_analytic_frequency_kernel(self):
        frequency_difference = np.linspace(-56e6, 56e6, 2001)/299792458
        for width in (0., 6., 12., 24.):
            d, weights = uniform_delays(width, 16)
            actual = np.exp(-2j*np.pi*frequency_difference[:, None]*d)@weights
            expected = np.exp(-1j*np.pi*frequency_difference*width)*np.sinc(frequency_difference*width)
            np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=0)

    def test_antenna_correction_preserves_causal_delay_and_nominal_motion_reference(self):
        band, motion = physical_tests.PhysicalProfileTests().fixture()
        point = band.points[0]
        correction = np.array([[12., -7., 3.], [-4., 2., 1.]])
        tx = band.field.data['transmitter_ecef_m']+correction[0]
        rx = band.field.data['receiver_ecef_m']+correction[1]
        state = SplineMotion({**band.model, 'parameters': motion}).apply(
            band.time, band.prior, band.reference_transmitter, band.receiver)
        position = state['position_ecef_m']+state['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        offset = (band.prior['position_ecef_m']-band.field.data['centre_ecef_m']+
                  state['translation_offset_m']+state['body_to_ecef']@point)
        delta, omega, _ = band.field.path_kinematics(position, velocity, transmitter=tx,
            receiver=rx, position_offset=offset)
        raw = band.field.path_response(delta+6.3, omega, range_bias_m=band.range_bias)
        binned, _ = average(raw, band.field.data['row_information'], band.field.metadata['block_rows'])
        expected = band.field.apply(binned).ravel()
        actual = response(band, motion, [point], [6.3], antenna_offsets=correction)[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


if __name__ == '__main__':
    unittest.main()
