import unittest
import numpy as np

from antenna_motion import Antennas
from export_shared_motion import exported_model
from geometry_profile import GeometryProfile
from imaging import corrected_sites
from spline_motion import SplineMotion
import test_physical_profile as physical_tests


class SharedMotionExportTests(unittest.TestCase):
    def test_licensed_and_fitted_offsets_are_combined_without_changing_motion_gauge(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        angle = .35
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        profile = GeometryProfile([band], truth*.3, Antennas(rotation, [0]),
            base_offsets=[[[300., -120., 40.], [0., 0., 0.]]])
        q = profile.initial.copy(); q[profile.antenna_slice] = [.3, -.2, .1, -.05, .1, .2]
        q[profile.range_slice] = -3.7
        model = exported_model(profile, q)
        tx = band.field.data['transmitter_ecef_m'][None]; rx = band.field.data['receiver_ecef_m']
        actual_tx, actual_rx = corrected_sites(tx, rx, model['antenna_fit'])
        expected = profile.offsets(q, 0)
        np.testing.assert_allclose(actual_tx[0], tx[0]+expected[0], atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual_rx, rx+expected[1], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(SplineMotion(model).parameters, profile.expand(q))
        self.assertEqual(model['range_bias_m'], [-3.7])
        self.assertFalse(model['phase_centres_validated'])
        self.assertFalse(model['channel_calibration_validated'])

    def test_point_templates_survive_export_with_receiver_and_site_offsets(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        profile = GeometryProfile([band], truth*.4, Antennas(np.eye(3), [0]),
            base_offsets=[[[25., -35., 4.], [0., 0., 0.]]])
        q = profile.initial.copy(); q[profile.antenna_slice] = [.03, -.04, .01, .02, -.01, .04]
        q[profile.range_slice] = 2.4
        model = exported_model(profile, q)
        before = band.design(profile.expand(q), antenna_offsets=profile.offsets(q, 0), range_bias_m=q[profile.range_slice][0])[0]
        offsets = np.array([model['antenna_fit']['transmitter_offsets_enu_m'][0], model['antenna_fit']['receiver_offset_enu_m']])
        after = band.design(np.array(model['parameters']), antenna_offsets=offsets, range_bias_m=model['range_bias_m'][0])[0]
        np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
