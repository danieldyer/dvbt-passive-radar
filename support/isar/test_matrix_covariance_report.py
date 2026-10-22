import tempfile
import unittest
from pathlib import Path

import numpy as np

from matrix_covariance_report import summary


class MatrixCovarianceReportTests(unittest.TestCase):
    def test_reported_decibels_match_exact_power_ratios_for_all_aperture_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'channel.npz'
            amplitude = np.broadcast_to([2.+0j, 1., 4.], (2, 2, 1, 3))
            thermal = np.broadcast_to([1., 2., 3.], amplitude.shape)
            np.savez(path, amplitudes=amplitude, thermal_variance=thermal,
                long_amplitudes=amplitude[:, 0], long_thermal_variance=thermal[:, 0])
            experiment = {'channels': [{'data': str(path)}]}
            expected = [0., 10*np.log10(4.), -10*np.log10(4.)]
            for options in ({}, {'long': True}, {'local': True}):
                result = summary(experiment, **options)
                np.testing.assert_allclose(result['aggregate_gain_db'], expected)
                np.testing.assert_allclose(result['per_channel_gain_db'], [expected])
                np.testing.assert_allclose(result['per_range_channel_gain_db'], [[expected, expected]])
                np.testing.assert_allclose(result['descriptive_resampling_range_db'], np.array([expected, expected]).T)


if __name__ == '__main__':
    unittest.main()
