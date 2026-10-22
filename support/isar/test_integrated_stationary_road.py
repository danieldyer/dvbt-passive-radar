import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from range_fit import Field, Pulse
from integrated_stationary_road import RecordedBins


class IntegratedStationaryTests(unittest.TestCase):
    def test_irregular_original_row_averages_preserve_stationary_rf_phase(self):
        rng = np.random.default_rng(92741)
        time = (np.arange(400)+.5)*.001
        keep = rng.random(len(time)) > .12
        time = time[keep]
        information = rng.uniform(.5, 2., len(time))
        index = np.floor(time/.02).astype(int)
        total = np.bincount(index, weights=information)
        mean_time = np.bincount(index, weights=time*information)/total
        seed = {'epoch': .2, 'duration_s': .4, 'scale': 1., 'path': [1000., 30., 2.], 'rf_hz': 219.5e6}
        offsets = np.arange(-2, 3)
        field = Field({'time': mean_time, 'offset_samples': offsets, 'value': np.zeros((len(mean_time), 5), complex),
                       'variance': np.ones((len(mean_time), 5))},
                      Pulse(np.sinc(.8*np.arange(-64, 65))), seed)
        bins = RecordedBins(field, time, information)
        delay, coefficient = 1005.7, .7-.4j
        actual = bins.stationary(field, [delay])[:, :, 0]*coefficient
        expected = []
        for group in range(len(mean_time)):
            choose = index == group
            path = 1000.+30*(time[choose]-.2)+2*(time[choose]-.2)**2
            value = coefficient*np.exp(-1j*field.k*(delay-path+1000.))[:, None]
            value = value*field.pulse(offsets[None, :]-(delay-path[:, None])*8e6/299792458)
            expected.append(np.average(value, weights=information[choose], axis=0))
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        midpoint = coefficient*np.exp(-1j*field.k*(delay-field.seed_path+1000.))[:, None]
        midpoint = midpoint*field.pulse(offsets[None, :]-(delay-field.seed_path[:, None])*8e6/299792458)
        self.assertGreater(np.linalg.norm(midpoint-actual)/np.linalg.norm(actual), .2)


if __name__ == '__main__':
    unittest.main()
