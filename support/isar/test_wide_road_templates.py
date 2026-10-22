import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from wide_road_templates import WidePulse
from geographic_io import read_profiles
from range_fit import Pulse


class WideRoadTemplateTests(unittest.TestCase):
    def test_measured_tails_match_independent_fractional_fourier_response_and_derivative(self):
        frequency = np.linspace(-.42, .42, 801)
        weight = 1+.7*np.sin(3*frequency)+.3*np.cos(20*frequency)
        def exact(lag, derivative=0):
            return np.exp(2j*np.pi*np.asarray(lag)[:, None]*frequency)@((2j*np.pi*frequency)**derivative*weight)/weight.sum()
        samples = exact(np.arange(-256, 257))
        wide = WidePulse(samples)
        lag = np.array([-200.23, -97.7, -45.2, 0., 36.68, 82.9, 213.15])
        np.testing.assert_allclose(wide(lag), exact(lag), atol=2e-5, rtol=0.)
        np.testing.assert_allclose(wide(lag, 1), exact(lag, 1), atol=2e-5, rtol=0.)
        old = Pulse(samples[256-64:256+65])
        central = np.linspace(-30., 30., 501)
        np.testing.assert_allclose(wide(central), old(central), atol=2e-14)
        with self.assertRaises(ValueError):
            wide(np.array([225.]))

    def test_reader_preserves_legacy_and_extended_record_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            for radius in (64, 256):
                path = Path(temporary)/f'{radius}.gprf'
                header = json.dumps({'psf_samples': 2*radius+1, 'psf_lag_radius': radius}).encode()
                base = np.zeros(1024, '<c8').tobytes()
                psf = (np.arange(2*radius+1)+1j).astype('<c8')
                row = struct.pack('<IIQQdddd', 2, 0, 7, 0, 1., 1.1, 4., 1.05)+base+psf.tobytes()
                path.write_bytes(b'GEOProf1'+struct.pack('<I', len(header))+header+row+row)
                metadata, records = read_profiles(path)
                self.assertEqual(len(records), 2)
                np.testing.assert_array_equal(records['psf'][1], psf)
                self.assertEqual(records['bin'][1], 7)
                path.write_bytes(path.read_bytes()[:-1])
                with self.assertRaises(ValueError):
                    read_profiles(path)


if __name__ == '__main__':
    unittest.main()
