"""AIS import and interpolation must not invent timing or attitude precision."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ais import read_csv, utc_seconds
from motion import WGS84_A


class AisTests(unittest.TestCase):
    def read(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'track.csv'
            path.write_text(text)
            return read_csv(path, reference_height_ellipsoid_m=0., mmsi=563052200)

    def test_utc_offset_and_required_timezone(self):
        self.assertEqual(utc_seconds('2026-10-13T19:33:36+11:00'), utc_seconds('2026-10-13T08:33:36Z'))
        with self.assertRaises(ValueError):
            utc_seconds('2026-10-13T08:33:36')

    def test_velocity_units_and_heading_are_independent(self):
        speed = 10*1852/3600
        delta = np.degrees(10*speed/WGS84_A)
        track = self.read('utc,latitude_deg,longitude_deg,sog_knots,cog_deg,true_heading_deg\n'
                          '2026-10-13T08:33:30Z,0,0,10,90,0\n'
                          f'2026-10-13T08:33:40Z,0,{delta},10,90,0\n')
        point = track.sample(track.epoch+5, max_gap_s=10)
        np.testing.assert_allclose(point['longitude'], np.radians(delta/2), atol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(point['velocity_ecef_mps'][0]), speed, places=10)
        self.assertEqual(point['true_heading_radians'][0], 0.)

    def test_gaps_and_missing_heading_preserve_exact_reports(self):
        track = self.read('utc,latitude_deg,longitude_deg,true_heading_deg\n'
                          '2026-10-13T08:33:30Z,-37.9,144.9,359\n'
                          '2026-10-13T08:33:40Z,-37.9,144.91,1\n'
                          '2026-10-13T08:35:40Z,-37.9,144.92,511\n')
        midpoint = track.sample(track.epoch+5, max_gap_s=10)
        self.assertLess(abs(np.angle(np.exp(1j*midpoint['true_heading_radians'][0]))), 1e-14)
        known = track.sample(track.seconds, max_gap_s=10)
        np.testing.assert_allclose(known['true_heading_radians'][:2], np.radians([359, 1]))
        self.assertTrue(np.isnan(known['true_heading_radians'][2]))
        with self.assertRaises(ValueError):
            track.sample(track.epoch+50, max_gap_s=10)
        self.assertTrue(np.isnan(track.sample(track.epoch+50, max_gap_s=120)['true_heading_radians'][0]))
        for seconds in [track.epoch-1, track.seconds[-1]+1]:
            with self.assertRaises(ValueError):
                track.sample(seconds, max_gap_s=120)

    def test_reject_wrong_units_nonfinite_and_duplicate_times(self):
        header = 'utc,latitude_deg,longitude_deg,sog_knots,mmsi\n'
        for bad in ['103', 'nan', 'inf', '-1']:
            with self.assertRaises(ValueError):
                self.read(header+f'2026-10-13T08:33:30Z,0,0,{bad},563052200\n'
                          '2026-10-13T08:33:40Z,0,0,10,563052200\n')
        for second_time, mmsi in [('08:33:30', '563052200'), ('08:33:40', '563052201')]:
            with self.assertRaises(ValueError):
                self.read(header+'2026-10-13T08:33:30Z,0,0,10,563052200\n'
                          f'2026-10-13T{second_time}Z,0,0,10,{mmsi}\n')

    def test_dateline_is_unwrapped(self):
        track = self.read('utc,latitude_deg,longitude_deg\n'
                          '2026-10-13T08:33:30Z,0,179.999\n'
                          '2026-10-13T08:33:40Z,0,-179.999\n')
        self.assertAlmostEqual(track.sample(track.epoch+5, max_gap_s=10)['longitude'][0], np.pi)


if __name__ == '__main__':
    unittest.main()
