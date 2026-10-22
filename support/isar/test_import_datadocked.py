import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from ais import read_csv
from import_datadocked import convert


class ImportTests(unittest.TestCase):
    def test_reverse_history_is_sorted_without_fabricating_heading(self):
        with TemporaryDirectory() as directory:
            source, destination = Path(directory)/'provider.json', Path(directory)/'ais.csv'
            rows = [{'lat': -37.8, 'lng': 144.9, 'course': 357, 'speed': 10,
                     'time': f'2026-10-13T08:3{i}:00Z'} for i in (4, 3)]
            source.write_text(json.dumps({'response': {'status': 'success', 'data': rows}}))
            result = convert(source, destination, requested_mmsi=563052200)
            track = read_csv(destination, reference_height_ellipsoid_m=5, mmsi=563052200)
            self.assertEqual(np.diff(track.seconds)[0], 60)
            np.testing.assert_allclose(track.speed, 10*1852/3600)
            self.assertTrue(np.isnan(track.heading).all())
            self.assertTrue(np.isnan(track.turn_rate).all())
            self.assertFalse(result['identity_echoed_by_response'])
            with self.assertRaises(FileExistsError):
                convert(source, destination, requested_mmsi=563052200)

    def test_duplicate_report_is_not_silently_dropped(self):
        with TemporaryDirectory() as directory:
            source, destination = Path(directory)/'provider.json', Path(directory)/'ais.csv'
            row = {'lat': -37.8, 'lng': 144.9, 'time': '2026-10-13T08:34:00Z'}
            source.write_text(json.dumps({'response': {'status': 'success', 'data': [row, row]}}))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                convert(source, destination, requested_mmsi=563052200)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
