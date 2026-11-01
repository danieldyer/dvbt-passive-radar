import unittest

import numpy as np
from validate_capture_videos import admitted_ends


class CaptureTimingTests(unittest.TestCase):
    def rows(self):
        rows = np.zeros((12, 7))
        rows[:, 1] = np.arange(12)
        rows[:, 5] = 1
        return rows

    def test_erasure_keeps_cpi_extent_but_not_all_erased_map(self):
        rows = self.rows()
        rows[2:6, 5] = 0
        self.assertEqual(list(admitted_ends(rows, 1, 3, 1, True)), [2, 3, 6, 7, 8, 9, 10, 11])
        self.assertEqual(list(admitted_ends(rows, 1, 3, 1, False)), [8, 9, 10, 11])

    def test_segment_change_restarts_cpi_and_publish_cadence(self):
        rows = self.rows()
        rows[6:, 6] = 1
        self.assertEqual(list(admitted_ends(rows, 1, 3, 5, True)), [2, 8])

    def test_missing_row_restarts_integration(self):
        rows = self.rows()
        rows[6:, 1] += 2
        self.assertEqual(list(admitted_ends(rows, 1, 3, 1, True)), [2, 3, 4, 5, 10, 11, 12, 13])

    def test_map_epoch_uses_first_row_and_uniform_cpi_grid(self):
        rows = self.rows()
        rows[4:, 1] += 1e-5
        expected = list(admitted_ends(rows, 1, 3, 1, True))
        self.assertEqual(expected[2], 4.0)
        self.assertEqual(expected[4], 6.00001)


if __name__ == "__main__":
    unittest.main()
