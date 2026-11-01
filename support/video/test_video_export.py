import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_export as recording


class RecordingVideoTest(unittest.TestCase):
    def test_display_cache_keeps_channel_phase_metadata(self):
        meta = {
            "channel": 3,
            "sequence": 17,
            "time_end_s": 29.7,
            "phase_epoch_s": 26.7,
            "physical_rf_hz": 219500000,
            "noise_range_factor": [1, 2],
            "noise_doppler_power": [3, 4],
            "map_peak_db": 41.37,
        }
        db = np.array([[-450, 1.723, 41.37], [-24.5, 12.57, 2.888]], dtype="<f4")
        stream = io.BytesIO()
        recording.write_preview(stream, meta, db)
        stream.seek(0)
        records = list(recording.read_previews(stream))
        self.assertEqual(len(records), 1)
        m, actual = records[0]
        for key in (
            "channel",
            "sequence",
            "time_end_s",
            "phase_epoch_s",
            "physical_rf_hz",
            "map_peak_db",
        ):
            self.assertEqual(m[key], meta[key])
        self.assertNotIn("noise_range_factor", m)
        np.testing.assert_allclose(actual, db, rtol=0, atol=0.016)
        truncated = io.BytesIO(stream.getvalue()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            list(recording.read_previews(truncated))


if __name__ == "__main__":
    unittest.main()
