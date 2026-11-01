import json
import tempfile
import unittest
from pathlib import Path

from analyse_diagnostics import summarize


class DiagnosticsTests(unittest.TestCase):
    def test_innovations_and_exact_time(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "diagnostics.jsonl"
            observations = [
                [1, 0, 6000 + i, 0, 1, 0, 0, 1, z, 1, z * z, True, False]
                for i, z in enumerate([1.0, -1.0, 1.0, -1.0])
            ]
            observations += [
                [1, 0, 6005, 0, None, 0, 0, 1, 1000, 1, 1e6, True, False],
                [1, 0, 5000, 0, 1, 0, 0, 1, 1000, 1, 1e6, False, True],
            ]
            utc = 1_700_000_000_123_456_789
            rows = [
                {"type": "configuration", "corrected_adc_rate_hz": 1000},
                {"type": "timing_observations", "rows": observations},
                {"type": "utc_anchor", "host_utc_ns": utc},
                {"type": "status", "elapsed_s": 6},
                {"type": "complete"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = summarize(path)
            self.assertTrue(result["complete"])
            self.assertEqual(result["utc_anchors"][0]["host_utc_ns"], utc)
            clock = result["clock_innovations"]["0"]
            self.assertEqual(
                (clock["offsets"], clock["count"], clock["rejected"], clock["late"]), (6, 4, 1, 1)
            )
            self.assertEqual(clock["mean_nis"], 1)
            self.assertAlmostEqual(clock["lag_one_correlation"], -1)

    def test_interrupted_log_is_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "diagnostics.jsonl"
            path.write_text('{"type":"configuration","corrected_adc_rate_hz":64000000}\n')
            self.assertFalse(summarize(path)["complete"])


if __name__ == "__main__":
    unittest.main()
