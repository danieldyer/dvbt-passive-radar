"""Reject invalid service settings before metadata, hardware or output access."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

CAPTURE = sys.argv.pop(1)


class CaptureCliTests(unittest.TestCase):
    def test_invalid_settings(self):
        cases = [
            (["--integration", "0"], "Detection requires"),
            (["--integration", "11"], "Detection requires"),
            (["--update", "-1"], "Detection requires"),
            (["--doppler", "0"], "Detection requires"),
            (["--range-metres", "100001"], "Detection requires"),
            (["--seconds", "nan"], "Invalid number"),
            (["--seconds", "-1"], "valid capture options"),
            (["--fec-workers", "6", "--frame-slots", "5"], "valid capture options"),
            (["--fec-workers", "1.5"], "Invalid FEC option"),
            (["--unpaced"], "Simulation controls require"),
            (["--archive", "typo"], "valid capture options"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for options, error in cases:
                with self.subTest(options=options):
                    output = root / "capture"
                    result = subprocess.run(
                        [
                            CAPTURE,
                            "--metadata",
                            str(root / "absent.json"),
                            "--output",
                            str(output),
                        ]
                        + options,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(error, result.stderr)
                    self.assertFalse(output.exists())

    def test_help(self):
        result = subprocess.run(
            [CAPTURE, "--help"], capture_output=True, text=True, check=False, timeout=10
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--no-tui", result.stdout)


if __name__ == "__main__":
    unittest.main()
