import importlib.util
import io
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "video", Path(__file__).resolve().with_name("render_video.py")
)
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def packet(sequence=1, end=1):
    values = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype="<c8")
    meta = {
        "sequence": sequence,
        "time_end_s": end,
        "ranges": 2,
        "frequencies": 2,
        "format": "cf32le",
        "order": "range,doppler",
    }
    header = json.dumps(meta).encode()
    return struct.pack("<4sIQ", b"RDM1", len(header), values.nbytes) + header + values.tobytes()


class Chunked(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))

    def readinto(self, buffer):
        return super().readinto(buffer[:7])


class VideoTest(unittest.TestCase):
    def test_stream_short_reads_and_sequence(self):
        frames = list(video.read_frames(Chunked(packet() + packet(2, 1.2))))
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1][1][1, 0], 5 + 6j)
        with self.assertRaisesRegex(ValueError, "Missing"):
            list(video.read_frames(io.BytesIO(packet() + packet(3, 1.2))))
        with self.assertRaisesRegex(ValueError, "Truncated"):
            list(video.read_frames(io.BytesIO(packet()[:-1])))

    def test_interleaved_channel_sequences(self):
        def channel_packet(channel, sequence, end):
            values = np.ones((2, 2), dtype="<c8")
            meta = {
                "channel": channel,
                "sequence": sequence,
                "time_end_s": end,
                "ranges": 2,
                "frequencies": 2,
                "format": "cf32le",
                "order": "range,doppler",
            }
            header = json.dumps(meta).encode()
            return (
                struct.pack("<4sIQ", b"RDM1", len(header), values.nbytes)
                + header
                + values.tobytes()
            )

        data = (
            channel_packet(0, 1, 1)
            + channel_packet(1, 1, 0.9)
            + channel_packet(0, 2, 1.2)
            + channel_packet(1, 2, 1.1)
        )
        self.assertEqual(len(list(video.read_frames(io.BytesIO(data)))), 4)
        with self.assertRaisesRegex(ValueError, "Missing"):
            list(video.read_frames(io.BytesIO(data + channel_packet(1, 4, 1.3))))

    def test_doppler_orientation_and_peak_pool(self):
        values = np.full((4, 8), 0.001 + 0j, dtype=np.complex64)
        values[2, 7] = 2  # positive Doppler must be at the top, range to the right
        db = video.power_preview(values, 2, 4)
        self.assertEqual(np.unravel_index(np.argmax(db), db.shape), (0, 1))
        self.assertAlmostEqual(float(db[0, 1]), 20 * np.log10(2), places=5)

    def test_pool_partial_edges(self):
        rng = np.random.default_rng(507)
        values = (rng.normal(size=(103, 281)) + 1j * rng.normal(size=(103, 281))).astype("<c8")
        actual = video.power_preview(values, 17, 39)
        expected = np.array(
            [
                [
                    10
                    * np.log10(np.max(abs(values[r : min(r + 7, 103), f : min(f + 8, 281)]) ** 2))
                    for f in range(0, 281, 8)
                ]
                for r in range(0, 103, 7)
            ]
        )
        np.testing.assert_allclose(actual, expected.T[::-1], rtol=1e-6, atol=2e-6)

    def test_archive_settings(self):
        args = SimpleNamespace(
            integration=3, range_km=100, doppler_hz=100, fps=None, start=0, seconds=0
        )
        video.validate_options(args, 16)
        self.assertAlmostEqual(args.fps, 3125 / 102)
        for key, value in (
            ("fps", float("nan")),
            ("range_km", 101),
            ("doppler_hz", 201),
            ("seconds", -1),
        ):
            invalid = SimpleNamespace(**vars(args))
            setattr(invalid, key, value)
            with self.assertRaises(ValueError):
                video.validate_options(invalid, 16)

    def test_quicktime_output_selection(self):
        prores = video.ENCODINGS["prores"]
        h264 = video.ENCODINGS["h264"]
        self.assertEqual(prores[0], ".mov")
        self.assertIn("prores_videotoolbox", prores[1])
        self.assertEqual(h264[0], ".mp4")
        self.assertIn("h264_videotoolbox", h264[1])
        self.assertEqual(h264[1][h264[1].index("-g") + 1], "1")


if __name__ == "__main__":
    unittest.main()
