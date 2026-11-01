#!/usr/bin/env python3
"""Shared fixed-scale frame drawing and hardware codec settings for archive export."""

import json
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ENCODINGS = {
    "h264": (
        ".mp4",
        (
            "-c:v",
            "h264_videotoolbox",
            "-q:v",
            "75",
            "-allow_sw",
            "0",
            "-g",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-bsf:v",
            "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
        ),
    ),
    # Apple's hardware codec encodes every frame independently for random access.
    "prores": (
        ".mov",
        (
            "-c:v",
            "prores_videotoolbox",
            "-profile:v",
            "standard",
            "-allow_sw",
            "0",
            "-vf",
            "scale=out_color_matrix=bt709:out_range=tv",
            "-pix_fmt",
            "p210le",
            "-color_range",
            "tv",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-colorspace",
            "bt709",
            "-bsf:v",
            "prores_metadata=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        ),
    ),
}


def read_exact(source, size, allow_eof=False):
    result = bytearray(size)
    position = 0
    while position < size:
        count = source.readinto(memoryview(result)[position:])
        if not count:
            if allow_eof and not position:
                return None
            raise ValueError("Truncated RDM frame")
        position += count
    return result


def read_frames(source):
    previous = {}
    while True:
        prefix = read_exact(source, 16, allow_eof=True)
        if prefix is None:
            return
        magic, length, size = struct.unpack("<4sIQ", prefix)
        if magic != b"RDM1" or not 0 < length <= 1024 * 1024 or size > 2**31:
            raise ValueError("Invalid RDM framing; stdout must contain only binary maps")
        meta = json.loads(read_exact(source, length))
        nr, nf = meta["ranges"], meta["frequencies"]
        if meta["format"] != "cf32le" or meta["order"] != "range,doppler" or size != nr * nf * 8:
            raise ValueError("Unsupported RDM layout")
        channel = meta.get("channel", 0)
        previous_sequence, previous_end = previous.get(channel, (0, -math.inf))
        if meta["sequence"] != previous_sequence + 1 or meta["time_end_s"] < previous_end:
            raise ValueError("Missing, reordered or non-monotonic RDM frame")
        previous[channel] = meta["sequence"], meta["time_end_s"]
        values = np.frombuffer(read_exact(source, size), dtype="<c8").reshape(nr, nf)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite radar map")
        yield meta, values


def power_preview(values, width, height):
    """Pool complex-map power before resizing; narrow Doppler peaks survive."""
    rd = max(1, math.ceil(values.shape[0] / width))
    fd = max(1, math.ceil(values.shape[1] / height))
    pooled = np.empty(
        (math.ceil(values.shape[0] / rd), math.ceil(values.shape[1] / fd)), dtype=np.float32
    )
    for first in range(0, pooled.shape[0], 32):
        last = min(first + 32, pooled.shape[0])
        z = values[first * rd : min(last * rd, values.shape[0])]
        power = z.real * z.real
        power += z.imag * z.imag
        if fd > 1:
            power = np.maximum.reduceat(power, np.arange(0, power.shape[1], fd), axis=1)
        if rd > 1:
            power = np.maximum.reduceat(power, np.arange(0, power.shape[0], rd), axis=0)
        pooled[first:last] = power
    return (10 * np.log10(np.maximum(pooled, 1e-30))).T[::-1]


def clock_label(seconds):
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - 60 * minutes:06.3f}"


class Renderer:
    def __init__(self, args):
        self.args = args
        self.width, self.height = 1920, 1080
        self.box = (118, 196, 1742, 936)
        paths = [
            ROOT / "build/vendor/imgui-1.91.9b/misc/fonts/Roboto-Medium.ttf",
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
        font = next((p for p in paths if p.exists()), None)
        self.fonts = {
            size: ImageFont.truetype(str(font), size) if font else ImageFont.load_default(size=size)
            for size in (18, 21, 24, 28, 38)
        }
        stops = np.array(
            [
                [0.018, 0.032, 0.07],
                [0.08, 0.22, 0.40],
                [0.08, 0.65, 0.66],
                [0.94, 0.65, 0.20],
                [1, 0.98, 0.82],
            ]
        )
        x = np.linspace(0, 4, 4096)
        self.colours = np.stack(
            [np.interp(x, np.arange(5), stops[:, c]) for c in range(3)], axis=-1
        )
        self.colours = np.round(self.colours * 255).astype(np.uint8)
        self.db_min, self.db_max = args.db_min, args.db_max
        self.base = self.background()
        self.map_layer = None
        self.meta = None

    def text(self, draw, xy, text, size=24, colour="#c2d0df", anchor=None):
        draw.text(xy, text, fill=colour, font=self.fonts[size], anchor=anchor)

    def background(self):
        image = Image.new("RGB", (self.width, self.height), "#0c131e")
        draw = ImageDraw.Draw(image)
        self.text(draw, (42, 28), "DVB-T  /  PASSIVE RADAR", 28, "#55d6c1")
        title = "MATCHED-FILTER DETECTION"
        self.text(draw, (485, 31), title, 24, "#899bad")
        if getattr(self.args, "signal_name", None):
            self.text(draw, (1878, 31), self.args.signal_name, 24, "#eef5f8", anchor="rt")
        draw.line((42, 78, 1878, 78), fill="#304050", width=1)
        x0, y0, x1, y1 = self.box
        draw.rectangle(self.box, fill="#040813", outline="#415568")
        for km in range(0, int(self.args.range_km) + 1, max(1, math.ceil(self.args.range_km / 6))):
            x = x0 + (x1 - x0) * km / self.args.range_km
            self.text(draw, (x, y1 + 18), str(km), 24, anchor="mt")
        self.text(
            draw, ((x0 + x1) / 2, y1 + 61), "Bistatic excess path length [km]", 28, anchor="mt"
        )
        for hz in np.linspace(-self.args.doppler_hz, self.args.doppler_hz, 5):
            y = y1 - (y1 - y0) * (hz + self.args.doppler_hz) / (2 * self.args.doppler_hz)
            self.text(draw, (x0 - 18, y), f"{hz:+g}", 24, anchor="rm")
        self.text(draw, (x0 - 18, y0 - 27), "Hz", 21, anchor="rm")
        bar = Image.fromarray(
            self.colours[np.linspace(4095, 0, y1 - y0).astype(int)][:, None, :]
        ).resize((24, y1 - y0))
        image.paste(bar, (1775, y0))
        for db in np.linspace(self.db_min, self.db_max, 7):
            y = y1 - (db - self.db_min) / (self.db_max - self.db_min) * (y1 - y0)
            self.text(draw, (1812, y), f"{db:.0f}", 21, anchor="lm")
        self.text(draw, (1770, y0 - 35), "Power [dB]", 21)
        scale = "Auto power scale" if self.args.auto_scale else "Fixed power scale"
        dc = "Stationary / DC removed" if self.args.remove_dc else "DC retained"
        self.text(
            draw,
            (42, 1030),
            f"{scale}  ·  {dc}  ·  Positive Doppler upward  ·  No grid overlay",
            18,
            "#899bad",
        )
        return image

    def set_map(self, meta, values):
        x0, y0, x1, y1 = self.box
        db = power_preview(values, x1 - x0, y1 - y0)
        self.set_display_map(meta, db)

    def set_display_map(self, meta, db):
        x0, y0, x1, y1 = self.box
        index = np.clip((db - self.db_min) / (self.db_max - self.db_min) * 4095, 0, 4095).astype(
            np.int32
        )
        self.map_layer = Image.fromarray(self.colours[index]).resize(
            (x1 - x0, y1 - y0), Image.Resampling.NEAREST
        )
        self.meta = dict(
            meta,
            display_db_min=self.db_min,
            display_db_max=self.db_max,
            map_peak_db=float(db.max()),
        )

    def frame(self, recording_time, duration, valid=True):
        image = self.base.copy()
        x0, y0, x1, y1 = self.box
        if self.map_layer is not None and valid:
            image.paste(self.map_layer, (x0, y0))
        draw = ImageDraw.Draw(image)
        self.text(
            draw,
            (42, 104),
            f"{clock_label(recording_time)}  /  {clock_label(self.args.start + duration)}",
            38,
            "#eef5f8",
        )
        if self.meta is not None and valid:
            m = self.meta
            state = (
                "SLIDING WINDOW"
                if m["integration_s"] >= self.args.integration - 0.02
                else "WINDOW FILL / COHERENCE LIMIT"
            )
            self.text(
                draw,
                (685, 106),
                f"{m['integration_s']:.3f} s coherent   ·   {state}",
                28,
                "#55d6c1",
            )
            detail = (
                "Linear range migration corrected"
                if m.get("motion_model") == "constant_bistatic_path_rate"
                else f"coherence {m['coherence']:.5f}"
            )
            self.text(
                draw,
                (685, 148),
                f"Window {m['time_start_s']:.3f}–{m['time_end_s']:.3f} s  |  bins {m['frequency_step_hz']:.4f} Hz  |  {detail}",
                21,
                "#aabaca",
            )
        else:
            self.text(
                draw,
                ((x0 + x1) / 2, (y0 + y1) / 2),
                "Acquiring / no complete validated map",
                28,
                "#aabaca",
                anchor="mm",
            )
            self.text(
                draw,
                (685, 112),
                f"{self.args.integration:g} s sliding coherent window",
                28,
                "#55d6c1",
            )
        progress = max(0, min(1, (recording_time - self.args.start) / duration))
        draw.rectangle((42, 1063, 1878, 1067), fill="#253545")
        draw.rectangle((42, 1063, 42 + 1836 * progress, 1067), fill="#55d6c1")
        return image


def duration_from_metadata(path, start, seconds):
    m = json.loads(path.read_text())
    available = m["input_samples"] / m["input_sample_rate"] - start
    return min(seconds, available) if seconds > 0 else available


def validate_options(args, guard):
    """Validate the single archive CLI's numeric settings and derive cadence."""
    if guard not in (4, 8, 16, 32):
        raise ValueError("Invalid archived guard interval")
    if args.fps is None:
        interval = (8192 + 8192 / guard) / 8e6
        args.fps = 1 / (max(1, math.floor(1 / (30 * interval))) * interval)
    for key in ("integration", "range_km", "doppler_hz", "fps", "start", "seconds"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"--{key.replace('_', '-')} must be finite")
    if (
        not 0.5 <= args.integration <= 10
        or not 1 <= args.range_km <= 100
        or not 4 / args.integration <= args.doppler_hz <= 200
        or not 0.1 <= args.fps <= 60
    ):
        raise ValueError("Invalid integration, range, Doppler or frame rate")
    if args.start < 0 or args.seconds < 0:
        raise ValueError("Invalid time interval")
    args.auto_scale = False
    return args
