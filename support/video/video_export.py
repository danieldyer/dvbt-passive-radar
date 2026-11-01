#!/usr/bin/env python3
"""Render all metadata-defined DVB-T channels with fixed noise-floor/peak scales.

One production invocation shares acquisition/channelisation/clock correction.
Only pooled display images are spooled; scientific complex maps are untouched.
The two display passes measure the whole-recording maximum before encoding.
"""

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import time
import zlib
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import render_video as video


def write_preview(stream, meta, db, compression=1):
    m = {k: v for k, v in meta.items() if k not in ("noise_range_factor", "noise_doppler_power")}
    m["display_shape"] = list(db.shape)
    header = json.dumps(m, separators=(",", ":")).encode()
    # float16 dB has <=0.016 dB error at the strongest usual returns, and much
    # less at the noise floor. This is a visualization cache, never a DSP input.
    payload = zlib.compress(np.asarray(db, dtype="<f2").tobytes(), compression)
    stream.write(struct.pack("<II", len(header), len(payload)))
    stream.write(header)
    stream.write(payload)


def read_previews(stream):
    while True:
        prefix = video.read_exact(stream, 8, allow_eof=True)
        if prefix is None:
            return
        length, size = struct.unpack("<II", prefix)
        if not 0 < length <= 1024 * 1024 or size > 32 * 1024 * 1024:
            raise ValueError("Invalid display cache record")
        meta = json.loads(video.read_exact(stream, length))
        raw = zlib.decompress(video.read_exact(stream, size))
        db = np.frombuffer(raw, dtype="<f2").reshape(meta["display_shape"]).astype(np.float32)
        yield meta, db


def fingerprint(input, metadata):
    s = input.stat()
    return {
        "input": str(input.resolve()),
        "bytes": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }


def read_pooled(source):
    previous = {}
    while True:
        prefix = video.read_exact(source, 16, allow_eof=True)
        if prefix is None:
            return
        magic, length, size = struct.unpack("<4sIQ", prefix)
        if magic != b"RDP1" or not 0 < length <= 1024 * 1024 or size > 32 * 1024 * 1024:
            raise ValueError("Invalid pooled preview framing")
        meta = json.loads(video.read_exact(source, length))
        shape = meta["display_shape"]
        if (
            meta["preview_format"] != "power_f32le"
            or meta["preview_order"] != "doppler_down,range"
            or size != math.prod(shape) * 4
        ):
            raise ValueError("Invalid pooled preview layout")
        c = meta["channel"]
        sequence, end = previous.get(c, (0, -math.inf))
        if meta["sequence"] != sequence + 1 or meta["time_end_s"] < end:
            raise ValueError("Missing pooled preview")
        previous[c] = meta["sequence"], meta["time_end_s"]
        power = np.frombuffer(video.read_exact(source, size), dtype="<f4").reshape(shape)
        db = 10 * np.log10(np.maximum(power, 1e-45))
        if not np.isfinite(db).all():
            raise ValueError("Nonfinite preview")
        yield meta, db


def collect(args, signals, work, command, source_duration):
    duration = source_duration - args.start
    if args.seconds:
        duration = min(duration, args.seconds)
    if duration <= 0:
        raise ValueError("Start is beyond recording")
    states = [
        {
            "id": s["id"],
            "name": s["name"],
            "rf_hz": s["frequency_hz"],
            "maps": 0,
            "peak_db": -math.inf,
            "noise_medians": [],
            "spool": str(work / (s["id"] + ".display")),
        }
        for s in signals
    ]
    streams = []
    processor = None
    started = time.monotonic()
    progress = -10
    env = dict(
        os.environ,
        TMP=str(work),
        XDG_CACHE_HOME=str(video.ROOT / "build/cache"),
        XDG_CONFIG_HOME=str(video.ROOT / "build/config"),
    )
    try:
        with ExitStack() as files, (work / "pipeline.log").open("w") as log:
            streams = [files.enter_context(Path(s["spool"]).open("xb")) for s in states]
            processor = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=log, env=env, cwd=video.ROOT
            )
            source = processor.stdout
            for meta, db in read_pooled(source):
                c = meta["channel"]
                s = states[c]
                if meta["physical_rf_hz"] != s["rf_hz"]:
                    raise ValueError("Map RF disagrees with metadata channel order")
                peak = float(db.max())
                s["peak_db"] = max(s["peak_db"], peak)
                s["maps"] += 1
                # Exclude stationary-suppression notch and the first 0.6 km.
                rows = np.linspace(args.doppler_hz, -args.doppler_hz, db.shape[0])
                background = db[
                    np.abs(rows) > 3 / args.integration,
                    math.ceil(0.6 / args.range_km * db.shape[1]) :,
                ]
                s["noise_medians"].append(float(np.median(background[::4, ::4])))
                write_preview(streams[c], dict(meta, map_peak_db=peak), db, 0)
                if meta["time_end_s"] >= progress + 10:
                    progress = meta["time_end_s"]
                    print(
                        f"Collect {progress:.1f}/{args.start + duration:.1f} s | maps {[s['maps'] for s in states]} | elapsed {time.monotonic() - started:.1f} s",
                        flush=True,
                    )
            if processor.wait() != 0:
                raise RuntimeError(f"Pipeline failed: {work / 'pipeline.log'}")
        if any(s["maps"] == 0 for s in states):
            raise RuntimeError("A metadata channel produced no validated maps")
        checked = video.duration_from_metadata(
            work / "pipeline/metadata.json", args.start, args.seconds
        )
        if abs(checked - duration) > 1e-7:
            raise RuntimeError("DSP and display WAV durations disagree")
        for s in states:
            noise = s.pop("noise_medians")
            median = float(np.median(noise))
            s["noise_median_db"] = median
            s["noise_median_p10_p90_db"] = [float(np.percentile(noise, p)) for p in (10, 90)]
            s["power_limits_db"] = [median - 1.5, s["peak_db"] + 0.25]
        manifest = {
            "source": fingerprint(args.input, args.metadata),
            "processor_command": command,
            "recording_start_s": args.start,
            "recording_duration_s": duration,
            "fps": args.fps,
            "integration_s": args.integration,
            "range_km": args.range_km,
            "doppler_hz": args.doppler_hz,
            "remove_dc": args.remove_dc,
            "signals": states,
            "collect_elapsed_s": time.monotonic() - started,
            "colour_scale": "Fixed per channel: whole-recording median displayed background minus 1.5 dB, full maximum plus 0.25 dB",
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        for process in (processor,):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def encode(args, manifest, state):
    suffix, codec = video.ENCODINGS[args.encoding]
    output = args.output / (state["id"] + suffix)
    if output.exists():
        report = json.loads(output.with_suffix(".json").read_text())
        if report["source"] != manifest["source"]:
            raise ValueError("Completed video belongs to another recording")
        print(f"Already complete: {output}", flush=True)
        return
    for path in (output.with_suffix(".json"), output.with_suffix(".png")):
        if path.exists():
            raise ValueError(f"Unfinished output already exists: {path}")
    args.db_min, args.db_max = state["power_limits_db"]
    args.auto_scale = False
    args.signal_name = f"{state['name']}  ·  {state['rf_hz'] / 1e6:g} MHz"
    renderer = video.Renderer(args)
    duration = manifest["recording_duration_s"]
    frames = math.ceil(duration * args.fps - 1e-9)
    temporary = args.work / (state["id"] + ".partial" + suffix)
    command = [
        shutil.which(args.ffmpeg),
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "1920x1080",
        "-framerate",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        *codec,
        "-movflags",
        "+faststart",
        "-metadata",
        f"title={args.signal_name} passive radar",
        str(temporary),
    ]
    encoder = None
    index = 0
    records = []
    poster = False
    started = time.monotonic()
    progress = -10
    try:
        with (
            (args.work / (state["id"] + ".encoder.log")).open("w") as log,
            Path(state["spool"]).open("rb") as spool,
        ):
            encoder = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log
            )

            def emit_until(end):
                nonlocal index, poster
                while index < frames and args.start + index / args.fps < end - 1e-10:
                    t = args.start + index / args.fps
                    valid = renderer.meta is not None and t - renderer.meta["time_end_s"] <= max(
                        0.5, 2 / args.fps
                    )
                    image = renderer.frame(t, duration, valid)
                    encoder.stdin.write(image.tobytes())
                    if not poster and t >= args.start + min(60, duration / 2) and valid:
                        image.save(output.with_suffix(".png"))
                        poster = True
                    index += 1

            for meta, db in read_previews(spool):
                emit_until(meta["time_end_s"])
                renderer.set_display_map(meta, db)
                records.append(dict(meta, display_db_min=args.db_min, display_db_max=args.db_max))
                if meta["time_end_s"] >= progress + 30:
                    progress = meta["time_end_s"]
                    print(
                        f"Encode {state['id']} {progress:.1f}/{args.start + duration:.1f} s | elapsed {time.monotonic() - started:.1f} s",
                        flush=True,
                    )
            emit_until(args.start + duration + 1 / args.fps)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"Encoder failed for {state['id']}")
        if len(records) != state["maps"] or index != frames:
            raise RuntimeError("Incomplete display cache or video")
        report = {k: v for k, v in manifest.items() if k != "signals"}
        report.update(
            signal=state,
            output=str(output),
            video_frames=index,
            video_duration_s=index / args.fps,
            power_limits_db=state["power_limits_db"],
            auto_scale=False,
            paced=False,
            encoding=args.encoding,
            encoder_command=command,
            encode_elapsed_s=time.monotonic() - started,
            maps=records,
            upper_clipped_maps=sum(m["map_peak_db"] > args.db_max for m in records),
        )
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)
        # Only this tool's disposable cache is removed after successful export.
        Path(state["spool"]).unlink()
        print(f"Video: {output} ({index} frames)", flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            try:
                encoder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
