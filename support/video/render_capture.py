#!/usr/bin/env python3
"""Render every channel from a saved live profile archive, without raw I/Q.

Uses production Vulkan focusing and the existing fixed-scale ProRes exporter.
Only complete, consecutive, valid CPIs are shown; missing data is never filled.
"""

import argparse
import json
import re
import shutil
import struct
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import render_video as video
import video_export as recording


def archive_layout(path):
    """Validate framing and index timing without loading the complex payloads."""
    total = path.stat().st_size
    duration = 0
    profiles = 0
    with path.open("rb") as source:
        if video.read_exact(source, 8) != b"DVBTLIV1":
            raise ValueError("Invalid live archive magic")
        (length,) = struct.unpack("<I", video.read_exact(source, 4))
        if not 0 < length <= 1024 * 1024:
            raise ValueError("Invalid live header size")
        header = json.loads(video.read_exact(source, length))
        if header["archive_mode"] not in ("profiles", "both"):
            raise ValueError(
                "This renderer requires a profiles archive (--archive profiles or both)"
            )
        while True:
            tag, length, size = struct.unpack("<4sIQ", video.read_exact(source, 16))
            if not 0 < length <= 1024 * 1024 or size > 64 * 1024 * 1024:
                raise ValueError("Invalid live record bounds")
            meta = json.loads(video.read_exact(source, length))
            if source.tell() + size > total:
                raise ValueError("Truncated live payload")
            if tag == b"PROF":
                c = meta["channel"]
                rows = meta["rows"]
                if (
                    not rows
                    or len(rows) != meta["count"]
                    or meta["size"] not in (1024, 3072, 4096)
                    or meta["moments"] != 2
                ):
                    raise ValueError("Invalid profile dimensions")
                interval = (8192 + 8192 / header["applied_channels"][c]["guard"]) / 8e6
                duration = max(duration, rows[-1][1] + interval)
                profiles += 1
            source.seek(size, 1)
            if tag == b"DONE":
                if source.tell() != total:
                    raise ValueError("Unexpected bytes after DONE")
                break
    if not profiles:
        raise ValueError("Archive contains no saved profiles")
    return header, duration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Capture directory or capture.dvblive")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--processor", type=Path, default=video.ROOT / "build/release/live_profiles_preview"
    )
    p.add_argument("--integration", type=float, default=3)
    p.add_argument("--doppler-hz", type=float, default=100)
    p.add_argument("--range-km", type=float, default=100)
    p.add_argument("--fps", type=float)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--seconds", type=float, default=0)
    p.add_argument("--encoding", choices=tuple(video.ENCODINGS), default="prores")
    p.add_argument("--remove-dc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--encode-workers", type=int)
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if args.input.is_dir():
        args.input = args.input / "capture.dvblive"
    header, duration = archive_layout(args.input)
    signals = header["signals"]
    if args.encode_workers is None:
        args.encode_workers = min(3, len(signals))
    if not 1 <= args.encode_workers <= len(signals):
        p.error("Invalid encode worker count")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", s["id"]) for s in signals):
        p.error("Signal ids must be safe filenames")
    if not shutil.which(args.ffmpeg):
        p.error("FFmpeg is required")
    if not args.resume and not args.processor.is_file():
        p.error("Build the optional live_profiles_preview target first")
    try:
        video.validate_options(args, signals[0]["guard_denominator"])
    except ValueError as error:
        p.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    args.work = args.output / "processing"
    args.metadata = args.input.parent / "metadata.json"
    if not args.metadata.is_file():
        args.metadata = args.output / "source_metadata.json"
        if not args.metadata.exists():
            args.metadata.write_text(json.dumps(header, indent=2) + "\n")
    if args.resume:
        manifest = json.loads((args.work / "manifest.json").read_text())
        if recording.fingerprint(args.input, args.metadata) != manifest["source"]:
            raise ValueError("Archive changed since collection")
        for key, field in (
            ("integration", "integration_s"),
            ("range_km", "range_km"),
            ("doppler_hz", "doppler_hz"),
            ("fps", "fps"),
            ("start", "recording_start_s"),
            ("remove_dc", "remove_dc"),
        ):
            if getattr(args, key) != manifest[field]:
                raise ValueError("Resume settings disagree with collection")
        requested = duration - args.start
        if args.seconds:
            requested = min(requested, args.seconds)
        if abs(requested - manifest["recording_duration_s"]) > 1e-7:
            raise ValueError("Resume duration disagrees with collection")
    else:
        args.work.mkdir()
        command = [
            str(args.processor.resolve()),
            str(args.input),
            "--output",
            str(args.work / "pipeline"),
            "--integration-seconds",
            str(args.integration),
            "--update-seconds",
            str(1 / args.fps),
            "--range-metres",
            str(args.range_km * 1000),
            "--doppler-hz",
            str(args.doppler_hz),
            "--start",
            str(args.start),
            "--seconds",
            str(args.seconds),
        ]
        if not args.remove_dc:
            command.append("--no-remove-dc")
        manifest = recording.collect(args, signals, args.work, command, duration)
        manifest["source_kind"] = "live_profiles"
        manifest["weights"] = (
            "Original archived carrier whitening and inverse-noise slow-time information"
        )
        (args.work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.collect_only:
        return
    if args.encode_workers == 1:
        for state in manifest["signals"]:
            recording.encode(args, manifest, state)
    else:
        with ProcessPoolExecutor(max_workers=args.encode_workers) as pool:
            jobs = [
                pool.submit(recording.encode, args, manifest, state)
                for state in manifest["signals"]
            ]
            for job in jobs:
                job.result()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, struct.error, KeyboardInterrupt) as error:
        import sys

        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
