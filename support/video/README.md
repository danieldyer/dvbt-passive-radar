# Video export

This optional tool turns the recording CLI's lossless `RDM1` map stream into an
unpaced 1920×1080 video. Rendering, encoding, Python dependencies and renderer tests
are contained here; the C++ pipeline has no dependency on them. The renderer
uses the computed complex maps and their sample timestamps without modifying
the scientific outputs.

Run from the repository root after building `dvbt_detect`:

```sh
python3 -m venv build/video-venv
build/video-venv/bin/python -m pip install -r support/video/requirements.txt
source build/sdk/setup-env.sh
build/video-venv/bin/python support/video/render_video.py INPUT.wav \
  --output renders/replay.mov --encoding prores --auto-scale
```

Use Python 3.11 or later (on this Mac, `/opt/homebrew/bin/python3`). An existing
environment with the listed dependencies can also be used.
FFmpeg with `libx264` is required. Existing outputs are not overwritten. The
export also saves a PNG preview, JSON settings and map timing, plus a sibling
`.processing` directory with pipeline diagnostics and both process logs. Use
`--seconds 12` for a trial or `--help` for all options. The command and processor
paths work from other directories too; supply absolute input and output paths.

Defaults use the production matched detector, a three-second coherent interval,
0–30 km excess path, ±200 Hz Doppler and stationary suppression. The frame
interval is a whole number of DVB-T symbols with at least 30 frames/s; for the
default 8K, 1/16 guard signal it is 30 symbols, or 3125/102 ≈ 30.637 frames/s.
Use `--fps` to override, `--integration` to change the coherent interval, or
`--no-remove-dc` to retain stationary returns. Processing runs at maximum speed.
`--backend legacy` selects the previous `dvbt_reconstruct` visualization for
comparisons. The matched backend streams maps without writing every map to disk.

For a fixed colour scale, omit `--auto-scale` and supply `--db-min` and
`--db-max`. These are power-dB limits, equivalent to `20*log10(magnitude)`.
The C31 noise-floor/maximum render uses `--db-min 1.2415573596954346
--db-max 42.077110290527344`. The lower limit is 1.5 dB below the median displayed
noise floor measured in validated maps near 60, 150 and 275 seconds. The upper
limit is 0.25 dB above the maximum of the previous complete recording render.
Those limits remain unchanged for every frame. The sidecar includes each map's
actual peak to check clipping. Fit provenance is saved in
`renders/C31_3s_fixed_noise_floor_max_scale_20261007.json`.

For forward and backward scrubbing in QuickTime on Apple Silicon, use
`--encoding prores --output renders/replay.mov`. This selects Apple's hardware
ProRes 422 encoder (`prores_videotoolbox`), 10-bit 4:2:2 Rec.709, independent
frames, and a MOV container with its index at the beginning. It requires FFmpeg
with that encoder and supported Apple hardware; software fallback is disabled.
ProRes files are larger than the default H.264 MP4. Changing the coherent
interval recomputes the maps from I/Q.
The JSON sidecar records the exact processor and encoder commands.
See the [Apple ProRes white paper](https://www.apple.com/final-cut-pro/docs/Apple_ProRes.pdf)
for the codec's independent-frame design and editing performance.

Run the stream, coordinate, WAV-duration and CLI checks with the same environment:

```sh
build/video-venv/bin/python -m unittest discover -s support/video -p 'test_*.py'
```

On macOS, check exact-frame seeking through AVFoundation, the system playback
framework, including distant random seeks and consecutive forward/reverse frames:

```sh
swiftc -module-cache-path build/cache/swift support/video/check_quicktime.swift -o build/check_quicktime
build/check_quicktime renders/replay.mov
```

This checks returned frame timestamps and reports seek latency. It does not
automate QuickTime Player's user interface.
