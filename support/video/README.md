# Video export

Saved live captures can be rendered without recovering raw I/Q:

```sh
cmake --build --preset release --target live_profiles_preview -j 6
build/web-venv/bin/python support/video/render_capture.py captures/session-001 \
  --output renders/session-001 --integration 3 --doppler-hz 100
```

This requires `--archive profiles` or `--archive both` at capture time. It reuses the production
Vulkan focus, range-migration correction, noise calibration and GPU preview.
The saved complex moments, common-clock coordinates and information weights
are preserved; they are not resampled or phase-corrected a second time. Only
full consecutive valid CPIs are admitted. Initial training and later FEC gaps
appear as unavailable intervals. Existing carrier whitening and stationary
subtraction are already part of the profiles and cannot be replaced exactly
by this exporter.

For new 100 km live captures, add `--range-km 100` to render the full saved
extent. The reader accepts both 4,096-slot profiles and older 1,024-slot profiles;
older captures cannot supply the newly added ranges.

Defaults are a three-second CPI, ±100 Hz, 0–30 km, a symbol-aligned rate above
30 fps, whole-recording fixed noise-floor/maximum colour bounds and hardware
ProRes for QuickTime seeking. It accepts the capture directory or its
`capture.dvblive`. `--collect-only` and `--resume` work as for recorded-I/Q
exports. Video sidecars retain the nominal CPI and the equivalent window
duration `1/enbw_hz`; unequal information weights can make the latter shorter.

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

For a metadata-defined multichannel recording, build the optional GPU display sink
and render every signal through one production channel bank:

```sh
cmake --build build/release --target dvbt_preview -j 8
build/video-venv/bin/python support/video/render_recording.py INPUT.wav \
  --metadata recordings/baseband_202000000Hz_20-55-20_10-10-2026.json \
  --output renders/mt_dandenong_vhf_20261010
```

This uses three-second coherent windows, ±100 Hz, 0–30 km and at least 30 fps,
with stationary suppression and hardware ProRes encoding. All channels share
the same ADC clock correction. The Vulkan preview kernel pools calibrated power
as the CPU display reference, checks finite samples, and preserves scientific
map/phase metadata. Display reduction is separate from the numerical pipeline
and does not change its maps.

The first pass caches only pooled display images in float16 dB. It measures the
whole-recording median displayed background, excluding the DC notch and ranges
below 0.6 km, and the true maximum for each channel. The lower bound is 1.5 dB
below that median and the upper is 0.25 dB above the maximum. Both stay fixed for
the entire channel's video. `processing/manifest.json` records the fit and source
identity. Float16 affects only this display cache, with errors below 0.016 dB
for typical peak values; scientific complex data is never quantized by the tool.

The cache needs substantial temporary disk space (roughly 2.3 MB per map/channel
at the default settings). `--cache-compression 1` trades extra CPU time for less
disk space; the default zero is fastest. Each channel's cache is removed once
its video finishes. `--collect-only` prepares caches and scales; `--resume` encodes
an already collected recording after checking source identity and settings.
Completed videos are not overwritten. This also accepts stereo int16 WAVs.

Build `dvbt_preview` for the default GPU display path. It borrows the ready
complex focus buffer, estimates exactly the same robust noise calibration, then
uses a Vulkan kernel to standardize and peak-pool power without copying or
serializing the full complex map. Only compact `RDP1` display pixels cross the
pipe. Scientific `RDM1` output remains available through `dvbt_detect`.
`--preview-backend cpu` uses `dvbt_detect` followed by `rdm_preview` as a reference.
The device buffer is borrowed only inside the focus worker, and all preview
framing and shaders remain in this support directory.

`--encode-workers 3` exports three independent channels concurrently; use one
for serial comparisons. ProRes uses Apple's hardware VideoToolbox encoder in
either mode. This does not change map cadence, coherent integration or the
whole-recording fixed colour bounds. A 31-map/s video requests substantially
more focusing work than the one-map/s production benchmark.

```sh
cmake --build build/release --target dvbt_preview preview_tests
build/release/preview_tests
```

The optional GPU test compares power and noise scaling against the CPU reference,
checks odd edge cells, unchanged scientific samples, reused buffers, nonfinite
input rejection, and device-versus-host focusing and calibration. Add `--benchmark`
to measure per-map GPU focus, CPU calibration and GPU preview costs.

For a complete recording comparison without encoding, build `dvbt_detect`,
`rdm_preview` and `dvbt_preview`, then run `benchmark_preview.py INPUT.wav
--metadata RECORDING.json --output build/preview-benchmark`. This includes
initialization and acquisition and consumes the preview pipe without writing
large map files. Use `--map-rate 1` to distinguish production publication cadence
from the default video cadence. Run without competing GPU jobs.

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
