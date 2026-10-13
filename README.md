# DVB-T passive radar

C++23 recorded-IQ and direct UHD capture with Vulkan channelization, shared clock resampling,
CPU FEC and residual cross-spectrum kernels. The file
pipeline decodes and reconstructs non-hierarchical **8K, 7 MHz DVB-T**: QPSK,
16-QAM, 64-QAM, all five code rates and all four guard intervals. TPS determines
the constellation and code rate; select the guard interval for acquisition.

The supplied 553 MHz recording contains **C31 at 557.625 MHz**, with BCH-validated
8K/QPSK/3/4/GI 1/16 and CRC-validated C31 service metadata. The nominal channel's
125 kHz frequency offset matters: the NCO offset is **4.625 MHz**. The recording is
float32 stereo I,Q at 56 MS/s. Its overflowing RIFF sizes are recognised using
both exact modulo-2^32 size checks; processing uses bounded memory.

```text
WAV I/Q + original sample index
  → shared forward FFT
  → symmetric FIR / common-clock retiming / channel IFFT / decimate / phase-correct NCO
  → N coherent channel streams, each with exported carrier correction
  → common FFT, then tee
       decoder: equalize (GPU) → hard QAM → deinterleave → Viterbi → errors-only RS (CPU)
                → RS / outer-interleaver reconstruction → convolutional re-encode
                → inner interleavers → QAM + known pilots + BCH-validated TPS
       measurement: keep un-equalized received carriers
  → minimum-phase response fit AFTER resampling
  → E = Y − Hmin X; residual cross spectrum = E conj(X)
  → complex SAR handoff with reference, validity, physical RF and clock/phase coordinates

Optional legacy visualization consumer:
     range IFFT → symmetric slow-time filter / decimate
  → coherence-limited Hann integration / Doppler FFT
  → native complex map stream → separate ImGui/WebAssembly viewer

Production matched detection consumer:
     previous-batch carrier variance → finite waveform CAF with two Doppler moments
  → information-weighted constant-path-rate focus
  → complex detection map → separate support/video exporter
```

The channelizer and fractional resampler retain the occupied carriers with measured
errors below 1e-5. The FIR is applied once with its group delay removed in the
sample coordinates; this is not forward/backward filtering. See
[signal, timing and output conventions](docs/reference_reconstruction.md) and
[validation](docs/validation.md).

The [fixed-CPI sensitivity audit](docs/sensitivity_audit_20261006.md) measures
direct-path cancellation and examines waveform, window and Doppler covariance
bounds. Its optional tools live in [support/validation](support/validation/README.md).
The [implemented improvements and qualified bounds](docs/sensitivity_improvements_20261007.md)
include corrected noise weighting, finite-waveform correlation, stationary
prediction and constant-path-rate focusing. The new `dvbt_detect` consumer
defaults to a three-second CPI with inverse-noise weighting, ±200 Hz and 0–30 km. It supports
multiple channels on the shared clock and retains complex phase coordinates.

```sh
build/release/dvbt_detect /path/to/C31-recording.wav --output build/detections \
  --recording-rf-hz 553000000 --channel 557625000,4625000,16
```

This runs unpaced and outputs physical complex amplitudes. `--keep-dc` retains
stationary returns; `--physical-units` remains an accepted compatibility flag.
Output includes physical-unit, weighting-method and explicit phase-epoch metadata;
these do not reverse discarded observations or make a compact map a sufficient
statistic for every later scene model. Frozen stationary training precedes the
first complete map and does not change the spectral SAR handoff. See the
[information-preservation review](docs/information_preservation_review_20261011.md)
for measured numerical accuracy, cancellation limitations and archive boundaries.

Recording JSON can configure all channels and retain transmitter geometry:

```sh
build/release/dvbt_detect INPUT.wav --output build/vhf-maps \
  --metadata recordings/baseband_202000000Hz_20-55-20_10-10-2026.json
```

Both `dvbt_detect` and `dvbt_sar` accept `--metadata` and optional
`--select-channel nine`. Without `--metadata`, a JSON file with the WAV's basename
beside the WAV is loaded if present. See [recording metadata](docs/recording_metadata.md).
Stereo signed int16 and float32 are supported, including wrapped RIFF lengths.
The 61 MS/s VHF capture is converted to the common 8 MS/s channel grid inside
the existing spectral synthesis; every channel shares the nominal rate conversion
and subsequent timing correction. Metadata validates the WAV rate and format.

To export one fixed-scale QuickTime video per multiplex using a single shared
channel bank, see [support/video](support/video/README.md).

For live B200mini reception, the [UHD capture front end](docs/live_capture.md)
uses mapped int16 receive storage, reports timestamped losses and shows per-channel
timing/FEC telemetry in an interactive terminal UI. It saves complex range histories
or channelised I/Q. Press `1`–`5` to select a channel and `v` for its native ImPlot map:

```sh
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --output captures/session-001
```

[Current live performance and gap validation](docs/performance_20261013.md)
records the measured optimisations, complete warmup and bounded capture/display
backpressure.

The main interface stays in the terminal; `t` shows timing plots, `?` shows help,
`s` drains and retains results, and `q` exits. Default detection settings are 3 s,
±100 Hz and 0–30 km. [Simulation launcher](support/live/simulate.command) loops
until stopped. No raw ADC I/Q or display images are saved.

## Build

Requires CMake 3.28+, a C++23 compiler, Vulkan 1.2 and `glslc`, glslang, FFTW3f,
VOLK, Boost and GMP. This machine already has those CPU dependencies under
`/opt/homebrew`. The bootstrap builds FFTW 3.3.11 with SIMD enabled, GNU Radio
3.10.12 and spdlog into `build/deps` without
installing GUI applications or changing system packages:

```sh
/opt/homebrew/bin/python3 scripts/bootstrap.py
source build/sdk/setup-env.sh  # or your installed Vulkan SDK's setup script
cmake --preset release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build --preset release -j 8
ctest --preset release
```

The bootstrap requires Python 3.12+; the processing executable is entirely C++.
If upgrading an existing build from system FFTW, configure once with
`cmake --preset release -URADAR_FFTW_LIBRARY` to refresh the cached library path.
GNU Radio supplies comparison implementations and DVB-T tables. The pinned
`gr-dl5eu` synchronizer is instrumented in the build tree by
[scripts/patch_sync.py](scripts/patch_sync.py). VkFFT 1.3.4 supplies channelizer
and symbol FFTs. The channelizer, common ADC resampler, OFDM numerical stages,
remodulation and residual computation use Vulkan by default. Hard QAM demapping,
deinterleaving and inner/outer FEC run on the CPU. Carrier buffers stay on device through minimum-phase
estimation and complex residual formation. Reduced tracking/TPS and fixed-origin
control remain on the CPU. `GpuPipeline` exposes the resident spectra before
the legacy output/visualization consumer downloads anything. `SarBatch` attaches
declared physical RF frequencies, sample coordinates and the exact common-clock
trajectory. Use `dvbt_sar` for the direct raw-IQ-to-SAR output path without map
or video processing; see [the handoff API, CLI and format](docs/sar_handoff.md).
The architecture and qualification evidence are tracked in
[the pipeline plan](docs/sar_pipeline_plan.md). `--compute cpu` selects the older
single-channel CPU comparison pipeline. Exact scalar kernel references are
retained in the tests. See [the Vulkan frontend report](docs/vulkan_frontend.md)
for filter requirements, phase conventions and kernel-only measurements.
FEC uses the CPU hard-decision Viterbi and errors-only RS decoder for both
compute modes. `--fec viterbi` is accepted for explicit selection; BCJR and the
GPU FEC decoder have been removed. The mapped-carrier handoff preserves the GPU
OFDM and reconstruction path. See [CPU FEC and measurements](docs/fec.md).

Complete frame batches now decode on bounded CPU workers while GPU reception
continues; clock feedback stays on the receiver path. Callbacks run on the caller
thread in per-channel order and may arrive during a later `process`, `drain` or
`finish` call. The frontend also accepts resident raw-IQ buffers through
`process_clocked_device`. See [scheduling, ownership and measured performance](docs/pipeline_concurrency_20261006.md)
for the exact contracts and the distinction between resampling cost, GPU spans,
active CPU time and end-to-end throughput.

Use `-DRADAR_BUILD_PIPELINE=OFF` to build only the original Vulkan/VkFFT smoke
executable. See [dependency provenance and licensing](THIRD_PARTY.md).

The opt-in component benchmark uses the production DSP implementations and is
excluded from normal builds and CTest. It reports repeated warm timings to CSV
and detailed decoder/channelizer stage timings to stderr:

```sh
cmake --build --preset release --target radar_benchmarks -j 8
build/release/radar_benchmarks --gpu --input /path/to/C31-recording.wav \
  > build/benchmarks.csv 2> build/benchmarks.log
```

The optional recording benchmark assumes 56 MS/s, a +4.625 MHz channel offset and
GI 1/16. Without `--input` it benchmarks synthetic data across all 15 supported
constellation/code-rate combinations. `--gpu` includes Vulkan transfer and
dispatch costs. Component timings are separate from end-to-end video throughput.
See the [current resampling optimization](docs/resampling_optimization_20261006.md),
[concurrent pipeline measurements](docs/pipeline_concurrency_20261006.md),
and [current CPU/GPU execution breakdown](docs/performance_breakdown_20261006.md)
for the Vulkan/CPU spectral pipeline, and the
[earlier performance measurements and assembly review](docs/performance.md)
for the historical CPU implementation.

## Process a recording

Start with a bounded interval. Output directories must be empty.

```sh
source build/sdk/setup-env.sh
export TMP=/private/tmp
export XDG_CACHE_HOME="$PWD/build/cache"
export XDG_CONFIG_HOME="$PWD/build/config"
build/release/dvbt_reconstruct \
  /Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav \
  --output build/example-reconstruction --seconds 60 \
  --offset-hz 4625000 --guard 16 --response-order 8
```

Omit `--seconds` (or set it to zero) to process the entire recording.
`--start SECONDS` seeks without losing the NCO's original phase origin. Default
output is the native complex range–Doppler map plus compact timing/FEC/coherence
diagnostics. `--raw-output` also saves intermediate carriers, masks, reconstructed
I/Q and every response fit (about 275 MB per recorded second). `--no-maps`
disables the range–Doppler stage for diagnostics. Input rates must be integer
multiples of 8 MS/s; non-finite samples and invalid formats are rejected.

Repeat `--offset-hz` to process several channels from the same capture. For
example, `--offset-hz -14000000 --offset-hz 0 --offset-hz 14000000` produces
`channel_00`, `channel_01`, and `channel_02` under the output directory. Use the
actual broadcast centre offsets. A common `sample_clock.csv` records the warp
for every channel; `--clock-phase-degrees 3` is the default RMS design budget
across their occupied frequency span. `--frontend cpu` is available for a
single-channel comparison while retaining Vulkan residual processing.

Default map limits are 0–30 km excess path length, ±100 Hz Doppler, a 300-second
coherent history and a two-second refresh interval. Automatic integration uses
all available history meeting the measured coherence threshold. Relevant options:

```sh
--range-metres 30000 --doppler-hz 100 --history-seconds 300 \
--integration-seconds 0 --map-interval 2 --coherence-degrees 5
```

Add `--remove-dc` to subtract the Hann-weighted complex mean at each range before
the Doppler FFT. This suppresses stationary returns and their window leakage;
returns close to zero Doppler are attenuated over the CPI's resolution width.

See [map coordinates, integration and stream format](docs/range_doppler.md).
FEC-invalid rows or synchronization gaps reset coherent history; they are never
silently removed from its time axis.

## Export a video from the command line

The export command runs the pipeline and encoder **without real-time pacing**.
Movie timing comes from recording sample timestamps, independently of how quickly
the processor runs. It consumes a lossless map stream, so an encoder that falls
behind applies backpressure rather than dropping scientific map updates.

```sh
python3 -m venv build/video-venv
build/video-venv/bin/python -m pip install -r support/video/requirements.txt
source build/sdk/setup-env.sh
build/video-venv/bin/python support/video/render_video.py \
  /Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav \
  --output renders/replay.mov --encoding prores --auto-scale
```

The optional exporter, its dependencies and its tests live in
[support/video](support/video/README.md). They are separate from the C++ pipeline
and its build. FFmpeg must be installed with its `libx264` encoder. The exporter
does not require a browser, ImGui or WebAssembly. It uses a bundled font when
available and otherwise a system/default font.

For QuickTime scrubbing on Apple Silicon, select `--encoding prores` and a `.mov`
output. This uses Apple's hardware ProRes 422 encoder with independent frames
and 10-bit 4:2:2 Rec.709. It requires FFmpeg's `prores_videotoolbox` encoder.
Both encodings put the container index at the beginning for immediate access.

The default result is a 1920×1080 H.264 MP4, a PNG preview and a JSON sidecar containing the
settings and every map's timing/CPI metadata. A sibling `.processing` directory
retains the decoder/encoder logs and diagnostics. The matched backend streams
maps without writing every large complex map to disk.
Existing outputs are not overwritten. Use `--seconds 12` for a short trial,
`--start 60` to select another interval. `--backend legacy` selects the earlier
processor; its `--compute cpu` option avoids Vulkan.
`--db-min -100 --db-max -40` sets the fixed relative-power colour scale.
With `--auto-scale`, the top follows each map's peak after optional DC removal,
and the difference between those limits sets the displayed dynamic range (60 dB
by default). The colour bar and JSON metadata report each frame's actual limits.
Neither the movie nor the viewer draws a grid over the data.

The production detector and current video previews use physical complex-map power.
Empirical per-map range/Doppler normalization is confined to historical diagnostic
replay under `support/validation`. See the
[scientific pipeline review](docs/pipeline_review_20261011.md) for assumptions,
phase regressions and real SDR validation.

The matched backend defaults to a full trailing three-second coherent window,
0–30 km, ±200 Hz and stationary suppression. The frame interval is a whole number
of OFDM symbols and at least 30 frames/s: 30.637 fps for the default guard 1/16.
Startup is labelled as acquisition until stationary training and a complete
CPI are available. `--no-remove-dc` retains stationary returns. The legacy backend
retains its Hann/coherence-limited behavior. Video time starts at the requested
recording time, includes acquisition and the final support tail, and is quantized
to one video frame. Changing `--fps` changes movie cadence, never processing speed
through an artificial pacing timer. Run `support/video/render_video.py --help` through
the same Python environment for all options.

## Browser viewer

Build the separate C++ ImGui/WebAssembly application and local server once:

```sh
/opt/homebrew/bin/python3 scripts/bootstrap_web.py
```

While the terminal processor runs, use another terminal:

```sh
build/web-venv/bin/python web/serve.py build/example-reconstruction
```

Open **http://127.0.0.1:8765/**. Drag to pan, wheel to zoom, or type range/Doppler
limits. Set dB limits, pause/resume the display, change integration time during
processing, or download the native complex map. The browser crops the native
Doppler bins on zoom; previews preserve narrow peaks when pooling pixels. After
the recording finishes the final map remains available.

**Remove DC / stationary returns** works during processing and on completed maps,
including when zoomed away from zero Doppler. **Auto colour scale** follows the
displayed peak after that subtraction. The viewer applies the mathematically
equivalent complex Hann projection before calculating power, leaving its source
map and download unchanged. If the processor already used `--remove-dc`, the
checkbox is checked and locked because the source map no longer contains DC.

The response estimator fits a short causal complex cepstrum to magnitude only,
then determines one constant phase. It has **no delay/all-pass fitting term**.
`--response-order 0` restricts it to scalar amplitude/phase. Minimum phase alone
cannot distinguish hardware from the minimum-phase part of multipath, and an LTI
response cannot model arbitrary nonlinear PA distortion. Use `--raw-output` to retain the pre-subtraction
spectrum and every fitted response for inspection.

No receiver latency calibration is needed for excess delay relative to the direct
signal in this same recording. The synchronizer establishes a fractional symbol
origin once per acquisition, then continuously resamples onto the recovered clock.
An explicit `--reference-delay-samples VALUE` overrides that automatic origin
adjustment; the value is additional to the documented FFT window margin. Absolute
UTC and a surveyed antenna phase centre are not established by this implementation.
