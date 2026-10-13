# Live UHD capture

`dvbt_capture` receives directly from UHD, feeds the existing common-clock Vulkan
channel bank and DVB-T reconstruction pipeline, and reports capture and per-channel
recovery status in an interactive FTXUI terminal application. No GNU Radio scheduler or SDR++
process sits between UHD and this pipeline.

```sh
# Use the Vulkan SDK environment from the project's normal build instructions.
export VK_ICD_FILENAMES="$PWD/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --output captures/session-001
```

The output directory must be new. Ctrl-C stops receiving, drains complete decoder
work and outstanding archive records, and restores the terminal. `s` stops and
retains the results in the TUI; `q` stops and exits. Use `--seconds 60` for a bounded
run. `--no-tui` or redirected input/output produces periodic text reports.
UHD settings are read back after configuration; the applied rate, frequency, gain,
bandwidth, antenna and clock sources are recorded. The fractional part of UHD's
rate is retained in the initial common clock trajectory, alongside its integer
sample-coordinate rate. Physical transmitter RF, channel NCO and receiver CFO
remain separate.

Build with Homebrew UHD installed:

```sh
cmake --preset release -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build --preset release --target dvbt_capture -j 6
```

`RADAR_BUILD_UHD=OFF` excludes this front end. `RADAR_BUILD_LIVE_DISPLAY=OFF`
builds the capture/TUI/archive executable without the optional display library.
`RADAR_BUILD_TUI=OFF` builds a headless front end. UI and simulation tooling is in
`support/live`; video tooling remains in `support/video`. The TUI uses FTXUI 7.0.3,
matching the RAD002A terminal interface. The optional macOS map uses GLFW (Homebrew
`glfw`), ImGui 1.91.9b, ImPlot 0.16 and Metal. Dependencies fetched by CMake are
pinned and hashed. Production DSP libraries do not depend on these UI libraries.

## Interactive controls and continuous simulation

The TUI has Overview, Timing, Events, Source and Help pages. Capture/DSP work,
including warm-up, runs on a worker; terminal and native-window events run on the
main thread. Telemetry updates at 4 Hz. Timing plots retain 512 updates per channel;
the event log retains 128 events. Resize the terminal, use the mouse or Tab to select
controls, and use PgUp/PgDn to scroll. An 80×24 terminal is supported; 120×38 shows
more per-channel columns and timing plots.

| Key | Action |
| --- | --- |
| `1`–`9`, Up/Down | Select channel |
| `o`, `t`, `e`, `i` | Overview, Timing, Events, Source |
| `?` / F1, Escape | Help, return to Overview |
| `[` / `]` | Previous / next page |
| `v` | Open or select the channel in native ImPlot |
| `p` | Hold/resume timing plots; acquisition continues |
| `s` | Stop acquisition, drain, leave results open |
| `q` / Ctrl-C | Stop, drain and exit |

Launch the looping demonstration by opening
[`support/live/simulate.command`](../support/live/simulate.command), or run:

```sh
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --simulate build/live-validation/five_channels.wav \
  --archive none --output captures/simulation-new
```

Simulation loops until stopped. At each file seam, source sample coordinates remain
monotonic, all receivers reacquire, and coherent histories reset; the seam is a known
synthetic boundary and is not reported as missing samples. `--simulate-once` ends at
EOF. `--unpaced` uses backpressure to run simulation at maximum processing speed.
Use `--archive none` for an indefinite demonstration to avoid an unbounded archive.

## Configuration

The existing version-1 recording metadata retains the signals, transmitter sites
and geometry. Add a `capture` object, as in
[`live_mt_dandenong.json`](../recordings/live_mt_dandenong.json):

```json
"capture": {
  "device_args": "type=b200,num_recv_frames=256",
  "channel": 0,
  "sample_rate_hz": 61440000,
  "centre_hz": 202000000,
  "gain_db": 30,
  "bandwidth_hz": 56000000,
  "antenna": "RX2",
  "clock_source": "internal",
  "time_source": "internal",
  "buffer_seconds": 2
}
```

Add `serial=...` to `device_args` when selecting among several devices. The signal
array determines channel order, physical centres and acquisition guard intervals.
TPS supplies the actual constellation and code rate. Offsets are adjusted for the
applied receiver centre; declared RF remains unchanged. The receive ring rounds
up to a power of two, so the TUI shows its actual capacity. It must fit the GPU's
storage-buffer limit.

## Data ownership and losses

UHD receives `sc16` directly into a persistently mapped, host-coherent Vulkan ring.
A GPU pass gathers the required centred FIR support, converts it exactly to cf32
and pads the existing FFT input. There is no application CPU copy or conversion
of the received waveform. This is application-level zero-copy; UHD still performs
USB transport/copy-conversion internally. The separate contiguous GPU FFT input
is needed for the overlapping transforms and common-clock interpolation.

One producer owns UHD's `recv()`. One DSP consumer owns the channeliser and core.
The ring has bounded storage and retains the next output's left halo until GPU
reads finish. A full ring makes the receive thread drain into a discard buffer;
it counts every discarded sample rather than blocking USB reception. Capture
queue pressure and the bounded archive queue are visible in the TUI.
Consumed input is released as soon as channelisation finishes, before downstream
decoder, archive or display work. The default two-second ring rounds up to
512 MiB (2.18 seconds at 61.44 MS/s); an explicit `buffer_seconds` overrides it.
This absorbs transient stalls, not a sustained processing deficit. The summary
records its actual capacity and peak occupancy.

Hardware timestamps label samples relative to the first successful receive. The
gap count comes from successive sample ticks, including partial receive blocks.
Overflow and sequence-error indications are counted independently. Errors without
a quantifiable timestamp jump remain explicitly unquantified. Missing or backwards
timestamps stop acquisition instead of inventing coherent coordinates. A timeout
is reported but is not itself assigned a lost-sample count.

Every discontinuity leaves a gap on the common corrected sample grid. NCO phase
does not restart, and all affected receivers reacquire. Decoder/reference context,
stationary training and coherent detection histories cannot bridge the loss.
Synthetic warm-up runs through acquisition, three seconds of stationary training
and actual range compression before UHD starts. All four range-compression FFT
tail shapes are prepared at construction, including those needed after a gap.
Warm-up observations are cleared and never archived.

UHD documents [receive error semantics](https://files.ettus.com/manual_archive/v4.9.0.0/html/classuhd_1_1rx__streamer.html)
and the [internal copy-conversion](https://github.com/EttusResearch/uhd/blob/master/host/lib/transport/super_recv_packet_handler.hpp).
SDR++'s inspected USRP loop receives fc32 but does not inspect these metadata
errors; this front end checks them explicitly.

## Terminal telemetry

The dashboard shows applied RF settings, sample counts, hardware/host losses,
unquantified gaps, overflows, sequence errors, timeouts, receive queue depth,
processing headroom, archive bytes/queue, and the shared clock trajectory/budget.
Each channel reports:

| Field | Meaning |
| --- | --- |
| SEARCH / ANCHOR / TRACK | OFDM acquisition, pilot anchoring, or established pilot timing; TPS/FEC can still be pending |
| CFO Hz | Receiver frequency correction, separate from the declared NCO and RF |
| drift ppm | Differential pilot rate error remaining on the corrected channel grid |
| timing / RMS ns | Anchored pilot phase-slope offset and its 0.25 s exponential RMS, not absolute propagation delay |
| phase RMS deg | 0.25 s exponential RMS of differential pilot phase innovations |
| ADC estimate / contributor | Per-channel estimate after undoing applied retiming, and whether it is recent enough for the ensemble |
| clock fit RMS / observations / span | Residual and support of the per-channel clock-offset fit |
| pilots / pilot SNR | Pilot-pattern correlation and an interpolation-residual SNR indicator; the latter includes channel modelling error and caps at 60 dB |
| QAM / code rate | Last BCH-valid TPS mode; pending is shown before the first decoded batch |
| RS pass / corrected / invalid | Evaluated RS-word results, including repeated decoder context |
| rows / valid rows | Unique emitted symbols and those with every oracle carrier valid |
| resets / maps / last map | Receiver reacquisitions and detection publication progress |

Processing headroom is processed input seconds divided by time spent in the DSP
path, including final drain/flush, excluding receive waits and pre-stream warm-up.
The FEC durations sum CPU-only work intervals across workers; these are worker
wall durations, not an operating-system CPU-utilisation percentage. `--profile`
enables device timestamp queries and reports GPU ms/input-s separately. GPU
timestamps are disabled in ordinary operation and are null in its summary.

## What is saved

`metadata.json` retains the request; `summary.json` records applied values,
capture errors and channel results. `capture.dvblive` embeds the request, applied
settings, physical RF/NCO declarations and the corrected/source coordinate rates.
Its `EPOC` record retains the first device tick. Internal device time is not a
calibrated UTC reference.

| CLI mode | Scientific payload | Approximate steady data rate for five channels |
| --- | --- | --- |
| `--archive profiles` (default) | Complex pre-CPI range profiles and first time moments, weights, validity and clock/phase provenance | 75 MB/s plus metadata |
| `--archive channels` | Filtered, commonly resampled cf32 I/Q at 8 MS/s/channel | 320 MB/s plus metadata |
| `--archive both` | Both channelised I/Q and complex range profiles for every configured channel | 395 MB/s plus metadata |
| `--archive none` | Metadata and final summary only | Small |

To save both, add `--archive both`:

```sh
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --output captures/session-001 --archive both
```

Both payloads share one `capture.dvblive` file and common clock history. The
profile analysis and video tools also accept this combined archive.

Profiles retain per-symbol complex phase and range history for compatible motion
estimation, autofocus and migration correction. Their 1,024 slots retain lags
−64 through +928 at 37.474 m spacing; the intervening unused slots are zero.
This precedes the final map's fourfold range interpolation. They use
the existing finite-waveform matched filter and two within-symbol Doppler moments;
they are not integrated maps or displayed powers. Noise weights, information,
stationary predictor, fitted complex response and original/corrected timing accompany them.

This compact representation supports the existing detector's signal model; its
statistical sufficiency for later SAR/ISAR has not been established. Received
spectra, source symbols and the complete template/noise Gram operator are not
stored. It is not a lossless substitute for later demodulation, re-estimation of
upstream whitening/cancellation or arbitrary intra-symbol motion. `channels`
retains the observations before those OFDM and range-compression choices, but
after common retiming and channelisation; it cannot restore discarded ADC bands.
With the default DC suppression, profiles start after frozen stationary training;
invalid rows and discontinuities stay explicit. `channels` includes acquisition
and training data. No raw wideband ADC I/Q is written in any mode.

The archive is little-endian: `DVBTLIV1`, a uint32 JSON-header length, then header
JSON. Each record is a four-byte tag, uint32 JSON length, uint64 payload byte
length, metadata JSON and payload. `CLCK` publishes new immutable clock segments;
`GAPS` identifies capture discontinuities; `PROF` stores cf32
`[moment=0,1][row][delay]` followed by an optional stationary-response plane,
the fitted complex response, float32 carrier-noise variance, and optional float64
frozen-cancellation coefficient variances (`cancellation_variance_carriers`);
`CHIQ` stores cf32 `[channel][sample]`; `DONE` closes a completed archive.
Profile rows retain channel sample, corrected seconds, original ADC sample,
pilot correction phase, CFO, validity, source offset and source rate error.
The last two describe the transmitter on the common receive grid, not a separate
receive resampling. Older six-field rows imply zero source offset/rate.
Frozen-coefficient errors are correlated across future rows: the covariance is
`diag(D) + A diag(Vbeta) A^H`, conditional on the fitted response and independent
training innovations. They are not added to `D` as white noise. The saved
variances do not include hardware-model bias or unknown interference covariance.
Decoder segments are per-channel lock
episodes; their numbers need not equal capture-segment numbers.

[`support/live/read_archive.py`](../support/live/read_archive.py) is a streaming
reader and integrity checker. An absent `DONE` marks an interrupted archive.

## Optional native ImPlot display

Select a channel in the terminal and press `v`. A native GPU-rendered ImPlot
window opens; the terminal remains the main interface. Default settings are a three-second CPI, ±100 Hz, 0–30 km and a
0.25-second map update. Change them with `--integration`, `--doppler` and `--update`.
The pipeline still consumes every received sample independently of map cadence.

GPU maximum-power pooling produces compact, owned previews. Metal draws an ImPlot
image texture instead of constructing millions of heatmap vertices. Pan and zoom
with the mouse; double-click to fit. The channel selector changes the viewed
multiplex. Close the window to hide it without stopping capture, then use `v` to
reopen. The display currently shows one selected channel at a time. Hidden channels
keep a bounded range history but skip Doppler focusing. All focus plans are warmed
before receiving, including plans for channels opened later.
If a display map is still in flight when another refresh is due, live capture
skips that refresh. Scientific profiles continue uninterrupted, and the next map
uses the same full coherent interval. Offline detection retains every requested
map. A slow display therefore cannot accumulate a queue of stale map jobs.

The physical-power colour floor is fixed from the first preview median minus
1.5 dB per channel. The upper end only expands to retain new maxima. Adjust the
floor/peak controls or explicitly fit them again. There is no background grid.
`--remove-dc true|false` controls scientific stationary subtraction and constant
projection. The native `Hide DC band` checkbox hides only display pixels within
three Doppler resolution cells of DC. Display holding and colour changes do not alter scientific data.
No display is saved and no HTTP server or browser is started.

First maps require acquisition, three seconds of held-out stationary training,
and a full CPI. [TUI and native-display validation](live_tui_validation_20261011.md)
includes screenshots. The [original capture report](live_capture_validation_20261011.md)
records transport/DSP validation from the preceding ANSI/browser prototype.


The live display and production detector use physical complex map amplitudes;
no empirical per-map range/Doppler normalizer runs. The preview pools physical
power on the GPU and applies colour limits only. Its labels distinguish the
fixed timestamp CPI from the equivalent weighted aperture and map age. Unknown
oracle rows retain their times with zero information; actual sample gaps reset
the history. The archive declares this erasure policy so old recordings retain
their original replay semantics.

See [the scientific pipeline review](pipeline_review_20261011.md) for covariance,
clock-reference assumptions, regression evidence and remaining sensitivity limits.
The subsequent [information-preservation audit](information_preservation_review_20261011.md)
quantifies the response solver's convergence limitation, retiming accuracy and
conditional compression losses, and distinguishes profiles from a sufficient
likelihood representation.
