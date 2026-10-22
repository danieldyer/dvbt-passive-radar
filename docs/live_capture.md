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
For a fixture whose modulator and rational-sampling states are verified cyclic,
`--simulate-cyclic` counts file wraps without resetting acquisition or coherence.
Ordinary recordings must retain the default seam reset. The support fixture
generator verifies equality of successive encoded superframes before enabling
cyclic synthesis; see [live support](../support/live/README.md). A continuous
fixture is important for throughput tests: frequent seam resets otherwise reduce
the live-map workload by repeatedly emptying the coherent window.

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
  "buffer_seconds": 4
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
it counts every discarded sample rather than blocking USB reception.
Host-discarded samples never reach channelisation and are absent from both IQ
and profile archives; their discontinuities remain explicit.
Capture queue pressure and the bounded archive queue are visible in the TUI.
Consumed input is released as soon as channelisation finishes, before downstream
decoder, archive or display work. The default four-second ring rounds up to
1 GiB (4.37 seconds at 61.44 MS/s); an explicit `buffer_seconds` in 0.1–8 seconds overrides it.
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
coherent detection histories cannot bridge the loss.
Synthetic warm-up runs through acquisition, decoding and actual range compression
before UHD starts. All four range-compression FFT
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
| acquisition estimate / recent update | Per-channel rate after undoing applied retiming, and whether observations are recent; reference membership remains fixed |
| innovation RMS / NIS | Timing innovation and its squared value divided by conditional prediction variance; NIS should average near one under a calibrated Gaussian model |
| bounded / late observations | Innovations receiving bounded influence; observations arriving before an already assimilated epoch |
| predicted clock sigma | Common-clock posterior/forecast uncertainty, conditional on the declared noise model; not a measured absolute timing error |
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
Recent headroom is computed from signal/DSP-time differences between status
updates; the minimum is tracked after the first five signal seconds. The longest
processing block and peak ring backlog expose stalls hidden by an average speed.
RF health reports complex RMS (`E[|I+jQ|²]`, with unit-amplitude complex tone at
0 dBFS), maximum component amplitude, and I/Q-pair counts at the digital rails
(`max(|I|,|Q|) >= 32752/32768`) and near rails (`>=0.98`). Reduction is fused
into the GPU unpack pass. Padding and overlapping filter halos are excluded.
These statistics cover processed ADC samples and cannot certify analogue-chain
linearity or diagnose overload before the ADC. Pilot SNR is also not MER.

## Shared clock controller

The live controller uses the covariance-driven `ClockEstimator`, with common
offset/rate/drift and zero-sum transmitter offset/rate contrasts over a **fixed
declared ensemble**. Losing a band does not re-average the time reference. A
reacquired pilot anchor changes its nuisance offset, not the ADC phase origin.
Applied segments remain immutable and meet continuously. Channel-major pilot
observations are ordered by their actual epochs before assimilation. Late
observations are counted and logged, never applied retroactively.

Pilot offsets use their conditional likelihood covariance. Differential rate
observations supply acquisition only: adjacent rate differences are correlated
and are not counted as independent extra measurements. A six-sigma Gaussian
Huber influence bound limits finite-covariance outliers; the original innovation
and applied robust weight remain in the log. Hard deletion can permanently lock
a filter out after a mismodelled acquisition transient. The bounded update
allows recovery without an epoch reset; see
[Observation Quality Control with a Robust Ensemble Kalman Filter](https://journals.ametsoc.org/view/journals/mwre/141/12/mwr-d-13-00091.1.xml).
A fixed 0.25-second first-order phase servo follows the
estimated common rate without the old channel-dependent prediction horizons.

`capture.clock_model` optionally overrides these design priors:

```json
"clock_model": {
  "receiver_white_fm": 1e-10,
  "receiver_random_walk_fm": 1e-10,
  "receiver_drift_diffusion": 1e-22,
  "transmitter_white_fm": 1e-11,
  "transmitter_random_walk_fm": 1e-11,
  "settling_seconds": 0.25,
  "innovation_huber_sigma": 6,
  "unknown_sigma_samples": 0.01
}
```

White/random-walk values are fractional-frequency Allan deviations at one
second: `AVAR(tau) = white_fm²/tau + random_walk_fm²*tau`, excluding deterministic
drift. Drift diffusion has units s⁻³. Unknown timing covariance uses the declared
ADC-sample sigma; no calibrated robust threshold is claimed for those observations.
See [NIST SP 1065](https://www.nist.gov/publications/handbook-frequency-stability-analysis).
The defaults are **uncalibrated design priors**, not a measured B200mini TCXO
specification. The estimator attains its conditional model bound in the stated
tests when the robust bound is inactive; this does not establish unconditional CRLB efficiency in unknown
interference/propagation. Inspect innovation variance and temporal correlation
on a real capture before treating posterior uncertainty as calibrated. When the
Huber bound is active, reported covariance is nominal bookkeeping for the bounded
gain, not an unconditional confidence interval or a robust CRLB certificate.

## What is saved

`metadata.json` retains the request; `summary.json` records applied values,
capture errors and channel results. `capture.dvblive` embeds the request, applied
settings, physical RF/NCO declarations and the corrected/source coordinate rates.
Its `EPOC` record retains the first device tick. `UTCA` records bracket UHD
device-time queries against host UTC and monotonic nanoseconds. The shortest of
eight queries is retained before and after reception; query half-width and any
host-clock step are explicit. This bounds query correspondence, **not the host's
absolute UTC accuracy**. Simulation has no hardware UTC anchor. The sidecar
`diagnostics.jsonl` is written even with `--archive none`.

| CLI mode | Scientific payload | Approximate steady data rate for five channels |
| --- | --- | --- |
| `--archive profiles` (default) | Complex pre-CPI range profiles and first time moments, weights, validity and clock/phase provenance | 300 MB/s plus metadata |
| `--archive channels` | Filtered, commonly resampled cf32 I/Q at 8 MS/s/channel | 320 MB/s plus metadata |
| `--archive both` | Both channelised I/Q and complex range profiles for every configured channel | 620 MB/s plus metadata |
| `--archive none` | Metadata and final summary only | Small |

To save both, add `--archive both`:

```sh
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --output captures/session-001 --archive both
```

Both payloads share one `capture.dvblive` file and common clock history. The
profile analysis and video tools also accept this combined archive.

Profiles retain per-symbol complex phase and range history for compatible motion
estimation, autofocus and migration correction. At the default 100 km extent,
their 4,096 slots retain lags −64 through +4,000 at 37.474 m spacing, including
context for motion compensation; the intervening unused slots are zero.
`PROF.size`, `positive_context_samples`, `negative_context_samples` and
`range_step_m` describe this layout. `--range-metres 30000` selects the older
1,024-slot layout (lags −64 through +928, about 75 MB/s for five channels).
Existing 1,024-slot archives retain their original range support.
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
Live profiles precede stationary subtraction and begin with the first eligible
decoded batch, including the interval formerly reserved for stationary training.
The first profile needs a preceding reconstructed symbol; pre-lock acquisition
samples are available only in `channels`. Invalid rows and discontinuities stay
explicit. Display DC suppression does not change profile samples, admission,
weights or phase. No raw wideband ADC I/Q is written in any mode.

The archive is little-endian: `DVBTLIV1`, a uint32 JSON-header length, then header
JSON. Each record is a four-byte tag, uint32 JSON length, uint64 payload byte
length, metadata JSON and payload. `CLCK` publishes new immutable clock segments;
`GAPS` identifies capture discontinuities; `PROF` stores cf32
`[moment=0,1][row][delay]` followed by an optional stationary-response plane,
the fitted complex response, float32 carrier-noise variance, and optional float64
frozen-cancellation coefficient variances in older recordings (`cancellation_variance_carriers`);
`CHIQ` stores cf32 `[channel][sample]`; `DONE` closes a completed archive.
Profile rows retain channel sample, corrected seconds, original ADC sample,
pilot correction phase, CFO, validity, source offset and source rate error.
The last two describe the transmitter on the common receive grid, not a separate
receive resampling. Older six-field rows imply zero source offset/rate.
For older stationary-subtracted profiles, frozen-coefficient errors are correlated across future rows: the covariance is
`diag(D) + A diag(Vbeta) A^H`, conditional on the fitted response and independent
training innovations. They are not added to `D` as white noise. The saved
variances do not include hardware-model bias or unknown interference covariance.
Decoder segments are per-channel lock
episodes; their numbers need not equal capture-segment numbers.

[`support/live/read_archive.py`](../support/live/read_archive.py) is a streaming
reader and integrity checker. An absent `DONE` marks an interrupted archive.

## Optional native ImPlot display

Select a channel in the terminal and press `v`. A native GPU-rendered ImPlot
window opens; the terminal remains the main interface. Default settings are a three-second CPI, ±100 Hz, 0–100 km excess range and a
0.25-second map update. Change them with `--integration`, `--doppler`, `--range-metres` and `--update`.
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
New optional map jobs also yield when receive backlog exceeds one quarter of the
ring capacity. These skipped admissions are counted; scientific profiles and the
next full coherent aperture are unchanged.

The physical-power colour floor is fixed from the first preview median minus
1.5 dB per channel. The upper end only expands to retain new maxima. Adjust the
floor/peak controls or explicitly fit them again. There is no background grid.
`--remove-dc true|false` controls the live map's information-weighted constant
projection. The archived profiles retain that component. The native `Hide DC band` checkbox hides only display pixels within
three Doppler resolution cells of DC. Display holding and colour changes do not alter scientific data.
No display is saved and no HTTP server or browser is started.

First maps require acquisition, decoding and a full CPI.
[TUI and native-display validation](live_tui_validation_20261011.md)
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

## Persistent diagnostics and capture preflight

`diagnostics.jsonl` is a bounded asynchronous stream containing the complete
requested metadata and applied processing settings, raw rate/offset observations,
conditional variances, feedback-removed offsets, applied retiming, innovation
statistics/robust weights, per-symbol applied phase/CFO and source timing, FEC
validity, cancellation fit error, receive/archive queues, RF health and UTC
anchors. Unknown/nonfinite uncertainties are JSON null. A final `complete` record
distinguishes clean termination from an interrupted log. Writer failures propagate
to capture rather than silently losing provenance.

```sh
python3 support/live/analyse_diagnostics.py captures/session-001/diagnostics.jsonl \
  --output captures/session-001/health.json
```

An optional `--plot health.png` uses Matplotlib. NIS and lag-one correlation are
model diagnostics, not a certificate of achieved radar sensitivity.

Before a long recording, exercise the **same range, archive mode and map load**:

```sh
cmake --build --preset release --target capture_preflight -j 6
python3 support/live/preflight.py --metadata recordings/live_mt_dandenong.json \
  --seconds 60 --range-metres 100000 --archive both --map-channels 1 \
  --output build/preflight-new
```

The support executable uses the production receiver, clocks, archive writer and
full GPU map formation. It substitutes a checksum for native-window painting.
One map channel matches the current native display. `--map-channels 5` is a
heavier stress test, not the normal display workload.
The report requires actual map publications on every selected channel; requesting
maps alone is not a passing test. Temporary scientific payloads are removed after
integrity checks; status/phase diagnostics and the small report remain. Add
`--keep-data` to retain the payload or `--simulate FILE.wav` for a transport test.
Inspect average/recent headroom, maximum block time, peak queue occupancy and
loss counters together. A 30 km profile uses about 75 MB/s across five channels;
100 km uses about 300 MB/s, before channel-IQ payloads. Use
`--range-metres 30000` when that is the required scientific support.

Gain remains fixed for a capture. Compare short preflights at nearby gain settings
with `--gain-db 28`, `--gain-db 31`, etc., checking digital headroom, pilot quality and FEC;
avoid treating a strong direct-path signal alone as evidence of RF linearity.
The override uses a temporary metadata copy and retains the requested/applied gain
in the report; the original configuration is unchanged. Synthetic replay cannot
validate RF gain selection. Preflight sidecars are compressed as
`diagnostics.jsonl.gz`; the analyser accepts either compressed or plain JSONL.

## Capture changes validated on 19 October 2026

The release build passed all 15 CPU tests, six relevant Vulkan tests with validation
layers enabled, and three Python support tests. In addition to the component tests,
the complete waveform test injects a 5 ppm sample-clock error and checks the phase
against an independent DFT reference. The resulting timing-induced phase RMS was
0.146°; the recovered rate was 5.00142 ppm. A separate two-transmitter moving-echo
test measured 0.065° and 0.085° phase RMS, including an echo beyond the guard
interval. These are results for the declared synthetic cases, not hardware
oscillator calibration or bounds for arbitrary interference.

The controller also passed missing-band/reacquisition, changing observation
precision, an isolated timing fault and stale-observation tests. Changing the
assumed white-frequency-noise power by 100× either way gave 0.00291 and 0.00325
ADC-sample RMS one-second timing error, below the 0.00959-sample three-degree
budget in that test. A hard innovation-rejection gate initially failed waveform
acquisition by locking out subsequent observations; the production controller
instead bounds their influence continuously. No existing phase limits were relaxed.

Archive regressions prove bit-identical scientific profiles, weights and admission
with display DC removal enabled or disabled, including the first eligible batch.
The unpack/health reduction preserves exact sample conversion; its multichannel
phase test measured a maximum error of 0.0000141°. Injecting a 1,048,576-sample gap
reported exactly that loss, with no host discards, and all five receivers reacquired
without RS failures. Unknown timing precision is logged without entering the clock
estimator. The diagnostic writer's FIFO ordering, completion and exact integer
sample/UTC epochs are covered by CPU tests.

The following throughput measurements used the production pipeline and actual
archive writes on this M5. The paced runs used a verified cyclic five-channel
61.44 MS/s fixture, one continuous capture segment, eight file wraps without
reacquisition, a three-second CPI, ±100 Hz, 100 km profiles and a 0.25 s map cadence.
Native-window painting is excluded. Speed includes processing and archive drain;
CPU/GPU contention and filesystem caching can change these numbers.

| Workload | Speed / realtime | Peak receive backlog | Result |
| --- | ---: | ---: | --- |
| IQ archive, maps/profiles off, unpaced six-second before/after comparison | 1.966 → 1.975× | — | No material measured overhead from the added telemetry/controller |
| Both archives, one map, continuous 60 s | 1.313× | 0.194 s | 193 complete maps; no losses, reacquisitions or RS failures |
| Both archives, five maps, continuous 60 s | 0.991× | 1.267 s | 409 complete maps; no losses or RS failures; 556 optional map admissions skipped |

The five-map stress test **does not pass the realtime-margin check**. It demonstrates
display backpressure, not sufficient capacity for five maps at the requested
cadence. The current native UI computes one selected channel. Use a slower refresh
when increasing map load, and repeat the preflight with the intended settings.
The preflight returns a failing exit status if maps were requested but none were
published, including captures shorter than the full aperture.

Detailed evidence is retained in
[`build/capture-validation-20261019`](../build/capture-validation-20261019/), including
the [one-map report](../build/capture-validation-20261019/continuous-6144-100km-one-map/report.json),
[capture-health plot](../build/capture-validation-20261019/continuous-6144-100km-one-map/health.png),
[five-map stress report](../build/capture-validation-20261019/continuous-6144-100km-five-maps/report.json)
and [gap report](../build/capture-validation-20261019/final-6144-gap/report.json).
Large synthetic IQ/profile payloads are discarded after validation.

No UHD device was available for this validation. USB transport under load, actual
TCXO noise parameters, RF gain/linearity and the absolute accuracy of the host UTC
clock still require a hardware capture. The configured oscillator noise is an
explicit design prior; the recorded covariance is conditional on it and is not
an unconditional CRLB or a proof of end-to-end radar sensitivity.
