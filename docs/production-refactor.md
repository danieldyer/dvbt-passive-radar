# Production consolidation and qualification

Qualified production snapshot, 1 November 2026, 06:00 AEDT
(31 October 19:00 UTC). Results below describe the retained code unless explicitly
labelled as experiments. The requested optimisation period is complete.

## Integration review, 1 November 2026

The subsequent review adds CMake install rules and an explicit switch for the
bootstrap dependency prefix. Driver-free builds retain the replay transport,
archive writer and their tests. The executable embeds its kernels; shared
libraries and a Vulkan ICD must be present at runtime. PL64 Nix packaging and service activation remain
in the separate system repository.

Failure handling now distinguishes draining queued records from successful
completion: archive destruction never writes DONE. A receive failure leaves the
archive incomplete and exits nonzero after preserving its diagnostic summary.
Metadata/summary close failures are checked. Signal-driven cancellation uses a
lock-free atomic flag across receive/processing threads. Detector and FEC resource settings
are validated before hardware access; extreme sample timestamps and empty GPU
dispatches are rejected before arithmetic can overflow or divide by zero.

Clock snapshots include the segment covering each batch plus subsequent changes,
avoiding repeated full-history copies and validation as capture duration grows.
The archive merges overlapping channel snapshots without dropping or duplicating
clock changes. The immutable controller trajectory itself is retained in memory;
this change does not claim constant memory for an indefinite capture.

Regression coverage includes failed-producer archive completion, long clock
histories and overlapping snapshots, pre-hardware CLI errors and timestamp range
limits. A redundant test labelled CPU continuity actually invoked the GPU branch;
it was removed because the normal CPU input test already covers that continuity.
Fresh review validation on macOS/M5:

- 47/47 CTest cases: 19 CPU and 28 GPU, including the long phase-history gate.
- 18/18 headless CPU cases under AddressSanitizer and UndefinedBehaviorSanitizer;
  the uSDR mock transport test also passes under both sanitizers against the local
  native DMA driver headers. This is not a live PCIe hardware test.
- All 18 Python unit tests and Ruff lint/format checks pass (100-column format).
- A fresh, offline, driver-free headless build installs and runs outside the
  repository. Its dynamic dependencies contain no GNU Radio/UHD/TUI runtime.
- The installed executable processes eight synthetic seconds on five channels,
  writing channel I/Q and profiles to 100 km with zero losses and zero rejected
  RS words. A second run preserves exactly 262144 injected missing samples,
  reacquires every channel and keeps the archive segmented.
- SIGTERM drains five-channel profiles and writes DONE with exit 0. Deliberately
  truncating the temporary replay input after startup preserves an error summary,
  exits 1 and leaves a readable partial archive without DONE.
- Four replays of the retained lower-three-channel real RF fixture exactly match
  the stored acceptance counts and all three packet checksums in `fixtures.json`.

Logs and generated measurements are under ignored `build/review-*` paths.
The PL64 host did not resolve at `pl64.local` during this review, so the target's
Nix build and live RF/thermal qualification have not been repeated for these
review changes. Earlier target measurements below remain historical evidence.

## Retained production scope

One capture pipeline supports UHD/B200mini and native DMA-loan uSDR, coherent
multichannel channelisation/timing, DVB-T decoding and waveform reconstruction,
reference cancellation, complex range profiles, bounded archives, the terminal
interface and native map display. Archive-to-video export remains in
`support/video`. Independent CPU numerical references live in `tests/reference`.
The FEC library accepts a compact likelihood span and rate with no GPU or modulation dependency.
Hard/uniform decoding and overlap caching are test-only. The CPU file channeliser
and response solver are test-only too. Live and validation streams use the same
common-clock spectral channeliser: the separate
unclocked fold/unpack implementation and shaders were removed. The scalar FIR
reference remains independent. Production pilot tables,
PRBS and TPS BCH are constexpr C++ definitions: capture no longer links GNU Radio.
A capture-only build succeeds with GNU Radio discovery explicitly disabled.

Removed SAR/ISAR/calibration experiments, web tooling, the scaffold/demo and
superseded reconstruction CLIs, duplicate spectral archives, legacy Hann map
processing, stationary-predictor production logic, decoder policy switches and
old parameter sweeps. Dormant stationary-predictor parameters, buffers and
training-statistic outputs were also removed from production range compression;
archive metadata retains zero-sized compatibility fields. Generated qualification
evidence is ignored under
`build/production-qualification`; prototypes remain outside the repository.
The Nix checkout and deployed navigation service have not been changed.

The decoder has one policy: adaptive pilot radii 4/8/16/32, full-GI frequency
smoothing, compact soft likelihoods, full-traceback Viterbi, RS-pinned feedback
with a maximum of 32 passes, and bounded RS rescue. Defaults are five CPU workers,
nine frame slots and independent 15-frame batches. Results publish in channel
order. There is no quality downgrade or signal-dependent CPU/GPU routing.

## Accuracy and execution time

The full retained suite passes **47/47 on M5 and 48/48 on PL64** (the latter
also includes native uSDR driver tests): multichannel clocks/phase, finite-waveform
correlation, known payloads, FEC references, archives and queue failures. The longer
five-band phase-history regression covers 4080 symbols and multiple decoder/response
updates, with zero failed synthetic payloads and less than 0.09° echo phase RMS on
each band. Python video tests pass 16/16 and diagnostic tests pass 2/2. Actual TUI
keyboard/resize/terminal restoration and native ImPlot switching/snapshot checks
also pass. The terminal validation tool now derives channel selections from the
metadata and supports text-only Linux snapshots; both five- and three-channel
Mac workflows pass.
Address/undefined-behavior sanitizer runs pass independent all-rate constrained
Viterbi and posterior/RS-rescue tests.

The exact FEC optimisation reuses an optimal subpath when newly accepted RS
constraints leave its two boundary states on the previous selected path. It
also seeds that certificate from the initial full-traceback solution. Likelihoods,
traceback, feedback limits and rescue are unchanged. Independent all-rate tests
cover ties, partial terminal bytes, compatible/new/revised/removed constraints
and fresh batches. A temporary real-RF differential build compared every feedback
pass against the previous decoder on identical inputs: all decoded bytes agreed.

Saved lower-three-channel replay, including reads, equivalent resident upstream
kernel work, end drains and actual 100 km profile writes:

| Host / version | Wall ms per input second | Initial Viterbi CPU ms/s | Feedback CPU ms/s | RS CPU ms/s |
| --- | ---: | ---: | ---: | ---: |
| M5, consolidated baseline | 273.16 | 294.50 | 335.75 | 43.36 |
| M5, final | 214.29 | 282.83 | 81.22 | 37.11 |
| PL64, warmed baseline ABBA | 798.98–806.91 | 1366.35–1409.92 | 837.10–843.75 | 111.27–112.55 |
| PL64, final hot replay | 669.01 | 1151.14 | 240.81 | 95.13 |
| PL64, final verification replay | 657.49 | 1146.62 | 222.76 | 93.06 |
| PL64, three copies of marginal Nine | 759.21 | 1290.41 | 836.58 | 259.73 |

CPU values are summed worker consumption, not elapsed time on one worker.
M5 runs use two warmup and four measured eight-second replays. The standard PL64
comparisons use twelve warmup and eight measured replays; the three-Nine stress
and final verification use six warmups and four measurements. Short files pay a
drain each repetition.
This is a coupled workload, not complete replay of acquisition/clock recovery.
Run-to-run power/scheduling variation is visible; the API cleanup is not claimed
as a separate speedup. Feedback CPU work fell about 76% on M5 and 73% on PL64;
whole replay throughput improved about 27% and 22% respectively. Some initial
Viterbi timing improvement can reflect the changed shared-package workload; it is
not attributed to a faster initial-trellis algorithm. The larger channel FFT is selected only on qualified Intel
hardware; M5 keeps its smaller tile. Both use the same synthesis shader and clock
math. Intel channelisation measured 78 rather than 102 GPU ms/input second.
The conservative hot PL64 replay is **1.495× realtime**, and the restored final
verification is **1.521×** with the exact expected per-channel packet checksums. A separate stress
run placing the same marginal Nine signal in every receiver at **1.317×**. The
three stress receivers each accept 471300/471360 words and have identical packet
checksums. That run uses six warmups and four measurements. It changes only the
benchmark input, not the decoder or pass limit. FEC work is bounded by the fixed
policy, but these empirical margins are not a hard realtime guarantee for every
possible RF impairment or competing system workload.

Final verification stage accounting, for all three channels together:

| Processor | Component | ms/input s |
| --- | --- | ---: |
| CPU workers, summed | Initial full-traceback Viterbi | 1146.62 |
| CPU workers, summed | Constrained feedback and rescue trellises | 222.76 |
| CPU workers, summed | RS, packet checks and constraint updates | 93.06 |
| CPU workers, summed | FEC metric/constraint preparation | 24.94 |
| CPU main thread | Capture/dispatch work | 192.41 |
| GPU | Common-clock channelisation | 78.19 |
| GPU | OFDM acquisition, timing and sparse pilots/TPS | 40.32 |
| GPU | Frame pilot filtering, equalisation and demapping | 68.72 |
| GPU | Accepted-waveform reconstruction | 52.85 |
| GPU | Reference-response estimation/cancellation | 28.96 |
| GPU | Conditional noise/information weights | 9.67 |
| GPU | Finite-waveform range compression and moments | 147.87 |

Total process CPU consumption is 1732.85 ms/input s; main and FEC entries are
attributed subsets, not an exhaustive sum. CPU worker consumption is aggregated
across five workers. GPU timestamps sum to 426.58 ms/input s. These quantities
**must not be added to wall time**: stages overlap, and the CPU and integrated GPU
share the PL64 package power budget. Measured wall time is 657.49 ms/input s,
including reads, repeated end drains and archive completion. Initial Viterbi
still dominates FEC CPU work, while GPU range compression is the largest GPU
stage. That mixed balance makes isolated decoder throughput an inadequate
criterion for another offload.

The current saved workload accepts Nine **117825/117840** RS words per replay
(99.98727%) on both hosts, preserving each host's accepted packet checksums.
Earlier PL64 comparisons showed a few words of upstream numerical variation;
exact FEC equivalence uses identical likelihoods, not independent RF runs.
Broadcast RS acceptance is not known-payload bit-error truth.

A full five-channel 64 MS/s synthetic live-engine run on M5, with 100 km profile
writes and the fixed production decoder, measured **3.538× DSP headroom**. All
five channels accepted 44193/44193 words with zero receive/host losses. This is
a short deterministic integration check, not marginal live-RF sensitivity.

The native OFDM handoff computes only sparse TPS acquisition statistics directly
from the FFT. It carries native FFT storage and explicit phase metadata to the
frame ring, where gather and phase restoration are fused. This removes a full
rough-equalisation copy and full-spectrum rotation pass before adaptive frame
equalisation. Pilot selection and sparse TPS combining also share one workgroup,
removing their intervening global dispatch/barrier. PL64 receiver GPU time fell
from 56.82 to about 40.25 ms/input s; the later fusion alone is not claimed as a
separate whole-pipeline speedup.
A callback can split command groups, so the frame gather owns its small phase
vector; it never borrows a host phase buffer that the producer may overwrite.
A deliberate buffer-poisoning regression covers that lifetime. The initial
prototype failed the original phase gate; the owned-metadata fix restores it
without relaxing any phase threshold.

Range compression shares its already weighted residual/reference product with
the zero-moment output. A profile-stage ABBA test saved about 4% on M5 and 5% on
PL64. Removing dormant stationary-predictor support keeps one scientific path.
The common-clock FIR's broad random-input scalar-reference error is 6.55e-6 on
M5, inside its declared -100 dBFS mask; tone error is 1.8e-7, blocker leakage
-105.5 dBFS and large-coordinate NCO phase error 3.73e-7 rad. The former
unclocked-only 2e-6 random-input test limit was replaced by the existing production
spectral/interpolation mask; no production filter or phase gate was changed.

## Live PL64 soak

The exact decoder and larger Intel channel FFT completed **1800 seconds** of live
lower-three-channel VHF acquisition at 32 MS/s, saving profiles only to 100 km:

- DSP headroom: **1.6458×**; zero SDR missing samples, host discards, unknown gaps
  or thermal-throttle counter increments. Peak receive queue was 0.382 s of 4.19 s.
- Median pilot SNR: Seven 23.62 dB, SBS 20.73 dB, Nine 20.53 dB.
- RS acceptance: Seven 99.9983%, SBS 99.7944%, Nine 99.4967%. Nine suffered an early
  RF fade; live propagation differs from the saved marginal fixture. This is not
  an algorithm A/B accuracy comparison.
- Package RAPL power: mean 15.07 W. RAPL domains overlap and are not wall power.
- Package temperature: median 61 °C, maximum 76 °C. The last five minute medians
  were 62, 62, 64, 64, 63 °C, with a descriptive slope about +0.40 °C/min.
  **Thermal equilibrium is not established**, although throughput remained stable.

The preceding 900 s run achieved 1.5163× with zero losses/throttle events, before
larger Intel tiles. An earlier preflight with much poorer SBS RF overloaded the
old decoder and host-discarded samples despite no SDR loss. No quality downgrade
was used. A separate queue defect limited a 4.19 s sample ring to about 2.1 s of
DMA descriptors. Adjacent descriptors now coalesce only within identical,
contiguous acquisition coordinates. Admission rejects metadata exhaustion before
advancing the accepted timeline. Full capacity and gap boundaries have tests.

Compact telemetry, status-only diagnostic extracts and plots are under
`build/production-qualification/qualified-soak/analysis/` and `soak/analysis/`.
The owned temporary 245 GB soak payload was deleted after evidence preservation.
Target governors, package power limits and navigation services were not altered.

## Forty-minute qualification

The subsequent native-FFT/shared-codebook/exact-RS-reuse build completed **2400 s**
of live lower-three-channel reception, saving only 100 km profiles with profiling
enabled. It sustained **1.6849× DSP headroom**, with zero SDR missing samples,
host discards, unknown discontinuities or thermal-throttle counter increments.
Peak receive backlog was 0.377 s in a 4.19 s buffer.

Median pilot SNR was Seven 23.65, SBS 20.61 and Nine 20.27 dB. RS acceptance was
99.9973%, 99.5900% and 98.2534% respectively; Nine's initial RF fade dominated its
cumulative rejects. These are live conditions, not a paired algorithm comparison.
The fixed saved Nine fixture remains 99.98727% accepted.

Package RAPL averaged 15.05 W. Package temperature was median 62 °C, maximum
78 °C. Final five minute medians were 65.5, 67, 66.5, 66 and 67.5 °C, a
**+0.30 °C/min** descriptive trend: thermal equilibrium still is not established.
Headroom changed only about -0.0022× over those five minutes. A later twenty-minute
run of the final policy was intentionally stopped for a
separate memory-allocation experiment; it closed cleanly with zero receive/host
loss. The allocation experiment showed no benefit and was reverted. A final
thirty-minute live run follows the restored-policy verification replay.

Compact evidence and plots: `build/production-qualification/final-soak/analysis/`.
The completed temporary profile payload was deleted after preserving status,
telemetry, summary and the tested binary hash. The navigation service, governors
and power limits were left unchanged.

## Final production build: thirty-minute live run

After reverting the unsuccessful allocation experiment, the exact restored
production policy completed **1799.992 s** of live lower-three-channel VHF
capture, saving only 100 km profiles. Measured DSP headroom was **1.69346×**;
SDR missing samples, host discards, unknown gaps and thermal-throttle increments
were all **zero**. Peak receive backlog was **0.372 s** in the 4.194 s ring.
The writer closed successfully with its DONE trailer, producing 245.05 GB.

| Channel | Median pilot SNR | RS acceptance |
| --- | ---: | ---: |
| Seven | 23.58 dB | 99.9977% |
| SBS | 20.83 dB | 99.7193% |
| Nine | 20.35 dB | 99.7835% |

These live acceptance rates describe changing RF, not a paired sensitivity test.
The fixed marginal Nine fixture still passes 99.98727% with matching packets.
Package RAPL averaged **15.00 W**; package temperature was median **65 °C**,
maximum **80 °C**. Final five minute medians were 68.5, 66, 65, 66 and 68 °C,
with a descriptive **−0.10 °C/min** slope. The last ten-minute slope is −0.08
°C/min, while the last fifteen-minute slope is still +0.12 °C/min. This is flatter
than the earlier 40-minute run, but does not establish full thermal equilibrium
or indefinite operation at another ambient temperature. Headroom changed by less
than 0.001× over that final five minutes. No governors, power limits or navigation
services were changed.

The supported live Linux TUI workflow also passes all keyboard/resize/stop/quit,
raw-terminal restoration and complete-archive checks for three RF channels.
It processed 60.04 s at **1.469×** with zero losses. That measurement included the
Python terminal emulator and concurrent completed-log extraction, so it does not
isolate the capture TUI's own cost. An additional raw-terminal measurement
processed 52.77 s at 1.643× with zero losses; its harness failed an assumed minimum
55 s duration because it counted shader/device startup before RUNNING. It is
a diagnostic timing observation, not another passed workflow gate.

Evidence: `build/production-qualification/production-soak/analysis/soak.png`,
`summary.json`, `temperature-tail-windows.json` and `ui/pl64/`. The compact logs,
source manifest and tested binary hash are retained locally. The three final temporary
profile payloads (260.34 GB) were deleted after those checks. No capture process
remains active; the private RF regression fixtures are retained.

## Renderer

GPU history computes each archived row's range FFT once and keeps a bounded
spectral ring. Sliding windows gather spectra on GPU. Planner changes keep the
large M5 transforms in one global pass; exact tail batching avoids empty FFTs.
Mean projection and pixel pooling use coalesced layouts. Scientific units,
information weights, gap policy and phase epochs are preserved.

For the same three-channel eight-second, 100 km, ±100 Hz, 3 s CPI fixture at
30.637 fps: map generation improved from **22.11 to 14.83 s**, with all 438 maps
retained and peak resident memory about **1133 to 368 MB**. Full hardware ProRes
export produced three 244-frame videos; parallel encoding took about 2.6 s after
collection. Tests verify complex cached-history agreement, analytic constant-rate
focusing and exact-frame forward/backward AVFoundation seeks. Both supported
codecs use VideoToolbox without software fallback. Colour limits remain fixed
per channel over the recording.

## Archive compatibility

See [archive format study](archive-format.md). The MCAP/FlatBuffers proposal is
compatible with navigation's encapsulation, with negligible file-size overhead.
On PL64, navigation-equivalent dual CRC costs about 117 extra CPU ms/input s with
MCAP's portable CRC helper. A byte-identical standard zlib-ng helper experiment
reduces that increment to about 35 CPU ms/s, retaining both integrity checks.
This excludes fsync and is measurable shared-power load, not proof of unchanged
coupled throughput. No second production archive
backend or Nix integration has been introduced. Acquisition time still requires
an uncertainty-qualified mapping into the navigation clock domain.

## Remaining performance limits

See [FEC design review](fec-design-review.md). Candidates are compared at whole
pipeline level as well as in microbenchmarks. Slower or inexact alternatives are
not retained. After exact feedback reuse, initial Viterbi dominates FEC CPU time;
GPU correlation/channelisation are now a larger share of live elapsed DSP work.
Further work must assess that changed balance rather than assume the old pacing
bottleneck remains unchanged.
