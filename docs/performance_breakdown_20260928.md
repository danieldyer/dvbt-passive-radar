# Pipeline computational breakdown — 28 September 2026

This report describes the earlier CPU-heavy implementation. The current
raw-IQ-to-SAR measurements are in [the 30 September report](performance_breakdown_20260930.md).

These are fresh measurements of the current release implementation on this
Apple M5, rather than a restatement of the earlier optimization benchmarks.
The normal pipeline processes the first 60 recorded seconds in **27.26–27.62
seconds (2.17–2.20× real time)**. CPU work is concentrated in channelization,
sample-clock resampling, and DVB-T soft/FEC decoding. Reference remodulation is
comparatively inexpensive. High map/video cadence changes the bottleneck.

## Workload and method

* Apple M5: 4 performance and 6 efficiency CPU cores, 24 GiB RAM; macOS 26.4.
* AppleClang 21, Release `-O3 -DNDEBUG`, existing project-local NEON FFTW and
  MoltenVK. The production executable and signal-processing sources were not changed.
* First 60 seconds of `baseband_553000000Hz_12-51-54_26-09-2026.wav`:
  56 MS/s complex float32, channel offset +4.625 MHz, decimation by seven to
  8 MS/s, 8K DVB-T, QPSK, code rate 3/4, guard interval 1/16.
* Baseline map mode: 10-second coherent window, 0–30 km excess path,
  ±100 Hz, slow-time decimation four, DC removed, five requested maps per
  recorded second. Every scientific map is consumed; output is not paced.
* Baseline native runs write the latest map and stream every map to `/dev/null`.
  Video runs stream to the actual renderer/FFmpeg instead. Raw intermediate
  spectra and remodulated I/Q export are disabled.
* An isolated source copy under `build/perf-20260928/source` adds scoped wall
  and **thread CPU** clocks. Nested scopes subtract child times, so the CPU
  breakdown does not count the same operation twice. Timers group short
  Doppler FFTs in tiles of 32 to avoid a timer per tiny FFT.
* Primary stage figures average two instrumented 60-second runs. Their elapsed
  times were 28.32 and 28.72 seconds versus uninstrumented 27.62 and 27.26 seconds:
  roughly 4% additional elapsed time including instrumentation and run variability.
  The 54-map/s comparison was 34.32 seconds instrumented versus 34.24 seconds
  uninstrumented. Treat small differences as approximate, not optimization wins.
* Jobs ran serially. Normal filesystem caching and ordinary background desktop
  activity were retained; no cache flush, CPU pinning or exclusive-machine claim.
  The 60-second input contains 26.88 GB, larger than physical RAM.

## Additive CPU breakdown

CPU-seconds count work on all threads. They are **not elapsed seconds** and
cannot be added to predict wall time: the channelizer, synchronizer, processor,
and map worker overlap. The table includes input/output system CPU time where
it occurs within a measured scope, but excludes time asleep waiting for I/O,
GPU completion or another thread. GPU execution time itself is not CPU time.

| Stage | CPU seconds / 60 s input | CPU ms / recorded second | Share of CPU |
| --- | --- | --- | --- |
| Soft demapping, deinterleaving and FEC decoding | 10.74 | 179.0 | 21.5% |
| NCO and FFT channelization/decimation | 10.52 | 175.4 | 21.0% |
| Fractional sample-clock resampling | 6.43 | 107.1 | 12.9% |
| WAV input and validation | 4.09 | 68.1 | 8.2% |
| Carrier residual assembly and phase/coherence diagnostics | 2.97 | 49.5 | 5.9% |
| Map and diagnostic output | 2.86 | 47.7 | 5.7% |
| OFDM synchronization, FFT and equalization | 2.55 | 42.6 | 5.1% |
| DVB-T re-encoding and remodulation | 2.52 | 42.1 | 5.0% |
| Minimum-phase response fit and model preparation | 2.18 | 36.4 | 4.4% |
| Doppler maps, including DC projection/windowing | 2.04 | 33.9 | 4.1% |
| Range IFFT and slow-time filtering/history | 1.69 | 28.2 | 3.4% |
| Runtime, driver threads and instrumentation not assigned to scopes | 0.85 | 14.2 | 1.7% |
| Other instrumented management/setup | 0.35 | 5.8 | 0.7% |
| Vulkan residual/cross-spectrum host work | 0.22 | 3.6 | 0.4% |

Total measured process CPU time: **50.01 seconds per 60 recorded seconds**. The normal uninstrumented runs used 47.39 and 48.13 CPU-seconds, averaging **1.72–1.77 CPU cores** over elapsed time. This is not a ten-core-saturated workload.

## Granular components

The following rows subdivide the preceding totals; do not add both tables.
Names deliberately distinguish measured combined regions where finer timing
would distort short operations.

| Measured operation | CPU seconds / 60 s input | Share of process CPU |
| --- | --- | --- |
| 57,344-point channelizer forward FFTs | 7.525 | 15.05% |
| 96-tap fractional clock resampling, adjacent table phases interpolated | 6.427 | 12.85% |
| 64-state soft Viterbi trellis, including metric normalization | 5.725 | 11.45% |
| WAV seek/read and userspace/kernel copying | 3.322 | 6.64% |
| Carrier residual assembly, masking, phase/magnitude/coherence diagnostics | 2.967 | 5.93% |
| Serialize and write complex maps, both destinations | 2.581 | 5.16% |
| Inner interleaving, constellation mapping, pilot/TPS regeneration | 2.140 | 4.28% |
| Outer deinterleaving, RS decoding, parity verification and validity propagation | 2.105 | 4.21% |
| Soft QPSK metrics, carrier selection and inner deinterleaving | 1.852 | 3.70% |
| Minimum-phase FRF fit and iterations | 1.299 | 2.60% |
| Doppler weighted DC projection, Hann window, transpose/padding | 1.032 | 2.06% |
| Viterbi survivor traceback and decoded-byte assembly | 1.018 | 2.04% |
| Prototype-filter spectral multiplication and seven-way folding | 0.913 | 1.83% |
| Doppler FFTs, frequency selection and normalization | 0.909 | 1.82% |
| Absolute-phase NCO mixing at 56 MS/s | 0.888 | 1.78% |
| Reference phase alignment, model/measurement preparation, SI bookkeeping | 0.885 | 1.77% |
| 8,192-point channelizer inverse FFT, normalization and valid-output copy | 0.880 | 1.76% |
| 513-tap slow-time FIR, decimation and history bookkeeping | 0.823 | 1.65% |
| Fine carrier correction before the OFDM FFT | 0.790 | 1.58% |
| SIMD finite-I/Q validation | 0.764 | 1.53% |
| 8,192-point range IFFTs | 0.734 | 1.47% |
| 8,192-point OFDM demodulation FFT and FFT shift | 0.507 | 1.01% |
| Convolutional re-encoding, puncturing and validity propagation | 0.383 | 0.77% |
| Interpolate channel estimates for decoder equalization | 0.263 | 0.53% |
| Cyclic-prefix acquisition and tracking checks | 0.223 | 0.45% |
| Channel estimates at scattered pilots | 0.222 | 0.44% |
| Continual-pilot clock/carrier tracking | 0.122 | 0.24% |

The synchronizer's remaining equalization, tagging and bookkeeping costs about
0.43 CPU-seconds. Allocation, stream copies, CSV output, map publication and
all smaller categories are retained in the full
[stage CSV](../build/perf-20260928/stage-breakdown.csv).

"Remodulation" in normal operation means regenerating exact frequency-domain
carriers. It does **not** perform another waveform IFFT per symbol. The optional
`--raw-output` branch performs two additional IFFTs (reference and residual),
adds the reference CP, and writes substantial intermediate data. Those optional
operations are not included in the normal-workload figures.

FEC reconstruction processes 1,020-symbol batches while retaining 136 symbols
of context, giving 884 new symbols per full batch. Thus much of decoding and
reconstruction processes approximately 15.4% extra context in steady state.
This is a cost to understand, not permission to drop FEC/phase boundary checks.

## Elapsed time and worker balance

Across the two primary profiled runs, the source worker spent about 19.2–19.7 s
inside its work function, with 14.3–14.9 CPU-seconds. The synchronization worker
used 8.8–9.2 CPU-seconds. The reconstruction/output worker used 20.4–21.1
CPU-seconds and spent 20.7–21.4 s inside its work function. Map work runs on a
separate background worker. These inclusive worker totals overlap with the
component tables and must not be added to them.

The reconstruction/output worker is the largest serial CPU workload in this
configuration, while the source is also substantial. The actual elapsed time
exceeds either worker's active work because of dependencies, bounded buffers,
acquisition/setup, tail flushing, I/O and scheduling. Optimizing a single FFT
does not automatically improve whole-pipeline throughput by its CPU percentage.

WAV reads alone consumed about 8.1 s of scoped wall time versus 3.3 CPU-seconds.
The difference includes blocked/descheduled time; it is not a measurement of
pure SSD latency. GPU residual processing uses only about 0.22 CPU-seconds and
roughly 0.35 s of host-call wall time over the recording, including copies and
completion. The host timers do not isolate GPU shader execution.

## Map cadence and video export

All cases below process the same 60 seconds; native cases consume every map.
Video cases produce 1080p H.264 with autoscaling and DC removal. The two video
rows use the normal, uninstrumented production binary. CPU totals for video
include the Python renderer and FFmpeg plus the C++ processor.
The renderer uses NumPy/Pillow and the encoder is CPU `libx264`, preset `fast`,
CRF 18; this path does not use the M5's hardware video encoder.

| Workload | Elapsed seconds | Recording / elapsed | Total CPU-seconds |
| --- | --- | --- | --- |
| Native: 10 s CPI, ±100 Hz, 5 maps/s (run 1) | 27.62 | 2.17× | 47.39 |
| Native: 10 s CPI, ±100 Hz, 5 maps/s (run 2) | 27.26 | 2.20× | 48.13 |
| Native: 1 s CPI, ±300 Hz, ~54.07 maps/s | 34.24 | 1.75× | 62.70 |
| Video: 10 s CPI, ±100 Hz, 5 fps | 31.84 | 1.88× | 89.52 |
| Video: 1 s CPI, ±300 Hz, 54.066 fps | 76.51 | 0.78× | 381.91 |
| No maps: profiled scientific pipeline only | 27.22 | 2.20× | 43.41 |

In the profiled high-cadence native run, Doppler preparation/FFT/cropping used
about 6.6 CPU-seconds and complex-map serialization/writes used 9.46 CPU-seconds.
The processor waited 12.58 s for previous asynchronous map jobs, compared with
0.19–0.26 s in the 5-map/s runs. These waits overlap the worker work: they are
not an additional additive CPU cost. Disabling maps at low cadence reduced
CPU work but left throughput near the existing front-end/reconstruction limit.

The current code writes every map twice: once to `latest.rdm` using an atomic
replacement, and once to the requested lossless stream. This also happens for
offline video export, whose consumer already receives the stream.

* video-5fps-native: 296 scientific maps and 300 video frames; approximately **3.51 GB per map destination**, or 7.02 GB sent across both output paths, before JSON/header overhead. The file is repeatedly replaced rather than archived.
* video-54fps-native: 3,214 scientific maps and 3,244 video frames; approximately **12.88 GB per map destination**, or 25.77 GB sent across both output paths, before JSON/header overhead. The file is repeatedly replaced rather than archived.


For video, renderer/encoder work competes with DSP and backpressure propagates
up the pipeline. Subtracting the native elapsed time from video elapsed time
does not isolate a serial renderer cost. The earlier instrumented 54 fps video
run showed 43.8 s in processor waits for the map worker and 36.6 s in map writes
(including pipe backpressure), which explains why video pacing limits that run.

## Component cross-checks and optimization priorities

Fresh warmed production-component medians agree with the stage profile:

| Component | Median wall time |
| --- | ---: |
| Channelize one recorded second, including read/NCO/filter | 202.38 ms |
| Resample 65,536 samples with the 96-tap interpolator | 0.691 ms |
| Synchronize 1.3 seconds, including flowgraph setup | 245.46 ms |
| Reconstruct 1,020 real QPSK symbols | 194.49 ms |
| Produce a full ten-second Doppler map with DC removal | 4.934 ms |
| Vulkan residual batch, 6.95 million carriers, including transfers/wait | 5.713 ms |
| CPU residual batch | 8.359 ms |

For a 1,020-symbol real-recording batch the detailed component means were
27.64 ms demapping/deinterleaving, 100.64 ms Viterbi, 31.25 ms outer/RS work,
5.13 ms re-encoding, and 31.17 ms interleaving/remapping. These are separate
warmed experiments and are not additive predictions of pipeline elapsed time.

The fresh GPU FFT comparison still strongly favours batching: one 57,344-point
FFT takes 119 µs on CPU versus 468 µs for a Vulkan round trip, while a batch of
32 takes 4.05 ms CPU versus 2.30 ms Vulkan. A single 8,192-point FFT is about
12 µs CPU versus 642 µs Vulkan. These include host copies and GPU completion;
they do not benchmark a GPU-resident channelizer or establish the speedup of
such an implementation.

Based on these measurements, the next optimization candidates are:

1. **High-cadence output:** eliminate redundant latest-map disk publication when
   a lossless offline consumer is present, reduce avoidable map copies, and
   profile/optimize rendering and encoding separately. Preserve requested
   scientific updates; reducing their cadence would change the workload.
2. **Reconstruction/output worker balance:** investigate overlapping independent
   bounded reconstruction batches or separating FEC from downstream response,
   diagnostics and map ingestion, retaining exact ordering and boundary validity.
3. **Viterbi, channelizer and resampler:** these are the largest arithmetic
   components. Evaluate batched GPU-resident channelization against a complete
   CPU path, and further CPU/SIMD or parallel-batch decoding against its soft
   decision and overlap requirements. Per-symbol GPU dispatch is not supported
   by the benchmark as an optimization.
4. **Per-carrier diagnostics and data movement:** phase/magnitude diagnostics
   and residual/mask assembly cost more CPU than the whole remodulator. Any
   approximation or reduced diagnostic sampling would require renewed phase
   and coherence qualification; these measurements do not justify silently
   weakening those checks.

The minimum-phase fit and remodulator are not the leading costs. The GPU is
currently used for the residual/cross-spectrum arithmetic, a small part of the
overall workload; the 2× rate does not measure the M5's total compute capacity.

## Validation and reproduction

Both revised profiled 10-second-CPI runs have final complex maps and all map
metadata bit-for-bit identical to the uninstrumented baseline. The high-cadence
profile also matches its native comparison bit-for-bit. Every completed native
and video run has 55,076 fully valid symbols and zero RS failures. No phase
registration, rescaling or alignment was applied for these comparisons.

The full stage data, run totals and validation results are in:

* [stage-breakdown.csv](../build/perf-20260928/stage-breakdown.csv)
* [summary.json](../build/perf-20260928/summary.json)
* [component CSV](../build/perf-20260928/components.stdout) and
  [component stage log](../build/perf-20260928/components.log)
* [isolated instrumentation generator](../build/perf-20260928/instrument.py)
  and [serial benchmark runner](../build/perf-20260928/run_profiles.py)

Each run retains its exact command, `/usr/bin/time -l` output, logs, and output
metadata in the same folder. Files named `*-v2.profile.json` contain the final
lower-overhead instrumentation. The earlier `*.profile.json` files are retained
for comparison and are not the source of the primary CPU-percentage table.
Existing nonempty output directories are rejected; use new case/output names
when repeating the runs.
