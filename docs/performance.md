# Performance pass — 2026-09-26

For the current complete Vulkan raw-IQ-to-SAR path, see
[the 30 September CPU/GPU execution breakdown](performance_breakdown_20260930.md).

The channelizer/backend descriptions below are historical. The current Vulkan
frontend and shared-clock kernel measurements are in
[the 29 September report](vulkan_frontend.md).

For the newer, granular CPU/wall-time breakdown and 5 versus 54 fps comparisons,
see [the 28 September profiling report](performance_breakdown_20260928.md).

Measured on the Apple M5 (10 CPU cores, 24 GiB RAM), macOS 26.4,
AppleClang 21, Release `-O3`, MoltenVK/Vulkan SDK 1.4.328.1. The input is the
301.84-second C31 recording at 56 MS/s. Results apply to this machine and
recording; synthetic 64-QAM timings do not qualify a real 64-QAM receiver.

## Bottleneck and retained changes

A 15-second sampling profile of the original native pipeline identified the
channelizer as the busiest worker. Its FFTW library had no ARM SIMD codelets:
`fftwf_version` was `fftw-3.3.11`, and the hot FFT assembly used scalar arithmetic.
The baseline native 60-second interval took 41.14 seconds, including map output.
Installing the same FFTW version with NEON into the project prefix alone reduced
that exploratory run to 29.56 seconds. No system library was replaced.

The retained changes are:

* Build pinned, SIMD-enabled FFTW locally, and link both the pipeline and GNU
  Radio's FFT support to it. Keep the existing estimate planner and bounded memory.
* Retain the original 8,192-sample inverse block for channelization (57,344-point
  forward FFT at 56 MS/s). A five-size component sweep favoured 2,048, but whole
  pipeline checks took 29.31 seconds with that candidate versus 28.17 and 28.78
  seconds with 8,192. The smaller-block change was therefore discarded.
* Give the WAV stream a 64 KiB buffer and reduce finite-sample checks with SIMD.
  NaN/infinity rejection, including vector tails and both I/Q lanes, is tested.
* Evaluate the two adjacent fractional-resampler phases together, sharing input
  loads. Retain all 96 taps and 2,048 phases with linear interpolation.
* Update paired Viterbi successors with NEON butterflies; swap metric buffers
  and increment puncture phase instead of copying metrics or dividing per bit.
  Keep floating-point soft decisions and the complete bounded-batch traceback.
* Re-encode bytes using two small linear-code lookup tables (640 bytes total).
  Preserve puncturing and invalid-byte propagation through encoder memory.
* Separate the Cartesian QAM max-log search into its two axes. Compare against
  a full search over GNU Radio's constellation, including decision boundaries.
* Accumulate the response fit by symbol/carrier, exposing independent SIMD
  accumulators while retaining the order of accumulation for each carrier.

There are no recording-specific correction branches for the closed 623 MHz
investigation. The minimum-phase model, processing order, filter bandwidths,
coherent coordinates, FEC gating and scientific output cadence are retained.

## Component measurements

The opt-in `radar_benchmarks` target calls production implementations, warms each
operation and reports median/minimum/maximum wall times. Most cases use seven
trials; decoder and recording cases use three. Stage details are arithmetic
means including warmup, except Vulkan residual stages, which are warmed medians.
Component times are not additive predictions of the threaded pipeline's wall time.

| Work | Original | Optimized |
| --- | ---: | ---: |
| Channelize one recorded second, including read/NCO/filter | 414.7 ms | 204.4 ms |
| WAV read and finite validation, 14.4 million complex samples | 20.5 ms | 9.8 ms |
| 96-tap resampling, 65,536 samples | 1.424 ms | 0.692 ms |
| Synchronize 1.3 seconds, including flowgraph setup | 352.1 ms | 244.3 ms |
| Reconstruct 1,020 real QPSK symbols | 254.5 ms | 193.8 ms |
| Reconstruct 136 synthetic 64-QAM, rate-3/4 symbols | 249.7 ms | 81.4 ms |
| Order-8 response fit, 1,020 × 6,817 carriers, 16 iterations | 21.4 ms | 18.6 ms |
| Ten-second Doppler map, 801 ranges, with DC removal | 8.38 ms | 4.77 ms |

Within real-recording reconstruction, Viterbi falls from about 129 to 100 ms
and re-encoding from 40 to 5 ms per batch. Reed–Solomon and remapping each take
about 31 ms. The remaining demapping/deinterleaving stage takes about 27 ms.
The range IFFTs take 14.2 ms per 1,020 symbols; the 513-tap slow-time filter
and ring/history bookkeeping take 6.8 ms. Coherence suffix selection is measured
separately. CSV files include all 15 supported constellation/code-rate cases.

The Vulkan residual path takes 5.14 ms per 6.95-million-carrier batch, versus
7.91 ms for the CPU equivalent. Separate warmed medians are 2.13 ms upload,
1.92 ms dispatch/submission/completion, and 1.06 ms readback. These are host
wall times, not isolated shader timestamps, and their medians need not sum.

## Why the FFT and decoder stay on the CPU

The benchmark also runs VkFFT with reusable plans and command buffers, checks
its complex output against FFTW, and includes host copies and completion:

| Forward FFT work | CPU copy + transform | Vulkan round trip |
| --- | ---: | ---: |
| One 8,192-point OFDM FFT | 12 µs | 599 µs |
| 32 × 8,192-point FFTs | 410 µs | 622 µs |
| One 57,344-point channelizer FFT | 119 µs | 618 µs |
| 32 × 57,344-point FFTs | 4.04 ms | 2.26 ms |

An earlier run measured 352 µs for the single 57,344-point Vulkan FFT; submission
latency is variable. Batches of 32 do favour Vulkan, but require a different
batching boundary and do not measure a complete
GPU NCO/filter/decimator. This pass keeps the existing CPU channelizer and the
Vulkan residual stage. GPU batching remains a possible future improvement, not
a rejected capability. The benchmark gives Vulkan reusable commands and does
not charge shader compilation against its steady-state times.

The existing decoder/re-encoder is already a compact C++ implementation. Its
Viterbi trellis advances serially in time, while its 64 states now use SIMD.
A GPU decoder would require a separate batching/traceback design and validation
of overlap and soft-decision behaviour. This pass retains the verified C++ path;
it does not claim that a GPU decoder is impossible or benchmark an unwritten one.

## Assembly review

Disassembly and compiler vectorization remarks were saved for DSP, modem,
synchronization, range–Doppler and pipeline output. Verified changes include:

* FFTW's hot NEON codelets use packed floating-point arithmetic.
* WAV finite validation uses `fabs.4s`, vector comparisons and integer reductions.
* NCO multiplication, spectral folding and response accumulation vectorize across
  adjacent carriers. The response accumulation has no per-carrier integer divide.
* The resampler uses shared `ld2.4s` input loads and `fmla.4s` accumulators.
* Viterbi uses packed add/subtract/compare/select and interleaved `st2.4s` stores.
  Its trellis loop has no metric-array copy or puncturing remainder division.
* Slow-time FIR arithmetic and Doppler input/window loops are vectorized.

Remaining scalar work includes dependent traceback, lookup/permutation work,
Reed–Solomon decoding, and phase/magnitude diagnostics with transcendental calls.
There is no blanket fast-math flag, reduced precision, shortened filter, or
approximate phase diagnostic. Further gains would require measured algorithm or
backend changes, rather than simply enabling an overlooked compiler SIMD switch.

## Reproduction and artifacts

```sh
source build/sdk/setup-env.sh
cmake --build --preset release --target radar_benchmarks -j 8
TMP=/private/tmp build/release/radar_benchmarks --gpu --input INPUT.wav \
  > build/benchmarks.csv 2> build/benchmarks.log

build/release/dvbt_reconstruct INPUT.wav --output build/perf-run --seconds 60 \
  --compute vulkan --integration-seconds 10 --history-seconds 10 \
  --map-interval 0.2 --remove-dc --map-stream - > /dev/null

xcrun llvm-objdump --demangle --disassemble --no-show-raw-insn \
  build/release/dvbt_reconstruct > build/assembly.txt
```

The optional input benchmark assumes C31's 56 MS/s input, +4.625 MHz offset and
GI 1/16. Run timing jobs serially, without compilation or another processor run.
Native throughput includes all requested scientific maps; `--map-stream -`
applies backpressure instead of permitting skipped updates. It excludes video
colour mapping and H.264 encoding. Record filesystem/cache and system-load effects
when comparing machines; these are file-processing measurements, not cold-disk claims.

Artifacts under `build/perf/`: `baseline-sample.txt`, saved baseline executable
and scalar libraries, `components-{baseline,final}.{csv,log}`,
`channel-blocks/results.csv`, `optimized-assembly.txt`, `fftw-neon-assembly.txt`,
and `*.vectorization.txt`. The benchmark implementation is the single optional
source [tests/benchmarks.cpp](../tests/benchmarks.cpp); stage timers are opt-in.

## End-to-end validation

Native workload: first 60 recorded seconds, Vulkan residual, ten-second coherent
integration/history, ±100 Hz, 0–30 km, DC removal and every 0.2-second map streamed
to `/dev/null` as well as publishing the latest map. Jobs ran serially. The
original executable and scalar FFTW/GNU Radio FFT libraries were preserved;
dynamic-loader logs confirm the intended libraries in the comparison.

| Native processing | Three wall times | Median | Recording/wall time |
| --- | --- | ---: | ---: |
| Original | 41.03, 41.47, 43.02 s | 41.47 s | 1.45× |
| Retained optimization | 28.17, 28.78, 28.56 s | 28.56 s | 2.10× |

This is a **1.45× throughput improvement**, or **31% less elapsed time**. The
first two retained measurements used the separately linked block-size probe;
the last used the final production executable. All three produce the same complex
final map. The discarded smaller-block candidate's three-run median was 29.53 s.
The final production run used about 1.01 GB peak RSS, 42.03 user CPU seconds and
5.85 system CPU seconds; concurrent workers explain CPU time exceeding wall time.

Every native run has 55,076 fully valid symbols, 810 TPS frames, 352,359 RS checks
including overlap, zero RS failures and 296 scientific maps. The final maps use
2,297 slow-time samples. Against the original, direct complex comparison gives
4.77e−6 relative RMS and 3.81e−6 degrees aggregate phase difference, without
registration, fitted scaling or phase correction. Map times are identical and
the final coherence changes by less than 5e−12.

All six CTests pass in Release and Debug; the final Debug run explicitly enables
Vulkan synchronization validation. Six Python viewer/export tests also pass.
The phase test independently checks injected sample-clock error, fractional delay,
CFO and a delayed Doppler echo against an analytic DFT. The demapper's full-grid
GNU Radio comparison has maximum scaled metric error 1.88e−7.

Evidence: `final-results.json`, `repeated-results.json`, `final-60.json`,
`release-tests.log`, `debug-tests.log` and `python-tests.log` under `build/perf/`.

The complete **301.84-second** recording was also rendered with the same
ten-second sliding window, range/Doppler limits, DC removal, autoscaling,
1080p H.264 output and 5 fps as the original export. Total elapsed time fell from
**229.27 to 149.87 seconds (2.01× real time, 35% less elapsed time)**. This includes
the C++ processor, lossless map stream, colour rendering and video encoding.

The new export contains all **1,504 consecutive scientific maps** and **1,510
video frames**. It has **277,368 fully valid symbols**, **4,079 TPS frames**,
**1,775,036 RS checks including overlap** and **zero RS failures**. The final
complex map differs from the original by **4.94e−6 relative RMS** and
**1.58e−6 degrees aggregate phase**, without alignment or correction.
All map times agree within 5.7e−14 seconds. All 277,372 exported timing rows retain
their row, segment, batch and validity identities, including the invalid edge rows.
Maximum coordinate differences are 3.82e−6 input ADC samples, 2.54e−7 radians
applied carrier phase and 1.45e−12 input samples per resampled output sample.

FFprobe confirms 1920×1080 H.264, 5 fps, 302 seconds and 1,510 frames; FFmpeg
decodes the entire file without errors. The validation copy is
`build/perf/C31_optimized.mp4`, with PNG/JSON companions and retained processor
diagnostics. `full-video-validation.json` records every comparison. The original
video in `renders/` remains available at its original path.
