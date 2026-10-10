# Common-clock resampling optimization — 6 October 2026

The production frontend now folds most clock correction into channel synthesis.
It removes the original full-rate, 32/64-tap ADC resampling pass for ordinary
channels and affine clock windows. The existing channel inverse transform applies
the block-centre fractional delay; a short residual interpolation runs at the
channel rate inside the same workgroup. There is no added FFT round trip.

The subsequent [scheduling and resident-input pass](pipeline_concurrency_20261006.md)
keeps this numerical operation, overlaps FEC with reception, and exposes raw-IQ
GPU input without host copying. Its current measurements distinguish the
recording benchmark from RAM processing and the resampling-specific speedup.

## Measurement results

The clock-correction comparison uses alternating identity/drifting-clock runs of
**the same new frontend**, with the same buffers, transform sizes and shader.
The difference measures the extra work for a 5 ppm clock error; it is not a
standalone resampler timestamp. The nominal frontend already pays for spectral
phase-ramp evaluation and the 16 MS/s synthesis grid. Complete frontend timings
are reported alongside this difference. The old raw-resampling kernel is frozen
from the original implementation and measured separately with its original
32-tap single-channel / 64-tap multichannel tables. No file reads, host copies,
video work or readback is included in GPU timings.

Apple M5, Release/MoltenVK, 56 MS/s input. Three runs of 2,048 paired
measurements per channel count, after 32 warmup pairs. GPU time is in
milliseconds per input second:

| Channels | Original raw resampler | Added correction in fused frontend | Two-standard-error upper cost | Conservative reduction |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.78 | 0.114 | 0.244 | 44× |
| 5 | 20.38 | 0.711 | 0.833 | 24× |
| 7 | 20.37 | 0.965 | 1.105 | 18× |

The uncertainty combines the paired standard error and run-to-run variation.
The conservative comparison uses the smallest measured old-resampler cost
divided by the upper new cost. Mean-cost ratios are approximately 95×, 29×
and 21×; the one-channel difference is small enough that its precise ratio
is poorly resolved. There is no separate normal-path resampling dispatch.

Complete frontend costs are also useful, because the nominal synthesis
grid and phase ramp remain real work:

| Channels | New fused frontend in paired benchmark | Old complete frontend, median | New complete frontend, median |
| ---: | ---: | ---: | ---: |
| 1 | 12.76 | 38.06 | 18.69 |
| 5 | 22.16 | 47.26 | 26.62 |
| 7 | 27.20 | 55.71 | 32.22 |

The paired one-channel case uses the recording’s +4.625 MHz centre. The
complete-frontend benchmark uses centres `(c−3)*7 MHz + 123.375 Hz` for
all three bank sizes, including −21 MHz for its one-channel case. Complete
frontend figures are medians of three alternating old/new runs. GPU timers
exclude host copies/readback, but host activity and device power state can
still change GPU clocks; these entry points have different host workloads.
The new frontend consumes approximately 1.3–3.2% of a device second per
input second in these resident tests. This is not a 10× reduction in total
filtering/FFT workload.

### Full recording comparison

Two alternating, uninstrumented runs of the first 60 seconds of the C31
recording, with five seconds of acquisition warmup excluded:

| Metric | Frozen original | New |
| --- | ---: | ---: |
| Compute elapsed, ms/input second | 153.44 | 125.99 |
| Compute throughput | 6.52× realtime | 7.94× realtime |
| GPU execution, ms/input second | 59.94 | 43.82 |
| Active main-thread CPU, ms/input second | 50.39 | 49.34 |
| Host submit elapsed, ms/input second | 6.20 | 4.46 |
| Fence-wait elapsed, ms/input second | 107.72 | 83.32 |
| GPU submissions/input second | 220.1 | 160.5 |

Compute elapsed falls by **17.9%**. The uninstrumented repeats were
154.60/152.27 ms/s before and
126.00/125.98 ms/s after.
The GPU/CPU/submit/wait rows come from separate detailed-profile runs; their
instrumented compute elapsed was 157.48/132.13 ms/s. Wait time includes
device execution and completion latency; submit time is already included
in CPU/elapsed totals. These columns must not be added together.

Frontend device-time breakdown in those recording runs:

| Work | Owner | Original ms/s | New ms/s |
| --- | --- | ---: | ---: |
| Full-rate warp / boundary-only warp plus replacement FFT | GPU | 18.214 | 0.369 |
| Shared forward FFT | GPU | 11.340 | 16.186 |
| Filtering + inverse transform + output NCO, including new residual interpolation | GPU | 8.054 | 3.827 |
| Extra detailed-timestamp markers | GPU | 0.000 | 0.106 |
| Total frontend device time | GPU | 37.608 | 20.489 |
| Frontend parameters, control and driver calls | CPU | 3.496 | 3.948 |

The shared forward FFT is now the largest GPU component. Its recording
time did not improve: the new overlap/halo and device scheduling differ.
Filtering, inverse transformation and interpolation are intentionally
reported as one fused component. Artificially splitting their timestamps
would require changing the kernel being measured.

Increasing the batch from 163,840 to 256,000 channel samples reduced
submission/completion overhead without changing the per-window operator.
A batch spans 32 ms. The first smaller-batch version improved full compute
by only about 3%; retaining that version would have hidden the importance
of host/device completion overhead after arithmetic became cheap.

FEC is unchanged and remains on the CPU: Viterbi uses 30.34 ms/s and
RS uses 1.33 ms/s in the new detailed run. The source is the same C++23
auto-vectorized decoder used in the baseline. Reads, input validation,
consumer validation, startup, maps and video are excluded from the
compute figures. Both runs emitted 51,340 measured rows and examined
328,967 RS words with **zero rejected words and zero corrected bytes**.

## Architecture and phase convention

For corrected ADC coordinate `n`, all channels share the immutable map

```
u(n) = n + offset_s + (n - start_s) * (rate_s - 1).
```

A normal 56 MS/s batch has 800 overlapping 3,584-point forward transforms and
256,000 valid 8 MS/s output samples. VkFFT reads each overlapping window directly
from the original ADC buffer. Filtering and clock correction share one prototype
response table; per-channel tables and full-rate NCO passes are unnecessary.

Inside an affine segment with slope `a`, commuting a centred, zero-phase filter
through the time map requires evaluating its response at **H(a f_raw − f_c)**.
Using H(f_raw − f_c) would apply the wrong passband when the clock rate changes.
The response is real, generated from the same float32 prototype used before this
change. A 65,536-point offline/runtime-initialization table provides linear lookup;
no extra FFT is dispatched for this table during streaming.

Let `v` be the original ADC FFT-window start and let `f_bin` be the nearest FFT
bin to the requested centre `f_c`. The synthesized baseband field is

```
z(t) = filtered_raw(v + t) exp(−i 2π f_bin t / Fs).
y(n) = z(u(n) − v) exp(i 2π [f_bin (u(n) − v) − f_c n] / Fs).
```

The last factor is essential: resampling a mixed channel without it loses
interchannel phase coherence. The CPU forms a compensated double-precision
absolute anchor and a small local phase increment. GPU coordinates separate
integer positions from fractional offsets. Only bounded, within-block phase
products occur; no cumulative float oscillator recurrence is used. Physical RF,
channel NCO and receiver CFO remain distinct in downstream metadata.

The synthesis shader applies a frequency-domain phase ramp for the complete
fractional delay at the block midpoint. Its 1,024-point radix-4 inverse FFT
replaces the former 512-point channel inverse FFT, producing a 16 MS/s
intermediate. Filtering, translation, inverse transform, residual interpolation
and NCO correction remain in one kernel. Only final 8 MS/s samples leave shared
memory; this avoids intermediate global writes, reads, barriers and dispatches.

For ordinary oscillator drift, the residual displacement `δ` is at most 0.002
intermediate samples. The centred approximation is

```
x(i + δ) ≈ x(i)
         + δ Σ[r=1..3] c_r (x(i+r) − x(i−r))
         + δ²/2 (x(i+1) − 2x(i) + x(i−1)).
```

The three first-derivative weights are fitted offline over the occupied band.
Dense tests measure maximum complex error 4.48e-6 through ±3.328125 MHz
and 4.88e-6 through 3.340 MHz, including centre-bin/clock-scaling margin. At the
56 MS/s block length, the fast path covers approximately ±6.27 ppm local slope.
Larger displacement uses the checked-in 16-tap fractional-delay table: measured
maximum complex error 3.10e-7 through ±3.671875 MHz. The branch is uniform within
a workgroup; both paths use the same clock map and phase convention.

The coefficient-design script `scripts/design_clock_interpolator.py` reproduces
the 16-tap coefficients and the three differentiator weights. Python/NumPy are
only offline design tools, not processing or build dependencies. Clear scalar
interpolators and the original scalar ADC-interpolation/FIR reference remain in
the tests; there are no CPU intrinsics or CPU micro-optimization changes here.

## Non-affine windows and finite interpolation

A window crossing a clock-segment boundary cannot commute a single affine map
through the filter. Those windows retain the original full-band interpolation,
followed by a replacement shared forward FFT. This correction is common to all
channels. Clock updates activate beyond previously consumed FIR support, and
published segments remain immutable. The normal affine path has no replacement
FFT and no full-rate interpolation pass.

The original interpolation is also retained when a selected channel's prototype
transition reaches ADC Nyquist. That case needs the wrapped ADC bins after
mixing. Existing synthetic OFDM tests exposed this distinction; preserving it
restores their original tolerances. These exceptional cases are not covered by
the normal-path speedup, nor are large clock errors that force many FFT windows
outside their affine support halo.

This is a bounded numerical approximation to bandlimited interpolation, not an
assertion of exact interpolation at arbitrary precision. Spectral selection
truncates prototype stopband tails outside the 16 MS/s intermediate, while the
local interpolators have explicit finite support. Fourier resampling alone
assumes a periodic block; overlap-save halos and discarded edges are necessary
for a continuous recording. See [SciPy's Fourier resampling definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html).

The existing ±3.328125 MHz passband / ±3.671875 MHz rejection-edge mask remains.
Full-scale clocked blocker tests retain more than 100 dB rejection, against an
ideal 12-bit in-band quantization-noise design point of about −83.26 dBFS at
56 MS/s. This integrated-band design criterion does not assert a coherent-spur
floor for arbitrarily long radar integrations. The minimum-phase estimator still
runs after corrected channel synthesis; no propagation delay or multipath model
has been introduced into clock correction.

## Regression evidence

All **21 Release tests and 21 Debug tests** passed, with Vulkan validation in
Debug. The subsequently strengthened centre-offset/interpolation-margin tests
also passed in both configurations. Final measured bounds:

| Check | Measured maximum |
| --- | ---: |
| Relative phase across five distinct channels, ±1000 ppm tests | 0.001160° |
| Different chunk boundaries | 1.76e-7 complex amplitude |
| GPU versus scalar original ADC-interpolation/FIR reference | 5.27e-7 complex amplitude |
| 16-tap interpolation response error | 3.10e-7 |
| Fast residual response error through 3.340 MHz, including bin-offset margin | 4.88e-6 |
| Full-scale clocked blocker leakage | −105.67 dBFS |
| Supported ADC-rate waveform error | 8.59e-7 complex amplitude |
| Closed-loop OFDM timing-phase RMS across the occupied band | 0.0581° |

The 3-second all-output verification run hashes decoded packet bytes and status,
measured/reference/residual spectra, validity masks, estimated response, reference
origin, and symbol sample/phase/frequency coordinates. Production and isolated
profiling modes 0, 1 and 2 give the identical hash **`49681309accb94a2`**, with
2,584 emitted rows, 15,843 RS examinations and no rejected/corrected words.
Production source hashes stayed unchanged throughout the measurement run.

Frozen-original versus new floating-point spectra are not expected to be bitwise
identical: the transform and interpolation formulation changed. The independent
waveform/scalar/phase regressions bound that difference; the hash comparison
checks that instrumentation itself does not alter production outputs.

Regression coverage includes:

- Five distinct centres from −21 to +21 MHz, independent tones at occupied-band
  edges and interior frequencies, and a common epoch beyond 2^40 ADC samples.
  An integer modular-arithmetic oracle checks absolute NCO phase independently
  of production's compensated floating-point formula.
- Positive and negative drift, up to ±1000 ppm, continuous clock-segment updates,
  different chunk boundaries and seeks. The ±1000 ppm cases use bounded rate
  changes, rather than asserting a bandlimited oracle across an instantaneous
  2000 ppm discontinuity.
- An independent scalar reference that first interpolates original ADC samples,
  then applies the centred FIR/NCO. It does not share the spectral algorithm.
- Full-scale blockers at rejection edges and farther out of band, with clock
  updates; all supported ADC decimation ratios from 8 through 128 MS/s.
- Existing synthetic OFDM tests with fractional delay, CFO, sample-clock error
  and a delayed Doppler echo. Closed-loop receiver and complete residual-handoff
  phase tests retain their original thresholds.
- Existing reconstruction, FEC, minimum-phase response, residual and multichannel
  SAR-coordinate tests. No test threshold was relaxed.

## Reproduction and evidence

```
cmake --build --preset release --target frontend_benchmarks resampling_benchmarks -j8
build/release/resampling_benchmarks 2048
build/release/frontend_benchmarks --stages
ctest --preset release
ctest --preset debug
```

On this machine set `VK_DRIVER_FILES` to the SDK's MoltenVK ICD manifest when
running GPU executables. `resampling_benchmarks` warms each case with 32 pairs,
then measures 1,024 alternating pairs by default (2,048 in the final report),
using GPU endpoint timestamps. It reports paired standard errors, including GPU timing variation. The copied old kernel is
benchmark-only (`tests/shaders/raw_clock_reference.comp`); production contains
only the narrow boundary-correction kernel and the fused synthesis shader.

Raw logs, CSVs, isolated profile sources/drivers and frozen baseline sources are
under `build/resampling-20261006/`. The full-pipeline driver excludes recording
reads, input validation, output validation, acquisition warmup, map formation,
visualization and video encoding from compute elapsed time. Active main-thread
CPU, driver/submit/wait time and device timestamps are reported separately;
CPU and GPU totals must not be summed as an elapsed throughput estimate.
