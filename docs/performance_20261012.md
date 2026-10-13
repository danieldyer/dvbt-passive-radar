# Pipeline optimisation after the receiver accuracy fixes

The five-channel path from native int16 input through complex range profiles
reached **404.9 ms per input second, or 2.47× realtime, including input reads**.
A second run measured 419.8 ms/s (2.38×). Those are observed results, not a
guaranteed throughput floor: concurrent work on this machine subsequently
increased device times substantially. Consecutive longer controls measured
607–634 ms/s. CPU work stayed approximately constant while GPU times increased.
The user confirmed intermittent competing CPU/GPU workloads.

The complete five-channel map workload was also measured separately. It is
heavier than saving the complex profiles needed for subsequent image formation.
No physical UHD capture was performed for this performance qualification.

## What caused the regression

The preceding correctness changes increased work in the response solver,
decoder equalisation and noise/cancellation estimation without an adequate
throughput regression gate. There were also avoidable costs in the surrounding
implementation:

* The finite-waveform correction beyond the cyclic prefix evaluated nested
  sample/delay loops, with quadratic cost in the corrected prefix length.
* VkFFT selected two global passes for 8K complex-float transforms on this
  device. A register-based decomposition fits in one pass at the same precision.
* Reconstruction expanded every packet byte to a 32-bit word and every trellis
  bit to another word, then repeated puncturing-index calculations in the shader.
* CPU demapping/deinterleaving read complete equalised carrier arrays back
  through the CPU before Viterbi. It consumed about 175 ms/s summed over workers
  in the starting benchmark.
* The response solver used many shared-memory reduction barriers and large
  per-thread arrays. It also produced two full residual arrays that the live
  matched-profile consumer never read.

The previous timing harness additionally retained a **4.88 GB** int16 conversion
scratch buffer after preloading **9.76 GB** of float-complex samples. That memory
pressure was unrelated to the bounded live input path. The preload now releases
its conversion scratch, and the benchmark has an explicit native-int16 live
input mode. This harness correction is separate from the DSP optimisations.

The historical 2.511× result in
[the live TUI validation](live_tui_validation_20261011.md) used a short simulated
capture, two selected displays and acquisition/training within the measurement.
It was not a steady-state five-map measurement. The later real-device review
already reported roughly 1.298× headroom before the most recent receiver fixes.
These workload differences do not excuse the regression, but prevent a false
before/after attribution to one change.

## Production changes

**Finite range compression.** The previous-symbol boundary correction is now
an exact, zero-padded linear convolution, evaluated with a 2048-point FFT.
Both complex moments are retained. The filter, delay support and fractional
source-clock convention are unchanged. Cached 16/32/64/128-row FFT shapes avoid
executing a full 128-row tile for a short tail. The correction is skipped when
the entire supported delay interval lies within the guard.

**FFT decomposition.** The shared plan helper selects `registerBoost=2` for
8192-point complex-float transforms on Apple devices reporting 32 KiB shared
memory. The probe confirms two axis uploads become one. LUT twiddles and float32
arithmetic are retained. Five forward/inverse round trips give relative RMS
error **6.85e-7**, versus **7.90e-7** with the old decomposition. Other devices
and transform lengths keep VkFFT's defaults.

**Reconstruction.** RS packet bytes are packed directly; the convolutional
encoder produces eight three-bit code/validity entries in one 32-bit word.
Packet staging is four times smaller and trellis scratch eight times smaller.
The fixed puncturing and interleaving permutations are composed when configuring
the MCS, outside the per-sample kernels.
All fifteen supported MCS combinations retain exact carriers and validity masks.

**Demapping.** A dispatch in the existing frame command group writes packed
byte-valued hard decisions in convolutional-code order. CPU workers read that
mapped buffer directly. Viterbi, outer deinterleaving and RS remain the existing
CPU implementation. The simple CPU demapper remains the reference/fallback.
Thresholds, signed zeros, subnormal signs and nonfinite erasures are explicitly
cross-checked before FEC can hide an error.

**Minimum-phase fitting.** The solver keeps the same constrained model,
power-weighted amplitude objective and sixteen iterations. An orthogonal
log-magnitude projection supplies a better initial guess. Order specialisation,
native subgroup sums when supported, and a device-limited workgroup of up to
512 threads reduce barriers and register pressure. A portable shared-memory
reduction remains available. The partial-band orthonormal basis and amplitude
refresh that removed the numerical cancellation floor are preserved.

**Unused products.** Live capture and matched detection request only the
aligned reference from the response application stage. The legacy cross and
normalised residual arrays remain available to consumers that need them. The
reference-only path is tested for bit-identical phase and validity.

These changes do not alter the shared ADC time map, per-transmitter phase
coordinates, six-frame live decoder batch, coherent aperture, noise weighting,
stationary prediction, retained range support or map normalisation.

## Measurement boundary and results

Input: the first 20 or 30 seconds of
`baseband_202000000Hz_20-55-20_10-10-2026.wav`, before its first known gap.
Five distinct VHF channels use the recording metadata, 64-QAM rate 3/4,
8K mode and guard 1/16. The input is 61 MS/s int16 IQ, mapped to the existing
common 64 MS/s grid. Processing warms for ten seconds before measurement.

Five FEC workers and seven bounded frame slots are used. Decoder batches stay
at **six TPS frames**; they were not enlarged to obtain these results.

`compute_ms_per_s` includes native input reads in live-input mode, GPU unpack,
channelisation/retiming, receiver, FEC, reconstruction, response fitting,
cancellation/noise estimation and complex profile production. It excludes only
the support consumer's packet-validation/hash loop. The separate full wall
measurement includes that loop too. Archive writes, UHD transport, the TUI,
display projection and video encoding are outside this benchmark.

| Workload/run | ms/input s | Realtime factor | Including validation/hash |
|---|---:|---:|---:|
| Complex profiles, first profiled run | 404.9 | 2.47× | 429.4 ms/s |
| Complex profiles, second profiled run | 419.8 | 2.38× | 444.2 ms/s |
| Longer consecutive controls under competing load | 607.3–634.3 | 1.58–1.65× | 631.0–658.6 ms/s |
| Final rebuilt binary, profiling disabled | 524.4 | 1.91× | 549.2 ms/s |
| All five maps, 3 s CPI, ±100 Hz, one update/s/channel | 581.7 | 1.72× | 605.5 ms/s |

The map run delivered **50 complete maps** over 9.952 measured input seconds.
It copies every scientific map cell to the CPU; the native display normally
uses a GPU projector. It must not be confused with the profile-only result or
with displaying only one selected channel.

The selective-query controls initially appeared to implicate profiling.
The subsequent all-query control also slowed down, ruling out that inference.
No timestamp, driver-configuration or power-management workaround was added to
production. Interleaved measurements on a shared GPU do not isolate hardware
throughput, so the best result is explicitly labelled as such.

### CPU and GPU ownership

Device timestamps from the 404.9 ms/s profile run, summed across **all five
channels**, in milliseconds per input second:

| Component | Execution | Time |
|---|---|---:|
| Native IQ unpack | GPU | 5.46 |
| Shared-clock frontend/channelisation | GPU | 28.39 |
| OFDM acquisition/tracking, FFT and receiver kernels | GPU | 27.82 |
| Frame gathering, pilot equalisation, hard demapping/deinterleaving | GPU | 41.47 |
| Oracle re-encoding/remapping | GPU | 15.16 |
| Minimum-phase response reduction/solve/application | GPU | 32.06 |
| Noise statistics | GPU | 6.42 |
| Finite-waveform range profiles, both complex moments | GPU | 63.62 |

Measured active CPU time is **165.6 ms/s on the main thread**, **928.3 ms/s
across the process**. The worker timers report **714.4 ms/s Viterbi** and
**30.3 ms/s outer deinterleaving plus RS**, summed over all five workers. Those
worker timers are elapsed intervals inside CPU work, not thread CPU clocks;
they can include descheduling. CPU demapping is now zero on this path.

The main thread, workers and GPU overlap. None of these columns should be added
to the end-to-end wall time, and fence waits are not additional active CPU work.
Input reads account for 43.6 ms/s of the wall interval in this run; this is not
an optimisation of storage I/O.

For context, the starting all-map benchmark measured 791.9 ms/s **excluding
reads**, with 306.0 ms/s main-thread CPU and 1280.8 ms/s process CPU. The later
all-map run measured 581.7 ms/s **including reads**, with 200.2 and 949.5 ms/s
respectively. Reconstruction GPU time was 53.4 → 22.6 ms/s, response work
82.3 → 48.4, and profile work 137.3 → 77.7. These are observed runs with different
memory pressure and external load, not an isolated causal speedup for each patch.

## Accuracy gates

The complete Release suite passes **37/37 tests**. It covers the scalar FEC
reference, all puncturing patterns, RS correction, multichannel scheduling,
independent transmitter clocks, common-clock retiming, exact reconstruction,
minimum-phase/all-pass separation, finite beyond-guard waveforms and direct
backprojection comparisons. No tolerance was relaxed for these optimisations.

The strong 20 µs multipath control was rerun for all fifteen MCS combinations,
eight blocks each, through the new GPU demapper. **592,220,160 coded-bit decisions
and 261,768 RS word decisions** contain no bit errors, rejected words or wrong
accepted payloads. The entire CSV, including edge-word and traceback counters,
matches the pre-optimisation control exactly.

Noisy F1 multipath controls retain the same packet-rejection counts at 20,
21 and 22 dB data-carrier Es/N0: respectively 36,789, 123 and zero out of 54,256.
A separate 26 dB control also has zero rejected or wrong accepted words.
With reordered FFT arithmetic, pre-FEC error counts differ by −2, +3 and +1
among 118,444,032 decisions at the three paired noise levels; the packet
waterfall observations are unchanged. Float waveform outputs are checked
numerically, not claimed to be bit-identical after changing FFT decomposition.

Every 20-second recording benchmark checks all accepted packets and rejects
any FEC failure. In the measured interval, **1,170,631 decoded word decisions**
have the same byte checksum as the starting implementation:
`196319515850604808`. These decisions include overlapping decoder context;
they are not a count of unique transmitted packets. The 30-second repeats also
agree with one another, at 2,317,514 decisions and checksum
`4946505363446608333`.

The response initialisation has additional tests at gains 0.01, 1 and 100,
with coloured reference power, erasures and both noiseless/noisy observations.
The sixteen-sweep GPU result is compared with an independent 128-sweep
double-precision CPU fit. Relative response RMS differences are below 8.1e-7 in
those controls. Omitting unused residual products leaves the aligned reference
bit-identical. Finite-waveform tests compare both moments with independent
time-domain sums, including guard crossings and source-clock phase wraps.
All four cached FFT tile sizes pass; the drifting-source finite-waveform control
has maximum target phase error **0.00301°**. The five affected GPU tests also
pass with Vulkan synchronization validation enabled and no reported hazards.

Raw results are in [data/performance_20261012](data/performance_20261012).
`gpu-demap-live-profiles.log`, `live-profiles-repeat.log`,
`live-profiles-series-*.log`, `final-live-profiles.log` and `live-all-maps.log`
are the end-to-end controls;
`ctest-release.log`, `vulkan-validation.log` and `fft-plans-final.csv` record
regressions, synchronization validation and FFT decomposition checks. The
CPU-sampled intermediate run is excluded from throughput conclusions.

## Reproduce

```sh
cmake --build --preset release -j6
cmake --build --preset release --target detection_pipeline_benchmark fft_plan_probe -j6
export VK_ICD_FILENAMES="$PWD/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
build/release/support/validation/detection_pipeline_benchmark \
  /Users/danielrd/data/dvbt_passive_radar/baseband_202000000Hz_20-55-20_10-10-2026.wav \
  20 5 0 0 1 3 recordings/baseband_202000000Hz_20-55-20_10-10-2026.json 6 live
ctest --preset release --output-on-failure -j1
```

Change consumer `3` to `1` for all five maps; change timing `1` to `0` to disable
profiling. Run sequentially and record competing machine load. The stage
optimisations are qualified; a contention-free sustained 2.5× live-capture
guarantee is not established by these shared-machine measurements.
