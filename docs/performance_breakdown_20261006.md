# Pipeline execution breakdown — 6 October 2026

The current raw-IQ-to-complex-residual-spectra pipeline uses Vulkan for channel
synthesis/retiming, OFDM numerical stages, reconstruction and response/residual
arithmetic. Hard demapping, deinterleaving, Viterbi and Reed–Solomon run on bounded
CPU workers. Reduced pilot/TPS tracking and the common-clock controller run on
the main thread.

The repeated recording benchmark measured **62.45 ms per input second** with
profiling disabled, approximately **16.0× real time** under its compute-only
measurement boundary. The RAM-input check measured **99.23 ms/s** excluding its
packet-checking consumer, or **100.70 ms/s including it**. The recording and RAM
figures have different boundaries; neither is substituted for the other below.

## Workload and method

* Apple M5, 10 CPU cores, 24 GiB RAM, macOS 26.4 (25E246), Apple Clang 21,
  Release `-O3 -DNDEBUG`, Vulkan/MoltenVK.
* Real C31 capture: `baseband_553000000Hz_12-51-54_26-09-2026.wav`,
  56 MS/s complex float32, +4.625 MHz NCO, 8 MS/s channel output,
  8K/QPSK, code rate 3/4, guard interval 1/16.
* `GpuChannelBank -> GpuPipeline -> SpectralBatch`, with the common clock's
  three-degree design budget, order-8 minimum-phase response and both complex
  residual products. No range/Doppler accumulation, map formation, SAR imaging,
  video, viewer, archive output or DC projection runs in this benchmark.
* Five input seconds warm acquisition, plans and buffers; queued FEC work is
  drained before the timed interval. The 60-second recording pass measures
  **54.975997625 input seconds**, including the final decoder/reconstruction drain.
* Active CPU uses `CLOCK_THREAD_CPUTIME_ID` in exclusive component scopes, with
  main and worker threads reported separately. Process CPU is measured
  independently using `CLOCK_PROCESS_CPUTIME_ID`. No scopes are inserted in the
  hot Viterbi implementation.
* GPU endpoint timestamps subdivide the actual streamed jobs. Readiness fences,
  submission calls and main-thread decoder waits are recorded separately.
* Instrumentation lives entirely in `build/perf-20261006/source`; the production
  DSP files remain unchanged. Runs are serial under normal desktop conditions.

Recording passes subtract input reads/validation and consumer work. FEC can
overlap excluded I/O, so the 62.45 ms/s result is **not strict I/O-free throughput**.
That subtraction does not remove worker CPU time or device work from their
component counters.

The RAM passes preload ten seconds of IQ before any processing begins. They
include the host-span copy into mapped GPU input, all numerical work, waits and
final draining. This eliminates excluded file reads during decoding. The timed
interval is 4.975951375 input seconds. Consumer time is shown both included and
excluded: asynchronous FEC can also progress during the validating consumer.

## Repeated throughput measurements

All values are milliseconds per input second. Component tables use the detailed
passes directly; they are not rescaled to the timing-disabled totals.

| Workload / instrumentation | Individual runs | Mean |
| --- | --- | ---: |
| Recording, one channel, timing disabled | 60.884, 64.012 | **62.448** |
| Recording, one channel, aggregate profile | 69.681, 66.477 | 68.079 |
| Recording, one channel, detailed profile | 67.231, 65.049 | 66.140 |
| Recording, five chains, detailed profile | 187.546, 180.426 | 183.986 |
| RAM, one channel, timing disabled | 109.872, 106.414, 81.412 | **99.233** |
| RAM, one channel, aggregate profile | 92.261, 78.190, 91.482 | 87.311 |
| RAM, one channel, detailed profile | 91.554, 94.282 | 92.918 |
| RAM, five chains, timing disabled | 194.892, 182.792 | **188.842** |
| RAM, five chains, aggregate profile | 205.044, 200.578 | 202.811 |
| RAM, five chains, detailed profile | 204.802, 243.022 | 223.912 |

The timing-disabled RAM means including the packet-checking consumer are
100.699 ms/s for one channel and 196.107 ms/s for five. Direct production-binary
checks measured 102.581/184.563 ms/s excluding that consumer and produced identical
per-channel packet checksums to the isolated driver.

The recording aggregate profile adds 9.0% relative to the timing-disabled mean.
The short RAM passes have enough variability that their profiling overhead
cannot be estimated reliably by subtracting those means. In the final five-chain
RAM pass, input copying alone used 46.75 ms/s active CPU, versus 21.91 ms/s in
the earlier detailed pass. The report preserves both runs; it does not attribute
that variation to a proven memory or device-clock cause.

All five-chain runs repeat the **same C31 waveform** through five decoding chains
sharing the channel bank. They measure concurrent processing load, not five
independent transmitters or a five-channel 64-QAM workload.

## Component balance: settled 60-second recording

The following is the final detailed run. CPU columns are active thread time;
the GPU column is device timestamp coverage. CPU and GPU work overlap, so these
columns do not partition elapsed time. Small CPU entries include parameter setup,
command recording and driver calls; query collection can appear under pipeline
control rather than the kernel whose results it reads.

| Component | Main CPU | FEC workers | GPU |
| --- | ---: | ---: | ---: |
| Input padding / adapter preparation, excluding reads | 0.160 | 0 | 0 |
| Channelization, common retiming and clock control | 4.002 | 0 | 20.139 |
| OFDM reception and pilot tracking | 5.080 | 0 | 5.312 |
| Frame copies, TPS/BCH and snapshot capture | 0.179 | 0 | 2.796 |
| Hard QAM and inner deinterleaving | 0 | 6.523 | 0 |
| Depuncturing, Viterbi and traceback | 0 | **30.940** | 0 |
| Outer deinterleaving, RS and packet validation | 0 | 1.369 | 0 |
| Reference reconstruction | 0.125 | 0 | 4.012 |
| Minimum-phase response fit | 0.003 | 0 | 4.385 |
| Residual and complex cross spectra | 0.002 | 0 | 3.153 |
| Pipeline, completion and decoder queue control | 0.925 | 0.027 | 0 |
| **Scoped totals** | **10.478** | **38.859** | **39.797** |

Compute elapsed for this run was **65.049 ms/s**. Independently measured process
CPU was **55.834 ms/s** after excluding reader/consumer CPU, leaving **6.497 ms/s**
outside the component scopes. That difference includes unscoped process work
such as driver/background threads and profiler bookkeeping; it is not assigned
to a particular component without evidence.

Viterbi uses approximately **79.7% of active FEC CPU**, hard demapping 16.8%, and
outer/RS work 3.5%. Excluding demapping, Viterbi versus outer/RS is approximately
**95.8% / 4.2%**. The RS measurement is clean checking cost: no codewords required
correction in this recording. No BCJR or GPU FEC runs.

### Five concurrent chains

These values are per shared input second, summed across five worker threads.
They are not elapsed time per channel.

| Component | Main CPU | FEC workers | GPU |
| --- | ---: | ---: | ---: |
| Input padding / adapter preparation | 0.212 | 0 | 0 |
| Channelization, common retiming and clock control | 10.237 | 0 | 37.062 |
| OFDM reception and pilot tracking | 22.406 | 0 | 40.722 |
| Frame copies, TPS/BCH and snapshot capture | 0.752 | 0 | 18.946 |
| Hard QAM and inner deinterleaving | 0 | 48.247 | 0 |
| Depuncturing, Viterbi and traceback | 0 | 180.582 | 0 |
| Outer deinterleaving, RS and packet validation | 0 | 7.086 | 0 |
| Reference reconstruction | 0.489 | 0 | 12.386 |
| Minimum-phase response fit | 0.010 | 0 | 15.618 |
| Residual and complex cross spectra | 0.009 | 0 | 13.144 |
| Pipeline, completion and decoder queue control | 2.907 | 0.315 | 0 |
| **Scoped totals** | **37.021** | **236.229** | **137.878** |

Compute elapsed was 180.426 ms/s; process CPU excluding reader/consumer was
284.225 ms/s, including 10.974 ms/s outside the component scopes. Simultaneous
channels do not simply multiply the one-channel numbers: shared frontend work,
worker contention and execution conditions differ.

## GPU operation detail

These endpoint intervals subdivide the same final recording runs. Fused
operations remain grouped: there is no defensible separate timestamp for normal
resampling, prototype filtering, channel IFFT or NCO inside one workgroup.
The boundary row is an exceptional-path cost, not the entire retiming cost.

| GPU operation | One channel | Five chains |
| --- | ---: | ---: |
| Shared channel forward FFT | **15.168** | 20.217 |
| Boundary-only full-band warp and replacement FFT | 0.420 | 0.436 |
| Fused prototype, channel IFFT, retiming and NCO | 4.432 | 16.291 |
| Receiver history copy | 0.747 | 6.096 |
| OFDM window extraction and carrier correction | 0.472 | 5.761 |
| Batched OFDM FFT | 1.865 | 14.742 |
| Pilot measurements | 0.703 | 5.320 |
| Equalization and TPS reduction | 0.836 | 4.884 |
| Apply tracked pilot phase | 0.636 | 3.865 |
| Frame-ring and owned-snapshot carrier copies | 2.796 | 18.946 |
| Convolutional re-encoding and validity | 1.018 | 3.073 |
| Interleaving, QAM, pilots and TPS regeneration | 2.972 | 9.238 |
| Response sufficient-statistic reduction | 1.659 | 6.499 |
| Iterative minimum-phase solver | 2.639 | 8.920 |
| Reference alignment, residual and cross spectra | 3.153 | 13.144 |
| Remaining timestamp-marker intervals | 0.282 | 0.447 |
| **Total timestamp coverage** | **39.797** | **137.878** |

The shared channel forward FFT is about 38% of measured one-channel GPU time;
the whole frontend is about 51%. The former standalone full-rate resampler is
absent from normal affine processing. Reconstruction produces carrier-domain
reference data; no waveform IFFT is hidden in its modulation cost.

CP acquisition and integer-CFO search run on the GPU during warmup, and are not
recurring costs in these settled intervals. Main-thread pilot tracking itself
uses 2.43/16.63 ms/s for one/five chains; the rest of the reception CPU row is
control, window parameters and submission work.

The profiler also recorded the union of absolute GPU endpoint intervals. Its
coverage agrees with the sum to timestamp/rounding precision: no measurable
interval overlap occurred on this queue in either final pass. This verifies
that these GPU totals do not double-count overlapping command spans. Endpoint
coverage can still contain dependency stalls and barriers; it is not an ALU
utilization counter or a device-wide utilization reading.

## Submissions, waits and overlap

| Diagnostic | One channel | Five chains |
| --- | ---: | ---: |
| Submissions per input second | 94.84 | 103.43 |
| Host submission elapsed, ms/s | 3.36 | 6.41 |
| Main-thread GPU fence wait, ms/s | 54.25 | 143.30 |
| Worker GPU-readiness wait, summed ms/s | 2.32 | 33.97 |
| Main-thread decoded-batch collection elapsed, ms/s | **0.59** | **0.59** |

Submission CPU and wait CPU are already contained in the component CPU scopes.
Fence wait elapsed includes GPU execution and queue/driver completion latency.
It is not additional CPU arithmetic and must not be added to GPU time.

FEC consumes about 38.9 ms/s active CPU in the single-channel recording pass,
but the main thread spends only 0.59 ms/s collecting/waiting for decoded batches.
The short RAM pass sees 7.37 ms/s there; after multiplying by each pass's input
duration, both waits are about 33–37 ms total. This is consistent with a final
decoder tail dominating those waits. It is evidence that the former synchronous
FEC duration no longer adds directly to the steady-state critical path.

Consequently, Viterbi is still the largest CPU arithmetic cost, while its CPU
share alone does not predict the throughput gain from optimizing it further.
GPU frontend execution and main-thread submission/completion remain relevant
to single-channel elapsed time. This interpretation also accounts for the
recording benchmark's excluded-I/O overlap.

## RAM-input component check

The final one-channel RAM pass measured 94.282 ms/s excluding its consumer,
23.423 ms/s scoped main CPU, 48.352 ms/s worker CPU and 53.304 ms/s GPU coverage.
Active process CPU was 78.408 ms/s. The main differences from the long recording
pass are visible rather than assigned to file reading:

| Component | Main CPU | FEC workers | GPU |
| --- | ---: | ---: | ---: |
| Host input copy/padding | **13.894** | 0 | 0 |
| Frontend parameters and common-clock control | 3.658 | 0 | 24.985 |
| OFDM reception and pilot tracking | 4.688 | 0 | 7.506 |
| Frame copies, TPS/BCH and snapshot capture | 0.160 | 0 | 4.824 |
| Hard QAM and inner deinterleaving | 0 | 9.161 | 0 |
| Viterbi | 0 | 37.509 | 0 |
| Outer/RS and packet validation | 0 | 1.596 | 0 |
| Reference reconstruction | 0.139 | 0 | 5.556 |
| Minimum-phase response fit | 0.004 | 0 | 6.113 |
| Residual and cross spectra | 0.003 | 0 | 4.320 |
| Pipeline and decoder queue control | 0.878 | 0.087 | 0 |

The short pass emits 5,372 measured rows and examines 34,699 RS words. Complete
frame batches can straddle the warmup boundary, so work per nominal input second
is not exactly the long-run steady-state workload. The GPU and CPU times also
vary across passes. The additional host-span copy belongs to this input adapter;
the production recording adapter reads directly into mapped input storage, and
the resident-input API avoids it. It is not attributed to interpolation math.

Five-chain RAM detail, including its larger copy variation, is retained in
`build/perf-20261006/components.csv` and `summary.json` rather than being dropped
from the reported repetition range.

## Validation and reproducibility

All single-channel recording passes examine **328,967 RS words**, with **zero
failures and zero corrected bytes**, and emit **51,340 measured rows**. Five chains
have those counts per channel. RAM repetitions and direct production checks
have identical per-channel packet checksums and observation counts.

The three-second full-output check hashes corrected packet bytes/status, measured
and reference spectra, both residual products, response, masks, reference origin,
sample coordinates, applied phase and CFO. Production and profile modes 0/1/2
all reproduce **`49681309accb94a2`**, with 2,584 rows, 15,843 RS examinations and
94 clock segments. Production source SHA-256 hashes remained unchanged. The
instrumentation generator was regenerated and confirmed byte-for-byte identical
to the measured isolated source. No numerical implementation was changed.

Artifacts are in `build/perf-20261006/`: raw JSON/logs, `summary.json`,
`components.csv`, `gpu_operations.csv`, production/instrumented source manifests,
the isolated source, generator, measurement scripts and analysis script.

```sh
python3 build/perf-20261006/instrument.py
cmake --build build/perf-20261006/profile-build --target pipeline_profile pipeline_benchmarks -j8
python3 build/perf-20261006/measure.py
python3 build/perf-20261006/measure_coverage.py
python3 build/perf-20261006/analyze.py
```

These profiling scripts use the bundled MoltenVK ICD and the known C31 recording;
GPU execution requires the usual device access on this machine. Production
throughput can also be checked with the opt-in `pipeline_benchmarks` target.
The [scheduling report](pipeline_concurrency_20261006.md) defines ownership and
callback contracts; [the resampling report](resampling_optimization_20261006.md)
documents the numerical fusion and coherence regressions. Earlier timings in
[the 2 October profile](performance_breakdown_20261002.md) describe the previous
synchronous pipeline and full-rate resampler.
