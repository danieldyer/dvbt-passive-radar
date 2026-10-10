# Pipeline execution breakdown — 2 October 2026

These measurements describe the earlier synchronous pipeline and full-rate
resampler. See [the 6 October CPU/GPU breakdown](performance_breakdown_20261006.md)
for the current fused frontend and concurrent CPU FEC workers.

The current raw-IQ-to-complex-spectra path takes **149.3 ms per
recorded second**, approximately **6.70 times real time**,
with detailed instrumentation disabled. FEC runs entirely on the CPU using the
new auto-vectorized C++23 implementation; frontend, OFDM carrier processing,
reference reconstruction and response/residual arithmetic run on Vulkan.

The measurements below supersede the [30 September GPU-FEC profile](performance_breakdown_20260930.md).
The intervening decoder replacement changed the algorithm from BCJR/soft RS to
hard-decision Viterbi/errors-only RS. For a controlled comparison of just the
latest CPU optimizations, see [the FEC optimization report](fec_cpu_optimization.md).

## Workload and measurement boundaries

- Apple M5, 10 CPU cores, 24 GiB RAM; macOS 26.4 (25E246), Apple Clang 21,
  Release `-O3`, Vulkan SDK 1.4.328 and MoltenVK. Existing target flags are retained.
- First 60 seconds of `baseband_553000000Hz_12-51-54_26-09-2026.wav`:
  56 MS/s complex float32, +4.625 MHz NCO, 7:1 decimation to 8 MS/s,
  8K/QPSK, rate 3/4, guard interval 1/16.
- `GpuChannelBank -> GpuPipeline -> SpectralBatch` with the common ADC clock,
  three-degree phase budget, order-8 minimum-phase response fit and both complex
  residual products. CPU FEC reads completed, coherent mapped carrier buffers.
- Five input seconds warm acquisition, buffers and plans. The timed interval is
  **54.995712 input seconds**, including decoder overlap and final flushing.
- Reads and finite-value validation, consumer checksumming, startup and output
  are excluded. Range/Doppler window accumulation and map formation, SAR imaging,
  DC projection, viewer and video encoding are **not run**. This measures the
  numerical spectral handoff, not complete video rendering.
- Runs are serial under normal desktop conditions. Two runs have scopes/device
  timing disabled except for excluded boundary work; two use exclusive wall and
  thread-CPU scopes plus device timestamps. A separate pass subdivides GPU jobs.
- The instrumented source tree is isolated under `build/perf-20261002/source`.
  `fec.cpp` is not instrumented; the FEC scopes surround calls in `CpuFec`.
  Production DSP source hashes remain unchanged throughout this profiling pass.

| Single-channel measurement | Run 1 | Run 2 | Mean, ms/input s |
|---|---:|---:|---:|
| Compute elapsed, timing off | 149.301 | 149.297 | **149.299** |
| Compute elapsed, aggregate profile | 154.541 | 155.917 | **155.229** |

Aggregate instrumentation adds **4.0%** elapsed time in this one-channel run.
The detailed timestamp pass takes 156.24 ms/s. Component tables retain
these measured values; they are not rescaled to the timing-off total.

## Component balance

All times are **milliseconds per second of input recording**. The elapsed column
partitions the synchronous path and includes host preparation and waits. GPU
device time and active calling-thread CPU time are contained in that elapsed
column: **do not add the columns**. CPU figures for GPU stages include command
recording/submission, driver calls and parameter preparation, not just DSP.

| Component | Arithmetic / control ownership | GPU device | Active main CPU | Elapsed incl. waits | Elapsed share |
|---|---|---:|---:|---:|---:|
| Shared resampling + channelization | GPU; CPU warp/NCO parameters and clock control | 37.33 | 3.24 | 55.82 | 36.0% |
| OFDM reception + pilot tracking | GPU copies/FFT/pilots/equalization; CPU tracking | 5.78 | 6.97 | 34.35 | 22.1% |
| Frame accumulation + TPS/BCH | GPU carrier copies; CPU frame/TPS/BCH control | 2.98 | 0.97 | 12.39 | 8.0% |
| Demapping + FEC | CPU | 0.00 | 38.03 | 38.06 | 24.5% |
| Reference reconstruction | GPU re-encoding/remapping; CPU packet staging/tables | 4.64 | 0.19 | 5.28 | 3.4% |
| Minimum-phase response estimation | GPU reduction/solver; CPU setup | 5.06 | 0.06 | 5.40 | 3.5% |
| Residual + complex cross spectra | GPU | 3.50 | 0.06 | 3.84 | 2.5% |
| Other pipeline control | CPU | 0.00 | 0.11 | 0.08 | 0.0% |
| **Total** | | **59.29** | **49.62** | **155.23** | **100%** |

Tiny CPU scopes have visible timing overhead; independently read wall and CPU
clocks need not give exact ordering for sub-millisecond aggregates. Scope
boundaries leave at most 0.011 ms/input s unassigned to a component.

## CPU FEC detail

| CPU stage | Active CPU ms/input s | Elapsed ms/input s | Share of active FEC CPU |
|---|---:|---:|---:|
| Hard QAM slicing + inner deinterleaving | 6.282 | 6.282 | 16.5% |
| Depuncturing + Viterbi forward recursion + traceback | 30.420 | 30.443 | 80.0% |
| Outer deinterleaving + RS checking/correction + sync validation | 1.331 | 1.332 | 3.5% |

Excluding demapping, the Viterbi/outer-RS split is **95.8% / 4.2%**,
about **22.8:1**. No GPU FEC or BCJR runs. The Viterbi figure includes
traceback and depuncturing; internal clocks are deliberately avoided because
they previously altered compiler optimization of the hot recursion.

Every single-channel run examines **328,967 RS words**, rejects none, corrects
none, and emits 51,340 new rows during the measured interval. This is clean
RS checking cost, not error-heavy correction cost. The independent CPU-only
noise tests cover correction behavior separately.

The earlier caller-only audit measured 29.89 ms/s Viterbi and 1.30 ms/s outer/RS.
The full profile measures 30.42 and 1.33; these are different runs and levels of
instrumentation, not an additional optimization or an arithmetic change.

## GPU operation detail

These are device timestamps from the **separate detailed run**, including adjacent
barriers. They subdivide GPU work, not total host elapsed time. Fused work stays
grouped. Small timestamp boundaries and changes in execution conditions mean
these figures need not sum to the aggregate-profile column above.

| GPU operation | One channel, ms/input s | Five chains, ms/shared-input s |
|---|---:|---:|
| Common ADC fractional resampling | 17.761 | 17.141 |
| Shared channelizer forward FFTs | 11.701 | 11.253 |
| Prototype filtering + spectral folding/decimation | 6.361 | 9.864 |
| Decimated inverse FFTs | 0.802 | 3.937 |
| Valid output extraction + NCO | 0.793 | 3.190 |
| Receiver history copy | 0.824 | 4.055 |
| OFDM window extraction + carrier correction | 0.939 | 4.283 |
| Batched OFDM FFTs | 1.790 | 10.976 |
| Pilot identification / channel measurements | 0.713 | 4.393 |
| Decoder equalization + TPS reduction | 0.798 | 4.110 |
| Apply tracked pilot phase | 0.715 | 2.298 |
| Frame-ring / staging carrier copies | 3.035 | 21.400 |
| Convolutional re-encoding + validity propagation | 1.114 | 5.518 |
| Interleaving/remapping, QAM, pilots and TPS regeneration | 3.186 | 15.357 |
| Response sufficient-statistic reduction | 1.791 | 9.632 |
| Iterative minimum-phase response solver | 2.896 | 14.570 |
| Reference alignment, residual + cross-spectrum products | 3.295 | 17.772 |

Acquisition CP correlation and integer-CFO search run on the GPU but do not recur
in this settled interval. Reduced pilot phase/slope tracking runs on the CPU;
its one-channel active cost is about 2.36 ms/s. Common-warp parameter preparation
uses about 1.34 ms/s, and the ensemble update about 0.10 ms/s.

Reconstruction produces frequency-domain carriers for the spectral consumer;
there is no waveform IFFT hidden in the modulation figure. These are measurements
in the actual streamed pipeline, not substituted resident microbenchmark times.

## CPU execution, submissions and waits

With detailed scopes disabled, active **process CPU time is 57.36 ms/input s**
after excluding input and consumer work. This sums CPU execution across threads;
it is about 5.74% of one CPU core at real-time input rate, not
CPU saturation. During the scoped profile it is 61.20 ms/s:
49.62 ms/s assigned to the calling thread and 11.58 ms/s in other
runtime/driver threads or unassigned execution. No per-function attribution is
claimed for the latter.

- **220.1 GPU submissions per recorded second.** End/submit handling takes
  5.44 ms/s of calling-thread CPU, already included in the table.
- Fence waits occupy **106.30 ms/s elapsed**, but only **0.73 ms/s active CPU**.
  That wait includes device execution and scheduling; it must not be added to
  either GPU time or total elapsed time.
- About **95.94 ms/s** lies outside GPU timestamps. Unlike the old all-GPU
  decoder path, this now includes **38.03 ms/s of CPU demapping/FEC**.
  It is therefore incorrect to label the entire difference as driver overhead.
- OFDM reception plus frame/TPS handling takes
  **46.74 ms/s elapsed for 8.76 ms/s GPU execution**.
  These short, synchronously completed jobs are a substantial submission/scheduling
  cost. Faster arithmetic alone will not eliminate their completion boundaries.

## Multiple channels

Five chains decode the same C31 signal from one capture, with a shared frontend
and clock. This exercises all five real decoding/reconstruction chains; it is
not a measurement of five different transmitters. It uses the first 20 seconds,
with **14.995712 seconds** measured after warmup, so final-batch/overlap costs
are not identical to those of the 60-second one-channel measurement.

| Compute measure per second of shared capture | One channel | Five chains |
|---|---:|---:|
| Elapsed, timing off | 149.30 ms | 485.62 ms |
| Elapsed, aggregate profile | 155.23 ms | 487.72 ms |
| GPU execution | 59.29 ms | 166.98 ms |
| Active process CPU, timing off | 57.36 ms | 241.03 ms |
| Shared frontend, GPU | 37.33 ms | 46.08 ms |
| Demapping + FEC, active CPU | 38.03 ms | 197.34 ms |

The two five-chain timing-off runs are **493.77 and 477.46 ms/s**, averaging
**485.62 ms/s**, approximately **2.06 times real time**.
The aggregate-profile runs are 487.48 and 487.97 ms/s; the detailed pass is
481.87 ms/s. Run variation is larger than the net timing overhead here, so a
precise overhead percentage for five chains would be misleading.

Each chain examines 93,179 words, rejects/corrects none and emits 14,552 measured
rows. Viterbi costs 157.86 active CPU ms/shared-input s across the five chains;
outer/RS costs 6.91 ms/s and demapping 32.57 ms/s. All three are still processed
on the calling CPU thread. The shared frontend grows from 37.33 to 46.08 GPU
ms/s, substantially less than a fivefold increase.

## Optimization implications

1. **Viterbi remains the main CPU target:** 30.42 ms/input s, approximately
   19.6% of profiled total elapsed. A hypothetical zero-cost Viterbi could remove
   at most that fraction before considering overlap or changes to other stages.
   The current measurements do not establish an optimal GPU decoder design.
2. **RS is now a minor cost:** outer gathering plus RS is
   0.86% of total elapsed on this direct-path recording. A dedicated GPU
   implementation has much less potential benefit than Viterbi or batching.
3. **Frontend arithmetic and GPU completion boundaries both matter.** Common
   ADC resampling and the channelizer together use 37.33 GPU ms/s; the many short
   OFDM/frame submissions contribute considerable elapsed overhead. Queueing,
   batching and overlapping independent work merit measurement before assuming
   that moving all remaining CPU code to shaders will provide the largest gain.

## Validation and reproduction

The normal Release libraries and the three isolated timing modes produce the
same full-output hash, **`1048c0a7ca676eae`**, in independent three-second runs.
The check covers decoded packets/status, measured/reconstructed carriers, masks,
fitted responses, both residual products, and phase/sample coordinates. Each
emits 2,584 rows and passes 15,843 RS examinations. No phase alignment or output
rescaling is applied for the comparison. All longer timed runs also report zero
rejections and identical checksums within each workload.

The preceding CPU changes pass **19/19 Release and 19/19 Debug tests**, including
GPU reconstruction and phase-coordinate tests, plus the expanded independent
FEC oracle under AddressSanitizer/UndefinedBehaviorSanitizer. Compiler reports,
assembly, controlled baseline comparisons and noise sweeps are documented in
[the CPU optimization report](fec_cpu_optimization.md).

Artifacts:

- [Summary JSON](../build/perf-20261002/summary.json),
  [single-channel stage CSV](../build/perf-20261002/single-stages.csv),
  [five-chain stage CSV](../build/perf-20261002/five-stages.csv).
- [Instrumentation generator](../build/perf-20261002/instrument.py),
  [driver](../build/perf-20261002/driver.cpp),
  [serial runner](../build/perf-20261002/run_profiles.py),
  [summary generator](../build/perf-20261002/summarize.py).
- Raw per-run JSON/logs, exact commands, build logs and production SHA-256 manifest
  are in `build/perf-20261002/`. The production-library check is for output
  validation; its read timing is not instrumented and is not used for throughput.

```sh
python3 build/perf-20261002/instrument.py
cmake -S build/perf-20261002/source -B build/perf-20261002/profile-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/sdk/macOS;/opt/homebrew" \
  -DFETCHCONTENT_SOURCE_DIR_VKFFT="$PWD/build/release/_deps/vkfft-src" \
  -DFETCHCONTENT_SOURCE_DIR_DL5EU="$PWD/build/vendor/gr-dl5eu-d6ab8c12c1283f144b17e4c0ca2b6727a084339a"
cmake --build build/perf-20261002/profile-build --target pipeline_profile -j 8
cmake --build --preset release --target fec_pipeline_benchmark -j 8
python3 build/perf-20261002/build_production_check.py
python3 build/perf-20261002/run_profiles.py
python3 build/perf-20261002/summarize.py
```

The runner selects the bundled MoltenVK manifest. Mode 0 disables internal timing,
mode 1 enables aggregate scopes/device timestamps, and mode 2 adds GPU substage
timestamps. Full-array hashing is confined to the validation runs and excluded
from measured compute time.
