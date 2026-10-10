# CPU FEC audit, 2026-10-02

Historical baseline: the later [CPU optimization report](fec_cpu_optimization.md)
replaces the intrinsic-based implementation with auto-vectorized C++23 and adds
further measurements. The figures below describe the earlier inlining fix.

The previous thirteen assembly/compilation passes did **not** establish a CPU
performance limit. This follow-up audit found a missed inlining decision in the
hottest Viterbi loop. Two explicit C++23 lambda attributes fix it; no trellis,
decision, puncturing, RS or coordinate algorithm changes.

Apple Clang 21's `-Rpass-missed=inline` says the rate-3/4 puncture-cycle lambda
costs 1430 against an inline threshold of 569. Assembly confirms a function call
once per three decoded bits, with captures and metrics materialized in memory.
`[[gnu::always_inline]]` on the cycle and per-step lambdas removes that call and
lets the compiler retain loop state in registers. The fix is in `src/fec.cpp`.

## Evidence for the improvement

All figures below were measured on the M5 using Release `-O3`, without changing
architecture flags. The isolated baseline and candidate use identical data,
production headers and libraries. Two rounds reverse their measurement order;
each constellation has one warmup and five measured decodes of 1,020 symbols.

| Rate 3/4 | Previous Viterbi, ms/input s | Fixed Viterbi, ms/input s | Reduction |
|---|---:|---:|---:|
| QPSK | 46.64 | 28.09 | 39.8% |
| 16-QAM | 93.36 | 56.95 | 39.0% |
| 64-QAM | 141.19 | 84.85 | 39.9% |

All known-data checks passed. The scalar exact-ML oracle checks every puncture
schedule, ties, erasures and partial-byte endings. The actual recording gives
54.82 -> 33.60 **active CPU ms/input s** for Viterbi, a 38.7% reduction. Its
328,967 examined RS packets, including frame overlap, still produce checksum
`16720166110442452894` with no rejected or corrected packets.

This is direct evidence of an improvement, not proof that other CPU layouts,
larger-radix recursions, inter-stream SIMD or multi-core scheduling cannot improve
it further. No lower bound from hardware counters or instruction throughput has
been established. The earlier statement about avoiding missed vectorization
opportunities was too broad: SIMD instructions were present, but the surrounding
inlining decision still mattered substantially.

## Active CPU balance on the recording

Same C31 input: first 60 seconds of
`baseband_553000000Hz_12-51-54_26-09-2026.wav`, 56 MS/s, NCO +4.625 MHz, QPSK 3/4,
GI 1/16. Five seconds warm up the pipeline; 54.995712 input seconds are measured.
The table averages two candidate runs. Thread CPU time uses
`CLOCK_THREAD_CPUTIME_ID`, so GPU waits and descheduling are excluded.

| Decoder stage | Active CPU ms/input s | Elapsed ms/input s | Share of these CPU stages |
|---|---:|---:|---:|
| QAM slicing + inner deinterleaving | 6.30 | 6.30 | 14.7% |
| Depuncturing + Viterbi + traceback | **33.60** | **33.62** | **78.3%** |
| Outer deinterleaving + RS + sync validation | **3.00** | **3.00** | **7.0%** |
| Total | **42.90** | **42.92** | **100%** |

Viterbi costs **11.2 times** outer-deinterleaving/RS. Excluding the demapper,
the split is **91.8% Viterbi / 8.2% outer+RS**. At real-time playback the decoder
stages consume about 4.29% of one CPU core; that is not a measurement of the
whole process's CPU usage.

RS requires zero corrections in this particular recording: its cost is principally
outer-byte gathering and checking the valid words. The figure is not a worst-case
RS decode cost. The earlier known-truth noise sweep in `fec.md` demonstrates
substantially higher RS cost when correction is required.

The CPU-only synthetic cases, without streamed frame overlap, give the following
fixed implementation totals. They are elapsed CPU-stage times, rather than the
active CPU measurements in the recording table:

| Rate 3/4 | Demap, ms/s | Viterbi, ms/s | Outer+RS, ms/s |
|---|---:|---:|---:|
| QPSK | 5.21 | 28.09 | 2.56 |
| 16-QAM | 12.24 | 56.95 | 5.06 |
| 64-QAM | 16.17 | 84.85 | 7.54 |

## Measurement limits and next decision

A fully instrumented Viterbi body changed generated code and slowed the inlined
candidate by roughly 19%. Those internal timings are **not** used for the final
balance. Instead, the coarse profiler instruments only the `CpuFec` caller and
links the exact same uninstrumented `fec.o` as the timing baseline. The coarse
profile's recording-stage wall time agrees closely with its active CPU time.

Whole-pipeline compute elapsed was 158.60 ms/input s uninstrumented and
156.72–160.53 ms/s with coarse profiling: approximately **6.3x real time**.
File read/validation, consumer checksumming, maps, output and video are excluded.
The previous CPU version rerun during this audit took 179.82 ms/s; normal GPU
scheduling variation means the decoder's 39% reduction does not translate to a
39% whole-pipeline reduction. These runs are serial, without concurrent builds or
other benchmark work.

Viterbi is the clear first candidate for a Vulkan experiment. Under the explicitly
idealized assumption that the other stages and scheduling stay fixed, removing
its approximately 34 ms/s entirely would improve the current roughly 159 ms/s
pipeline by about **1.27x**. Removing outer+RS alone would improve it by less than
2%. These are bounds for this one-channel workload, not GPU performance forecasts;
more channels and higher-order QAM increase the Viterbi work.

The forward trellis has a time-step dependency. A GPU port needs measured useful
parallelism and validated traceback/batch boundaries; SIMD use on CPU alone does
not predict a GPU speedup. This audit does not implement a new Vulkan decoder.

## Artifacts and reproduction

`build/fec-audit-20261002/` contains:

* Frozen `baseline`, `inline`, `profile`, `inline-profile`, `outer` and
  `inline-outer` variants, assembly, inlining remarks and known-data executables.
* `prepare.py`, `outer_profile.py` and `audit.hpp` for the isolated builds.
  `production-sha256.json` identifies the pre-fix source snapshot; the frozen
  baseline remains available after applying the fix.
* `*-component-{0,1}.csv`, `*-cpu-{0,1}.jsonl`, `*-real-*.json` and `summary.json`.
* Release/Debug CPU tests, Release GPU phase/reconstruction tests, and
  AddressSanitizer/UndefinedBehaviorSanitizer logs.

The regular `fec_benchmarks` and `fec_pipeline_benchmark` targets use the fixed
production implementation. The audit-only instrumentation stays out of production.
