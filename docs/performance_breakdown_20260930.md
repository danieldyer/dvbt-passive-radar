# Vulkan pipeline execution breakdown — 30 September 2026

Historical GPU-FEC measurements. The [2 October profile](performance_breakdown_20261002.md)
covers the current CPU FEC implementation and supersedes this execution breakdown.

The then-current raw-IQ-to-SAR compute path takes **248.2 ms per recorded second**
for one C31 channel, or approximately **4.03 times real time**, excluding input
read/validation, startup and output work. This is a compute-only rate, not a
file-processing or video-rendering rate. GPU execution accounts for about
**194.6 ms/s**. The principal arithmetic cost is now convolutional decoding and
its soft information, followed by the frontend and RS candidate decoding.

All waveform and carrier-array DSP runs on Vulkan. CPU work consists of reduced
pilot/clock tracking, TPS/BCH decisions, coordinate/table preparation, buffer
management, command recording/submission and runtime/driver work. The CPU
comparison implementations are not invoked by this processing path.

## Workload and measurement

- Apple M5, 4 performance plus 6 efficiency CPU cores, 24 GB RAM; macOS 26.4,
  build 25E246. Release C++23, MoltenVK and the existing project dependencies.
- First 60 seconds of the supplied 553 MHz recording: 56 MS/s complex float32,
  NCO +4.625 MHz, 7:1 decimation to 8 MS/s, 8K/QPSK, rate 3/4, GI 1/16.
- Production `GpuChannelBank -> GpuPipeline` with the ensemble clock enabled,
  three-degree clock budget, order-8 response fit, BCJR/ML plus soft RS, resident
  reconstruction and both complex residual products.
- A minimal consumer checks RS status and samples a checksum. No carrier-array
  download, archive writing, range/Doppler map, SAR image formation, DC projection,
  viewer or video encoder runs. Consumer checking is timed and excluded.
- The first five input seconds warm the plans, buffers, acquisition and decoder.
  The measured interval is 54.9957 seconds. Decoder overlap and final flushing
  remain included. Output batches can carry context from before the timed
  interval; this is not an infinite-stream asymptotic measurement.
- Two runs with instrumentation disabled and two with exclusive wall/thread-CPU
  scopes plus GPU timestamps. A separate detailed pass adds timestamps between
  GPU stages. Scopes subtract nested work, including callbacks into later stages.
  Timed input reading and finite-value validation are removed from both wall and
  CPU totals. GPU timestamps are device execution intervals, not host-call clocks.
- Instrumentation is confined to `build/perf-20260930/source`; production DSP
  sources are unchanged. Jobs ran serially under normal desktop conditions.

| Single-channel measurement | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| Compute elapsed, instrumentation off | 248.84 ms/s | 247.61 ms/s | **248.22 ms/s** |
| Compute elapsed, instrumented | 252.29 ms/s | 252.66 ms/s | **252.47 ms/s** |

The scoped/timestamp profile adds about **1.7% elapsed time**. The detailed
timestamp pass measured 254.04 ms/s. All component tables below are instrumented
measurements; their small differences from the uninstrumented rate are retained,
not silently rescaled away.

## Additive component elapsed times

Units are **milliseconds per second of the input recording**. Elapsed columns
partition the synchronous compute path. GPU time is contained within elapsed
time; do **not** add them. Main-thread CPU time counts active CPU execution,
including command/driver calls, but not sleeping on a fence. It is also not an
additional elapsed-time term. Other runtime/driver threads are discussed below.

| Component | Numerical work / remaining CPU work | GPU execution | Main-thread CPU | Elapsed incl. host/waits | Share of elapsed |
|---|---|---:|---:|---:|---:|
| Shared resampling + channelization | GPU; CPU warp/NCO parameters and clock control | 35.81 | 3.32 | 53.85 | 21.3% |
| OFDM reception | GPU history, FFT, pilots, equalization and phase application; CPU tracking | 5.70 | 7.05 | 34.84 | 13.8% |
| Frame accumulation + TPS/BCH | GPU carrier copies; CPU frame/BCH control | 2.98 | 1.00 | 11.96 | 4.7% |
| Demapping + FEC | GPU; CPU tables and reduced metadata | 143.32 | 0.28 | 144.06 | 57.1% |
| Reference reconstruction | GPU encoding/remapping; CPU TPS/table preparation | 1.81 | 0.13 | 2.17 | 0.9% |
| Minimum-phase response estimation | GPU reduction/solver; CPU setup | 2.47 | 0.06 | 2.73 | 1.1% |
| Residual + complex cross spectra | GPU | 2.51 | 0.06 | 2.78 | 1.1% |
| Other pipeline control | CPU | 0 | 0.12 | 0.08 | <0.1% |
| **Total** | | **194.61** | **12.00** | **252.47** | **100%** |

The sub-millisecond CPU figures are approximate: short scopes and CPU-clock reads
make instrumentation overhead visible. In particular, tiny independently read
wall and CPU intervals should not be interpreted as exact timing bounds.

## GPU kernels in more detail

These subdivide the GPU column, measured in the separate detailed run; they must
not be added to the preceding table. Timestamp endpoints include the associated
device barriers. Fused work is labelled together rather than assigned invented
sub-kernel timings.

| GPU operation | GPU ms / recorded second |
|---|---:|
| Common ADC fractional resampling | **17.180** |
| Shared channelizer forward FFTs | **11.336** |
| Prototype filtering and spectral folding/decimation | **6.060** |
| Decimated inverse FFTs | 0.790 |
| Valid output extraction and NCO | 0.763 |
| Receiver history copy | 0.821 |
| OFDM window extraction and carrier correction | 0.933 |
| Batched OFDM FFTs | 1.791 |
| Pilot identification/channel measurements | 0.699 |
| Decoder equalization and TPS reduction | 0.792 |
| Apply tracked pilot phase | 0.712 |
| Frame-ring/staging carrier copies | 2.978 |
| QAM likelihoods, inner deinterleaving and depuncturing | **1.928** |
| ML sequence + log-MAP/BCJR + joint-byte probabilities | **112.096** |
| Outer deinterleaving access and parallel RS syndrome screening | **0.708** |
| RS errors/erasures trials and candidate selection | **28.481** |
| Convolutional re-encoding and validity propagation | 0.444 |
| Interleaving/remapping, QAM, pilots and TPS regeneration | 1.363 |
| Response sufficient-statistic reduction | 1.183 |
| Iterative minimum-phase solver | 1.272 |
| Reference alignment, residual and cross-spectrum products | 2.529 |

Approximately 0.07 ms/s remains in GPU stage boundaries and timestamp intervals.
CP acquisition and integer-CFO search also run on the GPU, but do not recur in
the settled, continuously locked interval. Their absence from this table does
not mean acquisition runs on the CPU. The initial fixed-origin adjustment uses
reduced statistics on the CPU once per acquisition and is outside the warm
interval as well.

The trellis entry is one fused kernel with sequential trellis sweeps: the profile
does not separately time its ML forward recursion, backward recursion and byte
posterior calculation. Remodulation reconstructs frequency-domain carriers; it
does not perform an unnecessary waveform IFFT for the SAR spectral handoff.

## Why real-recording RS costs more than the clean benchmark

A separate run over the same interval counted **328,967 RS examinations**, including
decoder overlap. **42,246 (12.84%)** required correction, with **45,337 corrected
bytes** and at most three bytes corrected in any accepted word. There were **zero
rejected words**. Almost every corrected word ran all nine candidate trials;
666,927 trials were counted overall, including clean screening results.

Thus zero final RS failures does not imply negligible RS work. The detailed real
run spends about **29.2 ms/s** in screening plus candidate decoding. The clean
synthetic QPSK benchmark spends only **0.60 ms/s** there.

Fresh isolated high-SNR measurements, rate 3/4, GI 1/16, 1,020-symbol batches,
five warm repetitions, GPU timestamps:

| Modulation | Demap/gathers | Trellis + soft information | RS | FEC total | Reconstruction |
|---|---:|---:|---:|---:|---:|
| QPSK | 1.90 | 95.17 | 0.60 | **97.67** | 1.53 |
| 16-QAM | 3.70 | 189.03 | 1.17 | **193.90** | 2.82 |
| 64-QAM | 6.76 | 283.99 | 1.75 | **292.51** | 4.49 |

These are per **second represented by the decoded batch**, without streaming
overlap, file operations, host transfers or other pipeline stages. The settled
recording run decodes about 1.173 seconds of symbols per measured input second,
including overlap and interval-boundary effects. This explains most of the
95.17-to-112.10 ms/s trellis difference. Additional RS candidate work explains
most of the remaining FEC difference. Synthetic 16/64-QAM timings do not establish
real-recording throughput at those modulations or under heavier error conditions.

## CPU work and synchronization

With timing scopes disabled, total **process CPU execution**, after excluding
input and the consumer, is **22.02 ms per recorded second**. This is CPU time
summed over threads, not the 248.22 ms elapsed time. With profiling enabled it is
25.50 ms/s: 12.00 ms/s assigned to the calling thread and 13.50 ms/s in other
runtime/driver execution or work outside the scoped regions.

The calling thread's non-GPU work includes:

- Reduced continual-pilot phase/timing tracking: about **2.4 ms/s**.
- Common-warp parameter preparation: about **1.4 ms/s**; ensemble updates add
  approximately **0.1 ms/s**.
- Buffer/descriptor and oscillator-anchor preparation, command recording,
  submission, query retrieval, TPS/BCH and frame management.

The profile records **222 GPU submissions per recorded second**. Submission/end
handling uses about **5.7 ms/s** of main-thread CPU. Fence-wait calls contain
241.2 ms/s of elapsed time but only 0.77 ms/s of CPU time. **The 241.2 ms/s is not
extra work to add:** it includes waiting for the already-counted GPU kernels.

Total elapsed outside GPU timestamp intervals is approximately **57.9 ms/s**.
It includes host work, submission/fence/driver latency and scheduling. It should
not all be described as CPU computation, nor as pure GPU arithmetic. OFDM history,
FFT processing, phase application and frame copies are short GPU jobs, each with
a synchronous completion boundary; this is where much of the elapsed overhead
appears.

An independent eight-second `sample` run, excluded from the timing averages,
shows `vkWaitForFences` on the application thread and Metal command/completion
dispatch queues on background threads. This supports the runtime interpretation
but does not precisely apportion all 13.50 ms/s among individual driver routines.
The application is not CPU-core saturated.

## Multiple channels

Five simultaneous decoder/reconstruction chains were measured on the first
20 seconds, excluding the first five. All five select the same C31 signal, so
each runs real decoding with the same error workload. This measures scaling of
independent chains sharing one capture/clock; it is **not** five independent real
transmitters, and the shorter interval has different batch-boundary costs.

| Compute-only measure, per second of shared capture | One channel | Five chains |
|---|---:|---:|
| Elapsed, instrumentation off | 248.22 ms | 981.53 ms |
| Elapsed, instrumented mean | 252.47 ms | 993.86 ms |
| GPU execution | 194.61 ms | 878.14 ms |
| Process CPU execution, instrumentation off | 22.02 ms | 50.73 ms |
| GPU resampling + channelization | 35.81 ms | 45.33 ms |
| GPU FEC | 143.32 ms | 749.04 ms |

FEC grows to **85.3% of GPU execution** and **75.8% of compute elapsed time**
in the five-chain case. The shared frontend grows much less than the decoders.
Current compute-only throughput is therefore about four times real time for one
channel, and approximately real time for these five chains. These are measurements
of the present implementation, not a claim about the M5's hardware limits.

A separate synthetic frontend benchmark uses distinct, arbitrarily offset
channels spaced 7 MHz apart within the 56 MS/s capture:

| Output channels | Channelizer alone, GPU ms/s | Shared clock resampling + channelizer, GPU ms/s |
|---|---:|---:|
| 1 | 26.10 | 38.00 |
| 5 | 30.68 | 48.06 |
| 7 | 36.09 | 55.12 |

This spans a wider ADC passband than C31 alone and uses different input/cache
conditions; its interpolation work and substage timings need not match the file
pipeline. The displayed totals are directly measured, not sums of independently
warmed timings. The standalone 8 MS/s resampler benchmark also remains available,
but is not the production common-ADC resampling path and is not included twice.

## Implications

1. **Trellis work is the largest arithmetic target:** 112.1 ms per input second
   in the recorded stream, before the RS stage. Optimize this with the existing
   posterior/known-truth checks; the profile does not justify dropping soft
   information or silently replacing log-MAP with max-log.
2. **RS candidate work is a substantial second FEC cost:** 28.5 ms/s here.
   The workload is mostly lightly corrupted words running nine trials. Any
   reduced search must preserve the existing candidate/acceptance behavior or
   explicitly qualify a changed decoding rule.
3. **Submission boundaries are a separate latency target:** roughly 58 ms/s
   lies outside the GPU intervals. Chaining compatible operations and reducing
   unnecessary waits would address this; merely accelerating the small shaders
   does not remove their host completion cost.
4. **The frontend is now shared and relatively modest:** 35.8 ms/s of GPU time
   for C31, of which 17.2 ms/s is clock resampling. Remodulation, minimum-phase
   fitting and residual arithmetic are each only a few milliseconds per second.

No pipeline arithmetic, decoder decisions, filter design or batching policy was
changed during this profiling pass. The older CPU-heavy report also included
map/output work and different algorithms, so its roughly two-times-real-time
file rate is not a like-for-like denominator for the new compute-only rate.

## Validation, artifacts and reproduction

Three-second runs with instrumentation disabled, aggregate timing enabled, and
detailed GPU timing enabled have identical full-output hashes for measured
carriers, reconstructed carriers, masks, fitted responses, both residual
products and phase/sample coordinates. They each emit 2,584 symbols and pass
15,843 RS examinations. No phase alignment or rescaling was applied. All timed
recording runs also report zero RS failures.

- [Summary JSON](../build/perf-20260930/summary.json)
- [Single-channel stage CSV](../build/perf-20260930/single-stages.csv)
- [Five-chain stage CSV](../build/perf-20260930/five-stages.csv)
- [RS activity](../build/perf-20260930/rs-activity.json)
- [FEC kernels](../build/perf-20260930/fec-resident.csv),
  [frontend kernels](../build/perf-20260930/frontend-detail.csv)
- [CPU stack sample](../build/perf-20260930/cpu-stacks.txt)
- [Isolated instrumentation generator](../build/perf-20260930/instrument.py),
  [driver](../build/perf-20260930/driver.cpp),
  [serial runner](../build/perf-20260930/run_profiles.py),
  [summary generator](../build/perf-20260930/summarize.py)

The folder also contains exact commands, raw per-run JSON/logs, build logs and
a SHA-256 manifest of the production DSP sources. The isolated tree is generated
from those sources; instrumentation locations are checked before replacement.

```sh
source build/sdk/setup-env.sh
python3 build/perf-20260930/instrument.py
cmake -S build/perf-20260930/source -B build/perf-20260930/profile-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$VULKAN_SDK;/opt/homebrew" \
  -DFETCHCONTENT_SOURCE_DIR_VKFFT="$PWD/build/release/_deps/vkfft-src" \
  -DFETCHCONTENT_SOURCE_DIR_DL5EU="$PWD/build/vendor/gr-dl5eu-d6ab8c12c1283f144b17e4c0ca2b6727a084339a"
cmake --build build/perf-20260930/profile-build --target pipeline_profile -j 8
cmake --build --preset release --target frontend_benchmarks fec_benchmarks \
  reconstruction_tests response_tests -j 8
python3 build/perf-20260930/run_profiles.py
python3 build/perf-20260930/summarize.py
```

The driver timing modes are 0 (scopes/device timing off except excluded boundary
work), 1 (exclusive host scopes and aggregate GPU timing), and 2 (detailed GPU
timestamps as well). Full-array hashing is confined to the validation runs;
it is not charged against the compute measurements.
