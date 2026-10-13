# Live pipeline optimisation and sample-loss robustness

The retained changes improve the five-channel recorded-IQ replay from **1.69× to
2.01× realtime** with one range–Doppler map at four updates/s: **591 → 497 ms per
input second**, a 19% throughput improvement across three alternating before/after
pairs. A complete simulated capture, including channelisation and saving both IQ
and complex profiles, runs at **2.59–2.65× realtime**. These are different
measurement boundaries, described below; neither is a new physical-UHD throughput
qualification.

The startup overrun was reproduced and fixed. The old executable discarded
**20,971,520 samples (0.32768 s)** during a paced ten-second simulation even though
its average processing headroom was 1.89×. With the **same half-second receive
buffer**, the new executable discarded zero samples, reached 2.64×, and used at
most 115 ms of queue. A larger buffer is useful additional protection, but was
not needed to pass this comparison.

## Retained changes

**Complete warmup before receiving.** The previous warmup covered 1,360 OFDM
symbols, about 1.48 s at guard 1/16. Stationary training needs three seconds, so
warmup never exercised range compression. Its first FFT plans were compiled
while live samples accumulated. Both original captures first lost host samples
around 4.4 s; the paced simulation reproduced the same startup failure. Warmup
now covers 4,080 symbols and actually produces profiles. All four bounded
range-compression FFT shapes are constructed upfront, so a short batch after a
gap cannot compile a new shape. Warmup observations are reset before reception.

**Avoid numerical work below floating-point resolution.** Pilot phase/timing
still uses the same weighted complex likelihood and Newton/backtracking solve.
It now stops when the predicted likelihood improvement is below double precision
instead of repeatedly searching for an unresolvable improvement. A paired CPU
comparison over 1,000 fading/coloured-noise cases measured approximately **1.57×**
speedup after the first run. Maximum differences from the previous solver were
**1.35e-8 samples** in delay and **8.13e-8 degrees** in phase. Existing analytic,
fading, traversal-order and explicit-whitening regressions pass unchanged.

**Remove unused buffers and padded work.** The frame ring retains measured
carriers and pilots. It no longer copies coarse equalised carriers that are
replaced by the subsequent frame equaliser. Ring capacity follows the configured
six-frame decoder batch instead of reserving for fifteen; frame snapshots also
allocate their actual row count. For five channels and seven frame slots these
changes eliminate approximately **905 MiB of requested Vulkan buffer capacity**.
That is an allocation calculation, not a claim that process RSS fell by this
amount: measured maximum RSS remained around 1.1 GB. The OFDM FFT batch uses the
actual guard lengths and the proven three-symbol acquisition-history bound,
instead of padding every channel for eight extra shortest-guard symbols.

**Reduce capture backpressure.** The capture loop releases consumed ADC storage
immediately after the synchronous channeliser finishes reading it, retaining the
next filter halo. Decoder, archive and display work therefore cannot retain
already-consumed input. Archive records move their owned payload into the writer
queue instead of copying it again to prepend the header; the archive format is
unchanged. The macOS capture DSP thread requests user-initiated QoS, as the
benchmark already did. The existing receive thread remains separately scheduled.

**Bound optional display work.** Live map refreshes are best-effort: when the
previous map is still busy, a refresh is skipped. Every scientific profile is
retained, and each displayed map still uses the full requested CPI and the same
phase epoch convention. Offline detection continues to deliver every requested
map. A regression deliberately blocks the display projector and verifies that
profile production returns without waiting or shortening the coherent aperture.

**Absorb transient stalls.** The default capture metadata and settings now request
a two-second ring: 512 MiB, rounded to 2.18 s at 61.44 MS/s. Explicit metadata
settings remain respected. The summary records actual ring duration and peak
occupancy, plus separate FEC and GPU component timings. A larger ring cannot
compensate for a sustained processing deficit.

The decoder still uses six TPS frames, five workers and seven slots. The FEC
algorithm, minimum-phase model/iterations, common-clock trajectory, noise
weighting, stationary cancellation, finite-waveform reference, coherent aperture
and map amplitude definitions are unchanged. No processing gain is traded for
throughput.

## Real-capture timing

`support/validation/live_pipeline_benchmark.cpp` reads the channelised IQ in
`captures/live-vhf-002/capture.dvblive`. It warms for 15 s and measures the next
20 s, a continuous part of the recording. It processes all five distinct VHF
signals and forms 80 maps for one channel, using a 3 s CPI, ±100 Hz, 0–30 km,
and a 0.25 s update interval. The source is already channelised and retimed:
frontend work is excluded. Reads are included; output archiving, the native
window, pooling and video encoding are excluded. Only the support tool's packet
checksum/validation loop is subtracted from the reported wall and CPU intervals.

The preserved old executable and the final executable ran in alternating order,
three runs each. No other project benchmarks ran alongside them. Other work on
the machine was not controlled. Values below are medians, in ms/input s,
**summed over all five channels** unless stated otherwise.

| Component | Where measured | Before | After |
|---|---|---:|---:|
| End-to-end replay | Wall time | 591.05 | 496.96 |
| Input reads, included above | Wall intervals | 105.24 | 103.04 |
| Main thread | Active CPU | 218.03 | 172.86 |
| Whole process | Active CPU | 947.32 | 923.75 |
| Viterbi, all five workers | Summed worker elapsed intervals | 674.28 | 694.32 |
| Outer deinterleaving + RS, all workers | Summed worker elapsed intervals | 31.49 | 31.25 |
| OFDM receiver, FFT and tracking kernels | GPU timestamps | 51.62 | 46.92 |
| Frame gather, equalisation and hard demapping | GPU timestamps | 80.02 | 58.32 |
| Oracle reconstruction | GPU timestamps | 26.59 | 22.88 |
| Response estimation/application | GPU timestamps | 58.21 | 48.98 |
| Noise statistics | GPU timestamps | 13.17 | 12.20 |
| Finite-waveform profiles, both moments | GPU timestamps | 89.17 | 77.31 |
| Focusing, one map channel at 4 Hz | GPU timestamps | 50.59 | 44.89 |

Before wall times were 584.2–592.2 ms/s; after times were 491.1–497.7 ms/s.
CPU, GPU and worker intervals overlap and must not be added to wall time. Worker
elapsed intervals can include descheduling and are not per-worker CPU-utilisation
percentages. The Viterbi implementation was not changed. Improvements in device
times of unchanged stages are observations of the complete schedule and working
set, not independently isolated kernel speedups.

Every measured final run delivered 93,024 new symbol-row decisions,
2,324,307 accepted RS word decisions and zero rejected words. Word counts include
overlapping decoder context. All five per-channel packet checksums agree between
final runs; per-channel hashing avoids dependence on cross-channel worker
completion order.
The other recording, `live-vhf-001`, was independently replayed through profiles:
its measured interval accepted 2,320,905 RS word decisions with zero rejections.

## Complete capture and gap tests

The simulated source is `build/live-validation/five_channels.wav`: native int16,
64 MS/s, five DVB-T channels and about ten seconds of samples. It feeds the real
receive ring, common-clock channeliser, receiver, reconstruction, profiles and
archive writer. No SDR hardware or live display was used in these runs.

| Run | Ring requested | Archive | Processing headroom | Host discard | Timestamped loss |
|---|---:|---|---:|---:|---:|
| Old executable, paced | 0.5 s | Both | 1.89× | 20,971,520 | 0 |
| Final executable, paced | 0.5 s | Both | 2.64× | 0 | 0 |
| Final executable, unpaced | 2 s | Both | 2.59× | 0 | 0 |
| Final executable, paced, injected gap | 2 s | Both | 2.65× | 0 | 1,048,576 |

Headroom is input duration divided by active DSP/archive-call wall time, including
final drain, excluding pre-stream warmup and receive waits. The old run includes
the initialisation that incorrectly occurred during reception. Its comparison
therefore demonstrates the startup fix as well as steady processing changes;
the longer real-IQ benchmark above is the steady-state comparison.
In the final unpaced run, native-int16 unpack costs 8.18 GPU ms/input s and the
shared channeliser/retimer costs 36.54 GPU ms/input s, for all five channels.
Its full DSP/archive-call interval is 385.68 ms/input s. Profile work starts only
after training, so its finite-run component totals should not be substituted for
the steady-state real-IQ table.

The injected gap starts at 4.5 s and removes exactly 1,048,576 source samples.
All five receivers reacquire once, all report zero RS failures, and both IQ and
valid complex profiles resume. The archive has two explicit capture segments and
finishes cleanly. Existing conservative recovery is retained: actual gaps break
decoder/context and coherent histories, and retrain the stationary estimate.
They are not disguised as continuous phase-coherent observations. Separate
ring-overrun tests check unread-sample preservation, exact host-loss accounting,
absolute resumed timestamps and a single new accepted segment.

## Validation and reproduction

The Release regression suite passes **37/37**. The four affected GPU paths also
pass with Vulkan synchronization validation enabled, without reported hazards.
Coverage includes short/acquisition batches, all supported FEC modes, coloured
pilot noise, multichannel clock/phase coordinates across gaps, finite correlations
beyond the guard, fixed-CPI focusing, a blocked display consumer and byte-exact
combined archives. No numerical tolerance was relaxed.

```sh
cmake --build --preset release -j6
cmake --build --preset release --target live_pipeline_benchmark -j6
export VK_ICD_FILENAMES="$PWD/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
build/release/support/validation/live_pipeline_benchmark \
  captures/live-vhf-002/capture.dvblive 35 1 1
python3 support/live/validate_capture.py \
  --fixture build/live-validation/five_channels.wav \
  --output build/new-gap-validation --gap-at 4.5 \
  --gap-samples 1048576 --archive both
ctest --preset release --output-on-failure -j1
```

Raw timings, summaries and validation logs are in
[data/performance_20261013](data/performance_20261013). `paired-summary.json`
summarises the alternating real-IQ measurements. `pilot-convergence.txt` records
the isolated CPU comparison. `profile-kernel-comparison.txt` is an **unshipped**
experiment: a template cache gained only about 3% in range compression and its
extra scratch plane was removed. The scoped changes above are what remain in
production. Performance under uncontrolled CPU/GPU contention and physical UHD
transport remains something to verify in the next real capture.
