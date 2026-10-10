# CPU DVB-T FEC

The [extended CPU optimization report](fec_cpu_optimization.md) documents the
current implementation: one auto-vectorized C++23 decoder, no FEC intrinsics,
and four-byte RS remainder slicing. It supersedes the earlier
[inlining audit](fec_cpu_audit.md). Measurements below describe the initial CPU
replacement and are retained as history.

The production decoder is C++23 hard-decision Viterbi followed by errors-only
RS(204,188). BCJR, byte posteriors, GMD/erasure trials and all four GPU decoding
shaders have been removed. BCJR previously supplied confidence only: the pilot,
carrier and common ADC-clock loops do not use its trellis and are unchanged.

`fec.hpp`/`fec.cpp` contain the independent convolutional/GF/RS primitives.
`cpu_fec.cpp` supplies DVB-T hard QAM slicing, inner deinterleaving, puncturing
and outer-byte coordinates. `CpuFec` has host-symbol and completed mapped-OFDM
entry points and owns reusable scratch storage. Both compute modes use it.
`GpuReconstruction` stages the corrected words and retains GPU re-encoding,
QAM/pilot/TPS reconstruction. Measured spectra and all sample/phase coordinates
remain untouched; invalid packets invalidate every dependent reference carrier.

There is no runtime dependency on an external FEC decoder. GNU Radio's libfec
remains an independent test oracle. The local `sources/fec` implementation was
used as a style reference, especially its compile-time code parameters and field
arithmetic; this implementation is newly written, with dynamic bounded batches
rather than fixed-size message templates.

## Decisions and boundaries

* K=7, 64 states, generators 0x4f/0x6d in the incoming-bit-at-LSB convention.
  Each of the five puncturing schedules is a template instantiation. Punctured
  bits and non-finite carrier observations are erasures, not guessed zeros.
* Unknown initial and final states; full survivor traceback over each batch.
  There are no independently decoded trellis windows or stitch boundaries.
  Frame-ring overlap and outer-interleaver acquisition masks retain their existing
  sample coordinates. No convolutional-code termination is assumed.
* Byte path metrics are exact Hamming distances modulo a common offset. A
  six-step path can reach every state with cost at most twelve. Subtracting the
  minimum at most 96 steps apart bounds metrics at 204, so byte arithmetic cannot wrap.
  Ties select the lower predecessor, and the terminal tie selects the lower state.
* Survivor storage is eight bytes per decoded bit: about 74 MB at QPSK 3/4 for
  1,020 symbols, up to 259 MB at 64-QAM 7/8. One decoder reuses this storage across
  channels. This buys exact full-batch ML decisions rather than finite-depth
  traceback approximations.
* RS uses compile-time primitive-field and generator tables, a cheap remainder
  screen, then Berlekamp–Massey, Chien search and Forney correction only for words
  with nonzero parity. Search is restricted to the 204 transmitted positions;
  unmatched roots reject a word. Corrected parity and 0x47/0xb8 sync are checked.
  Failed RS decoding leaves its input word unchanged.
* This intentionally gives up soft-decision performance near the noise threshold.
  RS is not a CRC and can miscorrect beyond its eight-error radius. No MPEG-TS
  extraction is needed for reconstruction, and no extra integrity claim is made.

## Initial compiler and assembly review

Apple Clang 21, Release `-O3`, C++23, without fast-math or host-specific `-march`.
Thirteen successive compiler/assembly experiments are saved under
`build/cpu-fec-20261002/pass*/`. Reports used `-Rpass`, `-Rpass-missed` and
`-Rpass-analysis` for loop vectorization, plus SLP reports. `assembly.py` in that
folder reproduces reports using the production compile commands.

The initial trellis stayed scalar. Separating its butterflies, making branch
costs constexpr, making the cost loads contiguous and packing byte flags exposed
SIMD operations, but Clang still scalarized survivor packing in several variants.
The initial ARM implementation used an explicit NEON compare/pack primitive.
That architecture split has since been removed: ordinary C++ butterfly and
pairwise-sum loops now auto-vectorize, with reports confirming 16-byte arithmetic
and 16/8-byte packing widths. The current report includes the retained and
rejected experiments, assembly and new measurements.

RS's original overlapping sixteen-byte shift/XOR also compiled into scalar byte
loads and updates. Expressing the 16-byte LFSR as two `uint64_t` lanes generates
64-bit shifts, paired table loads and XORs; no ISA-specific RS code is needed.
QAM dimensions are template parameters, eliminating dynamic axis/Gray-code loops.
The remaining scalar table gathers, traceback dependency chain and irregular
error-locator work are data dependent. This review did not establish an overall
CPU limit: the subsequent inlining audit linked above found a substantial gain.

## Initial validation

All 20 Release tests and all 20 Debug tests (with Vulkan validation) passed.
The FEC oracle tests also passed AddressSanitizer and UndefinedBehaviorSanitizer.
After removing unused soft-QAM/noise helpers, the affected CPU reconstruction and
FEC tests were rerun in both configurations. The independent QAM test now checks
the hard slicer against GNU Radio over a grid including decision boundaries,
plus NaN/infinity erasures.

* Exact independent scalar Viterbi comparison: all five punctures, random errors,
  erasures, ties, short/partial-byte endings and normalization boundaries, on both
  the then-separate NEON and portable implementations. The current single C++
  implementation retains the independent scalar oracle and expands its boundary
  cases; see the current report for validation results.
* Independent libfec parity/correction comparison: 500 random words, 0–9 injected
  errors each, plus rejection of a root in the shortened prefix.
* Independent GNU Radio transmit chain comparison, all fifteen constellation/rate
  combinations, damaged-word masks and mapped-buffer preservation.
* Existing GPU remodulation, receiver, common-clock, two-transmitter SAR and
  inside/beyond-guard-interval phase tests retained.

Known-truth AWGN sweep at rate 3/4 (204 symbols, excluding 16 edge packets at
both ends) found zero wrongly accepted words. At QPSK 6 dB, 16-QAM 12 dB and
64-QAM 18 dB it rejected 612/1091, 2187/2225 and 2494/3359 words respectively.
At 8/14/20 dB respectively it rejected none in this finite test. These are test
points, not claimed universal decoder thresholds. Error-heavy RS work costs more
than the clean screen; raw results are in `noise-curve.csv`.

## Reproduction

```sh
cmake --build --preset release
ctest --preset release
build/release/fec_benchmarks
build/release/fec_benchmarks --curve
cmake --build build/release --target fec_pipeline_benchmark
build/release/fec_pipeline_benchmark INPUT.wav 60 1
```

The pipeline benchmark uses the supplied C31 recording parameters (NCO +4.625 MHz,
7 MHz bandwidth, GI 1/16). Its channel argument repeats that same signal to measure
scaling; it does not fabricate independent transmitters. Five seconds warm up the
pipeline. File read/validation and consumer work are subtracted, and no video,
map or archive is produced. CPU FEC values are elapsed time in CPU-only stages;
GPU values use device timestamps. Neither is additive to the already inclusive
compute elapsed time.

## Initial M5 measurements, 2026-10-02

The first 60 seconds of `baseband_553000000Hz_12-51-54_26-09-2026.wav`, minus
five seconds warmup: 54.995712 measured input seconds, QPSK 3/4, 51,340 emitted
rows. Old and new runs examine **328,967 RS words**, including overlap, and
produce the **same corrected-packet checksum** `16720166110442452894`, with
**zero rejected words**. The new decoder needs zero RS corrections; the old
windowed GPU decoder corrects 45,337 bytes to obtain those same packets. This
observation does not establish a general hard-over-soft decoding advantage.

| Work | Previous GPU-FEC version, ms/input s | New version, ms/input s | New execution owner |
|---|---:|---:|---|
| QAM slicing + inner deinterleaving | included below | 6.20 | CPU |
| Depuncturing + Viterbi + full traceback | included below | 54.73 | CPU |
| Outer deinterleaving + RS | included below | 2.96 | CPU |
| **Combined FEC** | **142.31 GPU** | **63.89 CPU** | **CPU** |
| Shared resampling/channelizer | 36.42 | 36.78 | GPU |
| OFDM | 5.64 | 5.82 | GPU |
| Frame copies | 2.93 | 3.04 | GPU |
| Re-encoding/remapping | 1.80 | 5.24 | GPU |
| Minimum-phase fit + residual spectra | 4.91 | 9.84 | GPU |
| **Entire pipeline compute elapsed** | **251.33** | **185.00** | **CPU + GPU + submission/waits** |

New values average two final runs, 185.26 and 184.73 ms/s. This is **5.41× real
time**, versus 3.98× for the old version rerun during this change: **26.4% less
compute elapsed time**, or 1.36× throughput. FEC takes **55.1% less stage time**.
The GPU remodulation/response stages are slower in the CPU-fed schedule; these are
measured device times, and their cause has not been isolated. Consequently the
whole-pipeline gain is smaller than simply subtracting the FEC savings. No CPU
process/thread accounting was repeated here; CPU columns measure elapsed time in
synchronous CPU-only decoder work, not a new active-CPU profile.

Five duplicate C31 chains over 20 seconds, minus five seconds warmup, take
**633.01 ms/input s** (1.58× real time). Combined CPU FEC is 333.76 ms/s; the shared
GPU frontend is 45.37 ms/s. All 465,895 examined words pass. These channels are
processed sequentially by the shared CPU decoder; no worker pool is introduced.
The previous five-chain measurement was 981.53 ms/s (historical, not rerun here).

Clean known-truth component measurements at rate 3/4, 1,020 symbols, five warm
repeats, use actual batch duration rather than streamed overlap accounting:

| Constellation | CPU demap, ms/s | CPU Viterbi, ms/s | CPU outer/RS, ms/s | CPU FEC total, ms/s |
|---|---:|---:|---:|---:|
| QPSK | 5.10 | 46.16 | 2.53 | 53.80 |
| 16-QAM | 11.85 | 92.36 | 5.03 | 109.24 |
| 64-QAM | 15.98 | 138.54 | 7.56 | 162.07 |

All packets checked against known transmitted data pass, with zero incorrect
accepted packets. Trellis input size scales with constellation and code rate.
The production frame overlap means these component numbers should not be
substituted directly for streamed per-input-second cost.

Artifacts: `build/cpu-fec-20261002/{final-60s-1.json,final-60s-2.json,old-60s.json,
final-20s-five.json,synthetic.csv,noise-curve.csv,summary.json}`. The old benchmark
links preserved pre-change libraries; production GPU FEC sources are removed.
