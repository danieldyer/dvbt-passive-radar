# Historical Vulkan soft FEC

This implementation was removed on 2026-10-02. The current decoder is
[CPU hard-decision Viterbi and errors-only RS](fec.md). Commands and implementation
details below describe the previous version and are retained as measurement history.

The Vulkan recording path now uses noise-scaled QAM likelihoods, inner
deinterleaving, depuncturing, log-MAP/BCJR, a retained maximum-likelihood trellis
path, outer deinterleaving and soft Reed–Solomon decoding. All likelihoods stay
on the GPU between stages. Corrected randomized RS packets and their validity
are the output; no MPEG decoding is needed.

The production Vulkan path also uses GPU CP acquisition, symbol FFTs, pilot
measurements and equalization, with reduced CPU tracking control feeding the
shared ADC clock. Frame accumulation and remodulation now run on shared device
storage too. `GpuFec::decode_device()` consumes those resident carrier batches
and returns corrected RS words directly to `GpuReconstruction`, with explicit
download methods for comparison tests. The production `GpuPipeline` continues
through GPU minimum-phase estimation and complex residual formation. The
architecture and coherent SAR handoff qualification are tracked in
[the pipeline plan](sar_pipeline_plan.md).

## Decoder choices

Use `--fec bcjr` (the Vulkan default), or `--fec viterbi` to compare with the
CPU soft-input Viterbi and errors-only RS baseline. `--compute cpu` selects
the latter. Explicit `--compute cpu --fec bcjr` is rejected. `metadata.json`
records the actual decoder. A 32-lane Vulkan compute subgroup with shuffle,
arithmetic and ballot operations is currently required; tested on Apple M5.

The transmitted DVB-T code is already a concatenation of RS(204,188) and a
punctured, 64-state convolutional code. Both specified interleavers must still
be inverted. The implemented approach is one pass through the concatenation,
without an iterative RS-to-convolutional-code feedback loop.

Soft-input Viterbi maximizes sequence likelihood. BCJR minimizes individual
bit error probability and provides posterior information; replacing the
Viterbi path with separate bit-MAP decisions can worsen RS packet recovery.
The implementation therefore retains the ML survivor path during the same
forward trellis sweep and supplies its bytes to RS. BCJR supplies confidence.
See [MIT's soft versus hard Viterbi discussion](https://web.mit.edu/6.02/www/f2010/handouts/lectures/L9.pdf).

## Likelihoods and trellis

Positive LLR means bit zero. The QAM demapper sums likelihoods along the
Cartesian axis, rather than taking only the nearest point. It fuses the
symbol/bit gather and all five depuncturing patterns; punctured observations
have LLR zero. Non-finite carriers become erased observations.

MAP requires a meaningful noise scale. `Symbol::noise_variance` supplies
complex noise power in raw FFT units when known (as in AWGN tests). Otherwise
the decoder estimates variance from leave-one-out scattered-pilot residuals,
after removing the known early-window phase ramp. Residual channel curvature
can inflate this estimate. This is a practical estimator, not a guarantee of
perfect likelihood calibration for every frequency-selective or impulsive
channel. Finite high-confidence LLRs are limited to magnitude 10,000.

Each subgroup owns two states per lane. The recursion includes the log-MAP
Jacobian correction; it is not max-log BCJR. A window produces 2,048 bits,
with up to 1,024 extra trellis steps on each side. Boundary states are uniform
where context is unavailable. **Finite context is an approximation** to
full-record MAP/ML; it is explicitly checked against full-sequence double
precision references, not described as mathematically lossless for all inputs.
Scratch storage is reused in batches of 128 windows, bounding its size near
67 MiB. A demapper submission borrows the carrier planes; a second submission
runs trellis/RS using owned likelihood and confidence buffers. This keeps the
input spectra immutable without copying them into decoder scratch.

For each selected byte, the decoder computes

`P(selected eight-bit word | all observations in its trellis window)`

by constraining those eight trellis branches and summing over the 64 entering
states. This retains the correlation between bits in a byte. Multiplying eight
bit posterior probabilities gave noticeably less useful RS confidence in the
measured noise sweep. Joint byte probabilities are checked independently with
a double-precision constrained-trellis calculation.

## Soft RS

An initial parallel syndrome check handles clean words. Failed words use
errors-and-erasures BM/Chien/Forney decoding. At most nine trials erase
0, 2, ..., 16 of the least reliable bytes. The full byte posterior and its
coordinate follow the exact 12-branch outer deinterleaver.

For a byte with selected-word probability `p`, the GMD reliability is
`a = max(0, 2*p - 1)`. Additional candidates must satisfy

`sum(1 - a + 2*a*[candidate byte differs from selected byte]) < 17`.

This is the generalized-distance acceptance ball for minimum distance 17.
Without it, using 16 erasures can manufacture an apparently valid codeword
from an uncorrectable input. The conventional errors-only candidate remains
available. Shortening zeros, the randomized sync byte, and all 16 syndromes
are checked. Surviving candidates are ranked by bit-marginal likelihood cost;
this score is not a full joint probability of all 204 bytes. See
[Forney's original GMD work](https://ieeexplore.ieee.org/document/1053873/).

Passing these checks is not a payload CRC guarantee. RS can miscorrect.
Qualification therefore counts accepted wrong payloads separately from
rejected blocks using known transmitted data. Failure masks retain the same
time positions and convolutional-encoder dependencies as the CPU reference.

## Measurements, 2026-09-29

All 14 Release and Debug/validation tests passed. Tests include exhaustive
enumeration of all paths for a short trellis, full-sequence bit and byte
posterior comparisons at all five puncturing rates, all 15 constellation/rate
round trips, an independent GNU Radio transmitter, and phase-coordinate tests.
The posterior tests observed maximum bit-LLR discrepancy below 0.0001;
decisions with reference magnitude above 0.0001 agreed. Forty noisy rate/SNR
cases cover window boundaries; this is evidence over tested cases, not a
worst-case finite-window bound. The RS erasure tests recovered 32 blocks that
errors-only RS rejected, including up to 16 low-confidence erroneous bytes.

Known-truth AWGN test, rate 3/4, 204 OFDM symbols per point, exact noise variance
and perfect channel knowledge (pilots/TPS kept clean):

| Modulation | Es/N0 | Examined interior packets | CPU soft Viterbi + RS failures | GPU sequence + hard RS failures | GPU sequence + joint-byte GMD failures |
|---|---:|---:|---:|---:|---:|
| QPSK | 4.5 dB | 1,091 | 131 | 131 | 117 |
| 16-QAM | 10.5 dB | 2,225 | 634 | 637 | 619 |
| 64-QAM | 16 dB | 3,359 | 235 | 235 | 210 |

No accepted wrong payloads were observed in the sweep. These are single-seed
finite samples, not a measured dB coding gain or a low-error-rate bound. The
CPU baseline uses max-log QAM while the GPU uses exact log-sum QAM, so the
middle column isolates the added GMD benefit better than the CPU column.
Bit-MAP hard decisions alone were worse in this test (e.g. 156 QPSK failures).

Warm GPU timestamp measurements, milliseconds per second of GI 1/16 input,
rate 3/4, 1,020-symbol batches, five repetitions, high SNR:

| Modulation | Demap/deinterleave/depuncture | ML + BCJR + joint byte probability | RS | GPU total |
|---|---:|---:|---:|---:|
| QPSK | 1.74 | 95.78 | 0.59 | 98.12 |
| 16-QAM | 3.01 | 190.07 | 1.17 | 194.25 |
| 64-QAM | 5.81 | 283.35 | 1.75 | 290.91 |

These exclude file reads, CPU carrier upload, result copying, remodulation and
visualization. Existing CPU demap/Viterbi/RS wall totals in the same run were
141.33, 305.04 and 487.88 ms/s respectively; their algorithms and timing
boundaries differ, so this is not a like-for-like kernel speedup. Noisy failed
RS words cost more than the clean syndrome path. The trellis is now the main
FEC compute cost.

The final ten-second C31 recording test processed two identical channel
outputs. Each had 58,091 RS examinations (including overlap), zero failures
and 9,108 fully valid symbols. `timing.csv`, `coherence.csv` and `tps.csv` were
byte-identical across channels. This tests bookkeeping for a common captured
signal; it does not establish coherence between independent transmitters.

Reproduce after sourcing the Vulkan SDK environment:

```sh
cmake --build --preset release -j 8
ctest --preset release
build/release/fec_benchmarks --curve > build/fec-curve.csv
build/release/fec_benchmarks > build/fec-kernels.csv
```

Evidence is under `build/fec-20260929/`: `awgn-joint.csv`,
`resident-final.csv`, the Release/Debug CTest logs, and
`recording-final-pair-10/verification.json`.

Code-assisted phase/timing refinement remains a possible later stage. The
present decoder does not augment the 64-state trellis with RF phase or timing
states and does not replace CP/pilot acquisition. DVB-T interleaving must be
accounted for when mapping decoder information back to OFDM coordinates.
