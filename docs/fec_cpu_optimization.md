# CPU FEC optimization and auto-vectorization, 2026-10-02

The retained FEC implementation uses ordinary C++23 arrays, integer arithmetic,
`std::min`, compile-time tables and `std::bit_cast`. All FEC intrinsics and the
separate ARM implementation have been removed. Compiler inlining attributes and
two Clang loop hints prevent premature unrolling from defeating vectorization;
there are no instruction builtins, inline assembly or extended vector types.

This work compares against the already-inlined implementation in
[fec_cpu_audit.md](fec_cpu_audit.md), not the earlier, substantially slower CPU
replacement. The work does not establish a hardware performance limit.

## Reference repository audit

Reviewed `/Users/danielrd/sources/fec/include/FEC/{Convolutional,ReedSolomon,
GaloisField,Interleaver}.h`, the convolutional benchmark and associated tests.
The following distinctions matter when applying those ideas here:

| Reference idea | Treatment here |
|---|---|
| Compile-time generators, puncture masks and Hamming-distance tables | Retained; five specialized puncture schedules, shared arithmetic, exact erasures. Also tested larger combined branch tables and an erasure-free specialization; neither justified retaining another path. |
| Compile-time unrolling of encoded blocks and trellis states | Tested 4/8/16-cycle unrolling, whole-state unrolling and array-generation templates. Aggressive versions regressed, some severely. The retained code unrolls only a puncture cycle and the eight traceback steps. |
| Compact decisions and traceback | Retained eight bytes per input trellis bit and full-batch traceback. Tested alternate byte assembly and two-step/radix-4 updates; the latter passed the exact oracle but was slower. |
| Fixed traceback windows and a known initial encoder state | Not transplanted. A captured continuous DVB-T stream does not provide that initial state or frame termination. Finite-depth decoding is possible, but is a distinct algorithmic tradeoff rather than an exact-loop optimization. |
| Compile-time GF and generator arithmetic | Retained and extended with four-byte remainder slicing. The reference repository has an RS encoder, not an RS decoder to transplant. |
| Packed convolutional encoding/interleaving | Already outside this CPU decoder: reconstruction re-encoding is on the GPU. Packing the decoder's observations is not free, particularly with erasures. |

This covers the reference's principal applicable structural ideas. It does not
prove that every possible CPU algorithm or compiler transformation is exhausted.
Inter-channel SIMD, scheduling across cores and alternative traceback algorithms
remain separate, unmeasured architectural possibilities.

## Retained changes

* Ordinary C++ butterfly loops operate on byte metrics and produce two weighted
  survivor bits per byte. Two pairwise-sum loops pack the decisions; the weights
  are disjoint, so the sums cannot exceed 255. Inlining both metric helpers keeps
  the state from escaping through an outlined tail or normalization call.
* Normalize between groups of complete puncture cycles, at most 96 steps apart.
  A six-step path reaches every state and costs at most 12, bounding the metric
  spread. The common offset grows by at most 192 between normalizations, so byte
  metrics stay at or below 204. No saturation or approximate comparisons occur.
* Four-byte RS remainder slicing replaces four dependent byte updates with four
  independent table contributions. The sliced table is 16 KiB, plus the existing
  4 KiB single-byte table for tails. Loads use standard copies/bit casts, including
  unaligned inputs. Berlekamp-Massey, root search and correction remain unchanged.

The demapper variants (16-bit indices and expanded bit labels) did not justify
retention. Eight-byte RS slicing was slightly slower and used more table space.
The final implementation keeps one CPU decoder, without worker pools or a new
Vulkan FEC path.

## Compiler evidence

Apple Clang 21, C++23, Release `-O3`, existing target flags, no fast-math or new
architecture flags. `cpp-final/compiler.txt` records:

* Butterfly loop: **vectorization width 16**.
* First survivor packing loop: **vectorization width 16**.
* Second survivor packing loop: **vectorization width 8**.

Assembly contains packed byte adds, unsigned minima, comparisons, interleaving,
and pairwise addition generated from C++. The hot metrics remain in SIMD
registers; the hot loop contains no out-of-line trellis helper calls. There is no
claim that the time-recursive trellis loop itself can be vectorized over time.

Earlier C++ variants also received positive vectorization remarks but still had
avoidable state spills or scalar survivor packing. Fully unrolled versions could
turn the cyclic state into scalar values. Explicit inlining plus preserving the
small packing loops for the vectorizer was substantially better. Reports alone
were not used as a performance result.

An interim intrinsic-based variant was faster than the retained C++ by about 6%
in the isolated trellis microbenchmark. Its results are kept as experimental
evidence only; it is not the production implementation.

## Controlled component measurements

Two rounds in reverse order; each run synthesizes 1,020 OFDM symbols at rate 3/4,
warms the decoder, then times five decodes. All checked packets match known data.
Values are elapsed CPU-stage milliseconds per input second, without streamed
frame overlap:

| QAM | Baseline Viterbi | C++ Viterbi | Baseline outer/RS | C++ outer/RS |
|---|---:|---:|---:|---:|
| QPSK | 28.27 | 25.36 | 2.52 | 1.11 |
| 16-QAM | 56.49 | 50.72 | 5.03 | 2.23 |
| 64-QAM | 85.39 | 76.18 | 7.56 | 3.36 |

Viterbi takes about 10–11% less time; clean outer/RS takes about 56% less. The
all-rate microbenchmark includes both hard observations and erasures and checks
stable output hashes. The fifteen-point AWGN sweep has identical acceptance,
rejection and wrong-packet counts to the baseline. Error-heavy RS cost improves
much less: slicing accelerates parity checks, not the error-location algorithm.

## Recording measurements

First 60 seconds of the C31 recording, NCO +4.625 MHz, QPSK 3/4, 8K and GI 1/16.
Five seconds of warmup leave 54.995712 measured input seconds. Two baseline and
two final runs were interleaved, with no concurrent builds or other benchmarks.
The table averages each pair. Caller-only profiling links the same uninstrumented
`fec.o`; `CLOCK_THREAD_CPUTIME_ID` excludes GPU waits and descheduling.

| CPU stage | Baseline active CPU ms/input s | C++ active CPU ms/input s | Reduction |
|---|---:|---:|---:|
| Demapping + inner deinterleaving | 6.25 | 6.23 | unchanged |
| Depuncturing + Viterbi + traceback | **33.27** | **29.89** | **10.2%** |
| Outer deinterleaving + RS + sync validation | **2.96** | **1.30** | **56.1%** |
| Total | **42.48** | **37.42** | **11.9%** |

Excluding demapping, Viterbi/outer-RS is now **95.8% / 4.2%**, about **23:1**.
All four runs examine **328,967 words**, reject none, correct none, emit 51,340
rows and produce checksum **16720166110442452894**. These are clean-direct-path
RS timings, not worst-case error-correction costs.

Whole-pipeline compute elapsed is **158.91 -> 153.22 ms/input second**, a 3.6%
reduction, or about **6.5x real time**. Reads/validation, consumer checksumming,
visualization and output are excluded. CPU-stage savings are not additive to this
inclusive figure; GPU scheduling and submission costs remain. All frontend,
OFDM, reconstruction and response/residual kernels are unchanged and still run
on the GPU. FEC still runs entirely on the CPU.

## Validation and artifacts

All **19 Release and 19 Debug tests** pass, including reconstruction,
multichannel and phase-coordinate GPU tests. The expanded FEC oracle also passes
AddressSanitizer and UndefinedBehaviorSanitizer. The 15-point noise sweep matches
the baseline acceptance/rejection counts exactly, with no wrong accepted words.

The independent scalar exact-ML Viterbi oracle checks all five rates, hard errors,
erasures, ties, partial endings and both sides of normalization boundaries.
Additional cases cover all-zero/all-one input, complete erasures and erasure value
255. Independent polynomial division checks RS remainder lengths 0–1,024 at eight
starting alignments; libfec checks 500 random words with 0–9 errors and shortened
prefix rejection.

`build/fec-cpu-extended-20261002/` retains 53 compiled snapshots, including the
baseline and intermediate/final candidates, assembly, compiler remarks, the
exploration scripts, oracle logs, component CSVs and recording JSONs. Names
`final-*` refer to the **interim intrinsic** experiment; the retained implementation
is **`cpp-final`**. Production has only the retained implementation. The previous
`fec_portable_cpu` test was a duplicate after removing the architecture split, so
it was removed; the independent scalar oracle remains in `fec_cpu`.

The frozen baseline should not be regenerated from current production sources.
`qualify_cpp.py` rebuilds the retained candidate and caller-only profile against
existing project libraries. `measure.py baseline cpp-final` repeats the all-rate
CPU microbenchmarks. The normal `fec_benchmarks` and `fec_pipeline_benchmark`
targets exercise production.
