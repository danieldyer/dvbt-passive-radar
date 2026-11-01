# FEC optimisation: candidate selection

The qualified decoder is compact soft Viterbi, RS-pinned exact trellis feedback
and bounded RS rescue. Pilot denoising precedes FEC. Rejected decision-directed
outer equalisation is not part of production: it cost more than the saved inner
passes and worsened a known-payload below-waterfall case.

The current investigation distinguishes algorithmic work reduction, state layout,
memory traffic and device mapping. CPU implementation remains idiomatic C++23;
compiler reports and assembly must substantiate vectorisation. A few fixture
words of extra acceptance do not establish a general sensitivity improvement.

The retained policy is one CPU decoder: compact soft Viterbi, up to 32 RS-pinned
passes including bounded rescue. It has no selectable hard/soft mode, GPU fallback
or decision-directed turbo equaliser. Independent hard/uniform/scalar decoders
remain under `tests/reference` only. The initial and constrained passes share
one constexpr observation codebook. Rejected RS words are decoded again only
when their actual trellis bytes change; the previous failed word is already the
necessary exact cache key, so no separate cache is introduced.

The following alternatives were measured and rejected rather than layered into
production:

| Candidate | Result |
|---|---|
| Forward-metric checkpoints | Exact, but neutral/slower except about 4% on one sparse microbenchmark. |
| Rotating CPU state layout | Register spills/shuffles made it roughly 8–10× slower. |
| Modular initial metrics | About 2× slower on M5; little benefit on PL64. |
| Exact coalescence/streamed traceback | Correct prefix commits, but ancestor work made the isolated implementation about 20% slower. |
| Radix-4 butterflies | Exact after correcting the partial-tail case, roughly 2× slower on M5 and 2–3× on PL64. |
| Hard Viterbi bootstrap then soft constrained passes | Too few SBS/Nine bootstrap words; more total CPU work despite matching accepted fixture words. |
| Broader likelihood-ranked RS Chase | More work without improved Nine acceptance. |
| RS errors/erasures without a confidence gate | Admitted inconsistent words; rejected on accuracy. A gated variant recovered no extra words and was slower. |
| Prefer 128-bit GCC vectors | About 10–13% slower than the normal Alder Lake AVX2 build. |
| GPU overlapping block Viterbi | A corrected four-state-per-active-lane version agreed with the saved Nine path at 4096-bit blocks/512-bit overlap, but took about 341 ms on PL64 versus 241 ms for CPU, including preparation/readback. On M5 it took 113 versus 103 ms. Finite overlap is also not a universal ML guarantee. |
| Multiplication-based survivor-bit packing | Exact, but 3–7% slower on M5 and 14–29% slower on PL64. |
| Precomputed branch complements | Doubled the branch table for no meaningful PL64 gain. |
| Sixteen-column padded observation table | About 1–2% faster for some PL64 punctured rates but slower at rate 1/2 and neutral on M5; no useful whole-pipeline improvement. |
| Prefer cached coherent Vulkan memory globally | Same Viterbi outputs; direct CPU test took 127–128 ms for ordinary RAM and both compatible mappings. Full replay was 676 versus 669 ms/input s, so the allocation change was reverted. |

The GPU experiment exposed a portability trap: the Intel shader reported a
32-wide subgroup but only 16 lanes were active for that workgroup. An independent
probe confirmed this. Correcting the topology fixed the tested likelihood paths,
not the unfavorable whole-device tradeoff. Viterbi would compete with range
compression for the same 15 W package budget. No unqualified shader survived the
cleanup.

Relevant primary prior art:

- [Peng et al., GPU parallel block Viterbi](https://arxiv.org/abs/1608.00066):
  separates decoding phases, exploits trellis structure and optimises memory
  layout. Their hardware/rate numbers do not qualify PL64.
- [High-throughput memory-efficient parallel Viterbi](https://arxiv.org/abs/2011.09337):
  another GPU parallelisation approach to compare before committing to one layout.
- [Tomita et al., GPU SDR Viterbi](https://globals.ieice.org/en_transactions/fundamentals/10.1587/transfun.E98.A.2246/_p):
  chunk parallelism, coalesced access and warp shuffle. Exact RS-bounded intervals
  avoid relying on an assumed finite traceback convergence depth.
- [Three-point Viterbi on CPU/GPU/FPGA](https://onlinelibrary.wiley.com/doi/10.1002/cpe.3093):
  phase separation, packing and platform-specific mappings. Use these as design
  guidance, not as evidence of a speedup for this pipeline.

Each candidate needs all-rate differential/known-payload checks, measured M5 and
PL64 runtime, and a coupled pipeline check. Retain one production implementation
only after accuracy and sustained workload measurements justify promotion.

## Retained exact subpath certificate

Let P be the selected maximum-likelihood path under the previous byte constraints.
New RS constraints only add fixed bytes. An unknown section whose fixed start
state and closing byte agree with P can keep P's subpath: any cheaper replacement
would splice into a cheaper old complete path, contradicting optimality. The
start state is the preceding byte's lower six bits. A full closing-byte match is
a conservative certificate for the six-step endpoint collapse. An open end needs
no added endpoint. Constraint revisions/removals invalidate the certificate.
Tie order is checked against the independent scalar decoder, not inferred from
path cost alone.

The initial compact Viterbi solution seeds this certificate through a private
CpuFec interface for exactly the same observations/rate. Tentative or arbitrary
bytes cannot seed it. Boundary comparisons are saved before the closing byte is
overwritten by traceback; no per-byte changed-flag buffer is needed. Existing
selected bytes and the previous constraint mask suffice. This removes redundant
ACS after RS interval splitting without shortening traceback or changing the
32-pass/rescue policy.

M5 full-fixture feedback CPU time fell from 335.75 to 81.22 ms/input s. First PL64
coupled results improved from about 1.25× to 1.38× including short-file drains;
the simpler retained implementation sustained 1.516× in a 15-minute live soak.
Every real-data feedback output was also compared against the prior decoder with
identical inputs. Different RF runs or GPU rounding are not used as proof of
FEC equivalence.

## Qualification boundary

The fixed saved three-channel workload retains Nine 117825/117840 accepted RS
words (99.98727%) per eight-second replay, with identical accepted-packet
checksums. All-rate tests cover ties, extreme likelihoods, punctures, partial
terminal bytes and added/revised/removed constraints. Address/undefined-behavior
sanitizers pass the independent feedback and posterior/rescue tests. Compiler
reports and assembly show vectorized byte butterflies and survivor packing on
M5; target reports are regenerated after the final retained changes. Serial
traceback is expected and is not evidence of a missed vectorisation opportunity.

A single fixture is not a universal waterfall or capacity claim. RS acceptance
on broadcast data is not known-payload bit-error truth. The practical policy
preserves the qualified sensitivity rather than substituting a favorable
microbenchmark or a few extra apparent RS successes for accuracy evidence.

Survivor-memory context: [architectural traceback tradeoffs](https://ieeexplore.ieee.org/abstract/document/221067),
[IBM generalised traceback framework](https://research.ibm.com/publications/generalized-trace-back-techniques-for-survivor-memory-management-in-the-viterbi-algorithm--1),
[exact coalescence-based Treeterbi](https://academic.oup.com/bioinformatics/article/23/5/545/238397),
and [FLASH Viterbi](https://arxiv.org/abs/2510.19301). General HMM trees, beam
pruning and repeated divide-and-conquer passes do not automatically improve a
64-state binary trellis. No reported paper throughput substitutes for PL64 tests.
