# Coherent multi-channel SAR handoff

The complete raw-IQ-to-cross-spectrum Vulkan pipeline now has a physical
SAR handoff and a distinct-transmitter end-to-end qualification. The API,
CLI, archive and coordinate conventions are in [sar_handoff.md](sar_handoff.md).
The [6 October scheduling report](pipeline_concurrency_20261006.md) describes
bounded CPU FEC workers, resident input, current profiling and added regressions.
The output stops before target detection, tracking, autofocus or image formation;
no video, display scaling, DC suppression or CPI windowing belongs in this path.

## Required architecture

1. Ingest timestamped raw complex samples, recording their nominal sample rate,
   recording RF centre, original sample index and all selected channel centres.
2. Apply a single continuous ADC clock warp before channelization. Combine
   usable per-channel timing observations with separate static timing offsets.
   Preserve the agreed three-degree phase-error design budget across the full
   occupied span; export the exact warp rather than only an approximate ppm.
3. Run the shared Vulkan channel bank. Channel oscillators retain their common
   corrected-grid origin and explicit frequency. No independent per-channel
   resampler may silently introduce another time grid.
4. Acquire and track OFDM symbols, FFT and equalize for decoding on Vulkan.
   Keep the measurement FFT separate and unchanged. CPU work here should be
   small control decisions using reduced GPU observations, not waveform DSP.
   Export every applied phase/frequency correction and symbol sample coordinate.
5. Run demapping, interleaving inverses, trellis/FEC and reconstruction on device
   storage. Regenerate BCH-validated TPS and known pilots. Preserve erasure
   masks, unavailable history, stream discontinuities and convolutional memory.
6. Estimate the constrained minimum-phase response only after common retiming.
   Do not fit or subtract an arbitrary delay/all-pass response or multipath
   impulse response. Keep fitted responses and reference-origin choices explicit.
7. Form complex cross spectra and the reconstruction residual, with enough
   reference normalization and coordinate information to backproject different
   channels on one physical time/frequency grid. Hand off these complex arrays,
   validity and metadata directly; no display-oriented range/Doppler processing.

## Coherence contract to verify

A common ADC clock alone is insufficient evidence for the final SAR interface.
Each channel currently has its own carrier tracker and reference-origin/gain
choices. Those transformations must be either removed in the SAR products or
explicitly represented so the SAR consumer can recover a common convention.
Absolute RF frequency is required for carrier-phase backprojection, in addition
to the OFDM baseband carrier index. OFDM frames from different transmitters need
not have identical symbol epochs; timestamps must preserve those differences.

The final synthetic qualification must contain distinct RF channels, independent
transmitter phases and frequency offsets, one shared ADC clock error, known
direct paths and delayed/Doppler-shifted echoes. The handoff must reproduce the
expected excess-path phase at each channel's actual RF carrier frequency and
retain phase through batch boundaries, seeks and common-clock updates. Identical
duplicate-channel outputs are a useful bookkeeping test but do not meet this
broader gate. Acquisition loss must be an explicit discontinuity or invalid
interval, never fabricated phase continuity.

The relative direct-path reference can remove each transmitter's nuisance phase.
Any unknown physical inter-channel calibration must remain explicit; receiver
clock precision does not itself prove surveyed absolute RF phase or propagation
delay. The implementation must not advertise an unverified absolute coherence
claim simply because all single-channel tests pass.

The handoff must also specify its measurement operator: useful-window sample
coordinates, active-carrier selection, reference waveform and unavailable
intervals. A per-symbol delay phase ramp alone is not an exact echo model beyond
the guard interval. Qualification should include echoes on both sides of that
boundary and use the actual reconstructed reference across symbol boundaries;
retaining complex spectra is not by itself evidence that a circular OFDM delay
model is valid for the requested SAR processing.

The signal-model and multiband calibration discussion in
[Brandewie and Burkholder (2022)](https://doi.org/10.1049/rsn2.12263)
is a useful external check for this handoff: it distinguishes the direct-path
baseline from the target path and treats coherent combination of separate
transmitters explicitly. The implementation's normalization must be derived
and verified for its reconstructed-reference convention, rather than assumed
equivalent to a separately received reference antenna.

## Current authoritative state, 2026-09-30

| Requirement | Current evidence / remaining work |
|---|---|
| Shared raw warp and channelization | Production Vulkan; scalar and phase tests. `next_device()` now exposes the resident bank buffer and exact block coordinates. |
| GPU OFDM reception | Production `GpuReceiver`: device history, CP/integer-CFO acquisition, continuous oscillator, batched VkFFT, pilot detection/equalization, pilot phase tracking and shared-clock feedback. Analytic moving-echo phase test, synthetic independent channels, real ten-second duplicate-channel TPS/FEC qualification. |
| FEC | CPU hard-decision Viterbi and errors-only RS consume mapped, BCH-aligned OFDM batches. GPU reconstruction consumes the corrected packet span; see [current decoder](fec.md). |
| Reconstruction/remodulation | Production `GpuReconstruction` re-encodes and remaps corrected RS words, pilots and BCH-valid TPS on device. All fifteen constellation/rate combinations and validity masks checked against CPU references; independent GNU Radio TX comparison retained. |
| Minimum-phase fit | Production `GpuResponse`: compensated carrier reductions and magnitude-only causal-cepstral iteration, compared with the double-precision CPU reference through order 32. Fixed origin calibration stays separate. |
| Residual cross spectrum | Production shared-context cross spectrum and normalized target/direct ratio. `SarBatch` exposes resident products, declared physical RF, exact common-clock map and reference/phase metadata. `dvbt_sar` bypasses the legacy visualization consumer. |
| Full multi-channel coherence | Distinct-transmitter raw-IQ qualification checks finite-window moving-echo templates inside and beyond the guard interval, independent symbol epochs/oscillators, common ADC drift and batch boundaries. Physical calibration uncertainty remains explicit. |

`GpuOfdm` uses a known 128-sample early FFT-window ramp when interpolating pilot
channel estimates. Removing that known ramp before interpolation avoids treating
the window shift as rapid channel variation; it does not remove delay from the
measurement spectrum. Its raw FFT test compares every carrier to an independent
analytic spectrum, including large positive/negative CFO corrections and sample
coordinates above 2^40. Current observed maximum relative error is 3.45e-6.
The numerical component evidence is `build/ofdm-20260929/ctest-release.log` and
`ctest-debug.log`. Streaming qualification is in `build/receiver-20260929/`.
Neither set proves the complete SAR handoff described above.

## Streaming receiver qualification

The Vulkan production path now consumes `GpuChannelBank::next_device()` directly.
One receiver batches all channels on the same device/context. It retains partial
symbols between arbitrary input blocks and exports an absolute corrected-grid
sample plus every applied oscillator and pilot-phase correction. CP acquisition
resolves fractional CFO; continual-pilot correlation resolves integer carrier
offset. Reduced continual pilots drive the carrier loop and the shared ADC clock.
There is no independent per-channel resampling. Acquisition gaps invalidate lock
and produce an explicit discontinuity after the 160-symbol settling interval.
The CPU GNU Radio synchronizer remains the comparison path.

`receiver_tests` checks two independent carrier offsets of +35585.25 and
-23137.125 Hz, different symbol delays, blocks as short as 137 samples, an epoch
above 2^40, TPS/FEC and an input gap. It compares raw carriers with a separate CPU
FFT using the exported coordinates and phase, including post-FFT corrections.
`phase_tests --receiver` injects unknown 5 ppm ADC error, fractional delay, CFO
and a delayed 29 Hz echo. The common-clock loop converged to 5.00058 ppm; settled
timing variation was 0.0523 degrees RMS across the occupied band. The independent
closed-form Doppler/clock oracle measured 4.15e-6 relative RMS carrier error.
These numbers describe these synthetic cases, not a universal noise/clock bound.

The ten-second C31 run, with two simultaneous duplicate channels and no map
processing, produced 8,972 fully valid symbols per channel and zero RS failures.
The duplicate case checks bookkeeping, not coherent combination of distinct RF
transmitters. The current CPU FEC decoder reads completed mapped carrier batches;
OFDM, frame accumulation, remodulation, minimum-phase estimation and residual
formation use the shared device context. The host and mapped-OFDM decoder entry
points are compared packet for packet, with both input carrier planes checked for
preservation. This supersedes the earlier resident BCJR/GMD implementation; see
[the CPU FEC report](fec.md).

The 17-test Release and Debug/Vulkan-validation suites passed after receiver
integration. The subsequent resident FEC change passed its four affected OFDM,
FEC, noisy known-truth and reconstruction tests in both builds; logs are
`fec-device-release.log` and `fec-device-debug.log` in the same evidence directory.
Warm FEC GPU totals remained 97.15/192.72/289.89 ms per input second for
QPSK/16-QAM/64-QAM, rate 3/4, in `fec-kernels.csv`; this excludes remodulation,
host transfers, file reads and display processing.

## Resident reconstruction, 2026-09-30

`GpuFrames` maintains bounded per-channel rings and one shared contiguous batch
buffer. A GPU reduction combines TPS carriers, leaving only one complex TPS
observation per symbol for CPU BCH/frame control. Batches contain at most fifteen
68-symbol frames and retain two frames of decoder/interleaver context. Emission
ranges prevent duplicate output rows; input gaps and failed TPS explicitly split
segments. Equalized carriers and FEC likelihoods are never downloaded in this path.

`GpuReconstruction` uses two kernels: independent convolutional re-encoding from
the corrected RS stream, followed by puncturing/bit-symbol interleaving/QAM mapping
and pilot/TPS insertion. Invalid packets and unavailable history invalidate every
dependent carrier. The reference is canonical transmitted carrier data; receive
window phase, channel response and delay are not silently folded into it.

`reconstruction_tests` checks all fifteen constellation/rate combinations against
the CPU reconstruction and known transmitted carriers. A serial bit-validity
oracle checks deliberately erased RS words, including byte/trellis boundaries;
bad BCH signalling is rejected. `receiver_tests` chains independent channel
carrier offsets through ring wrap, FEC and GPU reconstruction and checks output
overlap for duplication or loss. The unchanged ten-second C31 pair has 57,335
RS examinations per channel, zero failures and 8,972 valid symbols. Timing,
coherence, TPS and response outputs are byte-identical to the CPU-remodulation
run. Evidence is in `build/reconstruction-20260930/`.

All eighteen Release and Debug/Vulkan-validation tests passed. Additional
continuity tests replay an input gap and corrupt one channel's TPS: that channel
rejects precisely the affected frame and starts a new segment, while the other
channel retains all three reacquired frames. The final targeted checks are in
`continuity-release.log` and `continuity-debug.log`. Warm reconstruction timings
are reproducible with `build/release/reconstruction_tests --benchmark` and saved
in `remodulation-kernels.csv`; they exclude FEC, transfers and file/output work.

## Resident response and numerical core, 2026-09-30

`GpuResponse` reduces valid measurements with compensated floating-point sums,
then runs the same magnitude-only coordinate iteration as the CPU reference in
one workgroup. It reconstructs phase from the causal cepstrum and aligns only a
scalar phase. The optional fixed reference-origin adjustment uses reduced
statistics once per acquisition segment; it is exported separately and is not
a delay parameter in the response estimator. Masks remain at their original
sample/carrier coordinates.

The resident outputs are the explicitly aligned reference X, fitted response H,
legacy cross spectrum `(Y-HX) conj(X)`, and normalized ratio `(Y-HX)/(HX)`.
The latter cancels the fitted direct-path gain/phase. X, H, the canonical
transmitted carriers, validity and window/oscillator metadata remain available;
the normalized array alone is not the complete physical SAR handoff. The complex
cross product is invertible to the measured spectrum where its reference is
valid, so a consumer need not assume a circular within-symbol delay model.

`GpuPipeline` composes the shared-context receiver, frame rings, FEC,
reconstruction, response and residual kernels. Its callback receives borrowed
device buffers before any legacy output work. `process_recording()` now uses
this core. Small TPS/tracking/origin control stays on the CPU; bulk DSP does not
make a host round trip.

`response_tests` checks orders 0, 1, 8 and 32 against the double-precision CPU
reference (observed maximum relative difference 7.95e-6), empty/partial masks,
pure-delay rejection, explicit origin sign and a delayed moving echo with
independent gain phases (maximum complex echo error 2.54e-6). The real C31 pair
retains zero RS failures and 8,972 valid symbols/channel; its fitted response
differs from the CPU path by at most 2.97e-6 relatively, and its fixed reference
origin by 1.95e-8 samples. Evidence is in `build/response-20260930/`.

`phase_tests --pipeline` runs the full core with a 5 ppm clock error, fractional
delay and moving echo, recovers the measured spectrum from the exported complex
cross product and reference, and compares it with the independent analytic
waveform oracle. Observed relative RMS error was 4.17e-6. All twenty Release and
Debug/Vulkan-validation tests passed. Kernel-only fit/apply timings are recorded
in `response-kernels.csv` and reproduced with `response_tests --benchmark`.
The warm order-8 measurement was 3.90 ms of fitting and 2.75 ms of residual
formation per input second (1,020-symbol batches, five repetitions, resident
inputs). This includes both cross products and aligned-reference output, and
excludes CPU control, file reads and the legacy output consumer.

The final integration step was the physical SAR handoff and its qualification:
declared RF frequencies, common physical time and phase conventions, exported
reference/measurement operator, and distinct transmitters with shared ADC drift
and echoes both within and beyond the guard interval. Neither the current
normalized product nor duplicate-channel equality proves those requirements.

The implemented handoff preserves these distinctions:

- Declared physical transmitter RF centre is separate from the NCO offset and
  estimated receiver CFO. Receiver oscillator error must not silently become
  an assumed physical transmitter-frequency shift.
- Window sample coordinates are on the common corrected ADC grid: bank origin
  plus decimation times the channel sample index. Export the exact clock map and
  every applied per-window phase/frequency correction; absolute UTC is optional
  and currently uncalibrated.
- A reference origin of `d` means `X[k]=S[k] exp(-j 2 pi k d/N)`. The canonical
  symbol's direct-arrival useful-start coordinate is window start plus `d`.
  Keep this fixed per-segment coordinate choice separate from H's coefficients.
- Preserve canonical transmitted symbols, TPS guard interval, masks and the
  original measured spectra as well as the derived products. In particular,
  zero residual at an invalid reference carrier does not mean a zero measurement.
- Test the exact finite-window/reference-waveform operator across symbol
  boundaries. Delays beyond the guard interval cannot be qualified merely by
  applying a phase ramp to the current symbol's canonical spectrum.

## Physical SAR handoff qualification, 2026-09-30

`SarBatch` attaches explicit per-channel RF declarations to the resident spectral
buffers and validates their common time grid. `SarWriter` is an optional output
boundary: it saves measured Y, canonical S, reference validity, H, checked TPS,
phase/window coordinates and the immutable clock trajectory. C and R are
available directly on device and derivable from the archive without storing
redundant planes. `dvbt_sar` runs this path unpaced with no map/video consumer.

The synthetic qualification starts from one 24 MS/s raw IQ file containing two
independently encoded 7 MHz transmitters, with different RF offsets, initial
phases and symbol epochs. A common 713.5 Hz receiver LO error is deliberately
kept out of the declared transmitter RF frequencies. The ADC has unknown 5 ppm
initial error and 0.2 ppm/s drift; the production ensemble clock loop publishes
65 continuous warp segments over the 1.48-second fixture. The production bank,
receiver, frame rings, CPU Viterbi/RS, GPU reconstruction, order-8 minimum-phase fit and
residual kernels all run unchanged.

Moving echoes have excess delays of 37.25 and 700.75 channel samples (GI 512),
and Dopplers of +29 and -43 Hz, including envelope range migration. The test
uses the exported reconstructed waveform across actual symbol boundaries to
form finite-window matched templates. It checks known transmitted carriers,
validity, emitted-row continuity across two decoder batches, complex target
amplitude/phase and coherent addition between transmitters. It also deliberately
uses an incorrect RF convention as a sensitivity check.

For 43 settled observations per channel, recovered target amplitudes were
0.0399963 and 0.0400003 for injected amplitude 0.04. Per-window phase RMS errors
were 0.0573 and 0.0889 degrees, with maxima 0.1028 and 0.1449 degrees. Aggregate
phase errors were -0.0105 and +0.00294 degrees. The incorrect RF assignment
produced a -22.5-degree error on the longer-delay echo. This qualifies the
numerical convention in the simulated cases; it does not establish surveyed
absolute RF phase for the real receiver or all possible propagation channels.

The coordinate/archive unit test additionally checks sample indices above 2^42,
a warp transition inside a receive window, signed RF bins, clock continuity,
invalid-reference retention, independent binary parsing and atomic no-overwrite
behavior. The existing frontend and receiver tests retain seek/gap coverage.
Evidence is under `build/sar-20260930/`; reproduce with `sar_tests --gpu` and
`ctest --preset release` / `ctest --preset debug`.

All 22 tests passed in Release and Debug, with Vulkan validation enabled for
Debug and no reported validation errors. A two-second C31 export starting at
one second produced 1,632 new symbols per channel for two duplicate channels.
An independent archive parser verified the nonzero origin (56,000,000 ADC
samples), 87 continuous clock segments, record lengths, masks, context/emission
bounds, completion marker and byte-identical spectral/coordinate data across
the duplicate channels. `c31-pair-seek-verification.json` records that check;
the distinct-transmitter synthetic test supplies the broader coherence evidence.
