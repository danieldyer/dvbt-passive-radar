# Vulkan frontend and common sample clock — 6 October 2026

The default frontend fuses the common sample-clock correction with channel
synthesis. One forward FFT serves all selected channels; each channel's inverse
FFT applies the fractional delay and feeds a short local drift correction. The
minimum-phase response fit still occurs after retiming. See the
[6 October optimization and regression report](resampling_optimization_20261006.md)
for current measurements. File reads and video encoding are excluded.
The [current full-pipeline CPU/GPU breakdown](performance_breakdown_20261006.md)
separates frontend arithmetic, control, input copying and synchronization.

## Filter specification and channel spacing

The current prototype preserves ±3.328125 MHz and rejects frequencies from
±3.671875 MHz outward. At 56 MS/s it has **1,049 real coefficients**, designed
offline by Remez and rounded to float32. Its passband amplitude error is below
5e-6 and its stopband maximum is about −106 dB. GPU blocker tests require at least
100 dB rejection and measure approximately −105.8 dBFS for full-scale tones.

Australian DVB-T allocations are normally spaced 7 MHz apart. The 8K, 7 MHz
waveform has 6,817 carriers at 976.5625 Hz spacing, so the outer carrier centres
span 6.65625 MHz. Equal-offset adjacent allocations leave 343.75 kHz between
their nearest carrier centres. These are carrier locations, not brick-wall
transmitter spectra. Allocated frequency offsets can alter the actual spacing;
the filter mask above is expressed relative to the selected centre, and does
not promise full rejection of a closer or overlapping transmission.
Sources: [ACMA, Table C.1 and section 4.2.8](https://www.acma.gov.au/sites/default/files/2021-02/Digital_Terrestrial_Television_Broadcasting_Planning_Handbook_including_technical_and_general_assumptions.pdf),
[ETSI EN 300 744, Table E.4](https://www.etsi.org/deliver/etsi_en/300700_300799/300744/01.06.02_60/en_300744v010602p.pdf).

The ideal 12-bit quantization-noise design point is about −74 dBFS over the full
56 MS/s complex Nyquist band, or −83.26 dBFS integrated over 6.65625 MHz. The
100 dB rejection requirement provides about 17 dB margin for a full-scale
blocker. A blocker-to-wanted ratio of 80 dB is not added to the ADC's full-scale
SNR: those quantities have different references. This is an integrated-band
noise criterion, not a guarantee that every coherent spur lies below the noise
in an arbitrarily narrow analysis bin.

MATLAB's `NumTapsPerBand=12` means 12 coefficients per branch, or 3,072 prototype
coefficients for 256 bands. That convention alone does not explain our length:
1,049 coefficients are still roughly 150 per decimation phase here. At fixed
rejection, the available transition width and passband ripple determine the
required branch length. [MathWorks documents this bandwidth tradeoff](https://www.mathworks.com/help/dsp/ref/dsp.channelizer-system-object.html).

Design comparisons at 56 MS/s, retaining the same 5e-6 stopband bound:

| Passband amplitude error | Prototype coefficients |
| --- | ---: |
| 5e-6 | 1,049 |
| 1e-4 | 905 |
| 1e-3 | 779 |

For anti-alias protection alone, an 8 MS/s output permits a stopband beginning
at 8−3.328125 = 4.671875 MHz. A 273-tap design meets that wider mask, but leaves
substantial adjacent-channel energy in the output. An analytic check of the
current rectangular 8,192-point demodulator FFT gives about −34.66 dBFS leakage
into all occupied bins for a full-scale tone at 3.671875 MHz plus half a carrier
spacing. That is a worst-case tone test, not a measured adjacent DVB-T waveform.
The wider mask is therefore not enabled; it would require additional downstream
rejection. The current strict mask is conservative, not a proof of minimum
cost for every possible multistage receiver design.

`scripts/design_channel_filter.py` reproduces the checked-in float coefficients
with SciPy 1.13.1 / NumPy 2.0.2. Neither is a build or runtime dependency. Rates
8–64 MS/s use those tables; higher integer multiples of 8 MS/s through 128 MS/s
use the conservative shared Kaiser designer. All supported rates are checked
against the mask. The older CPU file-processing path has its previous prototype;
the exact GPU cross-check is the scalar convolution in `tests/frontend_reference.hpp`.

## Implementation

The common-clock path uses overlap-save channel synthesis with arbitrary centre
frequencies. At 56 MS/s, each batch contains 800 overlapping 3,584-point forward
transforms. VkFFT reads the windows directly from the raw ADC buffer without a
packing pass. A shared real prototype response is evaluated at the frequency
appropriate to the common clock rate, rather than independently warping channels.

Each channel's 1,024-point inverse FFT synthesizes a 16 MS/s intermediate grid.
Filtering, spectrum selection, the block-centre fractional-delay phase ramp,
radix-4 inverse transform, residual interpolation and NCO correction share one
compute kernel. Intermediate samples stay in workgroup memory. Only the final
8 MS/s channel samples are written to global memory. The inverse transform
replaces the former 512-point inverse transform; this adds no FFT round trip.

Slow clock drift leaves at most 0.002 intermediate samples of residual shift in
the fast path. A seven-sample centred differentiator plus second difference
approximates that shift with measured complex error below 5e-6 through the occupied
band. Larger drift uses a 16-tap interpolator with measured error below 1e-6 through
the rejection edge. Coefficients are generated offline; no runtime fitting occurs.

Windows crossing a published clock segment use the original full-band interpolator
before their shared forward FFT. The same path handles a filter transition reaching
ADC Nyquist, where cropped spectral interpolation would discard wrapped bins.
Its support is sized for the outermost selected carrier. These exceptional windows
retain the original finite interpolation error bound and are not covered by the
fast-path speedup figures. A normal batch uses one submission/fence and persistent
mapped buffers. Clock segments and physical RF metadata remain stage-independent.

The explicitly unclocked path retains the existing 512-point inverse transform
and spectral alias folding. Scalar convolution and interpolation references in
the tests provide independent checks. All interpolation is a finite, measured
approximation to the bandlimited operation.

## Coherence contract

For corrected ADC coordinate `n`, the original recording coordinate is

```
u(n) = n + offset + (n - segment_start) * (rate - 1).
```

`sample_clock.csv` records the piecewise-affine segments. They meet continuously
and are never changed after publication. Updates activate beyond the previous
batch's right filter halo. Overlapping windows therefore use the same time map.
Every channel NCO is referenced to **n**. Fused retiming explicitly includes the
centre-dependent phase term needed when evaluating a mixed channel at **u(n)**.
Absolute phase uses compensated double precision; GPU coordinates separate integer
positions from small fractional shifts. No cumulative float NCO recurrence occurs.

Continual-pilot timing measurements are referred back through the applied map
before entering the estimator. Differential measurements acquire the clock;
one-second quadratic fits of anchored timing track its drift. Per-channel
constant offsets keep propagation delay out of the rate estimate. New channels
join the existing offset reference. Residual-weighted fits contribute to the
common rate; stale observations are excluded. Individual demodulators no longer
resample their own streams in the Vulkan pipeline. Carrier derotation remains
per-channel and is exported explicitly.

The default budget is **3° RMS**, configurable with `--clock-phase-degrees`.
The corresponding timing budget is `3/360 * Fs / occupied_frequency_span` ADC
samples: about 0.00959 samples for a 48.65625 MHz span at 56 MS/s. Fitted curvature
limits the prediction horizon to 25–100 ms; updates can occur at each processing
batch. This is an estimator design budget, not certification of an unknown
transmitter clock or propagation environment. `coherence_qualified` remains
false in metadata; measured residual timing/phase diagnostics are still exported.

`timing.csv` includes the original ADC coordinate, instantaneous input/output
ratio, corrected ADC coordinate, carrier phase and carrier frequency correction.
For an FFT window spanning a clock-segment boundary, use `sample_clock.csv` for
the exact coordinates of each sample rather than extrapolating its first ratio.

## Historical resident kernel measurements — 29 September

Apple M5, Release, MoltenVK; 8 warmup and 64 measured batches, GPU endpoint
timestamps. These exclude input generation, host copies, file reads, decoding,
visualization and encoding. They measure the frontend workload, not whole-pipeline
throughput. Values vary with GPU clock and other system load.

| Output channels at 8 MS/s | Channelizer only | Common ADC resampling + channelizer |
| ---: | ---: | ---: |
| 1 | 29.84 ms / signal second | 38.08 ms / signal second |
| 5 | 34.23 ms / signal second | 47.43 ms / signal second |
| 7 | 36.05 ms / signal second | 53.62 ms / signal second |

The five-channel frontend therefore consumes about 4.7% of one second of GPU
execution per second of 56 MS/s input. This does not imply the whole receiver
runs at that speed: synchronization, FEC and subsequent processing remain.

```
cmake --build --preset release --target frontend_benchmarks -j 8
build/release/frontend_benchmarks
build/release/frontend_benchmarks --stages
```

`--stages` adds intermediate timestamp queries; endpoint-only measurements are
preferred for the total. CSV artifacts are in `build/frontend-20260929/`.

The separate five-channel stage run measures 48.24 ms per signal second:

| GPU stage | ms / signal second |
| --- | ---: |
| Common raw ADC resampling | 22.68 |
| Shared forward FFT | 9.63 |
| Prototype response and alias folding, all channels | 9.22 |
| Decimated inverse FFTs | 3.65 |
| Valid-output extraction and NCO | 3.07 |

This identifies raw interpolation as the largest remaining frontend kernel.
The full and channelizer-only runs have different cache/clock behaviour, so
subtracting their independently measured totals is not a substitute for stage
timestamps.

## Validation

All ten CTests pass in both Release and Debug, with Vulkan validation enabled
for the Debug run. The tests cover scalar double convolution against
GPU output, prototype masks at all supported rates, strong blockers, arbitrary
channel centres, seeks, tails, NCO coordinates beyond 2^40, and interpolation
against analytic tones. Shared-clock tests additionally cover changing drift,
late channel admission, dropout, different fixed offsets, segment boundaries,
and different chunk boundaries through overlapping FIR supports.

The known three-channel waveform spanning 48.65625 MHz has maximum phase error
below 0.001° and scalar/GPU disagreement below 2e-8. A dense band/phase sweep
measures raw interpolation error below 2.5e-6 for the tested 32-, 64- and 192-tap
designs. The analytic OFDM phase test
uses the production pre-channelization warp, then verifies the exported NCO,
carrier correction and FFT coordinates against an independent finite-DFT oracle,
including a delayed Doppler-shifted echo.

The first ten seconds of the C31 recording yield 9,108 fully validated symbols
and no RS failures. After the first second, reconstruction diagnostics show
0.46° RMS common carrier-phase variation and 0.0016 output samples RMS timing
variation. These are single-channel recording measurements; they do not establish
coherence between independent real transmitters.

A subsequent 60-second run sent the same channel through two branches sharing
the raw ADC warp. Both yielded 55,076 fully valid symbols and zero RS failures
in 352,359 examinations including overlap. Their timing and coherence CSV files
were byte-identical. After the first second the phase standard deviation was
0.447° and timing standard deviation was 0.00136 samples at 8 MS/s. This checks
scheduler independence; distinct real-transmitter ensemble accuracy remains
unmeasured. Evidence: `build/frontend-20260929/common-clock-final-60-pair/verification.json`.
