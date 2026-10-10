# Reconstruction, phase and timing contract

The implemented pipeline works on the original recording without rewriting it.
Filtering, resampling and oscillator corrections precede the decoder/measurement
tee. The decoder alone uses equalization. The response compensator operates on
**un-equalized, resampled** measurement spectra and the regenerated reference.

## Channelization

The complex sample is `I + jQ`. For a channel at `offset_hz` above the recording's
centre, the NCO is `exp(-j 2π offset_hz n / input_rate)`. In the Vulkan frontend,
`n` is the common corrected ADC-grid coordinate after the raw clock warp. Its
zero origin is preserved across channels and seeks. The CPU comparison frontend
mixes on the original ADC grid before its single-channel clock correction.

At 56 MS/s the Vulkan frontend uses a 1,049-tap minimax prototype, flat through
3.328125 MHz and with a stopband beginning at 3.671875 MHz. Shared 3,584-point
forward transforms and 512-point inverse transforms implement filtered 7:1
decimation; only complete convolution outputs are returned. The older CPU
comparison path retains its 1,025-tap Kaiser prototype and 57,344-point FFT.
The result indexed `m` is centred at corrected ADC coordinate `origin + 7*m`.
The first and last incomplete FIR/interpolation supports are excluded. There is
no padded-edge data presented as a valid observation and no uncompensated FIR
phase delay. This offline zero-phase operation requires lookahead.

The nominal DVB-T output rate is 8 MS/s. The active carriers are `-3408..3408` on
an 8192-point FFT, spanning approximately ±3.328125 MHz. Further anti-aliasing
inside the occupied band would discard useful signal; the designed filter leaves
that band flat. The implementation supports integer input/output rate ratios.

## Common clock correction and phase bookkeeping

The receiver uses CP acquisition followed by continual-pilot tracking. A common
Kaiser-windowed sinc interpolator, with 2048 phase intervals and interpolation
between phases, corrects sample-clock drift before channelization. Its support
depends on the occupied ADC-band span (32 taps for the supplied 56 MS/s
recording, 64 for the wider benchmark banks).
Fractional coordinates and carrier phase are maintained in double precision;
GPU coordinates use small local integer/fractional parts. Both measurement and
decoder see the same resampled samples and oscillator corrections.

Clock acquisition initially compares identical continual carriers across adjacent
symbols. After settling, their averaged channel becomes a fixed phase anchor for
a carrier PLL and a common clock-offset fit across participating channels.
This avoids accumulating small biases from differences of noisy phase estimates.
This is a **dominant stationary
reference-path assumption**, not an independent measurement of the SDR oscillator;
changes in the direct path can also affect the recovered clock. Only common phase
and linear phase slope drive the clock loop. The frequency-dependent channel is
not equalized on the measurement branch.

The useful FFT window begins 128 output samples before the CP-derived origin.
This accommodates the channelizer and interpolation support in the tested
56 MS/s configuration. The
reference receives the corresponding `exp(-j 2π k*128/8192)` phase factor. Automatic
origin acquisition estimates one additional fractional offset, after removing the
minimum-phase shape, and freezes it for that synchronization segment. This sets
the direct-path excess-delay coordinate; it is separate from the FRF estimator.
A supplied `--reference-delay-samples` disables this automatic origin estimate.

The Vulkan frontend first applies one common ADC-clock warp, then channelizes
all selected bands. For a row in `timing.csv`, `input_sample` is the original
ADC coordinate, `corrected_adc_sample` is the corrected-grid coordinate, and
`input_samples_per_output_sample` is the instantaneous ratio at the window start.
For useful-window sample `j`, evaluate the common `sample_clock.csv` map at
`corrected_adc_sample + j*decimation`. A window can cross a rate-segment boundary,
so its first ratio is not necessarily valid throughout the window. The channel
NCO uses the corrected grid, after resampling, and has its phase origin at zero.
The single-channel CPU comparison path retains its affine `input_sample + j*r`
coordinates. See [the frontend contract](vulkan_frontend.md) for details.

Writing `p` for `applied_carrier_phase_rad`, the carrier derotation is
`exp(j*(p - 2π*carrier_error_hz*j/8000000))`. Phase is stored modulo 2π; the exact
per-window correction and input coordinates are saved. `reference_origin_samples`
records the reference's complete FFT-origin adjustment. Tracker reacquisition
starts a new segment rather than implying phase continuity across an unknown gap.

## FEC-assisted reconstruction

TPS DBPSK is BCH checked over all 68 symbols. Its parameters and frame progression
must agree with symbol-pilot timing. All reconstructed TPS signs come from the
checked bits and PRBS seeds, not independent noisy carrier decisions.

The CPU decoder reads completed, host-coherent equalized carrier buffers. It
slices QAM, inverts both inner interleavers, depunctures and runs full-traceback
64-state hard-decision Viterbi. Outer deinterleaving restores RS(204,188) words;
the errors-only decoder corrects up to eight bytes, rejects roots in the shortened
prefix, verifies parity and checks the randomized sync byte. All fifteen
non-hierarchical constellation/code-rate combinations are tested. See
[CPU FEC](fec.md) for implementation, assembly review and measurements.

Uncorrectable
packets and their convolutional-encoder dependencies propagate to carrier masks.
Successful RS decoding can still miscorrect: the reference is FEC-assisted, not
infallible ground truth.

The corrected randomized packet is the reconstruction boundary. No video decode
or MPEG transport file is necessary. Corrected codeword parity and the original
outer-interleaver coordinates are restored before convolutional coding, puncturing,
inner interleaving and QAM mapping. Acquisition/history/tail regions with unknown
state stay invalid. Processing uses overlapping 1020-symbol batches, emits each
observation once, and retains context between batches. RS examination totals
include overlap and must not be interpreted as unique packet counts.

Corrected packets are also descrambled for optional SI inspection. SDT/NIT sections
are CRC checked; the example identifies C31. PCR values are logged as **program
clock** timestamps, not RF emission-time anchors. A MIP PID observation is only
reported as a candidate; this implementation does not claim MIP timestamp/CRC
validation or absolute UTC. No MIP candidate was observed in the tested interval.

## Minimum-phase response and residual

After resampling, estimate the magnitude of the averaged transfer function and
iteratively fit

`Hmin(k) = exp(c0 + Σ(q=1..Q) cq exp(-j 2π k q / 8192))`.

The complex causal cepstral coefficients allow asymmetric complex-baseband
amplitude responses. Phase comes from this minimum-phase construction, followed
by a single constant phase alignment. Observed phase slopes are not fitted by the
response estimator. A pure delay has flat magnitude and therefore remains outside
this model, as an explicit test verifies. Fits are per overlapping batch and are
saved with `--raw-output`; varying fit coefficients are not a calibrated stationary hardware model.

For valid carriers, Vulkan evaluates `E = Y - Hmin*X` and `C = E*conj(X)`. It does
not whiten or equalize `Y`. CPU and Vulkan implementations are compared on complex
inputs and multiple buffer sizes. `Q=8` is the default smooth response model;
`Q=0` permits scalar gain/phase only. Increasing Q can absorb finer minimum-phase
multipath structure. Minimum phase by itself cannot preserve every delayed echo:
for example, a dominant direct signal plus a weak delayed copy can be minimum
phase. Nonlinear transmitter distortion is not generally an LTI response either.

## Output format

The default map stream is described in [range_doppler.md](range_doppler.md).
The intermediate binary products below require `--raw-output`.

All `.cf32` files are little-endian interleaved float32 I,Q; `.u8` is byte data.
Rows correspond to `timing.csv` and are ordered by original sample coordinate.

| File | Elements per row | Meaning |
| --- | ---: | --- |
| `measured.cf32` | 6817 complex | Resampled, carrier-corrected, un-equalized received spectrum |
| `reference.cf32` | 6817 complex | Regenerated unit constellation/pilots with fixed timing origin; invalid carriers zero |
| `residual_cross.cf32` | 6817 complex | `(Y − Hmin X) conj(X)`, invalid carriers zero |
| `valid.u8` | 6817 bytes | Per-carrier FEC/reference validity |
| `reference_iq.cf32` | 8192 + guard samples | IFFT of reference with CP prepended; scale is IFFT/N |
| `residual_useful_iq.cf32` | 8192 complex | IFFT of masked residual, useful window only |
| `timing.csv` | one record | Original ADC coordinates, clock ratio, phase, reference origin, segment and full-row validity |
| `response.csv` | 6817 records per fit | Complex Hmin in active-carrier order |
| `coherence.csv` | one record | Common phase/timing residual relative to a fixed received-channel template, also used to bound coherent integration |
| `tps.csv` | one record per checked frame | Full checked TPS bits and parameters |
| `broadcast_metadata.txt` | text | Checked SI identification, PCR observations and any MIP PID candidates |
| `metadata.json` | one object | Dimensions, frequency/rate settings and aggregate status |

`reference_iq` is a synthesized cyclic extension of each aligned reference window;
it is not a promise of sample-continuous transmitter output across gaps or invalid
rows. Residual I/Q omits the CP and unoccupied carriers. The measured stream is
not rewritten on the original 56 MS/s grid. For delays exceeding the guard interval,
OFDM orthogonality no longer gives a simple per-carrier channel model; the recorded
spectrum still contains that interference, but reconstruction/cancellation accuracy
needs separate qualification.

For delay processing, place carrier `k` at FFT index `(k+8192) % 8192` and take the
inverse FFT. Delay bin spacing is 125 ns; effective resolution is limited by the
6.656 MHz occupied bandwidth. Preserve masks and segment boundaries during slow-time
Doppler integration. A delayed copy gives positive excess delay under this convention.
Bistatic excess path length is `c*delay`, not `c*delay/2`.

Fixed receiver latency cancels when measuring excess delay against the direct signal
in the same recording. Transmitter/receiver geometry is needed for a geographic
interpretation, not for this common-stream clock correction. An address or map pin
does not establish a surveyed antenna phase centre.
