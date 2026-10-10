# Complex SAR handoff

`GpuChannelBank -> GpuPipeline -> SarBatch` is the spectral interface. OFDM and
residual formation use the shared Vulkan context; CPU Viterbi/RS reads completed
mapped carrier buffers and returns corrected packets to GPU reconstruction. `SarBatch`
adds physical frequency declarations and a snapshot of the immutable common
ADC clock trajectory. Consume its borrowed buffers before returning from the
pipeline callback; another batch may reuse them. Only rows
`[frame.first, frame.last)` are new observations. Other rows provide context
for FEC and for reference waveforms crossing symbol boundaries.

Frame decoding runs on bounded CPU workers; reception and common-clock feedback
continue independently. Each `process`, `drain` or `finish` call supplies its own
consumer. The callback runs synchronously on that call's thread and may deliver
frames accepted by an earlier call. Capture order is preserved within a channel;
channels can complete independently. `drain` completes queued work without
finalizing partial TPS frames; `finish` flushes the final complete frames and
queued work. Call `finish` before declaring an archive complete. Consumer and
input buffer lifetimes do not extend implicitly across calls.

The channel bank accepts original ADC samples from a host span or a resident
`BufferView` using `process_clocked_device`; both return the same borrowed
`ChannelBatch` coordinates and invoke the same numerical synthesis. Resident
input requires complete raw halo/padding and the shared context. See
[the resident-input and scheduling contracts](pipeline_concurrency_20261006.md).

The optional `dvbt_sar` command writes this handoff without running maps, video,
DC removal, detection or integration. It runs as fast as processing permits:

```sh
build/release/dvbt_sar \
  /Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav \
  --recording-rf-hz 553000000 --channel 557625000,4625000,16 \
  --seconds 2 --output capture.sar
```

Repeat `--channel RF_HZ,NCO_HZ,GUARD_DENOMINATOR` for other channels. The RF
frequency is an explicit declaration, not a measurement derived from receiver
CFO. CFO contains receiver LO error as well as transmitter offset. A nominal
RF declaration carries that uncertainty into physical carrier-phase prediction.
The command refuses to replace an existing output file. Omit `--seconds` to
process the remaining recording. Decoding starts after acquisition and settling;
unavailable intervals are not synthesized. A channel yielding no decoded frames
makes the command return status 2.

## Time, frequency and complex conventions

Let `N=8192`, `Fo=8e6`, `D=input_rate/Fo`, `m=window.sample`, and signed carrier
index `k=carrier-3408`. All symbols keep their individual epochs; different
transmitters need not have aligned frames.

* Window start on the common corrected ADC grid: `n=origin+D*m`.
* Relative time since the recording's sample zero: `t=n/input_rate`.
* Original recording coordinate at corrected coordinate `n`: `u=n+offset_at(n)`.
  The clock trajectory is piecewise affine and continuous. Evaluate its segment
  at **each coordinate**, including a window crossing a segment boundary.
* Channel NCO: `exp(-j*2*pi*nco_hz*n/input_rate)` on the corrected grid, before
  the receiver's additional carrier correction.
* Measured spectrum: unnormalized forward DFT of 8192 channel samples multiplied
  by `exp(j*window.phase-j*2*pi*window.frequency_hz*j/Fo)`. These applied
  corrections are exported; the measured plane is not equalized for decoding.
* Declared physical carrier frequency: `channel.rf_hz+k*Fo/N`.
* Canonical reference `S`: transmitted active carriers, with validity mask.
  Its direct-arrival useful-symbol origin is modelled at `m+d`, where
  `d=estimate.reference_origin` in channel samples. This is one explicit fixed
  coordinate per acquisition segment, separate from the minimum-phase response.
* Aligned reference `X=S*exp(-j*2*pi*k*d/N)`; fitted response `H`.
* Residual cross spectrum `C=(Y-H*X)*conj(X)` and normalized product
  `R=(Y-H*X)*conj(H*X)/max(abs(H*X)^2,1e-30)`.

Invalid reference cells have zero `X`, `C` and `R` in the device products. Their
measured `Y` is still retained. They must not be interpreted as zero received
energy. H and X retain amplitude/phase normalization and weighting information;
where valid, `Y=H*X+C/conj(X)`.

These are relative, direct-path-referenced coordinates. No UTC epoch, surveyed
propagation delay, absolute RF phase calibration, or transmitter oscillator
calibration is inferred from them. The three-degree parameter is a clock-loop
design budget, not a claim that every physical calibration error is bounded by
three degrees. Independent transmitter clock error remains a modelling limit
of assuming one nominal 8 MS/s waveform grid after common ADC retiming.

## Finite measurement operator

The output retains 6817 active carriers from each rectangular 8192-sample FFT
window. Samples outside those windows and out-of-band FFT bins are not included.
Acquisition/decoder gaps and partial reference validity remain explicit. This
is a specified measurement operator, not a lossless export of all input IQ.

For an arbitrary delayed moving echo, reconstruct each canonical symbol by
placing S in signed FFT bins, inverse transforming with `1/N` normalization,
and prepending its last `N/guard` samples. Place its useful start at `m+d`.
Evaluate the delayed reference across **actual adjacent symbols**, apply the
physical excess-path carrier phase and within-window motion, then apply the
same finite receive window/FFT/active-bin selection. A reference cell missing
from an IFFT can affect every time sample; do not silently fill it with zero
and claim an exact template. Decoder context rows may be necessary even when
they are not new output observations.

The diagonal approximation `S[k]*exp(-j*2*pi*f[k]*delay)` is not an exact model
for a delayed echo crossing a guard interval. Motion within an OFDM window can
also cause intercarrier coupling. Keeping the reference and measurement operator
allows a SAR consumer to account for these effects rather than hide them in a
range/Doppler display. Minimum-phase fitting can absorb minimum-phase parts of
the scene; the constraint alone does not guarantee preservation of every echo.

## Archive version 1

Little-endian, IEEE float32/float64, no struct padding. Complex float32 is
real followed by imaginary. The fixed header is:

| Field | Type |
|---|---|
| Magic `DVBTSAR1` | 8 bytes |
| Input rate, decimation, channel count, flags (zero) | 4 x uint32 |
| Corrected ADC origin | uint64 |
| Declared recording RF, clock phase budget in degrees | 2 x float64 |
| Per channel: physical RF, NCO, guard denominator, reserved zero | float64, float64, uint32, uint32 |

Records have a four-byte tag and uint64 body length. A reader can skip unknown
tags. `CLCK` appends uint32 count followed by that many `(uint64 start, float64
offset, float64 rate)` segments. Existing published segments never change.
Segments preceding a SPEC record cover its coordinates, including future
segments already scheduled at the time of that batch.

`SPEC` contains:

1. uint32 channel, segment, rows, first-new-row, end-new-row, TPS-frame count.
2. float64 reference origin (channel samples), relative response-fit error.
3. Per row: uint64 channel sample; float64 applied phase, applied CFO in Hz;
   uint32 modulo-4 index, discontinuity flag.
4. Per TPS frame: six uint32 values (frame, constellation, code rate, guard,
   mode, hierarchy), then 68 uint8 BCH-checked TPS bits. Enum values follow
   `Tps`; the channel header separately gives the guard denominator.
5. 6817 complex float32 response values H.
6. `rows*6817` complex float32 measured Y, then the same number of canonical S.
7. `rows*6817` uint32 reference validity values.

Row-major planes include decoder context. C and R are derived from these
arrays and the formulas above; redundant product planes are not written.
`DONE` contains a uint64 SPEC record count and marks normal completion. A file
without DONE is partial, even if its earlier records are complete. The writer
does not pad unavailable signal data or promise atomic completion of a recording.
