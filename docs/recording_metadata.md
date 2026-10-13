# Recording metadata

`dvbt_detect INPUT.wav --metadata RECORDING.json --output DIRECTORY` configures
one coherent channel bank for all `signals`. `dvbt_sar` accepts the same metadata.
`--select-channel ID` selects one signal. If `--metadata` is omitted, the CLI checks
for `INPUT.json` beside `INPUT.wav`. Explicit RF/channel options conflict with
metadata instead of silently overriding it.

The version 1 JSON declares:

- `recording.centre_hz`, `sample_rate_hz`, `sample_format` (`ci16_le` or `cf32_le`)
  and optional `iq_order` (`I,Q`). Rate and format must agree with the WAV.
- `signals`: unique `id`, `name`, physical `frequency_hz`, `bandwidth_hz` (7000000),
  `fft_size` (8192), `guard_denominator` (4, 8, 16 or 32), and `transmitter_id`.
  NCO defaults to physical RF minus recording RF; an explicit `nco_hz` can declare
  a known frequency translation while leaving physical RF unchanged. Receiver CFO
  never changes the declared transmitter frequency.
- `transmitters`: unique `id` and `location.latitude_deg` / `longitude_deg`.
  Descriptive fields such as site address, position accuracy, alternate licensed
  sites and sources are retained for later image formation.

The pipeline writes `recording.json` beside detections, or a `.json` sidecar beside
the SAR archive, adding `processing_signal_ids` in output-channel order. The input
metadata is not modified. Transmitter positions do not affect today's independent
range–Doppler maps; no antenna survey or calibrated phase centre is inferred.

The [VHF capture configuration](../recordings/baseband_202000000Hz_20-55-20_10-10-2026.json)
contains Seven 177.5 MHz, SBS 184.5 MHz, Nine **191.625 MHz**, Ten 219.5 MHz and
ABC 226.5 MHz. These centres come from
[Oz Digital TV](https://www.ozdigitaltv.com/transmitters/VIC/326-Mount-Dandenong).
The three nominal mast groups are Ornata Road (Seven/ABC), Eyre Road (Nine) and
Ridge Road (SBS/Ten). Site coordinates and licensed alternatives are from
[ACMA's July 2026 stations book, page 371](https://www.acma.gov.au/sites/default/files/2026-07/9%20Televison%20by%20Area%20Served.pdf).
ACMA lists alternatives and does not establish which antenna was operating
during this capture. The JSON records that distinction explicitly. Coordinates
have whole-arcsecond precision and are not surveyed antenna phase centres.

The new WAV contains signed 16-bit stereo I/Q at 61 MS/s. Conversion to float32
uses an exact factor of 1/32768, preserving every ADC bit. The channelizer uses
a 64 MS/s corrected coordinate grid (decimation eight) while sampling the
original ADC grid at the common rational rate 61/64. This interpolation is folded
into channel synthesis, using the existing shared forward FFT and per-channel
twice-oversampled synthesis. No separate 61-to-64 MS/s waveform is generated.
Pilot-derived drift is an additional shared correction; its 3-degree differential
phase budget covers the full span of the selected channels.

Scientific sample times and channel NCO phases refer to the corrected grid.
`SarBatch.original_adc()` also applies the nominal 61/64 conversion when returning
actual source-file sample coordinates. SAR archives preserve that ratio in an
optional `SRAT` record. This keeps the input format, sample rate conversion and
transmitter geometry separate from the detection and visualization consumers.
