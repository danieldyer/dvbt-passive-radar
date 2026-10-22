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
  `antenna_agl_m` gives a shared nominal antenna phase-centre height for all bands
  at that site. The offline geometry tools add bilinearly interpolated terrain
  elevation and geoid undulation; AGL is never passed to ECEF as ellipsoid height.
  `transmitter_terrain` records the cached DEM, source, vertical datum and format.
  Descriptive fields are retained for later image formation.
- Each signal's `candidate_transmitter_ids` limits the site search.
  `transmitter_site_roles` records the configured primary/backup hypothesis;
  `licensed_site_ids` and `licensed_station_types` separately preserve ACMA
  records. A configured role does not establish which antenna was transmitting.
- `receiver`: optional descriptive geometry, retained in capture sidecars.
  `location.latitude_deg` / `longitude_deg` identify the nominal WGS84 antenna
  phase centre; `antenna_agl_m` is height above local ground, not ellipsoid height.
  Geographic analysis adds terrain elevation and geoid undulation before ECEF
  conversion. The live Mount Dandenong configuration uses the user-supplied
  position −37.81040454194318, 144.98744506389454 at approximately 10 m AGL.

The pipeline writes `recording.json` beside detections, or a `.json` sidecar beside
the SAR archive, adding `processing_signal_ids` in output-channel order. The input
metadata is not modified. Transmitter positions do not affect today's independent
range–Doppler maps; no antenna survey or calibrated phase centre is inferred.

The single-channel [C31 live configuration](../recordings/live_c31.json) tunes
MGV RF channel 32 at **557.625 MHz**, as listed in TX Australia's
[Ornata Road specifications](https://www.txaustralia.com.au/tvcoverage/metropolitan/melbourne/tower/selby).
It requests 16 MS/s signed 16-bit I/Q with the SDR centred at 556.625 MHz,
placing C31 at +1 MHz from SDR DC. The acquisition bandwidth is 12 MHz.
The 8K, guard 1/16 settings match the previously decoded C31 recording;
modulation and code rate are recovered from TPS. Receiver position and gain
are inherited from the VHF setup. The Ornata mast location and 190 m AGL
height are nominal priors, not a measurement of the UHF antenna phase centre.

```sh
build/release/dvbt_capture --metadata recordings/live_c31.json \
  --output captures/live-c31-001 --archive both
```

`--archive both` saves channelised I/Q and range-compressed time-domain profiles.
UHD's actual applied settings are displayed and recorded by the capture program.

The [VHF capture configuration](../recordings/baseband_202000000Hz_20-55-20_10-10-2026.json)
contains Seven 177.5 MHz, SBS 184.5 MHz, Nine **191.625 MHz**, Ten 219.5 MHz and
ABC 226.5 MHz. These centres come from
[Oz Digital TV](https://www.ozdigitaltv.com/transmitters/VIC/326-Mount-Dandenong).
The user-supplied physical priors of 16 October 2026 are:

| Site | Latitude | Longitude | Antenna AGL |
|---|---:|---:|---:|
| Eyre Road | −37.83540006541602 | 145.3477595896221 | 130 m |
| Ornata Road | −37.83745859518615 | 145.34687782301492 | 190 m |
| Ridge Road | −37.834854231253686 | 145.34846826008254 | 138 m |

The configured primary/backup pairs are Ornata/Eyre for Seven, Nine and Ten,
and Ridge/Ornata for SBS and ABC: **32 assignments**. Ten-at-Ridge is licensed
but excluded from this search. The cached 13 October ACMA RRL records agree with
the commercial and ABC roles, but label SBS Ornata **Main** and Ridge **Standby**;
both SBS sites remain candidates. These records do not establish the active site
during a particular capture. Source: [ACMA RRL](https://www.acma.gov.au/radiocomms-licence-data).

The approved terrain source is the existing
[Geoscience Australia one-arcsecond SRTM-derived DEM](https://services.ga.gov.au/gis/rest/services/DEM_SRTM_1Second_2024/MapServer),
with EGM96 heights. It is not a native DTED Level 2 product. The transmitter
update retains the incumbent receiver position and height. The older fits used
a different terrain mosaic; the displacement audit separates this ground-height
change from the requested AGL changes.

Run `build/validation-venv/bin/python support/calibration/update_transmitters.py`
to materialize the metadata as
`docs/data/calibration_20261016_sites/{geometry-prior,sites-main,transmitter-update}.json`.
The final file contains signed north/east/down differences from the actual
previous optimiser site centres, per-channel licensed-height comparisons, and
all 32 assignments. The geometry snapshot freezes its configuration; historical
capture range/phase references are not rewritten. A changed configuration needs
a new output directory, selected with `--output`.

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

Live capture clock/process-noise settings are declared in `capture.clock_model`;
see [the controller and diagnostics contract](live_capture.md#shared-clock-controller).
The supplied live configurations use a four-second requested receive buffer.
Transmitter positions remain priors; these capture changes do not adopt a fitted
site assignment or redefine the receiver phase centre.
