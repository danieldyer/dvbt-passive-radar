# Mount Dandenong VHF recording, 10 October 2026

Input: `baseband_202000000Hz_20-55-20_10-10-2026.wav`.
The recording contains 64,076,235,445 complex signed-int16 samples at **61 MS/s**,
lasting **1050.430089 seconds**. Its legacy RIFF/data sizes wrap modulo 2^32;
the reader verifies that wrapping against the actual file size.
An identical JSON sidecar is installed beside the WAV for automatic CLI loading.

The [recording metadata](../recordings/baseband_202000000Hz_20-55-20_10-10-2026.json)
configures all five signals in one channel bank:

| Multiplex | RF centre | NCO relative to 202 MHz | Nominal site |
| --- | ---: | ---: | --- |
| Seven | 177.500 MHz | −24.500 MHz | TXA, 12 Ornata Road |
| SBS | 184.500 MHz | −17.500 MHz | BAI, 116–118 Ridge Road |
| Nine | 191.625 MHz | −10.375 MHz | TXA, 8 Eyre Road |
| Ten | 219.500 MHz | +17.500 MHz | BAI, 116–118 Ridge Road |
| ABC | 226.500 MHz | +24.500 MHz | TXA, 12 Ornata Road |

Frequencies and nominal mast assignments are from
[Oz Digital TV](https://www.ozdigitaltv.com/transmitters/VIC/326-Mount-Dandenong).
The coordinates and licensed alternatives are from
[ACMA's July 2026 stations book, page 371](https://www.acma.gov.au/sites/default/files/2026-07/9%20Televison%20by%20Area%20Served.pdf).
These are licensed site coordinates, not surveyed antenna phase centres;
the active antenna during this recording is not independently established.
The metadata retains that uncertainty and the alternative licensed sites.

## Processing and validation

Signed-int16 conversion uses exact float32 scaling by 1/32768. Nominal 61/64
retiming is folded into channel synthesis on a common 64 MS/s coordinate grid,
producing five 8 MS/s streams. Pilot-derived clock correction is shared by all
channels. Physical RF, channel NCO and receiver frequency correction remain
separate; source coordinates retain the nominal rate conversion.

The rational-rate regression uses the actual five channel centres, passband-edge
tones, positive/negative clock changes and overlapping blocks at a 1200-second
epoch. Maximum complex error was 1.17e−6, with overlapping-block discrepancy
1.45e−7; differential phase stayed below 0.01 degrees where amplitude was
nonzero. This verifies numerical bookkeeping, not transmitter phase calibration.
All 27 Release CTest cases passed, and the targeted Debug tests passed with
Vulkan validation. All ten video-support tests passed, including comparison of
native display pooling against the Python reference.

Independent FEC checks acquired all five channels as **8K, 64-QAM, rate 3/4,
guard interval 1/16, nonhierarchical**. TPS parameters passed their BCH check.
These counters include overlapping decoder context, so they are not counts of
unique transport packets:

| Interval checked | RS words examined per channel | Failed RS words |
| --- | ---: | --- |
| 0–20 s | 347,907 | 0 on all five |
| 500–520 s | 277,588–278,722 | 0 on all five |
| 40–75 s | 545,069–576,810 | 0 on all five |
| 990–1050 s | 751,204–754,595 | 1,024–2,449, concentrated near 1044 s |

The final check also found 17 failed words on ABC near 1028 s. Some validated
radar-map intervals are unavailable. These gaps are displayed as unavailable
instead of holding an old image indefinitely or inventing coherent samples.
Independent cyclic-prefix checks establish **timing discontinuities in the
input recording**, not simply weak signals. A fixed-clock check, with adaptive
feedback disabled, found matching abrupt CP shifts on all five channels near
51 s and repeatedly between 285–297 s, while CP correlation quality remained
about 0.99. A second check operated directly on raw int16 samples with NumPy,
without the channelizer, decoder or resampler: the composite CP peak moved from
ADC phase 49,271 to 16,272 between 50.8 and 51.1 s. That −32,999-sample shift
matches the independently measured −4,328-channel-sample shift multiplied by
61/8, within two source samples. CP offsets are measured modulo the OFDM symbol
period; these figures do not identify the absolute number of missing samples.

This is consistent with dropped or discontinuous capture data; the particular
hardware/software cause is not established. Strong successful FEC intervals
do not certify uninterrupted timing. Future coherent image formation must
respect reacquisition segments and cannot assume a recovered physical timeline
across these jumps merely from nominal WAV sample indices.
Machine-readable probe results and test logs are in `build/vhf-20261010/`.
The CLI's automatic sidecar loading was also checked on the actual recording;
it selected all five signals and the correct input/coordinate rates without
an explicit `--metadata` argument.

The optional `cp_probe` validation target writes per-channel CP phase and quality
on a fixed common clock. It is kept outside default production build targets.

The [full-recording timing plot](figures/vhf_20261010/timing_discontinuities.png)
and [event table](figures/vhf_20261010/timing_discontinuities.csv) cover 0–1050.4 s,
with 32,826 measurements per channel in nominal 32 ms blocks (median measured
CP-epoch spacing 31.55 ms). The
detector found 1,231 nearest-period jumps exceeding 4 µs and agreeing to within
1 µs on at least three high-quality channels. Median CP quality was 0.9951;
its first percentile was 0.9890. The CP search resolution is 0.5 µs. Full-symbol
losses are invisible, and multiple jumps inside one measurement interval can
merge. The accumulated nearest-period offset includes oscillator drift and
modulo ambiguity; it is **not** an absolute dropped-sample count or a repaired
physical capture timeline. The [PDF](figures/vhf_20261010/timing_discontinuities.pdf)
is suitable for export.

## Video export

All export tooling lives under `support/video/`. The complete five-channel
render uses a nominal three-second sliding coherent window, **±100 Hz Doppler**,
0–30 km excess path and **3125/102 ≈ 30.637 frames/s**. The actual symbol-aligned
coherent interval is 3.000704 s. ProRes MOV encoding uses independent frames for
QuickTime seeking. Each video covers the complete source timeline, including
acquisition and unavailable intervals.

The colour scale is fixed separately for each channel. Its lower limit is the
whole-recording median of displayed background medians minus 1.5 dB, excluding
the stationary notch and the first 0.6 km. Its upper limit is the largest
displayed power over the entire recording plus 0.25 dB. No scale adapts during
playback, and no grid overlays the data. Display caching/pooling does not modify
scientific matched-filter outputs.

```sh
cmake --build build/release --target dvbt_preview
build/web-venv/bin/python support/video/render_recording.py \
  /Users/danielrd/data/dvbt_passive_radar/baseband_202000000Hz_20-55-20_10-10-2026.wav \
  --metadata recordings/baseband_202000000Hz_20-55-20_10-10-2026.json \
  --doppler-hz 100 --output renders/mt_dandenong_vhf_20261010_pm100Hz
```

If collected caches already exist, use `--resume` to encode them. Input or
metadata changes are rejected on resume. Completed videos are preserved, and
the tool removes each disposable display cache only after its video succeeds.

The five completed MOVs are in
[`renders/mt_dandenong_vhf_20261010_pm100Hz/`](../renders/mt_dandenong_vhf_20261010_pm100Hz/):
[Seven](../renders/mt_dandenong_vhf_20261010_pm100Hz/seven.mov),
[SBS](../renders/mt_dandenong_vhf_20261010_pm100Hz/sbs.mov),
[Nine](../renders/mt_dandenong_vhf_20261010_pm100Hz/nine.mov),
[Ten](../renders/mt_dandenong_vhf_20261010_pm100Hz/ten.mov) and
[ABC](../renders/mt_dandenong_vhf_20261010_pm100Hz/abc.mov).
Each contains **32,183 frames, 1920×1080, 3125/102 frames/s, 1050.453120 s**,
with ProRes Standard, 10-bit 4:2:2 Rec.709. The final partial frame extends the
nominal WAV timeline by 23.03 ms. Each channel has a JSON sidecar and PNG preview.

| Channel | Valid scientific maps | Fixed display power bounds (dB) | MOV size (decimal GB) |
| --- | ---: | ---: | ---: |
| Seven | 15,086 | −0.061 to 48.920 | 12.53 |
| SBS | 15,092 | −0.343 to 44.554 | 12.40 |
| Nine | 14,987 | −0.267 to 52.044 | 12.48 |
| Ten | 15,083 | −0.113 to 48.616 | 12.51 |
| ABC | 15,089 | −0.031 to 50.446 | 12.55 |

All five frame counts, durations, codec/colour metadata, fixed bounds and zero
upper clipping were checked. Exact-frame AVFoundation seeking passed 34 distant,
forward and reverse requests per video (170 total), with median latencies of
13–31 ms. Initial and middle-recording decoded frames were inspected. Disposable
display caches have been removed after successful export; scientific source and
metadata remain. [Machine-readable validation](data/vhf_20261010/video_validation.json)
includes file sizes, map counts, bounds and seeking results.

The [render performance analysis](render_performance_20261010.md) explains the
publication-cadence difference and full-map pipe bottleneck. The new optional
GPU preview path reduces transfer volume sixteenfold and measured 1.54× faster
on identical five-channel data. These videos reuse the completed original map
collection and were exported concurrently using hardware ProRes encoding.
