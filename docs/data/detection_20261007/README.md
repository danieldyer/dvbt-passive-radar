# Saved detector validation

These small results accompany [the analysis](../../sensitivity_improvements_20261007.md).
Plots are in [the figure directory](../../figures/detection_20261007/).
Production code is separate from the opt-in [validation tools](../../../support/validation/README.md).

The source recording is
`/Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav`:
56 MS/s float I/Q, declared recording RF 553 MHz, channel RF 557.625 MHz,
NCO +4.625 MHz, 8K DVB-T, guard 1/16. The three disjoint source excerpts begin
at 50, 140 and 265 seconds. Raw recordings and large spectral archives remain
outside this results package.

| File | Meaning |
|---|---|
| `model-bound.json` | 360 sampled efficiencies against an explicitly specified waveform and diagonal covariance; stratified symbols over a three-second CPI |
| `full-cpi-bound.json` | Four observation-space checks using all 2,758 symbols in the CPI |
| `previous-filter-comparison.json` | Independent conditional expected-SNR comparison with the previous Hann, circular-correlation operator |
| `background.json` | Identical-CPI physical cancellation controls, band statistics and held-out local covariance diagnostics |
| `benchmark.json` | Output-free sequential CPU/GPU/wall measurements, one FEC worker, preloaded recording |
| `noise-diagnostic.json` | Earlier diagnostic carrier weighting and its assumptions; these are current-batch diagnostic weights, not the final production estimator |
| `carrier-noise.npz` | Small retained diagonal-variance input for the conditional bound |
| `test-release.log` | All 26 Release tests |
| `test-debug.log`, `matched-tests.log` | Debug Vulkan validation and Release numerical regression details |
| `phase-validation.log` | Earlier broader Debug multichannel, clock and phase regression run |
| `python-*-tests.log` | Support-analysis, video and viewer test results |
| `vectorization.log` | Compiler vectorization report for the CPU map calibration and scaling |
| `optimization-trials.md` | Retained and rejected implementation optimizations |
| `cli-smoke.json` | Raw-I/Q CLI and framed-map metadata checks |
| `video-validation.json`, `native-seeks.json` | Full recording, 9,248 frames, all-map phase/timing checks, complete decode and 34 exact macOS seeks |
| `full-video-render.log`, `full-video-pipeline.log` | Completed unpaced render and production processor diagnostics |
| `source-sha256.json` | Source-file hashes at completion |

The approximately 0.89 dB result is conditional on the listed model, mean guard
energy, three-degree RMS clock allowance and noise-estimation error. It is not
a uniform continuous-coordinate proof or an all-impairment real-recording
certificate. The covariance tests do not establish a bound on full temporal
covariance. Background reduction is not interchangeable with target SNR gain.
Noise standardization changes threshold units, preserves complex phase, and is
reversible with the exported factors.

See the validation-tool README for commands. Replays use independent frozen
stationary training and identical later CPIs for cancelled and uncancelled
controls. Benchmark consumers run serially, without compilation or plotting
alongside them. CPU process time, main-thread CPU, GPU timestamps and wall time
are reported separately and must not be summed.
