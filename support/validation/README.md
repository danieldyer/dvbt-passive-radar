# Sensitivity audit tools

These opt-in tools inspect the scientific spectral handoff without changing the
production pipeline. The dated measurements and their limitations are in
[the sensitivity report](../../docs/sensitivity_audit_20261006.md).
The subsequent production detector, corrected noise analysis and qualified
bounds are in [the improvements report](../../docs/sensitivity_improvements_20261007.md).

## Matched-detector validation

All numerical implementations are in `src` and `shaders`; these archive adapters,
independent bounds, plotting scripts and output-free benchmarks remain optional
support tools. Reuse the setup below, then build:

```sh
cmake --build --preset release --target dvbt_detect matched_tests \
  detection_replay detection_pipeline_benchmark -j8
ctest --preset release
mkdir -p build/detection-check
build/release/support/validation/detection_replay \
  build/sensitivity-20261006/excerpt-050.sar \
  build/detection-check/final-050 1 3 10 1
build/validation-venv/bin/python support/validation/optimality_bound.py \
  build/sensitivity-20261006/excerpt-050.sar \
  --output build/detection-check/model-bound.json
build/validation-venv/bin/python support/validation/detection_probe.py \
  build/sensitivity-20261006/excerpt-050.sar \
  --output build/detection-check/carrier-noise.json
build/validation-venv/bin/python support/validation/full_cpi_bound.py \
  build/sensitivity-20261006/excerpt-050.sar \
  build/detection-check/final-050.times.f64 \
  build/detection-check/carrier-noise.npz \
  --output build/detection-check/full-cpi-bound.json
MPLCONFIGDIR="$PWD/build/cache/matplotlib" XDG_CACHE_HOME="$PWD/build/cache" \
  build/validation-venv/bin/python support/validation/detection_analysis.py \
  build/detection-check/final-050 \
  --output build/detection-check/background.json \
  --figures build/detection-check/figures
```

Replay arguments after the output prefix are direct-cancellation enable, CPI
seconds, capture seconds and frozen-stationary enable. Use the last flag `0`
for the minimum-phase-only control; direct cancellation `0` produces the
uncancelled control. All three select the same final CPI from the input timeline.
The fourth replay argument controls acquisition-independent background prediction;
it does not alter the original archive. Repeat on excerpts 140 and 265 for the
disjoint recording controls. `--baselines` and `--uncancelled` add matched-CPI,
common-scale comparison panels to `detection_analysis.py`.

The 360-case bound uses stratified source symbols; `full_cpi_bound.py` checks all
2,758 symbols of selected three-second cases. Both explicitly specify covariance
and waveform assumptions. They are not all-impairment real-data certificates.
GPU interpolation, finite guard context, positive calibration and phase
bookkeeping are checked independently by `matched_tests`.

`matched_tests` also compares a full three-second weighted focusing aperture
with independent direct backprojection, including a 20 dB temporal information
span and weighted stationary projection. Noise-step controls retain a narrow
blocker through both rising and falling broadband noise. Replay writes
`.information.f64` alongside moments and times so batch normalization is explicit.
`test_sensitivity.py` independently checks the inverse-variance information bound.

The prior-filter comparison can be reproduced with:

```sh
build/validation-venv/bin/python support/validation/compare_filter_bound.py \
  build/sensitivity-20261006/excerpt-050.sar \
  build/detection-check/final-050.times.f64 \
  build/detection-check/carrier-noise.npz \
  build/detection-check/full-cpi-bound.json \
  --output build/detection-check/previous-filter-comparison.json
```

The output-free benchmark preloads I/Q, warms for ten seconds, then measures
the remainder. Its positional arguments are input seconds, channels, FEC workers,
frame slots, timing enable and consumer (`0` core, `1` new detector, `2` legacy):

```sh
build/release/support/validation/detection_pipeline_benchmark \
  /Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav \
  20 1 1 3 1 1
```

Run consumers sequentially with no concurrent compilation or analysis. The
benchmark reports device time, active process/main CPU time and wall time
separately. File reads, writes and encoding are excluded. A preloaded 20-second
56 MS/s float-I/Q buffer takes about 9 GB; use a shorter interval greater than
ten seconds if necessary. The two consumers have different grid sizes and the
legacy consumer publishes an additional map at finish.

## Setup

Use an isolated environment; nothing needs to be installed into the system
Python environment:

```sh
python3 -m venv build/validation-venv
build/validation-venv/bin/python -m pip install -r support/validation/requirements.txt
cmake -S . -B build/release
cmake --build build/release --target dvbt_sar sensitivity_response -j8
```

On this machine the Vulkan runtime is selected with:

```sh
export VK_DRIVER_FILES="$PWD/build/sdk/macOS/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="$PWD/build/sdk/macOS/share/vulkan/explicit_layer.d"
```

## Reproduce the C31 measurements

Run each excerpt independently (the audit used starts 50, 140 and 265 seconds):

```sh
mkdir -p build/sensitivity-20261006
build/release/dvbt_sar \
  /Users/danielrd/data/dvbt_passive_radar/baseband_553000000Hz_12-51-54_26-09-2026.wav \
  --output build/sensitivity-20261006/excerpt-050.sar \
  --recording-rf-hz 553000000 --channel 557625000,4625000,16 \
  --start 50 --seconds 32
build/validation-venv/bin/python support/validation/sensitivity.py summarize \
  build/sensitivity-20261006/excerpt-050.sar \
  --output build/sensitivity-20261006/excerpt-050.npz
build/validation-venv/bin/python support/validation/sensitivity.py maps \
  build/sensitivity-20261006/excerpt-050.npz \
  --output build/sensitivity-20261006/excerpt-050-maps.json
build/release/support/validation/sensitivity_response \
  > build/sensitivity-20261006/response-injection.csv
build/validation-venv/bin/python support/validation/sensitivity.py theory \
  --output build/sensitivity-20261006/theory.json
build/validation-venv/bin/python support/validation/matched_filter_probe.py \
  build/sensitivity-20261006/excerpt-050.sar \
  --output build/sensitivity-20261006/matched-filter.json
build/validation-venv/bin/python support/validation/covariance_bound.py \
  build/sensitivity-20261006/excerpt-050.sar \
  build/sensitivity-20261006/excerpt-140.sar \
  build/sensitivity-20261006/excerpt-265.sar \
  --output build/sensitivity-20261006/covariance-bound.json
build/validation-venv/bin/python support/validation/matched_filter_probe.py \
  build/sensitivity-20261006/excerpt-050.sar \
  --noise build/sensitivity-20261006/excerpt-050-carrier-noise.npz \
  --output build/sensitivity-20261006/matched-filter-coloured.json
build/validation-venv/bin/python support/validation/doppler_bound.py \
  build/sensitivity-20261006/excerpt-050.npz \
  build/sensitivity-20261006/excerpt-140.npz \
  build/sensitivity-20261006/excerpt-265.npz \
  --output build/sensitivity-20261006/doppler-bound.json
build/validation-venv/bin/python support/validation/check_noise_model.py \
  build/sensitivity-20261006/excerpt-050.npz \
  build/sensitivity-20261006/excerpt-140.npz \
  build/sensitivity-20261006/excerpt-265.npz \
  --output build/sensitivity-20261006/noise-model-check.json
build/validation-venv/bin/python support/validation/sensitivity.py windows \
  build/sensitivity-20261006/excerpt-050.npz \
  --output build/sensitivity-20261006/excerpt-050-windows.json
MPLCONFIGDIR="$PWD/build/cache/matplotlib" XDG_CACHE_HOME="$PWD/build/cache" \
  build/validation-venv/bin/python support/validation/plot_sensitivity.py \
  build/sensitivity-20261006 --output docs/figures/sensitivity_20261006 \
  --doppler-hz 200
build/validation-venv/bin/python -m unittest discover -s support/validation -v
```

`summarize` checks completion records, dimensions, emitted-row continuity and
the independent double-precision replay of the GPU's residual energy statistic.
Only complete reference rows are used. The four incomplete endpoint rows of
each excerpt are excluded; a gap inside the resulting sequence fails the audit.
It saves complex range profiles to avoid repeated transforms of the large
spectral archives. Each 32-second archive takes about 4.3 GiB and its cache
about 713 MiB. `maps` evaluates the final one- and ten-second windows, retaining
full ±300 Hz diagnostic coverage for the pilot ghost; `plot_sensitivity.py`
displays ±200 Hz by default. Its exact-length Doppler FFT differs slightly from
the production map's smooth-length padding. Both evaluate the same windowed DFT.

The Vulkan response probe puts a −40 dB echo **into the measured signal before
fitting**. It uses paired baseline subtraction to separate echo retention from
the estimator's numerical baseline error. It measures the complex
projection gain after refitting, not just total residual power. Its frozen-fit
column is the analytical unity paired-control result, not another GPU fit.
All delays use the diagonal OFDM model: the long-delay rows test estimator
identifiability, not physical beyond-guard waveform reconstruction. Existing
`sar_multichannel_gpu` tests independently exercise the finite waveform operator.

Noise proxies derived from received data include tracking errors, ICI, targets
and interference. The initial source-divided proxy is preserved for diagnostics;
its 2–5 dB apparent opportunity is superseded by the carrier/common-mode analysis.
Conditional efficiency does not certify thermal-noise or detection performance.
The stationary ghost diagnostic uses the
same batch's coherent transfer estimate and is a decomposition, not held-out
validation. Plot comparisons share a physical voltage normalization for a flat
unit channel; they do not rescale each image independently.

`matched_filter_probe.py` holds the CPI at 919 symbols, freezes the fitted
response, disables DC removal and uses the actual reconstructed recording
symbols. A finite time-domain delay/frequency-shift operator supplies the exact
echo template. It computes complex filter efficiency against a known white
covariance bound, including the existing diagonal template and each taper.
This isolates implementation losses from target motion, adaptive cancellation
and uncertain real covariance. The source file and output explicitly state
that no baseband range migration is included.

`covariance_bound.py` removes two common modes only in the diagnostic noise
estimate, not production data. It compares diagonal weights across disjoint
halves, computes low-rank and augmented-real conditional bounds, and saves the
estimated physical carrier noise. With `--noise`, the waveform probe evaluates
the exact matched-filter bound using the held-out diagonal variance. The full
recorded map variance exceeds this model; see the report's explicit failed
model check. No calibrated independent noise floor is asserted.

`doppler_bound.py` derives local complex-cell information from the window,
measures row noise-like power, and compares sample covariance, structured
shrinkage and fixed analytical weights. It uses separate 10/10/11-window
training/validation/evaluation blocks. A synthetic occupied-band white-noise
control uses the same transforms and estimator. Gains assume an on-bin tone in
the retained map, and overlap the Hann taper loss; they must not be added to the
waveform correction gains. The actual Doppler grid must be used for a padded or
fractional-bin implementation. Row normalization itself does not improve SNR.

Repeat `windows` for each excerpt to reproduce the recording comparisons.
The small result archive in `docs/data/sensitivity_20261006` includes JSON, CSV,
noise spectra and test logs; the large `.sar` files and range caches stay in
`build`. No production sensitivity optimisation is enabled by these tools.
