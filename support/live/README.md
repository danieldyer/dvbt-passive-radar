# Live capture support

To analyse a completed profile capture's actual timing, validity and matched
filter weights, use the existing analysis environment:

```sh
MPLCONFIGDIR="$PWD/build/cache/matplotlib" build/validation-venv/bin/python \
  support/live/analyse_capture.py captures/session-001/capture.dvblive \
  --output build/session-001-analysis
```

It writes JSON statistics, the timing/weight arrays and standalone plots of
oracle validity, temporal weighting, Doppler response and expected range
resolution. It distinguishes the fixed row-count CPI from its weighted
point-spread response; it does not equate the window-shape metric to a measured
SNR loss. See [video support](../video/README.md) for rendering saved profiles.

The production front end is `src/live_capture.cpp` and `dvbt_capture`. This
directory contains the FTXUI terminal application, optional native ImPlot/Metal
map window, and independent simulation and validation tools. See
[capture documentation](../../docs/live_capture.md).

Generate an independent five-channel 64-QAM/3/4/GI 1/16 fixture:

```sh
cmake --build --preset release --target simulate_capture
build/release/simulate_capture recordings/live_mt_dandenong.json \
  build/live-validation/five_channels.wav 10
```

This support fixture is at 64 MS/s so its occupied VHF carriers lie exactly on
the synthesis FFT grid. Production capture reads back UHD's applied rate; 61.44
MS/s clock conversion and NCO phase are tested independently. The fixture is a
temporary validation artifact, not saved raw data from production acquisition.

Run the same live ring and complete pipeline with a simulated transport:

```sh
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --simulate build/live-validation/five_channels.wav --output captures/simulation \
  --archive none
```

Open [`simulate.command`](simulate.command) to launch this in Terminal with a new
output directory. Simulation loops by default until `s` (stop and retain results),
`q` or Ctrl-C. A file seam creates a new acquisition segment without a false
sample-loss count. `--simulate-once` gives a finite replay. Select with `1`–`5` and
press `v` for a native ImPlot window; the TUI remains the main interface. Timing,
events and applied source settings have their own pages (`t`, `e`, `i`).

`--unpaced` runs simulation at maximum processing speed with backpressure.
Ordinary simulation is paced to the input clock and tests real receive pressure.
`--inject-gap-at 2.5 --inject-gap-samples 119999` tests precise timestamped loss.

`validate_capture.py` runs a paced five-channel acceptance test, asserts loss
counts, lock/FEC results and archive completion. It explicitly selects finite replay.

```sh
python3 support/live/validate_capture.py \
  --fixture build/live-validation/five_channels.wav --output build/capture-validation
python3 support/live/read_archive.py captures/simulation/capture.dvblive
```

The pseudo-terminal check sends real keyboard events, verifies resizing and timing
telemetry, runs beyond EOF, selects two native maps, stops while retaining results,
quits and checks raw-mode and alternate-screen restoration. It saves actual terminal
screens as text and PNG. `validate_shutdown.py` selects the Ctrl-C variant:

```sh
build/web-venv/bin/python support/live/validate_tui.py \
  --fixture build/live-validation/five_channels.wav \
  --output build/capture-tui-validation --native
build/web-venv/bin/python support/live/validate_shutdown.py \
  --fixture build/live-validation/five_channels.wav \
  --output build/capture-shutdown-validation --archive profiles
```

The archive reader and finite validation need only Python's standard library.
Terminal validation uses `pyte` and Pillow in the existing dedicated
`build/web-venv` (its historical name does not imply a browser dependency).
Simulation generation is C++. UI/replay regression tests are CTest CPU targets.

Validate the real native renderer with two known signed-Doppler targets, including
window close/reopen. The helper exports only its own framebuffer; it does not use
OS screenshot permissions. Ordinary capture/display performs no graphics readback.

```sh
cmake --build --preset release --target native_display_check -j 6
build/release/native_display_check recordings/live_mt_dandenong.json \
  build/native-display-validation
build/web-venv/bin/python support/live/validate_native_images.py \
  build/native-display-validation
```
