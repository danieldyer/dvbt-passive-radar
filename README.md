# DVB-T range-profile capture

A C++23/Vulkan receive pipeline for coherent passive radar. It captures one shared
SDR I/Q stream, channelises DVB-T signals, tracks their timing, decodes/reconstructs
the transmitted waveform, cancels the reference channel, and saves complex range
profiles with the coordinates needed by a later SAR/ISAR processor.

Supported capture drivers are UHD (including B200mini) and the native DMA-loan
uSDR driver used on PL64. The supported waveform is non-hierarchical 8K DVB-T,
including Australian 7 MHz QPSK/16-QAM/64-QAM signals and all code rates/guards.
This repository contains capture, archiving, live display, archive video export,
and their accuracy/performance qualification. Image formation experiments have
been removed.

## Build

CMake 3.28+, a C++23 compiler, Vulkan 1.2, glslc/glslang, FFTW float, Boost 1.75+
headers and an enabled SDR driver are required. GNU Radio 3.10 is required only
when building the independent tests (`BUILD_TESTING=ON`, the default). Capture kernels
require at least 32 KiB shared memory and 512 workgroup invocations. M5/MoltenVK
and PL64/Intel Vulkan are the qualification targets.

On macOS, `scripts/bootstrap.py` builds the pinned FFTW/GNU Radio dependencies
locally in `build/deps`; GNU Radio is a test dependency. A capture-only build
uses `-DBUILD_TESTING=OFF`. Set `VULKAN_SDK` to the installed SDK if required.

```sh
cmake --preset release
cmake --build --preset release -j6
ctest --preset release
```

UHD capture and the FTXUI TUI are enabled by default. Set `RADAR_BUILD_USDR=ON`
with `USDR_INCLUDE_DIR`/`USDR_LIBRARY` for the native acquire/release uSDR branch;
use `RADAR_BUILD_UHD=OFF` on a uSDR-only target. The optional macOS native ImPlot
window uses GLFW/Metal and can be disabled with `RADAR_BUILD_LIVE_DISPLAY=OFF`.
These are build/dependency choices; the DSP has one production decoder policy.
On the PL64 Linux host, `cmake --preset pl64` selects the measured Alder Lake
compiler target and uSDR backend; supply the driver include/library paths if
they are not already on the build environment search path. The matching build
and test presets are also named `pl64`. This does not modify the Nix image.

For the Nix closure, disable the local bootstrap search with
`RADAR_USE_LOCAL_DEPS=OFF` and use `BUILD_TESTING=OFF` for the runtime package.
Provide the pinned VkFFT source through `FETCHCONTENT_SOURCE_DIR_VKFFT`; a TUI
build also needs installed FTXUI 7 or `FETCHCONTENT_SOURCE_DIR_FTXUI`. The pins and
hashes are in CMake. This lets the Nix derivation fetch dependencies separately
and configure without network access. A service build can disable the TUI and
native display entirely; enabling uSDR still requires its native DMA API.

`cmake --install build/pl64 --prefix /desired/prefix` installs `bin/dvbt_capture`,
the two live configurations under `share/dvbt_passive_radar/recordings`, and the
README/provenance notices under `share/doc/dvbt_passive_radar`. Kernels are embedded
in the executable; runtime needs its linked libraries and Vulkan ICD. With both SDR
drivers disabled, the executable and transport/archive tests still support replay.

## Capture

```sh
build/release/dvbt_capture \
  --metadata recordings/live_mt_dandenong.json \
  --output captures/new-session \
  --archive profiles --range-metres 100000
```

The example is configured for five VHF channels on a B200mini, TX/RX input.
The lower-three-channel PCIe uSDR configuration is
`recordings/live_usdr_mt_dandenong.json`; use `build/pl64/dvbt_capture` for that preset.
For the Mac, `support/live/record-vhf-b200.command` also keeps it awake and selects
the local MoltenVK ICD when available. Metadata declares SDR settings, channel
RF centres, guard intervals and nominal antenna positions. Requested and applied
SDR values are retained separately. New output paths are required.

The TUI shows receive gaps/discards, queue pressure, per-channel lock/timing,
pilot SNR, RS acceptance and processing headroom. Keys 1–9 select a channel;
`v` opens its native range/Doppler display and `t` shows timing detail. Ctrl-C
stops reception, drains bounded processing queues, then closes the archive.
`--no-tui` runs the same capture engine headlessly.
SIGTERM follows the same drain path for service shutdown. Invalid detector and
worker/slot settings fail before opening hardware or creating output files.

The decoder always uses adaptive pilot filtering, compact soft likelihoods,
full-traceback Viterbi, RS-pinned feedback (at most 32 trellis passes), and bounded
RS rescue. There is no automatic quality downgrade. Default resources are five
FEC workers and nine frame slots; `--fec-workers` and `--frame-slots` set bounded
resource limits. Batch context and decoding algorithms are fixed.

Use `--archive both` to save channelised I/Q as well as profiles; `profiles` is
the normal space-efficient output. `--archive channels` is useful for RF/FEC
qualification. Raw wideband ADC samples are not saved. Optional display refreshes
may be skipped under pressure; scientific profiles are retained until an actual
capture/input discontinuity. Display DC projection never changes archived data.
Only successful explicit archive completion writes `DONE`; an exception drains
already queued data while leaving the archive detectably incomplete. Output-file
close errors propagate to a nonzero exit status. This is buffered output, with
no fsync/power-loss durability guarantee.

## Archive review

```sh
cmake --build --preset release --target live_profiles_preview -j6
python3 -m venv build/tools-venv
build/tools-venv/bin/python -m pip install -r support/video/requirements.txt
build/tools-venv/bin/python support/video/render_capture.py captures/new-session \
  --output captures/new-session/videos --integration 3 --doppler-hz 100 --range-km 100
```

The renderer uses the production Vulkan focus operator, reuses immutable row FFTs
in a GPU ring, and reads back only pooled display pixels. It runs unpaced and
exports each channel using hardware VideoToolbox ProRes, with independently
encoded frames for QuickTime scrubbing. H.264 export also uses VideoToolbox.
Colour limits are fixed per channel for the recording: background median minus
1.5 dB to the full maximum plus 0.25 dB. The broad noise floor therefore stays
near the bottom of the scale. Pixel pooling preserves the strongest displayed
cell. Neither the colour scale nor the display cache is a scientific DSP input.

## Qualification

`tests/` contains waveform, known-payload FEC, multichannel phase/clock, response,
profile, archive, driver and queue-failure regressions. CPU implementations under
`tests/reference` are independent checks, not selectable production backends.
`support/validation` contains the small repeatable benchmark/soak suite.

See [current qualification](docs/production-refactor.md) for measured accuracy,
throughput and limitations, including the archive-format compatibility study.
The navigation/SAR service and Nix image remain separate; no Nix integration is
performed by this repository yet. Nominal antenna-site assignments in metadata
are priors, not a completed phase-centre calibration.

GNU Radio-derived DVB-T mapping/synchronisation code carries GPL-3.0-or-later
terms. See [third-party provenance](THIRD_PARTY.md).
