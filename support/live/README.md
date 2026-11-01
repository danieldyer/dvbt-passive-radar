# Live capture tools

`dvbt_capture` runs the bounded capture/channelisation/decoder/profile pipeline.
UHD and native PCIe uSDR DMA transport use the same engine. JSON examples are in
`recordings/`; select the one matching the connected RF bandwidth and device.
No driver, gain or decoder mode is selected automatically from observed quality.

On the Mac, launch `record-vhf-b200.command` for the five-channel TX/RX-port B200
configuration. The FTXUI terminal interface is the normal status UI: 1–9 select
a channel, `t` opens timing details, `v` opens the native ImPlot map, `?` shows
help. Headless use adds `--no-tui`. Ctrl-C drains processing and closes output.

`--archive profiles` is the normal output; `both` additionally stores channelised
I/Q for qualification. The map window is display-only. DC projection, colour
scaling and skipped display refreshes do not change the saved complex profiles.
The summary and `diagnostics.jsonl` distinguish SDR losses, host discards, decoder
erasures and physical acquisition segments. Receiver UTC is not calibrated.

`simulate_capture` generates an independent deterministic DVB-T RF fixture for
transport/TUI testing; `--simulate INPUT.wav` feeds it through the complete live
pipeline. Replay loops until stopped; `--simulate-once` is finite, and verified
cyclic fixtures may use `--simulate-cyclic`. Timestamp gaps can be injected for
regression. These modes do not establish live RF sensitivity or device timing.

For a continuous coherent simulation, generate the verified cyclic fixture once
(about 1.9 GB, even when requesting one second), then replay it explicitly:

```sh
cmake --build --preset release --target simulate_capture -j6
mkdir -p build/live-validation
build/release/simulate_capture recordings/live_mt_dandenong.json \
  build/live-validation/five_channels_cyclic.wav 1 --cyclic
build/release/dvbt_capture --metadata recordings/live_mt_dandenong.json \
  --simulate build/live-validation/five_channels_cyclic.wav --simulate-cyclic \
  --output captures/cyclic-simulation --archive none
```

Use a new output path. Ordinary WAV replay deliberately starts a new acquisition
segment at each loop; do not mark an arbitrary recording as cyclic. The convenience
`simulate.command` uses ordinary replay and never assumes coherence at its seam.

Install `requirements.txt` in an isolated Python environment for the terminal
validation scripts; these dependencies are not used by the capture executable.

`preflight.py` runs a bounded headless capture with actual archiving, selected
map load, drop/lock checks and timing/phase diagnostics. `analyse_diagnostics.py`
and `read_archive.py` inspect completed runs. `validate_capture.py`,
`validate_shutdown.py` and `validate_tui.py` exercise the supported control paths.
The TUI test accepts one to nine metadata channels; Linux runs use `--text-only`
for portable terminal snapshots, while macOS can also save Menlo-font PNGs.
Generate artifacts in ignored `build/` or `captures/`; keep experiments outside
production source directories.
