# VHF rendering performance, 10 October 2026

The main avoidable cost was transporting full scientific maps to the display
tool. GPU focusing also performs much more work at the requested video cadence
than in the earlier production benchmark. Video encoding had not started while
the original collection was progressing slowly.

For the three-second CPI and ±100 Hz, each scientific map contains 3,203 × 2,509
complex float32 cells: **64.29 MB**. At 3125/102 maps/s on five channels, continuous
valid data requests approximately **9.85 GB/s** through the original display
pipe. The picture needs only a 1,602 × 628 peak-pooled power image: **4.02 MB**.
That is a sixteenfold reduction in transport. The earlier approximately 10×
realtime benchmark published one map per second on one channel. The video asks
for roughly 153 maps per second across all five channels.

## Evidence and measured improvement

A five-second CPU sample of the running collection found approximately 45% of
the detector's main-thread samples in scientific-output pipe writes. The native
preview reducer spent approximately 80% waiting for input and 11% waiting for
its output; Python spent approximately 91% waiting for input. These percentages
measure sampled **wall time on those threads**, including waiting, not CPU load
or GPU execution time. They show backpressure and do not establish that the
waiting processes consume that much CPU.

The direct comparison used the first 18 seconds of the actual int16 VHF capture,
all five different frequencies, the same receiver and finite-waveform matched
filter, 3 s CPI, ±100 Hz, 0–30 km and 30.637 maps/s. Two runs per path each produced
**330 maps per channel**, with identical first/last timestamps and float32 peak
levels. No frames were dropped or cadence reduced.

| Display path | Run 1 wall time | Run 2 wall time | Mean |
| --- | ---: | ---: | ---: |
| GPU scientific focus → CPU normalization → full RDM pipe → CPU pooling | 65.65 s | 77.62 s | 71.64 s |
| GPU scientific focus → calibrated GPU pooling → compact preview pipe | 45.43 s | 47.70 s | 46.57 s |

The change is **1.54× faster, with 35.0% lower elapsed time**. This benchmark
includes process startup, acquisition, shader/FFT setup, WAV reads and preview
consumption; it excludes cache writing, colour rendering and video encoding.
The initial approximately seven signal seconds have no complete detection map.
It is therefore not the earlier I/O-excluded steady-state numerical benchmark.
File reads were identical in the comparison and were not optimized.

In the new path, GPU timestamp queries measured 19.21–19.78 s of focus execution
over the 1,650 maps, 1.36–1.39 s generating matched profiles, and 0.226–0.231 s
estimating carrier noise. Robust map calibration consumed 2.88–3.21 CPU seconds
across the focus workers. Device timestamps and active CPU seconds overlap and
must not be added to predict wall time. Remaining wall time includes the other
receiver/reconstruction stages, initialization, host submission/copies and
preview consumption. This comparison does not attribute all of that remainder
to any one stage.

An isolated per-map test at the same dimensions measured 9.88 ms of GPU focusing
inside 11.26 ms wall time, 1.57 ms active CPU calibration, and 1.30 ms wall time
for GPU preview reduction plus compact readback. Even an ideal display sink
cannot make approximately 153 fresh focused maps/s into the earlier one-map/s
workload. Further large improvements would require reusing overlapping-CPI
focusing work, rather than another display or encoder setting.

Results are retained in [render_performance.json](data/vhf_20261010/render_performance.json)
and [gpu-preview-regression.log](data/vhf_20261010/gpu-preview-regression.log).

## Implementation and validation

`GpuFocus::calculate_device()` exposes a ready, borrowed GPU map. The ordinary
scientific API still copies and returns the complete complex map. An optional
matched-detector projector can consume that GPU result after the same robust
CPU noise calibration; its lifetime ends before the next calculation. The
projector does not retain receiver buffers or alter the sample clock.

The optional `support/video/dvbt_preview` sink applies the same positive noise
normalization and peak pooling in a Vulkan shader, then sends distinctly framed
`RDP1` previews. It never overwrites the complex samples. All shaders, framing
and video-specific reduction stay in `support/video/`; the production library
only exposes the general device-result interface. `--preview-backend cpu`
retains the simple reference path.

The GPU regression covers noise scaling, odd dimensions, final partial blocks,
narrow peaks on edge cells, reused buffers, nonfinite samples and overflow,
unchanged scientific values, device/host focus equality and borrowed-map
calibration equality. The maximum calibrated power discrepancy from the CPU
reference was **0.00000346 dB**. Debug Vulkan validation passed. All 27 production
CTest cases and all ten Python video-support tests passed after this change.

Encoding now runs up to three independent channel exports concurrently using
Apple's hardware ProRes VideoToolbox encoder. The completed collection is reused
for these videos; their numerical maps were not recomputed merely to change the
export scheduler. Fixed scales, frame cadence, map timestamps and unavailable
intervals are preserved. Each MOV uses independent frames and retains the
QuickTime seeking settings requested earlier.

All five exports completed: 32,183 frames each at 3125/102 frames/s. Combined
parallel export took approximately fourteen minutes. Codec and frame counts,
fixed bounds with zero upper clipping, and 170 exact AVFoundation seeks passed.
Details are in [video_validation.json](data/vhf_20261010/video_validation.json).
