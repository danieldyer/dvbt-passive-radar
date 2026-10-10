# Range–Doppler stream and viewer

The terminal processor publishes `latest.rdm` atomically every two seconds of
recording time by default. The separate local server serves this stream to the
C++/Dear ImGui/WebAssembly viewer. No browser component participates in decoding
or clock correction. Output replacement bounds disk use; this is a latest-frame
stream, not an archive of every overlapping map. Download a native map from the
viewer to retain it.

## Scientific coordinates

After the common clock resampler and the minimum-phase compensator, each valid
symbol supplies `C(k) = (Y(k) - Hmin(k) X(k)) conj(X(k))`. Embed the 6817 active
carriers in their signed 8192-point FFT bins and take `IFFT(C)/8192`. Positive
sample delay is positive **bistatic excess path length**, with a 37.47405725 m
sample spacing (`c / 8 MS/s`, not `c / 2 Fs`). The occupied bandwidth limits actual
range resolution to approximately 45 m. This is excess delay against the chosen
direct-signal origin, not transmitter-to-receiver absolute propagation time.

Slow-time samples are separated by the recovered transmitter symbol period:
1088 microseconds for the observed 8K/GI 1/16 mode. A symmetric Kaiser FIR filters
these profiles before slow-time decimation. With the default factor four, the
retained Doppler interval is ±100 Hz. Its 513-tap filter is applied once and its
centre defines the output time; its lookahead never appears as an excess delay
or an unreported Doppler phase rotation. No incomplete filter supports are used.
`--slow-decimation 1 --doppler-hz 400` retains a wider Doppler band.

A forward slow-time FFT gives positive frequency to an `exp(+j 2π f t)` echo.
Complex output is normalized by the sum of the Hann window, so a tone exactly on
a Doppler bin retains its complex amplitude and phase at the first profile's time.
The FFT is padded to an efficient size **without discarding coherent samples**.
`frequency_step_hz` is the FFT grid spacing; `integration_s` is the actual used
sample duration, and `enbw_hz = 1.5 / integration_s` describes the Hann window.
Zero padding does not improve physical resolution. Hann weighting trades 1.76 dB
of tone-in-white-noise SNR for lower Doppler sidelobes; the channelization filters
do not make this windowing tradeoff.

## Stationary / DC removal

The processor and video CLI accept `--remove-dc` (off by default). For each range
and the actual selected CPI, subtract `mu = sum(w[t] x[t]) / sum(w[t])` before
windowing and FFT. Here `w` is the same periodic Hann window used by the Doppler
transform. The resulting weighted DC is zero. This rejects the stationary
component's entire window spectrum, including leakage into adjacent bins when
the FFT is padded; it does not simply blank the zero bin. Original time/range
coordinates, clock correction and reconstruction are unchanged.

For a retained complex map `Z`, the GUI performs the equivalent operation
`Z_filtered[k] = Z[k] - Z[0] W[k]/W[0]`, where `W` is the identically padded FFT
of the Hann weights. It reads the original DC bin even when the visible crop
excludes it, and subtracts complex amplitudes before power or pixel pooling.
This allows toggling on completed recordings without rerunning their decoder.
The source map remains available for download. Maps carry `dc_removed`,
`dc_removal` and `fft_size`; legacy maps supply enough timing/sample metadata to
recover their FFT size. A source already filtered by the processor is never
filtered twice and the GUI cannot restore its removed component.

DC rejection attenuates very slow moving returns too: its transition has the
Hann spectrum's width, proportional to `1 / CPI`. A changing or accelerating
stationary return is not guaranteed to vanish. Both signs of resolved moving
echoes retain their complex phase and gain outside that narrow projection.
Autoscaling uses the power after subtraction, and the video records its chosen
colour limits for every map. There are no grid lines superimposed on either plot.

An OFDM per-carrier channel model is exact only within the cyclic-extension
support. Delays beyond that support produce inter-symbol/inter-carrier interference.
The displayed range extent is an analysis window, not a claim that cancellation
has been qualified out to its far edge. Reconstruction and resampling preserve
the measured interference; this implementation does not cancel it using an
unconstrained multi-tap channel fit.

## Coherent integration

Automatic mode uses the longest suffix of the retained history whose common
phase/timing diagnostics meet the selected coherence threshold. At each occupied
band edge it evaluates the magnitude of the mean unit phasor:

`mean(exp(j * (phase_error ± 2π * 3408 * timing_error / 8192)))`.

The default threshold is `exp(-0.5 * radians(5)^2)`. The diagnostics are measured
against the fixed received-channel template; they are not an independent clock
standard or a guarantee about a moving target's acceleration. The diagnostics
only choose the integration interval; they never rephase the radar echoes.
Failures of FEC validity or synchronization clear the slow-time FIR and coherent
history. Missing observations are never compacted into a fictitious uniform grid.

Default retained history is 300 seconds and maximum range is 30 km. Increase
`--history-seconds` to retain a longer coherent interval; memory scales linearly
with history and range extent. Set `--integration-seconds 0` for automatic mode,
or a positive duration for a shorter requested CPI. The measured coherence limit
still applies to a manual request. The viewer can change the requested CPI while
the processor runs; it takes effect on the next map. After EOF the viewer retains
pan/zoom/scaling and downloads, and disables integration changes.

Map transforms and writes run on a separate worker. If a transform is still busy,
a display refresh is skipped; received samples and their coherent history remain
intact. The final map is completed before a successful exit. The metadata reports
the actual duration, frequency bins and coherence for each emitted map.

## Wire format

Little-endian `latest.rdm` has a 16-byte prefix:

| Offset | Type | Meaning |
| --- | --- | --- |
| 0 | 4 bytes | ASCII `RDM1` |
| 4 | uint32 | JSON header byte length |
| 8 | uint64 | Payload byte length |
| 16 | UTF-8 JSON | Coordinates, dimensions, sequence, segment, timing, normalization |
| 16 + JSON length | float32 complex | Range-major, then ascending signed Doppler |

The format is `cf32le`, with real and imaginary floats interleaved. Dimensions and
payload size must agree exactly. Each `latest.rdm` replacement is a complete frame.

`GET /api/frame` returns the same framing with `db32le` power values, ordered by
Doppler then range. Optional `rmin`, `rmax` (metres), `fmin`, `fmax` (Hz), `width`,
`height`, and `remove_dc=0|1` select the view. It crops the **native** map before maximum-power
pooling to the available pixels. `native_frequency_step_hz` remains explicit;
zooming recovers individual native bins. This display pooling changes only the
preview, never the stored complex map. `GET /api/native` downloads that map.
`POST /api/integration` accepts `{"seconds": 0}` for automatic integration or a
positive value within the processor's retained-history limit.

The server listens on 127.0.0.1 only. Browser UI operations: drag to pan, wheel to
zoom, Shift-wheel for range alone, Ctrl-wheel for Doppler alone, and double-click
to reset. Numeric axis limits and dB limits are editable. Pause stops preview
fetches; the terminal processor continues collecting coherent history.


## Lossless export stream

`dvbt_reconstruct ... --map-stream FILE` appends consecutive, self-framed `RDM1`
records to a new binary file. `--map-stream -` sends the same records to stdout
and sends diagnostics to stderr. This mode waits for every requested map transform
and stream write; it does not skip refreshes when the consumer is busy. The live
viewer’s `integration.txt` override is ignored during an export, keeping its CLI
integration setting reproducible. There is no real-time pacing in either mode.

The video CLI consumes this stream directly without archiving all native maps.
It chooses maps using their recording timestamps, preserves their relative time
axis, and repeats a map only until the next update becomes available. Gaps longer
than the update tolerance show an acquisition/no-data frame. Positive Doppler is
upward and increasing excess range is rightward. The colour scale stays fixed
unless `--auto-scale` is selected.
The JSON sidecar retains the complete sequence of scientific map metadata.

A video's duration is rounded up to a complete video frame so it covers the entire
requested interval. This is independent of elapsed processing time. FFmpeg consumes
RGB frames through its [rawvideo input](https://ffmpeg.org/ffmpeg-formats.html#rawvideo),
with explicit frame rate and image dimensions.
