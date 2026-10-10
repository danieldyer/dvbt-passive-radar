# Validation on 2026-09-26

For the current channelizer and shared ADC-clock implementation, see
[the frontend validation and kernel report](vulkan_frontend.md) and
[the 6 October concurrency and coherence regressions](pipeline_concurrency_20261006.md). The
filter parameters and full-recording measurements below describe the earlier CPU
frontend.

The later [performance pass](performance.md) repeats the full-recording export
and phase/FEC comparison after optimization: 149.87 seconds for the 301.84-second
recording, with the same 1,504 scientific maps, 277,368 fully valid symbols and
zero RS failures. Historical timings below describe the original implementation.

Host: Apple M5, AppleClang 21, CMake 4.4.3, Vulkan SDK/MoltenVK 1.4.328.1.
Debug and Release tests cover CPU processing and Vulkan. Debug GPU checks enable
Khronos synchronization validation; no validation errors were reported.

## Automated checks

* **Channelizer:** an independently generated two-tone float32 WAV exercises file
  input, NCO sign and absolute phase, centred FIR delay compensation, decimation,
  overlap boundaries and the final partial block. Maximum complex error over four
  FFT blocks stays below **2e-6**, including a rejected alias tone. The inverse
  transform uses frequency-domain polyphase folding rather than computing and
  discarding six out of seven time samples.
* **Prototype response:** occupied-band amplitude ripple **8.89e-7**, centred
  phase error below **1e-15 rad**, stopband at/above 4 MHz **−121.6 dB**.
* **Fractional interpolator:** tested at DC and both outer occupied carriers over
  101 fractional phases; maximum complex error **5.67e-6**.
* **Complete synchronization bookkeeping:** synthesize actual DVB-T symbols,
  inject **5 ppm** sample-rate error, a **173.375-sample** delay, **85.25 Hz** CFO,
  an initial carrier phase, and a **37.25-sample / 29 Hz** Doppler echo. Compare
  un-equalized receiver output against a closed-form finite-sinusoid DFT using
  the exported sample/phase coordinates. The expected spectrum uses the known
  injected clock and signal, independently of the receiver's estimated clock.
  Relative complex RMS error: approximately **5e-6**. This exercises the WAV/NCO/centred channelizer, interpolation,
  fractional coordinates, continuous oscillator phase, FFT sign and retention of
  the delayed Doppler component across scheduler blocks and clock-loop acquisition.
* **DVB-T coding:** all **15** non-hierarchical constellation/code-rate combinations
  round-trip; 64-QAM/3/4 payload carriers also agree exactly with GNU Radio's
  independent full RS/interleaver/convolutional/QAM transmitter chain. TPS frame
  progression and BCH pass on generated signalling. Destructive symbol corruption
  triggers FEC masks while preserving output positions.
* **Response estimator:** recovers a known minimum-phase response and leaves a
  pure all-pass delay outside the fitted response.
* **Range–Doppler:** independent positive/negative complex tones verify FFT sign,
  phase and normalization. An active-carrier delayed-echo test exercises the range
  IFFT, slow-time FIR and decimation, checking excess-delay origin, absolute complex
  phase and rejection of an out-of-band Doppler alias. Separate phase and timing
  drift tests shorten the automatic coherent interval.
  DC removal rejects a strong complex stationary return and its Hann leakage
  while retaining the amplitude and phase of weaker signed Doppler echoes.
  A non-bin-centred echo with 1,013 samples and FFT padding agrees with an
  independent direct complex DFT after weighted-mean subtraction to **1e-8**.
* **Viewer:** native complex-map crops and maximum-power preview pooling agree
  numerically. Headless Chrome runs the actual ImGui/WebAssembly application;
  interaction checks cover pan, wheel zoom, pause/resume, live map progression,
  integration requests and retrieval of native Doppler bins on zoom.
  The completed-map DC projection agrees with time-domain subtraction, including
  crops excluding zero Doppler and legacy metadata; tests also check that an
  already filtered source is not filtered twice.
* **GPU:** residual subtraction and conjugate product agree with CPU results for
  multiple buffer sizes, including partial workgroups and buffer growth. The
  original VkFFT/cross-spectrum tests also pass.

Sources: [DSP/coding tests](../tests/dsp_tests.cpp),
[analytic phase test](../tests/phase_tests.cpp),
[range–Doppler tests](../tests/range_doppler_tests.cpp),
[preview tests](../tests/viewer_server_tests.py) and
[browser checks](../scripts/test_viewer.py). Run `ctest --preset release` or
`ctest --preset debug` with the SDK environment active. A sandbox without Metal
access cannot run the GPU tests; `ctest --preset release -L cpu` selects CPU checks.

The phase test caught and drove correction of an actual boundary problem: an FFT
window only 10 samples inside the cyclic extension is insufficient for a 96-tap
fractional interpolator. The implemented margin is 128 samples and its reference
phase is explicitly compensated.

## Supplied recording: complete 301.84-second check

Input: `baseband_553000000Hz_12-51-54_26-09-2026.wav`, all
16,903,040,000 complex samples at 56 MS/s. The original recording was not modified.

```sh
build/release/dvbt_reconstruct INPUT.wav \
  --output build/radar-live --history-seconds 360
```

* NCO offset: **4.625 MHz**; recovered nominal output rate: **8 MS/s**.
* BCH-validated TPS: **8K, QPSK, 3/4, GI 1/16**, non-hierarchical.
* CRC-validated service descriptor: **provider C31, service C31**.
* **4,079** checked TPS frames; **277,368** fully FEC-valid emitted symbols.
* **1,775,036 RS examinations, zero failures**. Examinations include overlapping
  context; this is not a count of unique packets.
* **213.70 seconds elapsed** for 301.84 seconds of input: **1.41× real time** on
  this host, including channelization, clock correction, decoding, response fits,
  native complex maps and diagnostics. The browser was connected for part of this
  run. Peak resident memory was approximately **5.86 GB**; the long coherent
  history and native 801-by-60,915 maps are substantial allocations.
* An earlier optimized 60-second run took **39.60 seconds**. These measurements
  qualify the supplied QPSK recording and this machine; they are not a throughput
  qualification for every constellation/code rate, CPU or SSD.

After excluding the first second and invalid rows, the fixed received-channel
coherence diagnostic measured:

| Quantity | Mean | Standard deviation | Observed range |
| --- | ---: | ---: | ---: |
| Common phase residual | 0.125° | **0.443°** | −1.683° to +2.036° |
| Timing residual at 8 MS/s | 0.00106 samples | **0.000858 samples** | −0.00455 to +0.00779 samples |
| Applied input/output rate error | −0.0857 ppm | 0.00770 ppm | −0.1270 to −0.0384 ppm |

Timing standard deviation is about **107 ps**, equivalent to approximately **0.129°**
at an outer occupied carrier. These are residual stability measurements relative
to a received-signal template, not absolute clock accuracy, transmitter certification,
antenna-position accuracy or UTC timing. The common-phase-only coherent amplitude
ratio is 0.999970 (approximately **0.00026 dB** loss); target acceleration, range
migration and amplitude variation can impose additional coherence limits.

The final map uses **301.219328 seconds** of data, has **0.00328331 Hz** FFT bins
and a Hann equivalent noise bandwidth of **0.00497976 Hz**. The measured coherence
proxy is approximately **0.9999666** at the occupied-band edges. The FFT includes
slight zero padding; physical resolution must be assessed using the CPI/window,
not bin spacing alone. Spatial bins are **37.47405725 m** of bistatic excess path.

The order-8 minimum-phase model gives roughly **12.5 dB** residual/input power
suppression. This is not a noise-floor claim or proof that every multipath component
is preserved: minimum-phase parts of multipath can enter the fitted response.
No candidate MIP PID appeared; observed PCR values are program clocks and were
not used for RF clock correction. Geographic targets have not been independently
identified or surveyed.

Retained evidence: `build/radar-live/latest.rdm`, `build/radar-live.log`,
`build/radar-live/validation_summary.json`, CPU/GPU CTest logs, and browser logs
under `build/`. A two-second run with all intermediate binary products remains
in `build/reconstruction-2s/`. Build artifacts are ignored; tests and this report
are the durable evidence. The generic metadata field `coherence_qualified` remains
false because these measurements do not qualify arbitrary recordings or durations.

To repeat the browser checks, install the optional test dependency in the local
server environment (`pip install playwright==1.58.0`) and run:

```sh
build/web-venv/bin/python tests/viewer_server_tests.py
source build/sdk/setup-env.sh
build/web-venv/bin/python scripts/test_viewer.py --launch INPUT.wav
```

The browser test uses installed Chrome, starts its own local server and a bounded
recording, and stops that test process after checking the controls. The production
viewer requires neither Playwright nor an installed browser automation runtime.

## Full video export and DC removal

The same 301.84-second recording was exported with a ten-second sliding CPI,
0–30 km excess range, ±100 Hz Doppler, DC removal and automatic colour scaling.
The grid-free H.264 video is 1920×1080 at 5 fps: **1,510 frames / 302.0 seconds**,
including the initial labelled window fill. Rendering ran unpaced and took
**229.27 seconds (1.32× real time)** including reconstruction and encoding.

All **1,504** scientific map sequences were contiguous; the largest recording-time
update interval was **0.200192 seconds**. The steady CPI is **9.996544 seconds**,
rounded down to complete slow-time samples. The final map spans 291.533618 to
301.525809 seconds; the remaining interval includes the symmetric filter's
lookahead tail and is displayed using that last valid map. FEC checks again
reported **zero RS failures** and **277,368 fully valid symbols**.

For the final ten-second map, the CLI's time-domain DC subtraction and the GUI's
complex spectral projection agree to **1.78e-8 maximum complex error**, or
**3.88e-8 relative to the unfiltered peak**. The filtered zero-bin amplitude is
below **5.88e-11**. The final map's peak falls from −6.80 to −44.61 dB; this is a
stationary-clutter suppression measurement, not a target/noise-floor claim.

Three CPU CTests and six Python checks pass. Chrome checks the actual DC checkbox,
completed-map and paused-map refresh, autoscaled display, pan/zoom and native-bin
requests without JavaScript errors. The saved video was probed for codec, size,
frame rate, duration and frame count, decoded in full, and visually inspected.

Artifacts: `renders/C31_10s_0-30km_pm100Hz_dc_removed.mp4`, its PNG/JSON companions,
and `build/video-dc-validation.json`. `build/dc-browser-tests.log` records the
completed-map GUI check; `build/dc-python-tests.log` and `build/dc-cpu-tests.log`
record the numerical checks. Repeat the Python checks with:

```sh
build/web-venv/bin/python -m unittest discover -s tests -p '*_tests.py' -v
```
