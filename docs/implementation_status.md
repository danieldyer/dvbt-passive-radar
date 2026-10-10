# Requirement audit — 2026-09-26

| Requested feature | Implemented result | Evidence |
| --- | --- | --- |
| Minimal CMake/C++23 and Vulkan compute foundation | Target-based CMake, pinned fetched dependencies, embedded SPIR-V; Vulkan residual product with CPU fallback | Debug/Release builds and GPU tests |
| DVB-T demodulation and reconstruction suitable for the Victoria recording | 7 MHz / 8K non-hierarchical receive path; all three constellations and five FEC rates; corrected RS packets re-encoded through the complete coding/interleaving/QAM chain | Independent GNU Radio TX comparison, 15-mode round trips, 4,079 BCH-valid TPS frames and CRC-valid C31 SI |
| File input, NCO, efficient decimation with no unreported filter phase | Float32 I/Q WAV reader, bounded reads, absolute-index NCO, centred symmetric FIR and frequency-domain polyphase folding | Two-tone/alias tests, partial blocks and seek phase checks |
| Correct SDR clock drift and maintain long coherence | Shared fractional resampler and continual-pilot carrier/timing loop before the measurement/decoder tee | Injected 5 ppm/CFO/fractional-delay/echo test; whole-recording phase and timing diagnostics |
| Minimum-phase compensation after resampling | Magnitude-only causal cepstral fit on resampled raw spectra, then matched-reference subtraction; no delay/all-pass term | Known-response and pure-delay exclusion tests; documented processing order |
| Preserve phase bookkeeping | Original ADC coordinates, applied phase and resampling ratio exported per symbol; fixed direct-signal origin per acquisition | Independent analytic DFT comparison plus signed complex range–Doppler tests |
| Residual cross spectrum in excess range and Doppler | Native complex maps, symmetric slow-time anti-alias filtering, actual CPI/FFT-grid/window metadata | Synthetic delayed/Doppler echoes and out-of-band alias test |
| Finest integration supported by measured coherence | Longest qualifying suffix of the configured retained history; gaps/invalid FEC reset history | Phase/timing drift tests; 301.219328-second CPI on the supplied recording |
| Real-time terminal processing | Entire 301.84-second recording processed in 213.70 seconds, including native maps | `build/radar-live.log`; 1.41× real-time throughput on the tested M5 |
| Separate browser GUI using WebAssembly and ImGui | Built Wasm application and local stream server; pan, zoom, dB scale, numeric range/Doppler limits, pause/resume and native downloads | Chrome rendering and interaction tests; native preview-crop/pooling tests |
| Adjustable integration time | On-screen automatic/manual controls update the running processor while retaining coherent history | Browser test changes to 2 seconds, observes the resulting map, and returns to automatic mode |
| Measured optimization without recording-specific recovery | Vulkan frontend/OFDM/reconstruction and auto-vectorized C++23 hard Viterbi/errors-only RS, constexpr tables, scalar oracle, compiler/assembly evidence | [Current execution breakdown](performance_breakdown_20261002.md) and [FEC report](fec_cpu_optimization.md); earlier CPU-only measurements remain in [performance history](performance.md) |

Six CTest checks pass in both Debug and Release; GPU Debug checks include Vulkan
synchronization validation. The server's numerical preview test and the browser
control test also pass. The browser was visually inspected with real recording data.

Scope and measured limits are explicit in [validation.md](validation.md) and
[range_doppler.md](range_doppler.md): throughput is measured for this QPSK recording;
clock stability is relative, not an absolute UTC calibration; minimum phase alone
does not separate every echo from hardware; and per-carrier cancellation beyond
cyclic-extension support is not qualified. No geographic target identification is
claimed. The final viewer remains available at http://127.0.0.1:8765/ while its local
server process is running.
