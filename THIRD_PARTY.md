# Dependencies and provenance

* [GNU Radio 3.10.12.0](https://github.com/gnuradio/gnuradio/tree/v3.10.12.0):
  GPL-3.0-or-later. Runtime and FFT/filter support. Phil Karn's Reed–Solomon codec
  is used only as an independent FEC test oracle;
  GNU Radio's DVB-T TX chain is also an independent test oracle. The DVB-T mappings
  in `src/modem.cpp` are adapted from gr-dtv and marked GPL-3.0-or-later.
* [gr-dl5eu](https://github.com/dl5eu/gr-dl5eu/tree/d6ab8c12c1283f144b17e4c0ca2b6727a084339a):
  GPL-3.0-or-later, pinned commit `d6ab8c12c1283f144b17e4c0ca2b6727a084339a`.
  Only configuration, pilot generation and synchronization sources are compiled.
  Build-tree modifications are explicit in `scripts/patch_sync.py`: wideband
  fractional interpolation, double phase accumulation, continual-pilot clock PLL,
  sufficient FFT margin, raw measurement output and exact timing tags.
* [FFTW 3.3.11](https://www.fftw.org/): GPL-2.0-or-later; pinned archive SHA-256
  in `scripts/bootstrap.py`. Built locally with ARM NEON or x86 SIMD support.
  VOLK: GPL-3.0-or-later, supplied by the system.
* spdlog 1.15.3: MIT; fetched by the local bootstrap when building GNU Radio.
* VkFFT 1.3.4: MIT; fetched by CMake at a pinned commit and archive hash.
* Dear ImGui 1.91.9b: MIT, pinned archive SHA-256 in `web/CMakeLists.txt`.
  Its bundled Roboto Medium font is Apache-2.0 (Christian Robertson).
* Emscripten 4.0.23: MIT / University of Illinois-NCSA; project-local SDK build tools.
  Its pinned SDL 2.32.8 port is zlib licensed. These build only the separate viewer.
* NumPy 2.4.0: BSD-3-Clause; used by the local server for native-map cropping and
  peak-preserving previews. Playwright 1.58.0 (Apache-2.0) is optional test tooling.
* Pillow 12.1.0: HPND license; optional video labels and heatmap rasterization.
  FFmpeg is a system dependency for MP4 export; this host uses FFmpeg 8.1 with
  the GPL libx264 encoder. The exporter invokes it as a separate process.
* Vulkan headers/loader and glslang are supplied by the Vulkan SDK; see its
  component licenses. MoltenVK supplies Vulkan on this macOS host.

This pipeline links GPL components. Distribution of a combined executable requires
compliance with their licenses; it is not a permissively licensed standalone modem.
Dependency archives are checked against SHA-256 hashes. The build tree contains
upstream notices and source, rather than copying whole frameworks into this project.

DVB-T signal definitions follow [ETSI EN 300 744](https://dvb.org/wp-content/uploads/2019/12/a012_dvb-t_june_2015.pdf).
The candidate transmitter location/frequency comes from
[TX Australia's Melbourne site table](https://www.txaustralia.com.au/tvcoverage/metropolitan/melbourne):
MGV, channel 32, 557.625 MHz, Ornata Road, Mount Dandenong, site 12013;
table coordinates S37.837 E145.347 (GDA94). C31 service identity was independently
confirmed from the supplied recording's CRC-checked SDT. The coordinates are the
operator's rounded site coordinates, not a surveyed radiating phase centre.
