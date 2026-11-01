# Dependencies and provenance

Production DSP uses these components:

* [FFTW 3.3.11](https://www.fftw.org/): GPL-2.0-or-later. CPU FFTs used for
  coefficient preparation. The local bootstrap checks its archive SHA-256.
* [VkFFT 1.3.4](https://github.com/DTolm/VkFFT): MIT. Pinned CMake commit/archive
  hash; Vulkan compute transforms compiled with glslang.
* Vulkan headers/loader and glslang: supplied by the Vulkan SDK or target system;
  consult their component notices. MoltenVK provides Vulkan on macOS.
* [Boost.JSON](https://www.boost.org/doc/libs/latest/libs/json/): Boost Software
  License 1.0. Compiled once from the installed Boost headers.
* UHD: GPL-3.0-or-later; optional B200mini capture driver. The optional uSDR
  backend uses the system `usdr-lib` native DMA acquire/release branch. Its driver
  source and notices are maintained in the separate PL64 packaging repository.
* FTXUI 7: MIT, installed package or hash-checked pinned source in CMake.
* Dear ImGui 1.91.9b and ImPlot 0.16: MIT; hash-checked source in
  `support/live/native.cmake`. GLFW: zlib/libpng license. These supply the optional
  macOS Metal display. ImGui's bundled Roboto Medium font is Apache-2.0.

DVB-T mappings, pilot definitions and TPS BCH arithmetic in `src/modem.cpp` and
`src/dvbt_pilots.hpp` follow ETSI EN 300 744 and GNU Radio gr-dtv. These files carry
GPL-3.0-or-later notices. The capture executable does not link GNU Radio.

Independent tests use [GNU Radio 3.10.12.0](https://github.com/gnuradio/gnuradio/tree/v3.10.12.0)
(GPL-3.0-or-later), including Phil Karn's RS codec and the DVB-T transmitter chain.
The test synchroniser and pilot oracle use
[gr-dl5eu](https://github.com/dl5eu/gr-dl5eu/tree/d6ab8c12c1283f144b17e4c0ca2b6727a084339a),
pinned to that commit with an archive hash. Audited test-only changes appear in
`scripts/patch_sync.py`. GNU Radio brings its runtime, FFT/filter, VOLK
(GPL-3.0-or-later), spdlog (MIT), PMT and other transitive dependencies into tests.
They are not selectable production decoders.

Archive video tooling uses NumPy 2.4.0 (BSD-3-Clause), Pillow 12.1.0 (HPND), and a
separate system FFmpeg process with the macOS VideoToolbox encoders. FFmpeg's
license depends on its system build; no FFmpeg library is linked into capture.
The optional archive-format study uses MCAP C++ 2.1.1 (MIT) and FlatBuffers
25.12.19 (Apache-2.0); neither is a production dependency yet.

The combined executable links GPL components and includes GPL-derived code;
distribution must comply with those terms. Pinned dependency sources and their
notices remain in the build tree. Signal definitions follow
[ETSI EN 300 744](https://dvb.org/wp-content/uploads/2019/12/a012_dvb-t_june_2015.pdf).
Antenna positions in metadata are user-supplied nominal priors with provenance,
not a surveyed or completed radar calibration.
