# DVB-T implementation choice

The assessment below preceded implementation. The current frontend uses the
selected dependencies with a custom soft decoder/reconstruction path and an
instrumented synchronizer. It has decoded the supplied C31 recording and passed
the tests in [validation.md](validation.md). See the [current build and usage](../README.md).

Assessment date: 2026-09-26. Recommend **GNU Radio 3.10 `gr-dtv`** as the first
complete C++ transmit/receive implementation to evaluate. The inspected release
is [v3.10.12.0](https://github.com/gnuradio/gnuradio/releases/tag/v3.10.12.0).
Use its coding/modulation blocks together with an evaluation of **`gr-dl5eu`'s
receiver synchronization/TPS blocks** for the clock-recovery requirement. The goal
is corrected symbols and a coherent regenerated reference, with decoding status
and timing metadata available throughout. MPEG-TS playback/export is unnecessary.
These are building blocks rather than a turnkey radar reference generator; the
Vulkan-only build remains available with `RADAR_BUILD_PIPELINE=OFF`.

| Candidate | Transmit / receive | Assessment |
| --- | --- | --- |
| [GNU Radio `gr-dtv`](https://github.com/gnuradio/gnuradio/tree/v3.10.12.0/gr-dtv) | Both, C++ blocks | Best match for functional coverage; requires GNU Radio runtime and supporting DSP libraries. |
| [dl5eu/gr-dl5eu](https://github.com/dl5eu/gr-dl5eu) | Receiver additions, C++ | Promising replacement synchronization, TPS and Viterbi blocks; requires GNU Radio. Adds clock tracking and useful telemetry. |
| [BogdanDIA/gr-dvbt](https://github.com/BogdanDIA/gr-dvbt) | Both | Older GNU Radio 3.7-era implementation; README requires SSE2. Prefer current in-tree `gr-dtv`, particularly on ARM. |
| [siro20/dvbt-encoder](https://github.com/siro20/dvbt-encoder) | Transmit only, C++/FFTW | Smaller modulator, but no demodulator; README has contradictory hierarchy claims and incomplete mode testing. Does not meet the paired TX/RX requirement. |

[leandvb](https://github.com/pabr/leansdr) is for DVB-S, so its lightweight design
does not make it a terrestrial DVB-T solution. Linux DVB tuner APIs and MPEG-TS
parsers also do not demodulate recorded IQ.

The in-tree [C++ sources](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/lib/CMakeLists.txt)
cover energy dispersal, Reed-Solomon, convolutional coding/Viterbi, interleaving,
constellation mapping, pilots/TPS, OFDM acquisition, and the corresponding receive
operations. Combine them with GNU Radio's FFT, cyclic-prefix and file-I/O blocks.
An application can wire these blocks directly in C++; Python and the GUI are not
required to orchestrate the processing.

The [configuration implementation](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/lib/dvbt/dvbt_configure.cc)
includes 2K/8K, QPSK/16-QAM/64-QAM, code rates 1/2, 2/3, 3/4, 5/6, 7/8, and guard
intervals 1/4, 1/8, 1/16, 1/32. These cover the common Australian DVB-T profile.
This is source-level coverage, not a claim that every combination has been tested
here or that all optional hierarchical/DVB-H features are complete.

For Victoria, use Australian **7 MHz channels**. Free TV Australia's
[OP-71, section 7](https://www.freetv.com.au/wp-content/uploads/2019/08/OP-71-Recommended-DVB-T-Transmitter-Modulator-Settings-Issue-1-December-2014.pdf)
documents 8K, 64-QAM, FEC 3/4 and guard interval 1/16 as the common broadcast
configuration. That document is from 2014 and is not a current survey of each
Victorian multiplex. Confirm the parameters of each recording; retain configurable
FEC and guard interval rather than hard-coding one transmitter profile.

| Parameter | Initial Australian profile |
| --- | --- |
| RF channel bandwidth | 7 MHz |
| Complex sample rate at the OFDM processing boundary | **8 MS/s** |
| FFT length | 8192 |
| Active carriers / data carriers | 6817 / 6048 |
| Constellation / hierarchy | 64-QAM / non-hierarchical |
| Inner code rate | 3/4, subject to the actual multiplex |
| Guard interval | 1/16 = 512 samples, subject to the actual multiplex |
| Useful symbol / guard duration | 1024 microseconds / 64 microseconds |
| Full symbol length | 8704 samples = 1088 microseconds |

The 8 MS/s rate follows from the 7 MHz elementary period `T = 1/8 microsecond`
in [DVB-T EN 300 744, Annex E](https://dvb.org/wp-content/uploads/2019/12/a012_dvb-t_june_2015.pdf).
The sample counts/durations above follow from that rate and the stated FFT/guard
lengths. Channelize and resample recordings made at other sample rates to this
boundary. A 7 MHz channel does **not** imply 7 MS/s complex input to the decoder.

Start from the upstream [8K transmitter](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/examples/dvbt_tx_8k.grc)
and [8K receiver](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/examples/dvbt_rx_8k.grc)
block connections. Their defaults use 8 MHz channel timing, code rate 2/3 and a
different guard interval: update all related blocks consistently for Australian
captures. Replace radio/GUI blocks with file sources/sinks for offline evaluation.

Efficiency needs measurement on the intended machine. `gr-dtv` uses VOLK in DSP
paths, but its [DVB-T Viterbi decoder](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/lib/dvbt/dvbt_viterbi_decoder_impl.cc)
has SSE2 and generic branches. Apple Silicon uses the generic branch in this
implementation; other blocks may still benefit from VOLK's ARM kernels. It is a
CPU modem, not a Vulkan modem. This source assessment initially had no off-air
benchmark; the project's implemented soft decoder and capture measurements are
documented separately in [validation.md](validation.md).

There are also receiver limitations worth keeping explicit. The
[reference-signal implementation](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/lib/dvbt/dvbt_reference_signals_impl.cc)
uses TPS for frame synchronization, but its public TPS getters/setters are empty
and mode parameters come from the configured blocks. It is not a turnkey
automatic multiplex-parameter detector. Pilot-based carrier correction exists;
sampling-clock tracking and robustness to actual SDR impairments need capture
tests. The source is GPL-3.0-or-later, which is relevant to dependency selection.

The [gr-dl5eu synchronization implementation](https://github.com/dl5eu/gr-dl5eu/blob/d6ab8c12c1283f144b17e4c0ca2b6727a084339a/lib/dvbt_ofdm_synchronization_impl.cc)
contains a fractional interpolator driven by a sample-clock loop, a carrier loop,
and optional channel-estimate, normalized frequency-error and sample-error
outputs. This is a closer fit for clock correction than stock acquisition alone.
It still needs extra telemetry for absolute input sample mapping and continuous
phase across acquisition resets. Its equalized output belongs in the decoder;
retain a separate measurement branch before channel equalization.

The accompanying [TPS decoder](https://github.com/dl5eu/gr-dl5eu/blob/d6ab8c12c1283f144b17e4c0ca2b6727a084339a/lib/dvbt_tps_decoder_impl.cc)
checks BCH parity, extracts mode/frame/cell fields and has message outputs for
constellation, transmission mode, FEC and guard interval. Preserve full TPS bits
and validity for reconstruction too. This is a small out-of-tree project,
GPL-3.0-or-later, inspected at commit
`d6ab8c12c1283f144b17e4c0ca2b6727a084339a`. Its synchronizer has now been built,
instrumented and substantially modified for this project's phase/timing contract.
Its Viterbi implementation also retains SSE2/generic branches.

A concrete integration gap is FEC validity: the stock
[RS decoder](https://github.com/gnuradio/gnuradio/blob/v3.10.12.0/gr-dtv/lib/dvbt/dvbt_reed_solomon_dec_impl.cc)
counts uncorrectable packets internally but still emits bytes, without a
per-packet validity output. Export correction count/failure and packet-to-symbol
mapping before using its output as a reference. A decoder returning bytes is
insufficient evidence of successful reconstruction.

The integration now provides a C++ reconstruction frontend and Vulkan residual
cross spectrum. See [reference_reconstruction.md](reference_reconstruction.md)
for the implemented signal/metadata contract and its limits. The common 64-QAM
profile above is not the supplied capture's profile: its BCH-checked TPS identifies
QPSK, 8K, 3/4, GI 1/16, and CRC-checked SDT identifies C31.
