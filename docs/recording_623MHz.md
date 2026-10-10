# Acquisition diagnostics: 623 MHz recording

Input: `/Users/danielrd/data/dvbt_passive_radar/baseband_623000000Hz_12-58-54_26-09-2026.wav`.
Tested on 2026-09-26. The original file was read only. Recovery work was stopped
at the user's request; the production changes made for this recording were reverted.

**Investigation closed at the user's request.** The upstream wall-port feed is
unknown; the user reports a badly corroded, loose socket and suspects a
distribution amplifier. Passive intermodulation is one unconfirmed possibility.
No further recovery or recording-specific compensation is planned. Diagnostic
scripts remain outside the production build and execution path; the generic
positive/negative-frequency channelizer regression is retained.

**No radar video was produced.** The current receiver did not sustain fully
FEC-validated reconstruction, so the requested ten-second coherent maps have not
been qualified for this recording. The existing C31 videos remain unchanged.

## Measurements

**Interpretation correction after the channel-selection audit:** the measurements
below describe an unidentified, inverted DVB-T candidate. They do not establish
the quality of the intended 610–617 MHz broadcast. Selecting an image, folded
alias or mixing product remains a plausible explanation. Neither its service
identity nor the physical origin of its frequency behaviour is established.

* The float32 stereo I/Q WAV declares 56 MS/s. Recovering its wrapped RIFF lengths
  gives **16,836,400,000 complex samples / 300.65 seconds**.
* The prominent lower-side DVB-T-shaped signal is centred at approximately
  **−8.085 MHz in the recorded complex spectrum**. This is a measured baseband
  offset, not a verified RF channel assignment. Probes at the user's suggested
  nominal-channel offsets (−9.5 and −9.375 MHz) did not acquire validated TPS.
* Matching the known continual-pilot pattern requires **spectral inversion**.
  Short-symbol pilot correlation reaches approximately **0.96** with inversion;
  the non-inverted search has no comparable match.
* Local cyclic-prefix correlation reaches approximately **0.92**, and prefix
  spacing is **8,704 samples at 8 MS/s**. A complex average over many symbols
  hides those boundaries because their phase changes strongly.
* In the first four seconds, high-confidence pilot/prefix measurements put the
  changing carrier offset at approximately **−11.7 to +18.2 kHz**, relative to
  the selected, inverted channel centre. Its spectrum has a strong component
  near **50 Hz**, plus harmonics. This establishes a periodic frequency error
  in the received waveform; it does not establish whether its cause is the
  transmitter, receiver, an image/interferer, or recording software.
* Bounded diagnostic recovery obtains BCH-validated **8K, QPSK, rate 3/4,
  GI 1/16** TPS frames. CRC-validated service identification was not obtained;
  the nominal 610–617 MHz channel and transmitter are therefore unconfirmed.

## Recovery checks

The regular receiver was tried on bounded intervals with normal and inverted
carrier ordering, multiple guard intervals, and the measured/nominal offsets.
All completed production probes emitted **zero fully FEC-valid symbols**.
Some unsuccessful acquisition probes were terminated at a bounded timeout.

Additional offline experiments on the first four seconds used prefix/pilot
frequency estimates, integrated frequency correction, periodic harmonic fits,
local fits, and decision-directed phase refinement. They were diagnostic only
and were not incorporated into the production signal path.

After the local harmonic correction, an independent symbol extraction found
7 BCH-valid TPS frames; the longest contiguous run was two frames. Its decoder
reported **388 failed RS checks out of 745**. A separate packet-only diagnostic
over a longer span reported **6,298 failures out of 12,085**. That diagnostic did
not use its generated reference waveform as a radar oracle.

Decision-directed refinement increased the count to **26 BCH-valid TPS frames**,
with a longest contiguous run of **six frames (0.444 seconds)**. Even this best
run had **878 failed RS checks out of 2,257**. These packet counts include decoder
context and do not establish unique-packet throughput or a recording-wide BER.
The normal receiver also failed to sustain reconstruction on the corrected
diagnostic WAV. A short valid TPS run alone does not qualify clock coherence or
ten-second range–Doppler integration.

No invalid packets were promoted to valid, no gaps were compressed out of time,
and no experimental correction was used to publish a radar map or movie.

## Independent channel-selection and sign audit

Moving a channel below baseband DC does **not** reverse its carrier ordering.
For a centre offset of −9.5 MHz, the production `--offset-hz -9500000` applies
`exp(+j 2 pi 9.5e6 n / 56e6)`, preserving the ordering within the channel.
Conjugation is not part of that operation.

[ACMA's October 2025 station list, printed page 362](https://www.acma.gov.au/sites/default/files/2025-10/9%20Televison%20by%20Area%20Served.pdf)
lists Melbourne Inner Suburbs SBS on channel 40 at **613.5 MHz**, from Bourke
Place. With the recording's stated 623 MHz tuning centre this predicts
**−9.5 MHz**. The previously selected candidate near **−8.085 MHz** is roughly
**1.415 MHz away**. The RF assignment is conditional on the filename's tuning
centre being accurate; a spectral shape alone cannot identify the service.

The current production channelizer was compiled into a diagnostic dumper and
compared with a separate **double-precision, direct time-domain 1,025-tap FIR**.
The reference used exact integer modular NCO phase and absolute input sample
coordinates, without FFT convolution or spectral alias folding. Checks sampled
510 output positions per case, including both sides of block boundaries:

| Channel centre | Recording time | Relative RMS complex error |
| --- | --- | --- |
| −9.5 MHz | 0 s | 2.79e−7 |
| −9.5 MHz | 60 s | 4.15e−7 |
| −9.5 MHz | 299 s | 5.34e−7 |
| −8.085 MHz | 60 s | 3.16e−7 |
| +9.5 MHz | 60 s | 4.55e−7 |

The maximum aggregate phase difference was 0.000026 degrees. These checks
found no NCO-sign, conjugation, decimation-folding or filter-delay discrepancy
at the sampled positions. They do not diagnose aliasing already in the WAV.

A second check evaluated DVB-T continual-pilot ordering using **57,344-point
FFTs of the original 56 MS/s samples**, before the production NCO, FIR or
decimator. An independent ideal channel selection located symbol boundaries;
the pilot measurements themselves came directly from full-rate FFT bins.
Both orientations received the same bounded carrier-offset/timing search.
Median pilot correlation over 32 symbols near 60 seconds was:

| Recording / candidate | Normal order | Reversed order |
| --- | --- | --- |
| 553 MHz recording / +4.625 MHz | 0.990 | 0.204 |
| 623 MHz recording / −8.085 MHz | 0.243 | 0.913 |

Thus the second candidate's reversed ordering is present in the recorded
samples; it is not introduced by our channelizer. This supports investigating
an image/alias but does not distinguish an ADC fold, mixer image, intermodulation
product or recording/tuning issue.

An independent broad scan of 0.13-second intervals at 60 and 200 seconds used
16 channel centres and all four 8K guard intervals. It reproduced the lower-band
8K/GI 1/16 candidate and did not identify another normal-order DVB-T service.
This bounded search is not proof that a weaker desired service is absent.

The permanent channelizer regression now exercises both +4.625 and −9.5 MHz
centres, asymmetric in-band tones, an out-of-band blocker, block boundaries and
seeking. Both signs have maximum complex error below 1.06e−6; all three CPU
CTest suites passed. The production pipeline itself was not changed.

Reproducible audit evidence is in `build/audit623.py`,
`build/audit623_channelizer.cpp`, `build/audit623-results.json`,
`build/raw623_pilots.py`, `build/raw623-pilots.json`, `build/scan623.py`,
`build/scan623-{60,200}.json` and `build/channel623-sign-tests.log`.

## Retained diagnostic evidence

The experimental `--invert-spectrum` option and its associated code, metadata
and tests were removed when recovery work stopped. The existing video exporter,
DC removal and GUI controls remain as previously validated. No recording-specific
recovery path remains in the production pipeline.

Numerical evidence and bounded experiments are under `build/`, including:

* `channel623-spectrum.json`, `channel623_cfo.csv`, `channel623-periodic.log`;
* `channel623-inverted.log`, `channel623-local-probe.log`, `channel623-dd-probe.log`;
* `channel623-fec-probe.log`, `channel623-metadata-probe.log`, `channel623-dd-fec.log`;
* `channel623-inversion-tests.log`, `channel623-python-tests.log`.

These results establish unsuccessful acquisition/reconstruction of the selected
candidate, not that the intended broadcast is damaged or the file irrecoverable.
These observations are retained as diagnostic evidence, not as a request for
further acquisition or phase-recovery work.
