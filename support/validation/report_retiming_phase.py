"""Render the phase-preservation audit from saved CTest output and waveform results."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('docs/data/clock_audit_20261018/phase-preservation'))
    args = parser.parse_args()
    out = args.output
    data = json.loads((out/'pipeline-phase-histories.json').read_text())
    log = (out/'regression-details.log').read_text()
    if not data['passed'] or log.count('Test Passed.') != 10:
        raise ValueError('Expected the successful ten-test suite and waveform phase test')
    gpu_phase = float(re.search(r'Independent band phase histories.*?differential phase error=([\deE.+-]+)', log)[1])
    controller_phase = float(re.search(r'common-clock change=.*?\(([\deE.+-]+) degrees', log)[1])
    echo_rms = max(e['phase_rms_degrees'] for ch in data['channels'] for e in ch['echoes'])
    echo_max = max(e['maximum_phase_error_degrees'] for ch in data['channels'] for e in ch['echoes'])
    carrier_rms = max(ch['restored_carrier_phase']['phase_rms_degrees'] for ch in data['channels'])
    carrier_max = max(ch['restored_carrier_phase']['maximum_phase_error_degrees'] for ch in data['channels'])
    packets = sum(ch['packets'] for ch in data['channels'])
    failed = sum(ch['failed_packets'] for ch in data['channels'])
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    colours = ['#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00']
    t = np.linspace(0, 1360*8704/8e6, 1000)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for c, (ch, name, colour) in enumerate(zip(data['channels'], names, colours)):
        carrier = (c+1)*.8*np.sin(2*np.pi*(.35+.17*c)*t)+(.3-.2*c)*t*t
        axes[0, 0].plot(t, np.degrees(carrier), color=colour, label=name)
        trace = np.array(ch['restored_carrier_phase']['phase_error_time_s_degrees'])
        axes[0, 1].plot(*trace.T, color=colour, linewidth=.8)
        for echo, style in enumerate(['-', '--']):
            phase = (1+.2*c)*np.sin(2*np.pi*(.6+.11*c+.13*echo)*t+.31*echo)+(.25*c-.4)*(1+echo)*t*t
            axes[1, 0].plot(t, np.degrees(phase), style, color=colour, linewidth=.8)
            trace = np.array(ch['echoes'][echo]['phase_error_time_s_degrees'])
            axes[1, 1].plot(*trace.T, style, color=colour, linewidth=.8)
    axes[0, 0].set(title='Injected whole-band phase histories', ylabel='Phase (degrees)')
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].set(title='Carrier phase restored from response + window metadata',
                   ylabel='Error against injected carrier phase (degrees)', ylim=(-.27, .27))
    axes[1, 0].set(title='Injected echo-only phases; solid/dashed: echoes 1/2', ylabel='Phase (degrees)')
    axes[1, 1].set(title='Range-compressed echo phase against independent oracle',
                   ylabel='Phase error (degrees)', ylim=(-.27, .27))
    for ax in axes.flat:
        ax.set_xlabel('Time (s)')
    fig.suptitle('Independent band phase histories survive as explicit observables\n'
                 'No fitted output phase offsets or slopes; independent transmitter clocks and two moving echoes')
    fig.savefig(out/'phase-preservation.png', dpi=160)
    plt.close(fig)
    sources = ['src/sample_clock.cpp', 'src/sample_clock.hpp', 'src/gpu_frontend.cpp',
               'src/spectral_clock.cpp', 'shaders/spectral_synth.comp', 'shaders/spectral_boundary.comp',
               'src/pilot_phase.hpp', 'src/receiver_clock.hpp', 'src/gpu_receiver.cpp',
               'src/gpu_response.cpp', 'src/matched_profiles.cpp', 'src/live_archive.cpp',
               'tests/pilot_phase_tests.cpp', 'tests/spectral_clock_tests.cpp',
               'support/validation/pipeline_coherence_probe.cpp', str(Path(__file__))]
    summary = dict(passed=True, regression_tests=10, gpu_band_phase_max_error_degrees=gpu_phase,
                   pilot_rotation_controller_max_change_degrees=controller_phase,
                   echo_phase_max_rms_degrees=echo_rms, echo_phase_max_error_degrees=echo_max,
                   restored_carrier_max_rms_degrees=carrier_rms,
                   restored_carrier_max_error_degrees=carrier_max, rs_packets=packets, rs_failed=failed,
                   source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources})
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Retiming and interband phase preservation</title>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:30px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}code{{background:#eee}}</style>
<h1>Retiming and interband phase preservation</h1>
<p><strong>No independent receive time warp was found.</strong> One SampleClock map drives all channel outputs and NCOs. New regressions distinguish raw band carrier phase from echo phase relative to the direct path. Production numerical behavior is unchanged; tests and clarification of the phase contract were added.</p>
<table><tr><th>Check</th><th>Measured result</th></tr>
<tr><td>Independent phase histories through actual GPU retiming/channelisation</td><td>{gpu_phase:.6f}° maximum differential phase error</td></tr>
<tr><td>Phase-only pilot rotations, including noisy asymmetric pilots and changing precision</td><td>{controller_phase:.5f}° maximum change of the common-clock correction across the occupied span</td></tr>
<tr><td>Five-band receiver/FEC/reconstruction/finite range compression</td><td>{echo_rms:.5f}° worst echo RMS; {echo_max:.5f}° maximum individual error</td></tr>
<tr><td>Whole-band carrier phase restored from exported response and window coordinates</td><td>{carrier_rms:.5f}° worst RMS; {carrier_max:.5f}° maximum error</td></tr>
<tr><td>Reed–Solomon</td><td>{packets:,} packets; {failed} failed</td></tr></table>
<h2>What can and cannot remove phase</h2>
<p>The pilot likelihood profiles a complex intercept independently of the carrier-frequency slope. Multiplying a complete band by exp(iα) changes the intercept but leaves the delay likelihood unchanged in exact arithmetic. The paired-controller regression retains the small effect of rotating FP32 samples; it does not claim bitwise identity.</p>
<p>Per-channel FFT windows follow each transmitter's symbol timing. Their absolute window positions remain on the common receive grid. Fractional source offsets phase-shift the regenerated waveform about its channel centre; there is no independent receive resampling. The receiver does apply carrier/CFO corrections for demodulation, exporting their exact applied coordinates.</p>
<p>The matched filter references the echo to its channel's direct-path response. A phase rotation shared by both cancels intentionally. Echo-only phase changes remain. The old stationary-profile test consequently supports stability of direct-path-referenced responses, not a claim that raw interband carrier phase was unaltered. Channelised I/Q is archived before carrier tracking; live profile records also store response and row phase/frequency coordinates.</p>
<img src="phase-preservation.png" alt="Injected phase histories and independent reconstruction errors">
<h2>Scope and limitations</h2>
<p>The channeliser test covers five bands, ten seconds of analytic phase histories, large sample epochs and clock-segment boundaries. The waveform test separately exercises five independent 64-QAM sources, arbitrary streaming blocks, source-clock offsets, FFT-window steering, two moving echoes (one beyond the guard interval), carrier tracking, FEC, reconstruction and finite range compression. It compares to the transmitted waveform, without fitting output phases or gains. These are compositional tests; the full DVB-T waveform test starts from channelised I/Q and does not itself close the ADC retiming feedback loop. The pilot/controller and GPU channeliser regressions cover those separate interfaces.</p>
<p>One common time map can change relative phase in proportion to band separation when compared at inconsistent time coordinates. Its segments and the NCO origin therefore remain essential metadata. This is a coordinate/reference choice, not freedom to impose an arbitrary separate phase history on each receive channel. Changing propagation delay can bias timing observations; these checks do not certify the physical oscillator model, a CRLB-tight estimator on arbitrary interference, or successful ship image formation.</p>
<p><a href="summary.json">Summary and source hashes</a> · <a href="pipeline-phase-histories.json">Waveform measurements</a> · <a href="regression-details.log">Ten-test output</a> · <a href="pipeline-test.log">Extended carrier-restoration check</a> · <a href="../index.html">Earlier clock-estimator audit and retained stress limits</a></p></html>''')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
