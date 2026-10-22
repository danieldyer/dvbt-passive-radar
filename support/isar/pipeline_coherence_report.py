"""Publish independent physical-waveform phase checks of the receive chain."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    labels = ('Seven', 'SBS', 'Nine', 'Ten', 'ABC')
    trials, rows, values = [], [], []
    for path in args.results:
        data = json.loads(path.read_text())
        if not data['passed'] or len(data['channels']) != 5:
            raise ValueError('Require a completed passing five-channel physical probe')
        case = 'Unequal transmitter clocks' if any(c['source_rate_error_ppm'] for c in data['channels']) else 'Nominal transmitter clocks'
        for c, channel in enumerate(data['channels']):
            if channel['channel'] != c or channel['failed_packets'] or channel['packets'] <= 1000:
                raise ValueError('The full decoder was not successfully exercised')
            if channel['source_rate_error_ppm'] and not channel['steered_fft_windows']:
                raise ValueError('Clock drift did not exercise FFT-window steering')
            for echo, value in enumerate(channel['echoes']):
                if (value['comparisons'] < 30 or value['phase_rms_degrees'] >= .37
                        or abs(value['phase_error_degrees']) >= .37
                        or abs(value['amplitude_ratio']-1) >= .02):
                    raise ValueError('Physical phase/amplitude gate failed')
                rows.append([case, labels[c], str(echo+1),
                    f"{channel['source_rate_error_ppm']:+.2f}", str(channel['steered_fft_windows']),
                    f"{value['phase_error_degrees']:+.5f}", f"{value['phase_rms_degrees']:.5f}",
                    f"{value['maximum_phase_error_degrees']:.5f}",
                    f"{100*(value['amplitude_ratio']-1):+.5f}"])
                values.append(value)
        trials.append({'name': case, 'source': str(path), 'result': data})
    sources = [*args.results, Path(__file__), Path('support/validation/pipeline_coherence_probe.cpp'),
        Path('support/validation/CMakeLists.txt'), Path('src/matched_profiles.cpp'),
        Path('build/release/support/validation/pipeline_coherence_probe'),
        *[Path('build/release')/name for name in ('libradar_dsp.a', 'libradar_frontend.a', 'libdvbt_sync.a')]]
    report = {'complete': True, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': __doc__, 'trials': trials,
        'maximum_mean_phase_error_degrees': max(abs(v['phase_error_degrees']) for v in values),
        'maximum_phase_rms_degrees': max(v['phase_rms_degrees'] for v in values),
        'maximum_individual_phase_error_degrees': max(v['maximum_phase_error_degrees'] for v in values),
        'maximum_absolute_mean_amplitude_error_percent': max(100*abs(v['amplitude_ratio']-1) for v in values),
        'maximum_relative_complex_rms': max(v['relative_complex_rms'] for v in values),
        'source_sha256': {str(path): fingerprint(path) for path in sources},
        'production_changed': False, 'end_to_end_snr_certified': False,
        'limitation': 'Channelised input on a common receiver grid, strong noiseless signals, flat per-band receiver gain. This exercises receiver acquisition through range-profile moment zero, not ADC-clock/channelisation, archive ingestion or image formation. No recorded-data geometry or scattering claim.'}
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(trials), figsize=(7*len(trials), 4.5), squeeze=False, layout='constrained')
    for axis, trial in zip(axes[0], trials):
        for e in range(2):
            y = [c['echoes'][e]['phase_rms_degrees'] for c in trial['result']['channels']]
            axis.bar(np.arange(5)+(e-.5)*.35, y, width=.35,
                label='14.7 km echo' if e == 0 else '26.4 km echo (beyond guard)')
        axis.axhline(.37, color='#555555', linestyle='--', linewidth=1, label='0.37° gate')
        axis.set(xticks=np.arange(5), xticklabels=labels, ylim=(0, .4),
            ylabel='Phase error RMS (degrees)', title=trial['name'])
        axis.legend(fontsize=9)
    figure.savefig(args.output/'pipeline-coherence.png', dpi=160)
    plt.close(figure)
    cells = lambda values, tag: '<tr>'+''.join('<'+tag+'>'+html.escape(v)+'</'+tag+'>' for v in values)+'</tr>'
    table = cells(['Case', 'Band', 'Echo', 'TX rate error (ppm)', 'Steered windows',
        'Mean phase error (°)', 'RMS (°)', 'Maximum (°)', 'Mean amplitude error (%)'], 'th')
    table += ''.join(cells(row, 'td') for row in rows)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Physical waveform coherence</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Physical echo phase through the receive chain</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="pipeline-coherence.json">Measurements and source identities</a> · {report['updated_utc']}</p>
<p class="note">Both five-channel probes pass. Worst phase RMS is {report['maximum_phase_rms_degrees']:.4f}°, largest individual error {report['maximum_individual_phase_error_degrees']:.4f}°, and largest mean amplitude error {report['maximum_absolute_mean_amplitude_error_percent']:.4f}%. Every decoder packet check passes. No output phase or amplitude alignment is fitted to obtain these results.</p>
<h2>Independent input and oracle</h2>
<p>Five independent 64-QAM DVB-T payloads are modulated to time samples at 8 MS/s. Each receives its own direct-path gain, initial phase, delay and carrier offset. Two echoes have excess-path rates +45 and −27 m/s, initial delays 391.375 and 703.25 samples, amplitudes 0.03 and 0.02, and fixed intrinsic phases. The injection changes the waveform delay and RF carrier phase together. The second echo exceeds the 512-sample guard interval.</p>
<p>The production chain performs acquisition, symbol FFTs, equalisation, demapping, deinterleaving, Viterbi/RS decoding, source reconstruction, minimum-phase response fitting and finite-waveform range compression. Dispatch sizes vary, frame/FEC boundaries are crossed, and the input epoch exceeds 2⁴⁰ samples. The second test gives each transmitter a different clock rate, from −0.31 to +0.25 ppm, and verifies that every channel actually shifts its FFT windows.</p>
<p>The comparison uses a CPU FFT of the injected echo waveform restricted to the measured occupied carriers, then direct time-domain correlation against the original transmitted reference. It does not use reconstructed source carriers or fitted channel responses. The exported applied FFT frequency accounts for residual within-symbol CFO. Only moment zero and two selected physical delays are compared, at 80 rows per echo and band.</p>
<h2>Results</h2><img src="pipeline-coherence.png" alt="Per-band physical phase RMS for nominal and unequal transmitter clocks">
<table>{table}</table>
<h2>Normalization and scope</h2>
<p>The initial oracle divided by each symbol's actual waveform energy and consequently reported an approximately 8% amplitude disagreement. The profile format instead uses the fixed unit-carrier scale 6817/8192 for this flat-channel, unit-noise case. Actual pilot boosting and symbol power remain in the pulse response. Correcting that independent oracle convention removes the discrepancy; the production code and acceptance thresholds were unchanged. The failed first comparison is retained in the build directory.</p>
<p>This validates physical phase and amplitude bookkeeping across the chained stages under the injected model. It is not a noise-limited efficiency or global CRLB certificate. Raw ADC retiming/channelisation, archive coordinates and image formation have separate tests. Real coloured interference, transmitter antenna patterns, scene correspondence and location calibration still require their own evidence. The independent clock test is not an assertion that the broadcast transmitters share a frequency or phase reference.</p></html>'''
    (args.output/'pipeline-coherence.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'pipeline-coherence.html').write_text(document+'\n')
    print(json.dumps({k: v for k, v in report.items() if k.startswith('maximum_')}))


if __name__ == '__main__':
    main()
