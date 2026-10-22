"""Publish finite-scene information, estimator-bias and coherent-block controls."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


@lru_cache(None)
def digest(path):
    return fingerprint(path)


def load(path, archives):
    data = json.loads(path.read_text())
    if not data['complete']:
        raise ValueError('Unfinished analysis: '+str(path))
    substitutions = {}
    for source, expected in data['source_sha256'].items():
        if digest(Path(source)) == expected:
            continue
        archive = next((p for p in archives if digest(p) == expected), None)
        if archive is None:
            raise ValueError('Missing original source version: '+source)
        substitutions[source] = str(archive)
    for channel in data.get('channels', []):
        if digest(Path(channel['arrays'])) != channel['arrays_sha256']:
            raise ValueError('Saved forward matrix changed')
    return data, substitutions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('average', 'per-row', 'per-block', 'monte-carlo', 'tests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--implementation-archive', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    paths = [args.average, args.per_row, args.per_block, args.monte_carlo]
    loaded = [load(path, args.implementation_archive) for path in paths]
    average, row, block, mc = [item[0] for item in loaded]
    for other in (row, block):
        for key in ('requested_interval_s', 'points_body_m', 'range_offsets_samples',
                    'averaging_rows', 'observation_centre'):
            if other[key] != average[key]:
                raise ValueError('Unmatched finite-scene experiment: '+key)
    tests = args.tests.read_text(); count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require completed support tests')
    channels = []
    for a, b, c in zip(average['channels'], row['channels'], block['channels']):
        if not a['channel'] == b['channel'] == c['channel']:
            raise ValueError('Channel order changed')
        columns = a['supports'][0]['columns']
        with np.load(a['arrays']) as x, np.load(b['arrays']) as y, np.load(c['arrays']) as z:
            for name in ('raw_root', 'exact_compressed', 'times_s'):
                np.testing.assert_array_equal(x[name], y[name])
                np.testing.assert_array_equal(x[name], z[name])
        channels.append({'channel': a['channel'], 'probes': a['probes'],
            'blocks': a['coherent_blocks'], 'rejected_blocks': a['rejected_blocks'],
            'complex_amplitudes': len(columns),
            'average_pulse_risk': a['supports'][0]['uncorrected_mean_gls_risk'],
            'per_row_risk': b['supports'][0]['uncorrected_mean_gls_risk'],
            'per_block_risk': c['supports'][0]['uncorrected_mean_gls_risk'],
            'information_bounds': [{'support': s['support'], 'bounds': s['bounds']} for s in c['supports']]})
    # The large physical amplitude maps remain in the scientific source files.
    for c in channels:
        for name in ('average_pulse_risk', 'per_row_risk', 'per_block_risk'):
            c[name].pop('mean_model_amplitude_mapping_real_imag', None)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'complete': True, 'support_tests': int(count[1]), 'channels': channels,
        'monte_carlo': mc,
        'source_sha256': {str(path): digest(path) for path in paths+[args.tests, Path(__file__)]},
        'verified_original_implementation_substitutions': [item[1] for item in loaded],
        'production_changed': False, 'phase_centre_calibration_accepted': False,
        'scope_limit': 'Known fixed waveform, fitted response, independent diagonal carrier noise, fixed scene support and motion. Sparse contiguous blocks across the aperture, not all symbol rows. No certification of nonlinear autofocus, unknown covariance, scattering support or site coordinates.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'finite-scene-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    grid = np.linspace(20, 55, 150)
    ten = channels[3]
    styles = [('average_pulse_risk', 'One aperture-average pulse', '#d35400'),
              ('per_row_risk', 'Known pulse per symbol', '#2874a6'),
              ('per_block_risk', 'Known pulse per 16 symbols', '#229954')]
    for key, label, color in styles:
        risk = ten[key]
        value = risk['noise_risk_over_raw_crlb']+10**(grid/10)*risk['average_squared_bias_over_raw_crlb_per_snr']
        axes[0].plot(grid, 10*np.log10(value), label=label, color=color)
    for case in mc['cases']:
        if case['channel'] != 3:
            continue
        estimate = case['estimators']['average_spectrum']
        v = estimate['mse_over_raw_crlb']; se = estimate['standard_error']
        axes[0].errorbar(case['snr_db'], 10*np.log10(v),
            yerr=[[10*np.log10(v/(v-1.96*se))], [10*np.log10((v+1.96*se)/v)]],
            color='#d35400', marker='o', capsize=4)
    axes[0].set(xlabel='Expected integrated raw-data SNR (dB)',
        ylabel='Amplitude MSE / raw-data CRLB (dB)', title='Ten: model bias grows with SNR')
    axes[0].legend(fontsize=8); axes[0].axhline(0, color='#777', lw=.7)
    x = np.arange(5)
    for i, (key, label, color) in enumerate(styles):
        values = [10*np.log10(c[key]['mse_over_raw_crlb_at_integrated_snr_db']['40']) for c in channels]
        axes[1].bar(x+(i-1)*.25, values, .25, label=label, color=color)
    axes[1].set(xticks=x, xticklabels=names, ylabel='Amplitude MSE / raw-data CRLB (dB)',
        title='All channels at 40 dB integrated SNR')
    axes[1].axhline(0, color='#777', lw=.7); axes[1].legend(fontsize=8)
    fig.suptitle('Controlled fixed-scene estimator test; not measured ship-image gain')
    fig.savefig(args.output/'finite-scene-validation.png', dpi=160); plt.close(fig)
    rows = ''
    for name, c in zip(names, channels):
        risks = [c[k]['mse_over_raw_crlb_at_integrated_snr_db']['40'] for k, _, _ in styles]
        bounds = c['information_bounds'][0]['bounds']
        loss = next(v['finite_profiles']['worst_information_loss_db'] for v in bounds
            if v['raw_relative_power_floor'] == 1e-10)
        rows += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
            [name, c['probes'], *(f'{r:.6f}' for r in risks), f'{loss:.6f}'])+'</tr>'
    max_z = max(abs(v['deviation_standard_errors']) for case in mc['cases'] for v in case['estimators'].values())
    maximum_block = max(c['per_block_risk']['mse_over_raw_crlb_at_integrated_snr_db']['50'] for c in channels)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite-scene waveform validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}code{{background:#f3f4f5;padding:2px}}</style>
<h1>Does an average waveform bias the scene and geometry fit?</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="finite-scene-validation.json">Results and source identities</a> · {result['updated_utc']}</p>
<p><strong>Yes, in the controlled multi-reflector test.</strong> An average pulse can retain almost all matched-filter information while still producing a biased amplitude estimate. Keeping known symbol-spectrum variation removes that bias here. Averaging the known response over each 16-symbol block retains the improvement, making a compact implementation plausible. No antenna calibration or production change is accepted from this test alone.</p>
<img src="finite-scene-validation.png" alt="Estimator MSE relative to raw information bound, by signal strength and channel">
<h2>What was tested</h2>
<p>Thirty time-selected blocks per band span 150.772–180.772 s of the capture. Each contains 16 consecutive reconstructed symbols: 2,400 probes total, with no rejected block. The physical fixed-north/east motion and eight existing reflector positions per band are held fixed. The same 17-cell range crop, saved observation-centre epochs, centre Doppler, phase deramp and information-weighted coherent averaging are used. Blocks start at the selected probe epoch; they need not share the original full-capture bin boundaries. This is a sampled-block experiment, not a full 30-second integration.</p>
<p>The exact forward reference includes current and previous transmitted symbols, finite guard interval, fitted channel response, saved source offsets, original C0/C1 coefficients and previous-row fractional-clock interpolation. All three candidate models use the same physical phase, within-symbol moments, compressed data and exact compressed noise covariance. Only the known spectral response changes: one average over the sampled aperture, the actual row spectrum, or its information-weighted average within each block. The average is calculated from these probes; it is not a claim that the production full-capture average is identical.</p>
<h2>Information retention and estimator efficiency are different checks</h2>
<p>For <code>y = Q a + n</code>, with known noise covariance <code>N</code>, raw information is <code>F = Qᴴ N⁻¹ Q</code>. After a fixed compression <code>L</code>, the whitened exact design is <code>A = C⁻½ LQ</code>, where <code>C = LNLᴴ</code>. Compression loss follows the generalized information eigenvalues of <code>AᴴA</code> relative to <code>F</code>.</p>
<p>A model <code>T</code> estimates amplitudes with <code>â = T†z</code>. Its bias is <code>(T†A − I)a</code>, even when the span of T retains the useful data. The reported risk is <code>E[(â−a)ᴴ F (â−a)] / p</code>, where p is the complex-amplitude count. It includes noise and bias. Independent equal-power complex amplitudes define the stated average risk; total expected raw-data SNR sets their scale. This is not a worst-case scene guarantee. Ratios below one can occur for a biased estimator and do not beat an unbiased CRLB.</p>
<table><tr><th>Band</th><th>Symbols</th><th>Average-pulse risk, 40 dB</th><th>Per-symbol risk</th><th>Per-block risk</th><th>Worst core compression loss, dB</th></tr>{rows}</table>
<p>Compression losses in the table use raw information modes above a relative power floor of 1e−10. The JSON also reports 1e−8 and 1e−12 floors, the 35-point union support, and unresolved mode counts. An almost unobservable reflector difference is not silently called well resolved. At 50 dB, the largest per-block average risk ratio is {maximum_block:.7f} ({10*np.log10(maximum_block):.5f} dB).</p>
<h2>Independent estimator trials</h2>
<p>{mc['trials_per_channel_and_snr']:,} proper-complex Gaussian trials per band and SNR compare average-pulse, per-symbol and exact finite-waveform GLS estimators at 40 and 50 dB. The largest discrepancy from the analytic MSE is {max_z:.2f} estimated standard errors. The saved trials include confidence intervals. The per-block candidate has the same analytic risk calculation and independently checked block operator; its dedicated Monte Carlo is not claimed here. {count[1]} support tests pass, including dense-operator checks of coherent weighting, covariance and phase deramp, and rejection of incomplete or cross-gap blocks.</p>
<h2>Consequence for phase-centre work</h2>
<p>This is an identified forward-model approximation to fix before interpreting small improvements from free transmitter or receiver coordinates. It does not prove that it caused the observed site offsets. A compact known-response kernel per coherent block can be retained without fitting additional parameters to the target data; its lag interpolation, derivatives, storage and runtime still need validation. The same correction then needs a matched comparison on actual recorded ship and vehicle data, including withheld-target transfer.</p>
<p>Unknown residual interference, response-estimation uncertainty, road association, reflector support, nonlinear motion efficiency, cross-band scattering and true antenna positions remain outside this conditional result. Nominal licensed sites remain in the <a href="siting-assumptions.html">current imaging hypothesis</a>.</p></html>'''
    (args.output/'finite-scene-validation.html').write_text(text)
    print(json.dumps({'report': str(args.output/'finite-scene-validation.html'),
        'maximum_per_block_risk_at_50db': maximum_block, 'maximum_monte_carlo_z': max_z}))


if __name__ == '__main__':
    main()
