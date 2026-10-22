"""Publish compact known-waveform validation and matched recorded-data controls."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import numpy as np

from finite_scene_report import digest, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('compact', 'mean-metric', 'block-metric', 'kernels', 'mean-fit',
                 'known-fit', 'tests'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--implementation-archive', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    paths = [args.compact, args.mean_metric, args.block_metric, args.kernels,
             args.mean_fit, args.known_fit]
    loaded = [load(path, args.implementation_archive) for path in paths]
    compact, mean_metric, block_metric, kernels, mean, known = [x[0] for x in loaded]
    for fit in (mean, known):
        if not fit['fit']['converged'] or fit['original_row_kkt'] > 1e-7:
            raise ValueError('Require the original constrained stationarity gate')
    manifests = [load(path.parent/'experiment.json', args.implementation_archive)[0]
                 for path in (args.mean_fit, args.known_fit)]
    if manifests[0]['availability'] != manifests[1]['availability']:
        raise ValueError('Different observed rows in the two controls')
    for manifest in manifests:
        if not manifest['antennas_fixed'] or not manifest['yaw_fixed_zero']:
            raise ValueError('Keep the declared fixed-site, fixed-attitude comparison')
    for other in (mean_metric, block_metric):
        for key in ('requested_interval_s', 'points_body_m', 'range_offsets_samples',
                    'averaging_rows', 'observation_centre', 'template_spectrum'):
            if other[key] != compact[key]:
                raise ValueError('Unmatched finite-scene control: '+key)
    tests = args.tests.read_text()
    count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require a completed passing support test run')
    channels = []
    for c, m, b in zip(compact['channels'], mean_metric['channels'], block_metric['channels']):
        if not c['channel'] == m['channel'] == b['channel']:
            raise ValueError('Channel order changed')
        risks = [s['supports'][0][key]['mse_over_raw_crlb_at_integrated_snr_db']['50']
                 for s, key in ((c, 'uncorrected_mean_gls_risk'),
                                (m, 'implemented_metric_gls_risk'),
                                (b, 'implemented_metric_gls_risk'))]
        channels.append({'channel': c['channel'], 'mse_over_raw_crlb_at_50db':
                         dict(zip(('exact_metric', 'mean_metric', 'block_metric'), risks))})
    comparison = {
        'mean': mean['after_summary'], 'known': known['after_summary'],
        'known_prediction_error_change_percent': 100*(
            known['after_summary']['prediction_error']/mean['after_summary']['prediction_error']-1),
        'known_stationary_residual_change_percent': 100*(
            known['after_summary']['stationary_only_error']/mean['after_summary']['stationary_only_error']-1),
        'kkt': [mean['original_row_kkt'], known['original_row_kkt']],
        'scope': 'Identical observations, mean range metric, motion/support constraints and alternate '
                 '0.5 s train/validation blocks. Known pulses also change stationary nuisance templates; '
                 'their projected matched energies have different baselines. No new antenna fit.'}
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
        'scope': __doc__, 'channels': channels, 'replay': kernels,
        'recorded_comparison': comparison, 'support_tests': int(count[1]),
        'source_sha256': {str(p): digest(p) for p in paths+[args.tests, Path(__file__)]},
        'verified_original_implementation_substitutions': [x[1] for x in loaded],
        'production_changed': False, 'phase_centre_calibration_accepted': False,
        'scope_limit': 'Conditional finite-scene information and estimator risk with fixed support, '
            'known symbols/response and independent Gaussian row noise. Real recorded comparison '
            'does not certify unknown interference, target scattering, motion or antenna coordinates.'}
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    rows = ''.join('<tr><td>'+names[c['channel']]+'</td>'+''.join(
        f'<td>{10*np.log10(v):.6f}</td>' for v in c['mse_over_raw_crlb_at_50db'].values())+'</tr>'
        for c in channels)
    fit_rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
        [name, f"{fit['after_summary']['prediction_error']:.3f}",
         f"{fit['after_summary']['stationary_only_error']:.3f}",
         f"{fit['original_row_kkt']:.3g}"])+'</tr>'
        for name, fit in [('Aperture-average pulse', mean), ('Known block pulse', known)])
    error = max(c['relative_l2_error'] for c in kernels['checks'])
    maximum = max(10*np.log10(c['mse_over_raw_crlb_at_50db']['mean_metric']) for c in channels)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Known-waveform kernel validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Known per-block waveform: numerical accuracy and recorded-data comparison</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="waveform-kernel-validation.json">Results and source identities</a> · {result['updated_utc']}</p>
<p>The compact response passes the controlled numerical checks. It has not established a recorded ship-focusing improvement or an antenna calibration. The same withheld observations decide that comparison.</p>
<h2>Original capture weights and reconstructed symbols</h2>
<p>All {kernels['known_rows']:,} requested original symbol rows were reconstructed, giving {sum(kernels['blocks_per_channel']):,} valid 16-symbol response blocks across five channels and approximately 30 seconds. Each kernel uses the response/noise weights saved in the original capture and the actual transmitted-symbol spectrum. No observed range profile is renormalized. The lag kernel is represented with fixed degree-10 polynomial pieces derived from the same windowed-sinc interpolant.</p>
<p>Independent double-precision spectrum accumulation and inverse FFT checked all nine full-block overlaps with the separately exported finite-waveform reference. Maximum relative kernel error was {error:.3g}. Unit tests also check lag derivatives, geometry derivatives, exact adjoints, row averaging and crop identity. The completed support suite contains {int(count[1])} passing tests.</p>
<h2>Controlled estimator loss</h2>
<p>The finite-waveform control uses 2,400 symbols in 30 contiguous blocks per band, eight fixed scene points per band, original saved range/phase bookkeeping and exact compressed noise propagation. The table reports information-weighted amplitude MSE divided by the raw-data CRLB, in dB, at 50 dB expected integrated SNR. It includes average bias under the stated scene ensemble; it is not a worst-case scene bound.</p>
<table><tr><th>Band</th><th>Exact covariance (dB)</th><th>Existing mean metric (dB)</th><th>Per-block metric (dB)</th></tr>{rows}</table>
<p>The largest loss with the existing mean metric is {maximum:.5f} dB in this conditional test. This small result supports keeping that metric for the matched recorded-data comparison. It does not prove that actual interference has the assumed covariance.</p>
<h2>Recorded fixed-site motion fits</h2>
<p>Both controls keep nominal transmitter/receiver coordinates, zero yaw, identical observed rows, identical mean covariance weights, the same eight-point support and motion/range-origin bounds. Training and validation alternate in 0.5-second blocks. All likelihoods and final stationarity checks use original symbol epochs. Known block pulses also change the stationary nuisance forward model, so projected matched energy has a different baseline and is not quoted as a direct imaging gain.</p>
<table><tr><th>Template</th><th>Withheld squared error</th><th>Stationary-only squared error</th><th>Constrained stationarity residual</th></tr>{fit_rows}</table>
<p>The known-block model changes withheld prediction error by {comparison['known_prediction_error_change_percent']:+.4f}%. A positive change is worse. Its stationary-only baseline changes by {comparison['known_stationary_residual_change_percent']:+.4f}%. Numerical accuracy and improved synthetic estimator risk alone do not establish an improvement to this recorded scene.</p>
<p>No production waveform, accepted site coordinates or current videos are changed by this experiment. Remaining work includes the recorded scattering/noise model and common vehicle/ship geometry validation. See the <a href="siting-assumptions.html">siting assumptions</a> and <a href="shared-origin-validation.html">shared-origin transfer checks</a>.</p></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'waveform-kernel-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'waveform-kernel-validation.html').write_text(document+'\n')


if __name__ == '__main__':
    main()
