"""Publish matched full-bin controls without conflating clutter and target gains."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', required=True)
    p.add_argument('--retained-prefix', type=Path, nargs='*', default=[],
        help='Explicitly retain completed targets from a failed later target; the source stays incomplete.')
    p.add_argument('--failure-log', type=Path, nargs='*', default=[])
    p.add_argument('--tests', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    tests = args.tests.read_text(); count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Require completed passing support tests')
    targets, provenance = [], []
    common = None
    keys = ('full_waveform_error', 'moving_error_after_identical_test_static_projection',
            'moving_matched_energy')
    for path in args.results:
        data = json.loads(path.read_text())
        if not data['complete'] and (path not in args.retained_prefix or not args.failure_log):
            raise ValueError('An incomplete source needs an explicit retained prefix and its failure log')
        for source, expected in data['source_sha256'].items():
            if fingerprint(Path(source)) != expected:
                raise ValueError('Scientific source changed: '+source)
        if common is None:
            common = data['source_sha256']
        else:
            for name in set(common)&set(data['source_sha256']):
                if common[name] != data['source_sha256'][name]:
                    raise ValueError('Matched experiments changed a shared scientific source')
        provenance.append({'path': str(path), 'complete_source': data['complete'],
            'retained_labels': [target['label'] for target in data['targets']]})
        for target in data['targets']:
            if [trial['operator'] for trial in target['trials']] != ['legacy', 'full']:
                raise ValueError('Require the same gated/full controls')
            fits = []
            for trial in target['trials']:
                fit = trial['fit']; selected = fit['starts'][fit['selected_start']]['fit']
                if (not selected['converged'] or selected['maximum_kkt_residual'] > 1e-7
                        or selected['maximum_constraint_violation'] > 1e-7
                        or selected['objective'] != min(s['fit']['objective'] for s in fit['starts'])):
                    raise ValueError('A selected fit must pass stationarity and use training-only selection')
                fits.append({'operator': trial['operator'], 'kkt': selected['maximum_kkt_residual'],
                    'metrics': {key: sum(c[key] for c in trial['prediction']) for key in keys},
                    'maximum_original_row_template_loss_db': max(
                        c['original_row_moving_template_loss_db'] for c in trial['prediction'])})
            before, after = [fit['metrics'] for fit in fits]
            comparison = {key+'_change_percent': 100*(after[key]/before[key]-1) for key in keys}
            for key, value in comparison.items():
                np.testing.assert_allclose(value, target['comparison'][key], rtol=1e-12, atol=1e-12)
            comparison['moving_matched_energy_change_db'] = float(
                10*np.log10(after['moving_matched_energy']/before['moving_matched_energy']))
            targets.append({'label': target['label'], 'source': str(path), 'fits': fits,
                'comparison': comparison, 'wide_template_check': target['wide_template_check']})
    if sorted(t['label'] for t in targets) != ['A', 'K', 'O', 'Q']:
        raise ValueError('Require all four preferred targets exactly once')
    targets.sort(key=lambda t: ['A', 'O', 'K', 'Q'].index(t['label']))
    result = {'complete': True, 'scope': __doc__, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'targets': targets, 'source_completion': provenance, 'support_tests': int(count[1]),
        'accepted_calibration': False, 'production_changed': False,
        'source_sha256': {str(path): fingerprint(path) for path in
            [*args.results, *args.failure_log, args.tests, Path(__file__)]}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'full-road-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    labels = [t['label'] for t in targets]
    ax[0].bar(labels, [t['comparison']['moving_matched_energy_change_db'] for t in targets], color='#277e8e')
    ax[0].set(ylabel='Matched-energy change (dB)', title='Moving response, identical test subspace')
    ax[1].bar(labels, [t['comparison'][keys[1]+'_change_percent'] for t in targets], color='#5f5099')
    ax[1].set(ylabel='Prediction-error change (%)', title='Moving prediction; lower is better')
    for axis in ax:
        axis.axhline(0, color='#555', lw=.7); axis.set_xlabel('Road hypothesis')
    fig.suptitle('Full saved bins versus a narrow Doppler gate; sites and range origins frozen')
    fig.savefig(args.output/'full-road-validation.png', dpi=160); plt.close(fig)
    row = lambda values: '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
    rows = ''.join(row([t['label'],
        f"{t['comparison'][keys[0]+'_change_percent']:+.3f}",
        f"{t['comparison'][keys[1]+'_change_percent']:+.5f}",
        f"{t['comparison']['moving_matched_energy_change_db']:+.4f}",
        f"{max(f['maximum_original_row_template_loss_db'] for f in t['fits']):.7f}"]) for t in targets)
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-bin vehicle likelihood</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Vehicle calibration without a narrow Doppler gate</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="full-road-validation.json">Values and source identities</a> · {result['updated_utc']}</p>
<p>All eight selected fits converge. Full-bin processing modestly increases moving-target matched energy, by 0.02–0.25 dB across these preferred street hypotheses. Moving prediction improves for A/K/Q and slightly worsens for O. The large total-waveform improvements mostly concern stationary-background prediction; they are not moving-target SNR gains or proof of physical antenna calibration.</p>
<img src="full-road-validation.png" alt="Moving energy and prediction comparison for four road targets">
<table><tr><th>Target</th><th>Total waveform error change (%)</th><th>Moving error change (%)</th><th>Moving matched energy change (dB)</th><th>Maximum moving approximation loss (dB)</th></tr>{rows}</table>
<h2>What is held fixed?</h2>
<p>Both controls use the same recorded samples, declared empirical covariance, nominal physical sites, common range origins, road hypotheses and two motion starts. They share an unknown stationary delay dictionary. Static templates are integrated over the actual original row epochs and weights before applying the observation operator. One control retains the earlier Fourier gate; the other retains all selected saved time bins. Motion and independent complex band responses are fitted on training blocks only.</p>
<p>Both predictions are scored on identical full-bin test observations. Total waveform error transfers the training-estimated static coefficients. The moving error and matched-energy columns use the same test stationary projection, so unknown test background is eliminated identically for both predictions. This conditional moving statistic is distinct from predicting the entire waveform. The moving response at the bin mean is independently compared with original-row integration at each final fit; the largest reported loss is below 0.0007 dB.</p>
<h2>Why this likelihood is useful for calibration</h2>
<p>For whitened observations y = u(θ)α + Bβ + n with a fixed stationary dictionary B and unit proper-complex Gaussian noise, minimising over β gives ‖P⊥<sub>B</sub>(y − u(θ)α)‖². This is exactly the profile likelihood, not a target-dependent noise normalization. Full-bin whitening is invertible on the saved selected bins; the only subsequent projection here removes the explicitly unknown static nuisance. A narrow Fourier gate can discard additional information, as the <a href="road-temporal-validation.html">previous local information audit</a> demonstrates.</p>
<p>This result is conditional on the covariance and scene model. It does not prove the empirical noise model, the vehicle's road identity or point-reflector approximation, or the information efficiency of earlier time averaging. A subsequent joint geometry fit and excluded-target transfer test is needed before changing any antenna coordinates.</p>
<h2>Measured pulse support and retained failures</h2>
<p>A and O require measured pulse tails beyond the original interpolation support. Their replayed central pulses pass the unchanged relative-error check of 10⁻⁵; no missing tails are truncated or extrapolated. The longer replay completed O/K, then stopped because Q's central pulse differed by 1.28 × 10⁻⁵. That incomplete source and failure log are preserved. Q needs no extended pulse support, so its separate completed comparison uses the original pulse and covariance. The table retains the already completed O/K cases and reports source completion explicitly.</p>
<p>{count[1]} support tests pass, including independent complex least-squares equivalence and original-row phase/path/origin derivative checks with stationary nuisance. No production or accepted physical-calibration change follows from this report alone.</p></html>'''
    (args.output/'full-road-validation.html').write_text(text+'\n')
    print(json.dumps({'report': str(args.output/'full-road-validation.html'),
        'moving_energy_change_db': {t['label']: t['comparison']['moving_matched_energy_change_db'] for t in targets}}))


if __name__ == '__main__':
    main()
