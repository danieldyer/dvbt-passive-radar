"""Publish the calibration observation-operator and stationary-clutter controls."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('control', 'integrated', 'sufficient', 'tests'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--full', type=Path)
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    paths = [args.control, args.integrated, args.sufficient]+([args.full] if args.full else [])
    records = [json.loads(p.read_text()) for p in paths]
    for record in records:
        if not record['complete']:
            raise ValueError('Publish only completed matched controls')
        for key in ('sources', 'implementation_sha256'):
            for path, expected in record['identity'][key].items():
                if fingerprint(Path(path)) != expected:
                    raise ValueError('Input or implementation changed: '+path)
    tests = args.tests.read_text()
    count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('A completed passing support-test log is required')
    control, integrated, sufficient = records[:3]
    full = records[3] if args.full else None
    labels = [t['label'] for t in control['targets']]
    if any([t['label'] for t in r['targets']] != labels for r in records):
        raise ValueError('Controls must use the same ordered target intervals')
    cases, information = [], []
    for i, label in enumerate(labels):
        a, b, c = [r['targets'][i] for r in records[:3]]
        errors = {'moving_only': sum(p['prediction_error'] for p in a['baseline']['prediction']),
            'stationary_midpoint': sum(p['prediction_error'] for p in a['with_stationary']['prediction']),
            'stationary_integrated': b['integrated_refit']['prediction_error'],
            'stationary_union': c['summary']['original_gate']['augmented_error']}
        convergence = {'moving_only': a['selected_fits_stationary'][0],
            'stationary_midpoint': a['selected_fits_stationary'][1],
            'stationary_integrated': b['selected_fit_stationary'],
            'stationary_union': c['selected_fit_stationary']}
        if full:
            d = full['targets'][i]
            errors['stationary_full_bins'] = d['summary']['original_gate']['full_bins']['prediction_error']
            convergence['stationary_full_bins'] = d['selected_fit_stationary']
        info = [r['conditional_information'] for r in c['channels']]
        information.append([-10*np.log10(max(1e-15, min(r['information_fraction_eigenvalues']))) for r in info])
        cases.append({'label': label, 'withheld_original_gate_prediction_error': errors,
            'error_ratio_db_vs_moving_only': {k: float(10*np.log10(v/errors['moving_only'])) for k, v in errors.items()},
            'selected_fits_stationary': convergence,
            'recorded_bin_reconstruction_relative_error_max': max(r['recorded_bin_relative_error'] for r in b['operator_checks']),
            'stationary_dictionary_midpoint_relative_error_by_channel': [r['stationary_dictionary_relative_midpoint_error'] for r in b['operator_checks']],
            'union_dictionary_projection_relative_error_max': max(r['stationary_dictionary_relative_projection_error'] for r in info),
            'conditional_information_by_channel': info})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(p): fingerprint(p) for p in [*paths, args.tests]},
        'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
        'cases': cases, 'tests': int(count[1]), 'full_bin_control': full,
        'accepted_calibration': False, 'production_changed': False,
        'scope': 'Offline calibration controls with nominal main-site coordinates frozen. Same four vehicle intervals, roads, observation values, empirical covariance and split. No recorded profile is filtered or removed.',
        'information_scope': 'Local proper-complex Gaussian information conditional on the full saved-bin covariance and stationary dictionary. Eliminate unknown complex moving amplitude and stationary coefficients, then compare all retained real-Jacobian modes above a relative 1e-8 singular-value cutoff. The worst-mode loss is not an image-SNR loss or a measured antenna error.'}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/'road-observation-validation.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
    variants = [('stationary_midpoint', 'Static model at bin mean'), ('stationary_integrated', 'Static model at original rows'),
        ('stationary_union', 'Additional static information')]
    if full:
        variants.append(('stationary_full_bins', 'All retained time bins'))
    x, width = np.arange(len(labels)), .8/len(variants)
    for j, (key, title) in enumerate(variants):
        axes[0].bar(x+(j-(len(variants)-1)/2)*width,
            [c['error_ratio_db_vs_moving_only'][key] for c in cases], width, label=title)
    axes[0].axhline(0, color='#777', lw=.8)
    axes[0].set(xticks=x, xticklabels=labels, xlabel='Vehicle interval',
        ylabel='Held-out squared prediction error / moving-only control (dB)',
        title='Identical original-gate test observations; lower is better')
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(information, aspect='auto', cmap='YlOrRd', vmin=0.)
    axes[1].set(xticks=range(5), xticklabels=['Seven', 'SBS', 'Nine', 'Ten', 'ABC'],
        yticks=range(len(labels)), yticklabels=labels,
        title='Worst local information loss after union projection')
    for i, row in enumerate(information):
        for j, value in enumerate(row):
            axes[1].text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=9,
                color='white' if value > .65*np.max(information) else 'black')
    figure.colorbar(image, ax=axes[1], label='Worst retained parameter-mode information loss (dB)')
    figure.suptitle('Vehicle calibration: observation and nuisance-model controls\n'
        'Coordinates frozen; no production change or accepted antenna correction')
    figure.savefig(args.output/'road-observation-comparison.png', dpi=160)
    plt.close(figure)
    rows = []
    for case in cases:
        for key, title in [('moving_only', 'Moving target only'), *variants]:
            values = [case['label'], title, f"{case['withheld_original_gate_prediction_error'][key]:.2f}",
                f"{case['error_ratio_db_vs_moving_only'][key]:+.3f}", case['selected_fits_stationary'][key]]
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>')
    full_note = ''
    if full:
        full_note = '''<h2>Full-bin reference</h2><p>The additional reference retains every selected 20-ms time bin and range sample. Its whitening is invertible. Stationary coefficients are profiled as Gaussian nuisance parameters, so the associated information reduction is the exact Schur complement of those unknown coefficients, rather than a Doppler notch. The report retains full-bin and original-gate prediction errors on identical test operators for every estimator. It also reports a separate moving-only prediction metric with the same stationary nuisance fitted on test samples; that is not full waveform prediction.</p>'''
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Vehicle calibration observation audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Vehicle calibration: preserve the observation operator before fitting sites</h1>
<p><a href="analysis.html">Index</a> · <a href="shared-road-ship-validation.html#road-reassessment">Road alternatives</a> · <a href="road-observation-validation.json">Measurements and source identities</a> · {result['updated_utc']}</p>
<p class="note">These checks concern the offline vehicle-calibration likelihood, not lost production data. E, Q, P and S were selected because their fitted Doppler approaches the stationary region. Roads and antenna positions remain frozen. No corrected antenna coordinates or video changes are adopted.</p>
<h2>Why the first clutter control was inaccurate</h2>
<p>The stored observations are information-weighted averages over 20-ms bins after a seed motion deramp. A stationary return rotates after that deramp, so evaluating its response only at the mean epoch is not the same operation as averaging its original-row phase and range response. The corrected dictionary uses the saved row times and weights. Reconstructing the archived data averages gives at most {max(c['recorded_bin_reconstruction_relative_error_max'] for c in cases):.3g} relative error. The moving template still uses the existing mean-epoch approximation; this control does not certify that separate approximation.</p>
<h2>Retain observations needed to estimate clutter</h2>
<p>Write the whitened saved-bin data as z, its moving-target Fourier operator as G with GGᴴ = I, and the stationary dictionary as B. Orthonormal columns U span (I − GᴴG)B. The augmented observations [Gz, Uᴴz] have identity covariance and retain the entire span of Gᴴ and B. For means in that span they have exactly the same Gaussian likelihood differences, linear estimates and Fisher information as the full data. No signal-derived normalization or arbitrary frequency weights are introduced.</p>
<p>The numerical implementation reprojects the small singular vectors before QR, preventing nearly dependent clutter columns from reintroducing correlated noise. Synthetic tests compare against a dense full-data Gaussian estimator and Fisher matrix, check colored-noise whitening and physical derivatives, and stress nearly dependent and fully redundant columns. The maximum recorded stationary-dictionary projection discrepancy is {max(c['union_dictionary_projection_relative_error_max'] for c in cases):.3g}. All {result['tests']} support tests pass.</p>
<img src="road-observation-comparison.png" alt="Held-out prediction errors for four vehicle controls and conditional motion-information losses">
<p>The bar chart compares the same held-out original-gate cells for every estimator. It includes the simple moving-only control: improving a poor clutter estimate does not by itself beat that control. Selection uses training likelihood, and both starts and their numerical stationarity remain in the source results. No target is silently excluded because its control fails.</p>
<table><tr><th>Interval</th><th>Estimator</th><th>Held-out error</th><th>Error ratio (dB)</th><th>Selected fit stationary</th></tr>{''.join(rows)}</table>
<h2>What the information check does and does not prove</h2>
<p>The union is sufficient only for signals in its declared subspace. Comparing actual physical motion and antenna derivatives against the full saved-bin model shows substantial information loss in some identifiable parameter combinations. The heatmap reports the worst retained mode per band, after eliminating stationary and moving complex amplitudes. It includes weak geometrical combinations; it is not a general image-SNR penalty, a phase-centre uncertainty, or a claim about the original raw I/Q stream. The covariance and stationary-scattering assumptions remain conditional.</p>
{full_note}<p>The appropriate next acceptance checks are robust prediction of the vehicle response and transfer of shared coordinates to excluded targets and ship apertures. Improved training fit, a small numerical gradient, or one conditional information result cannot substitute for those checks.</p></html>'''
    (args.output/'road-observation-validation.html').write_text(page)
    print(json.dumps({'report': str(args.output/'road-observation-validation.html'), 'tests': result['tests'],
        'full_bin_reference': full is not None}))


if __name__ == '__main__':
    main()
