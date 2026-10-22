"""Compare separated zero-yaw fits without selecting on withheld observations."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from finite_scene_report import digest, load
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments', type=Path, nargs='+', required=True)
    p.add_argument('--information', type=Path, required=True)
    p.add_argument('--implementation-archive', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    groups, substitutions, sources = {}, {}, [*args.experiments, args.information, Path(__file__)]
    for path in args.experiments:
        experiment, replaced = load(path, args.implementation_archive)
        substitutions[str(path)] = replaced
        for entry in experiment['fits']:
            source = Path(entry['path'])
            if digest(source) != entry['sha256']:
                raise ValueError('A fitted aperture changed: '+str(source))
            fit, replaced = load(source, args.implementation_archive)
            substitutions[str(source)] = replaced
            if (not fit['fit']['converged'] or fit['original_row_kkt'] > 1e-7
                    or fit['constraint_violation'] > 1e-7 or np.any(SplineMotion(fit).yaw_coefficients)):
                raise ValueError('Require stationary original-row zero-yaw fits')
            group = groups.setdefault(entry['window'], [])
            if group:
                reference = json.loads(Path(group[0]['source']).read_text())
                for key in ('interval_s', 'range_bias_m', 'selected_points_body_m', 'origins_frozen_from'):
                    if fit[key] != reference[key]:
                        raise ValueError('Initialization comparison changed '+key)
                np.testing.assert_allclose(fit['after_summary']['stationary_only_error'],
                    reference['after_summary']['stationary_only_error'], rtol=1e-12)
            group.append({'source': str(source), 'window': entry['window'],
                # The source-verified first driver only supported inherited motion.
                'initial_motion': fit.get('initial_motion', experiment.get('initial_motion', 'inherited')),
                'interval_s': fit['interval_s'],
                'objective': fit['fit']['objective'], 'kkt': fit['original_row_kkt'],
                **fit['after_summary']})
            sources.append(source)
    windows = []
    for window, trials in sorted(groups.items()):
        if len(trials) != len(args.experiments):
            raise ValueError('Every requested start must cover each aperture')
        selected = min(trials, key=lambda case: case['objective'])
        windows.append({'window': window, 'trials': trials, 'selected': selected,
                        'selection': 'Minimum training objective among original-row stationary fits.'})
    information, replaced = load(args.information, args.implementation_archive)
    substitutions[str(args.information)] = replaced
    for aperture in information['apertures']:
        if digest(Path(aperture['arrays'])) != aperture['arrays_sha256']:
            raise ValueError('An information root changed')
    values = []
    for case in information['summaries']:
        if not case['case'].endswith('cutoff-1e-10'):
            continue
        values.append({'case': case['case'], **{name: {
            'weak_directions': case[name]['directions_weaker_than_declared_prior'],
            'eigenvalues': case[name]['information_eigenvalues_before_prior']}
            for name in ('original', 'additional', 'combined')}})
    result = {'complete': True, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'scope': __doc__, 'windows': windows, 'information': values,
        'information_sources': [r['source'] for r in information['apertures']],
        'source_sha256': {str(path): digest(path) for path in sources},
        'verified_archived_sources': substitutions, 'accepted_calibration': False,
        'limitation': 'Road and ship initialization/support used this capture. Local model information is conditional on the declared noise metric. Fixed points retain geometric anchors; independent free per-band points are a conservative nuisance model. This is not an end-to-end efficiency certificate or a site impossibility result.'}
    rows = ''
    for window in windows:
        for trial in window['trials']:
            selected = trial['source'] == window['selected']['source']
            fields = [str(trial['window']), trial['initial_motion']+(' (selected)' if selected else ''),
                f"{trial['objective']:.9f}", f"{trial['kkt']:.2g}",
                f"{trial['prediction_error']:.2f}", f"{trial['matched_energy']:.2f}"]
            rows += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in fields)+'</tr>'
    info_rows = ''
    for case in values:
        for name in ('original', 'additional', 'combined'):
            value = case[name]
            fields = ['free per-band 3-D points' if 'True' in case['case'] else 'fixed points', name,
                      str(value['weak_directions'])+'/18', f"{max(value['eigenvalues']):.6g}"]
            info_rows += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in fields)+'</tr>'
    info_sources = ''.join('<li>'+html.escape(source)+'</li>' for source in result['information_sources'])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Separated ship aperture checks</title>
<style>body{{font:17px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Separated zero-yaw apertures and antenna information</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="aperture-reassessment.json">Values and source identities</a> · {result['updated_utc']}</p>
<p>The antenna locations and five common envelope origins stay fixed in these motion fits. Three disjoint 30-second windows near the beginning, middle and end of the capture supply additional views. Geographic north/east axes stay fixed; yaw, roll and pitch are not fitted.</p>
<h2>Initialization controls on identical observations</h2>
<p>Each window uses inherited motion, zero correction and a training-only phase-gradient rate seed. The rate seed only initializes the same physical trajectory model; it is not an additional phase correction or a motion prior. Each solve fits complex per-band reflectivities and the original-row physical response, with identical bounds, observations, support and stationary nuisance. Alternating half-second blocks test prediction. The fit is selected by training objective only, including cases where another start has lower withheld error. Original-row stationarity is necessary here but does not prove global optimality or correct physical motion.</p>
<table><tr><th>Window</th><th>Start</th><th>Training objective</th><th>Original KKT</th><th>Withheld squared error</th><th>Withheld matched energy</th></tr>{rows}</table>
<p>Matched energy is the independently fitted per-channel template statistic. It is not a calibrated coherent sum across the five bands. The coarse trajectories and selected reflector support used this recording, so this remains conditional prediction rather than independent truth validation.</p>
<h2>What extra apertures tell us about sites</h2>
<p>Each aperture's unknown motion and complex reflector amplitudes are eliminated before combining antenna information. Five range origins remain shared and are eliminated only after stacking apertures. The table compares data information with the declared antenna prior; no optimizer bound or motion penalty is counted as a measurement. Eigenvalues use prior-standardized coordinates, so a value of one means information equal to the prior in that direction.</p>
<table><tr><th>Point model</th><th>Apertures</th><th>Directions weaker than prior</th><th>Largest information eigenvalue</th></tr>{info_rows}</table>
<p>Fixed points retain conditional geometric anchors. The free-point control eliminates independent three-dimensional scatterer positions in each band, so it is more permissive than a verified shared physical scene. Additional views can increase information substantially without making all 18 antenna coordinates data-determined. Neither table establishes that another justified scene or road model cannot resolve the ambiguity. No site correction is adopted.</p>
<details><summary>Exact apertures used for the information table</summary><ul>{info_sources}</ul></details>
<p>The <a href="siting-assumptions.html">licensed main/backup alternatives and receiver prior</a> remain provisional. The preferred-road transfer check is separate from this ship-only information calculation. Known-waveform and cancellation audits are linked from the analysis index.</p></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'aperture-reassessment.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'aperture-reassessment.html').write_text(document+'\n')
    print(json.dumps({'selected': [r['selected']['source'] for r in windows],
                      'report': str(args.output/'aperture-reassessment.html')}))


if __name__ == '__main__':
    main()
