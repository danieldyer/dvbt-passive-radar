"""Publish matched preferred-street calibration and whole-target transfer."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path

import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--implementation-archive', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    p.add_argument('--name', default='preferred-road-validation',
        help='Report filename stem; keep matched experiments in separate files.')
    args = p.parse_args()
    if Path(args.name).name != args.name or not args.name:
        raise ValueError('Report name must be a filename stem')
    paths = [args.experiment/name for name in ('experiment.json', 'fixed.json', 'fitted.json')]
    trial, fixed, fitted = [json.loads(path.read_text()) for path in paths]
    if not trial['complete']:
        raise ValueError('The transfer experiment is still running')
    identity = trial['identity']
    refinement = trial.get('numerical_refinement')
    if refinement:
        for group in ('source_sha256', 'implementation_sha256'):
            for path, expected in refinement[group].items():
                if fingerprint(Path(path)) != expected:
                    raise ValueError('Numerical refinement provenance changed: '+path)
        if fitted.get('numerical_refinement') != refinement:
            raise ValueError('Fit and experiment describe different numerical refinements')
    if identity['excluded_label'] in identity['calibration_labels']:
        raise ValueError('Target was not excluded')
    if fingerprint(Path(identity['source'])) != identity['source_sha256']:
        raise ValueError('Original calibration source changed')
    substitutions = {}
    for path, expected in identity['implementation_sha256'].items():
        if fingerprint(Path(path)) != expected:
            archive = next((p for p in args.implementation_archive if fingerprint(p) == expected), None)
            if archive is None:
                raise ValueError('Original transfer implementation unavailable: '+path)
            substitutions[path] = str(archive)
    if 'site_assignment_source' in identity:
        if fingerprint(Path(identity['site_assignment_source'])) != identity['site_assignment_sha256']:
            raise ValueError('Licensed assignment changed')
    for path, expected in identity.get('additional_sources_sha256', {}).items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Additional observation-model source changed: '+path)
    for fit, expected in ((fixed, True), (fitted, False)):
        if (not fit['complete'] or not fit['fit']['converged']
                or fit['fixed_antennas'] != expected or fit['identity'] != identity
                or fit['original_row_maximum_kkt_residual'] > 1e-7
                or fit['original_row_constraint_violation'] > 1e-7):
            raise ValueError('Require matched original-row stationary calibration controls')
    tests = []
    for transfer in trial['transfer']:
        selected = transfer['starts'][transfer['selected_start']]
        if (not selected['fit']['converged'] or selected['fit']['maximum_kkt_residual'] > 1e-7
                or selected['fit']['maximum_constraint_violation'] > 1e-7):
            raise ValueError('Excluded-target fit failed stationarity')
        if selected['fit']['objective'] != min(s['fit']['objective'] for s in transfer['starts']):
            raise ValueError('Target start must be chosen on training objective')
        tests.append({'objective': selected['fit']['objective'],
            'kkt': selected['fit']['maximum_kkt_residual'],
            'prediction_error': sum(c['prediction_error'] for c in transfer['prediction']),
            'matched_energy': sum(c['matched_energy'] for c in transfer['prediction'])})
    comparisons = {
        'prediction_error_change_percent': 100*(tests[1]['prediction_error']/tests[0]['prediction_error']-1),
        'matched_energy_change_db': float(10*np.log10(tests[1]['matched_energy']/tests[0]['matched_energy']))}
    for key, value in comparisons.items():
        np.testing.assert_allclose(value, trial['summary'][key], rtol=1e-12, atol=1e-12)
    ship = [fit['validation']['ship_summary'] for fit in (fixed, fitted)]
    comparisons['ship_matched_energy_change_db'] = float(10*np.log10(ship[1]['matched_energy']/ship[0]['matched_energy']))
    comparisons['ship_prediction_error_change_percent'] = 100*(ship[1]['prediction_error']/ship[0]['prediction_error']-1)
    result = {'complete': True, 'updated_utc': datetime.now(timezone.utc).isoformat(),
        'identity': identity, 'comparison': comparisons, 'excluded_target_fits': tests,
        'calibration_kkt': [fit['original_row_maximum_kkt_residual'] for fit in (fixed, fitted)],
        'range_origins_m': [fit['range_origins_m'] for fit in (fixed, fitted)],
        'antenna_fit': fitted['antenna_fit'], 'accepted_calibration': False,
        'verified_archived_implementation': substitutions,
        'source_sha256': {str(path): fingerprint(path) for path in paths+[Path(__file__)]}}
    if refinement:
        result['numerical_refinement'] = refinement
    body_note = ''
    if 'fitted_reflector_offsets_along_across_up_m' in trial:
        limits = np.asarray(identity['reflector_extent_along_across_up_m'])
        reference = identity['reference_channel']
        body_rows, bound_hits = '', []
        channels = ('Seven', 'SBS', 'Nine', 'Ten', 'ABC')
        for name, targets in zip(('Fixed antennas', 'Fitted antennas'),
                trial['fitted_reflector_offsets_along_across_up_m']):
            hits = 0
            for label, values in targets.items():
                values = np.asarray(values)
                if values.shape != (5, 3) or np.any(abs(values[reference]) > 1e-12):
                    raise ValueError('Unexpected reflecting-point dimensions or trajectory gauge')
                if np.any(abs(values) > limits+1e-7):
                    raise ValueError('Reflecting points exceed their stated bounds')
                for c, offset in enumerate(values):
                    if c == reference:
                        continue
                    hit = int(np.isclose(abs(offset), limits, atol=1e-6, rtol=0).sum())
                    hits += hit
                    cells = [name, label, channels[c], *[f'{v:+.3f}' for v in offset], str(hit)]
                    body_rows += '<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells)+'</tr>'
            bound_hits.append(hits)
        result['reflecting_point_model'] = {
            'limits_along_across_up_m': limits.tolist(), 'reference_channel': reference,
            'offsets': trial['fitted_reflector_offsets_along_across_up_m'],
            'bound_hits_fixed_and_fitted_antennas': bound_hits,
            'nested_value_difference': trial['nested_value_difference'],
            'nested_gradient_maximum_difference': trial['nested_gradient_maximum_difference'],
            'recorded_derivative_checks': trial['recorded_derivative_checks']}
        body_note = f'''<h2>Band-specific vehicle reflecting points</h2>
<p>Each vehicle shares one translating road trajectory. {channels[reference]} defines that trajectory's reference point; the other bands can have constant physical offsets in a fixed along-road/across-road/up frame, bounded by ±({', '.join(f'{v:g}' for v in limits)}) m. These are extent hypotheses, not observed vehicle dimensions. No rotations, additional Gaussian offset prior or independent per-target timing offsets are fitted.</p>
<p>The body-offset model is present in both calibration controls and both excluded-target fits. Zero offsets reproduce the earlier common-point likelihood and its original parameter derivatives. The fixed/free antenna controls hit {bound_hits[0]}/{bound_hits[1]} body-offset bounds respectively. These fitted points must remain nuisance parameters when calculating antenna information; holding them known would overstate location accuracy.</p>
<details><summary>Fitted offsets, not measured vehicle structure</summary><table><tr><th>Control</th><th>Vehicle</th><th>Band</th><th>Along (m)</th><th>Across (m)</th><th>Up (m)</th><th>Bounds hit</th></tr>{body_rows}</table></details>'''
    offsets = np.r_[np.asarray(fitted['antenna_fit']['transmitter_offsets_enu_m']),
                    np.asarray(fitted['antenna_fit']['receiver_offset_enu_m'])[None, :]]
    rows = ''
    for name, offset in zip(('Seven', 'SBS', 'Nine', 'Ten', 'ABC', 'Receiver'), offsets):
        rows += '<tr>'+''.join('<td>'+html.escape(value)+'</td>' for value in
                               [name, *[f'{v:+.2f}' for v in offset]])+'</tr>'
    label = html.escape(identity['excluded_label'])
    site_description = ('the nominal licensed assignment' if 'site_assignment' not in identity else
        'the licensed assignment '+html.escape('/'.join(identity['site_assignment']))+' (Seven/SBS/Nine/Ten/ABC)')
    road_model = html.escape(identity.get('road_observation_model',
        'Legacy Doppler gate, bin-mean moving response, no explicit static nuisance.'))
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Preferred-road transfer</title>
<style>body{{font:17px/1.55 system-ui;max-width:1100px;margin:35px auto;padding:0 22px;color:#243040}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Albert Street and Victoria Parade calibration control</h1>
<p><a href="siting-assumptions.html">Siting assumptions</a> · <a href="{html.escape(args.name)}.json">Values and source identities</a> · {result['updated_utc']}</p>
<p>No physical antenna correction is adopted from this experiment. The control uses the user's preferred street hypotheses and keeps unknown motion and scattering responses separate from antenna geometry.</p>
<h2>Matched calibration and excluded-target test</h2>
<p>Targets O (Albert Street), K and Q (Victoria Parade), together with the strong 30-second ship aperture, fit common transmitter/receiver coordinates and five range origins. Target A (Albert Street) is completely removed from that likelihood, including its motion parameters and auxiliary priors. The other six road hypotheses are also omitted. Both controls use {site_description}; the separate 48-assignment sweep is not this comparison. Alternative-assignment controls retain the main control's absolute origin bounds.</p>
<p>The fixed control keeps nominal antennas. The second frees the same antenna coordinates under the existing broad priors, with the same observations, weights, motion constraints and scattering nuisance. Yaw remains zero. Both calibration fits pass the unchanged original-row stationarity gate. Their antenna positions and origins are then frozen while the excluded target's motion and independent complex band responses are fitted on its training blocks. Complementary blocks measure prediction; training objective alone selects the numerical start.</p>
<p>Vehicle observation model: {road_model} The ship response uses its original saved row epochs. These distinctions matter when comparing this experiment with a different vehicle likelihood.</p>
<p>For excluded target {label}, transferring the fitted geometry changes withheld squared error by <strong>{comparisons['prediction_error_change_percent']:+.4f}%</strong> and matched energy by <strong>{comparisons['matched_energy_change_db']:+.4f} dB</strong>. Positive error change is worse. Within the calibration set, ship withheld squared error changes by {comparisons['ship_prediction_error_change_percent']:+.4f}% and matched energy by {comparisons['ship_matched_energy_change_db']:+.5f} dB. These statistics do not certify coherent cross-band image gain.</p>
<p>The road identities are plausibility hypotheses, not independently observed vehicle locations. Target discovery, road association, support and inherited numerical initialization used this capture. This is a conditional transfer check, not independent ground-truth geolocation validation. Omitting a road likelihood does not undo all earlier data-dependent choices.</p>
{body_note}
<h2>Trial location shifts, not calibrated coordinates</h2>
<table><tr><th>Antenna</th><th>East (m)</th><th>North (m)</th><th>Up (m)</th></tr>{rows}</table>
<p>The <a href="aperture-reassessment.html">separated ship aperture and initialization checks</a> investigate additional temporal coverage. Neither a stationary optimizer nor a lower training objective proves physical site accuracy. Successful transfer across targets, road alternatives and justified scene models remains necessary before replacing the nominal geometry in production metadata.</p></html>'''
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/(args.name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    (args.output/(args.name+'.html')).write_text(document+'\n')
    print(json.dumps({'comparison': comparisons, 'report': str(args.output/(args.name+'.html'))}))


if __name__ == '__main__':
    main()
