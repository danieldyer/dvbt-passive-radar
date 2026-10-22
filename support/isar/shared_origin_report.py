"""Publish the common-origin sensitivity control without adopting a calibration."""
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


def read_fit(path):
    result = json.loads(path.read_text())
    if (not result['complete'] or not result['fit']['converged']
            or result['original_row_maximum_kkt_residual'] > 1e-7):
        raise ValueError('Report completed stationary fits only: '+str(path))
    for file, digest in result.get('numerical_refinement', {}).get('implementation_sha256', {}).items():
        if fingerprint(Path(file)) != digest:
            raise ValueError('Numerical refinement implementation changed: '+file)
    return result


def row(name, path, fit, reference):
    validation, baseline = fit['validation'], reference['validation']
    vehicle_error = sum(c['prediction_error'] for channels in validation['vehicles'].values() for c in channels)
    baseline_vehicle = sum(c['prediction_error'] for channels in baseline['vehicles'].values() for c in channels)
    return {'name': name, 'file': str(path), 'sha256': fingerprint(path),
        'numerical_refinement': fit.get('numerical_refinement'),
        'objective': fit['fit']['objective'], 'kkt': fit['original_row_maximum_kkt_residual'],
        'ship_matched_energy_change_db': float(10*np.log10(validation['ship_summary']['matched_energy']/baseline['ship_summary']['matched_energy'])),
        'ship_prediction_error_reduction_percent': float(100*(1-validation['ship_summary']['prediction_error']/baseline['ship_summary']['prediction_error'])),
        'vehicle_prediction_error_reduction_percent': float(100*(1-vehicle_error/baseline_vehicle)),
        'vehicle_error_reduction_by_target_percent': {label: float(100*(1-
            sum(c['prediction_error'] for c in channels)/sum(c['prediction_error'] for c in baseline['vehicles'][label])))
            for label, channels in validation['vehicles'].items()},
        'origins_m': fit['range_origins_m'], 'antenna_bound_hits': fit['antenna_bound_hits'],
        'antenna_fit': fit['antenna_fit']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', type=Path, nargs='+', required=True)
    parser.add_argument('--joint', type=Path)
    parser.add_argument('--information', type=Path)
    parser.add_argument('--legacy-control', type=Path)
    parser.add_argument('--transfer', type=Path, help='Completed whole-target exclusion directory')
    parser.add_argument('--tests', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    tests = args.tests.read_text()
    count = re.search(r'Ran (\d+) tests', tests)
    if not count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Use a completed passing test log')
    fixed = [(p, read_fit(p)) for p in args.fixed]
    identity = fixed[0][1]['identity']
    if not identity['fixed_antennas'] or any(f['identity'] != identity for _, f in fixed):
        raise ValueError('Fixed-location starts must have the identical objective and observations')
    for group in ('sources', 'implementation_sha256'):
        for file, digest in identity['legacy_identity'][group].items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('Source or implementation changed: '+file)
    for file, digest in identity['implementation_sha256'].items():
        if fingerprint(Path(file)) != digest:
            raise ValueError('Common-origin implementation changed: '+file)
    selected_path, selected = min(fixed, key=lambda x: x[1]['fit']['objective'])
    records = [row('Fixed antennas, '+f['origin_start']+' start', p, f, selected) for p, f in fixed]
    if args.joint:
        joint = read_fit(args.joint)
        joint_identity = dict(joint['identity']); joint_identity['fixed_antennas'] = True
        if joint['identity']['fixed_antennas'] or joint_identity != identity:
            raise ValueError('Free-antennas comparison changed the likelihood')
        records.append(row('Fitted antennas', args.joint, joint, selected))
    information = json.loads(args.information.read_text()) if args.information else None
    if information and (not args.joint or not information['complete']
            or information['fit_sha256'] != fingerprint(args.joint)):
        raise ValueError('Information audit must describe this completed joint fit')
    legacy_control = json.loads(args.legacy_control.read_text()) if args.legacy_control else None
    legacy_change = None
    if legacy_control:
        if not legacy_control['complete'] or legacy_control['source'] != str(selected_path):
            raise ValueError('Legacy feasible-point control must use the training-selected start')
        for file, digest in legacy_control['source_sha256'].items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('Legacy comparison source changed: '+file)
        baseline = json.loads(Path(legacy_control['legacy_control']).read_text())['validation']['ship_summary']
        current = selected['validation']['ship_summary']
        legacy_change = {'ship_matched_energy_db': float(10*np.log10(current['matched_energy']/baseline['matched_energy'])),
            'ship_prediction_error_reduction_percent': float(100*(1-current['prediction_error']/baseline['prediction_error']))}
    source_paths = [p for p, _ in fixed]+[args.tests, Path(__file__)]
    source_paths += [p for p in (args.joint, args.information, args.legacy_control) if p]
    transfer = None
    if args.transfer:
        transfer_paths = [args.transfer/name for name in ('transfer.json', 'fixed.json', 'fitted.json')]
        measured, fixed_transfer, free_transfer = [json.loads(p.read_text()) for p in transfer_paths]
        identity_transfer = measured['identity']
        if (not measured['complete'] or identity_transfer['source_sha256'] != fingerprint(selected_path)
                or identity_transfer['excluded_label'] in identity_transfer['calibration_labels']):
            raise ValueError('Transfer must exclude its target and use the selected source')
        for name, digest in identity_transfer['implementation_sha256'].items():
            if fingerprint(Path(name)) != digest:
                raise ValueError('Transfer implementation changed: '+name)
        for value in (fixed_transfer, free_transfer):
            if (not value['complete'] or not value['fit']['converged'] or
                    value['original_row_maximum_kkt_residual'] > 1e-7 or value['identity'] != identity_transfer):
                raise ValueError('Require matched stationary calibration controls')
        if not fixed_transfer['fixed_antennas'] or free_transfer['fixed_antennas']:
            raise ValueError('Transfer requires one fixed and one free geometry')
        transfer = {'directory': str(args.transfer), 'identity': identity_transfer,
            'summary': measured['summary'],
            'calibration_kkt': [v['original_row_maximum_kkt_residual'] for v in (fixed_transfer, free_transfer)],
            'range_origins_m': [v['range_origins_m'] for v in (fixed_transfer, free_transfer)],
            'antenna_fit': free_transfer['antenna_fit'], 'accepted_calibration': False}
        source_paths.extend(transfer_paths)
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'selected_fixed_start': str(selected_path), 'selection': 'Lowest training objective among the supplied stationary fixed-location starts.',
        'records': records, 'information': information, 'tests': int(count[1]),
        'joint_fit_completed': args.joint is not None,
        'whole_target_transfer': transfer,
        'legacy_feasible_control': legacy_control, 'change_from_legacy_fixed_fit': legacy_change,
        'parameter_count': selected['parameter_count'], 'origin_bounds_m': selected['range_origin_bounds_m'],
        'accepted_calibration': False, 'yaw_fitted': False, 'production_changed': False,
        'source_sha256': {str(p): fingerprint(p) for p in source_paths},
        'scope_limit': 'Conditional on the same road identities, measured fields, empirical covariance, motion bounds and fixed ship support. Common origins are a sensitivity hypothesis, not demonstrated timing calibration. Different starts need not find the same local minimum. A point-target matched-energy statistic is not full-scene coherent gain.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'shared-origin-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    channels = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for r in records:
        axes[0].plot(channels, r['origins_m'], marker='o', label=r['name'])
    axes[0].set(title='One absolute envelope origin per band', ylabel='Excess-path offset (m)')
    axes[0].legend(fontsize=8)
    labels = list(selected['validation']['vehicles'])
    width = .8/len(records)
    for i, r in enumerate(records):
        axes[1].bar(np.arange(len(labels))+(i-(len(records)-1)/2)*width,
            [r['vehicle_error_reduction_by_target_percent'][label] for label in labels], width=width, label=r['name'])
    axes[1].axhline(0, color='#777777', lw=.8)
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, title='Withheld vehicle prediction',
        ylabel='Error reduction vs selected fixed fit (%)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Shared vehicle/ship range-origin sensitivity; calibration remains provisional')
    fig.savefig(args.output/'shared-origin-comparison.png', dpi=160); plt.close(fig)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['name'],
        f"{r['objective']:.8f}", f"{r['kkt']:.2g}", f"{r['ship_matched_energy_change_db']:+.5f}",
        f"{r['ship_prediction_error_reduction_percent']:+.4f}", f"{r['vehicle_prediction_error_reduction_percent']:+.3f}",
        r['antenna_bound_hits']])+'</tr>' for r in records)
    info_text = ''
    transfer_text = ''
    if transfer:
        stats = transfer['summary']; label = transfer['identity']['excluded_label']
        transfer_text = f'''<h2>Transfer to a completely excluded vehicle</h2>
<p>Target {html.escape(label)} was excluded from the calibration observations, parameters and priors. Nine other vehicle candidates and the ship fitted the common geometry and five origins. Those quantities were then frozen while target {html.escape(label)}'s motion and independent complex band responses were fitted on its training time blocks.</p>
<p>Both calibration controls and the selected excluded-target fits passed the original stationarity checks. Freeing antenna positions changed withheld target matched energy by <strong>{stats['matched_energy_change_db']:+.4f} dB</strong>, with <strong>{stats['prediction_error_reduction_percent']:+.3f}%</strong> prediction-error reduction. This does not support transferring the fitted antenna corrections. Target selection, road association and initialization still used this recording, so this is conditional transfer, not an independent ground-truth position test.</p>'''
    status = ('The free-antenna comparison is complete; acceptance and transfer remain separate checks.' if args.joint
        else 'This snapshot contains the two fixed-antenna starts. The free-antenna fit is still in progress; it is not a completed phase-centre comparison.')
    legacy_text = ''
    if legacy_control:
        legacy_text = f'''<h2>Improved initialization, not evidence of improved antenna positions</h2>
<p>Compared with the previous fixed-site result, the selected solution has {legacy_change['ship_matched_energy_db']:.3f} dB more withheld ship matched energy and {legacy_change['ship_prediction_error_reduction_percent']:.3f}% lower squared prediction error. No antenna moved.</p>
<p>Evaluating this same feasible state in the unchanged legacy model, with vehicle origins again fixed at zero, gives original-row objective {legacy_control['new_state_legacy_original_row_objective']:.8f}, versus {legacy_control['legacy_original_row_objective']:.8f} at the old fit. This is an evaluation, not an optimized legacy refit. It establishes a better feasible point in the old problem, so the gain must not be attributed to the shared-origin assumption or claimed as successful site calibration. Further starts and transfer tests remain necessary.</p>'''
    if information:
        cases = [c for c in information['cases'] if c['cutoff'] == 1e-10]
        info_text = '<h2>Location information</h2><p>'+ ' '.join(
            f"With ship reflector coordinates {'free' if c['ship_point_coordinates_eliminated'] else 'fixed'}, "
            f"{c['combined']['directions_weaker_than_declared_prior']} of 18 antenna directions have less information from these observations than from the declared location prior."
            for c in cases)+'</p><p>Local motion and complex responses are eliminated first. The five common origins are eliminated only after combining targets. Optimizer bounds and motion priors do not count as measured location information. This is a local, model-conditional information calculation.</p>'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared range-origin control</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}</style>
<h1>Do different range-origin assumptions bias the site fit?</h1>
<p><a href="siting-assumptions.html">Siting assumptions</a> · <a href="shared-origin-validation.json">Results and source identities</a> · {report['updated_utc']}</p>
<p><strong>{status}</strong></p>
<p>The previous combined experiment fixed the vehicle envelope origins at zero and fitted a separate ship origin in each band. This control instead applies the same five absolute offsets to every vehicle and the ship. It keeps the same {selected['parameter_count']} parameters, measured samples, weights, road hypotheses, trajectory constraints and antenna priors. The offset changes the envelope; independent complex target/band coefficients already absorb a constant carrier phase.</p>
<p>The inherited seed-centred ±75 m origin bounds are unchanged and are not measured calibration uncertainty. Sharing a constant offset across separated time intervals is itself a hypothesis. This result cannot establish that the hardware delay reference is constant or that the fixed per-band ship reflector support is physically correct.</p>
<p>All changes below are relative to the stationary fixed-antenna start selected solely by training objective. Both supplied starts are retained. Different local minima expose initialization sensitivity; they do not establish a globally optimal fit. Free versus fixed antennas is the matched comparison of interest.</p>
<p>The zero-origin start initially failed the original-row stationarity check despite passing the two-node quadrature check. A four-node refinement also failed. The accepted numerical refinement evaluates the original rows directly and passes the unchanged 1e−7 gate. The original failed results and the four-node implementation are preserved separately. This numerical stationarity check does not establish a global optimum or physical calibration.</p>
<table><tr><th>Fit</th><th>Training objective</th><th>Original-row KKT</th><th>Ship matched energy (dB)</th><th>Ship error reduction (%)</th><th>Vehicle error reduction (%)</th><th>Antenna bounds</th></tr>{rows}</table>
<img src="shared-origin-comparison.png" alt="Fitted common range origins and withheld vehicle prediction by target">
{legacy_text}
{info_text}
{transfer_text}
<p>Regression validation: {count[1]} support tests passed, including independent finite-difference origin derivatives, known-offset recovery, equality with the legacy model at zero origins, and shared-nuisance information elimination against a full block-matrix projection. Neither calibration coordinates nor production imaging have been changed. Successful transfer and additional apertures remain necessary before accepting site corrections.</p></html>'''
    (args.output/'shared-origin-validation.html').write_text(document)
    print(json.dumps({'report': str(args.output/'shared-origin-validation.html'), 'records': [
        {k: r[k] for k in ('name', 'objective', 'ship_matched_energy_change_db', 'vehicle_prediction_error_reduction_percent')} for r in records]}))


if __name__ == '__main__':
    main()
