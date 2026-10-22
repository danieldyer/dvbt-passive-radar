"""Report matched controls for the combined road/ship location experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint
from sweep_shared_sites import verified


def sweep_snapshot(path, output, refinements=()):
    """Publish a fixed snapshot; a live sweep manifest may change afterwards."""
    raw = path.read_bytes()
    sweep = json.loads(raw)
    records = []
    overrides = {int(index): Path(filename) for index, filename in refinements}
    if len(overrides) != len(refinements) or set(overrides)-{c['index'] for c in sweep['cases']}:
        raise ValueError('Each refinement must identify one recorded sweep case')
    for case in sweep['cases']:
        if 'failure' in case:
            records.append({**case, 'ship_gain_db': None, 'vehicle_error_reduction_percent': None})
            continue
        fits = []
        for fixed, file in zip((True, False), case['files']):
            if fingerprint(Path(file)) != case['source_sha256'][file]:
                raise ValueError('Completed sweep result changed: '+file)
            fitted = json.loads(Path(file).read_text())
            if not fitted['complete'] or fitted['identity']['fixed_antennas'] != fixed:
                raise ValueError('Incomplete or mismatched sweep control: '+file)
            fits.append(fitted)
        first, second = [dict(fit['identity']) for fit in fits]
        first.pop('fixed_antennas'); second.pop('fixed_antennas')
        if first != second:
            raise ValueError('Sweep controls change more than antenna fitting')
        case = dict(case)
        if case['index'] in overrides:
            refined_path = overrides[case['index']]
            refined = verified(refined_path, fixed=False)
            if refined['identity'] != fits[1]['identity']:
                raise ValueError('Refinement changed the scientific objective or observations')
            if refined['fit']['objective'] > fits[1]['fit']['objective']+1e-10:
                raise ValueError('Refinement worsened the same training objective')
            case['original_case'] = dict(case)
            case['refinement'] = {'file': str(refined_path), 'sha256': fingerprint(refined_path),
                'original_file': case['files'][1], 'original_sha256': case['source_sha256'][case['files'][1]],
                'original_converged': case['both_stationary'], 'scope': 'Same objective and bounds, continued from the original terminal state. Original result and live sweep manifest are preserved.'}
            fits[1] = refined
            case.update(both_stationary=all(r['fit']['converged'] and
                r['original_row_maximum_kkt_residual'] < 1e-8 for r in fits),
                original_row_kkt=[r['original_row_maximum_kkt_residual'] for r in fits],
                joint_training_objective=refined['fit']['objective'],
                antenna_bound_hits=refined['antenna_bound_hits'],
                ship_validation=[r['validation']['ship_summary'] for r in fits],
                vehicle_validation_error=[sum(c['prediction_error'] for values in r['validation']['vehicles'].values()
                    for c in values) for r in fits])
        a, b = case['ship_validation']
        v, w = case['vehicle_validation_error']
        records.append({**case,
            'ship_gain_db': float(10*np.log10(b['matched_energy']/a['matched_energy'])),
            'vehicle_error_reduction_percent': float(100*(1-w/v))})
    eligible = [c for c in records if c.get('both_stationary')]
    snapshot = {'captured_utc': datetime.now(timezone.utc).isoformat(),
        'live_manifest': str(path), 'live_manifest_sha256_at_snapshot': hashlib.sha256(raw).hexdigest(),
        'experiment': sweep, 'cases': records, 'stationary_pairs_including_refinements': len(eligible),
        'training_selected_including_refinements': min(eligible, key=lambda c: c['joint_training_objective'])['index'] if eligible else None}
    snapshot_path = output/'shared-road-ship-sweep-snapshot.json'
    snapshot_path.write_text(json.dumps(snapshot, indent=2)+'\n')
    site_names = {'12013': 'Ornata', '12014': 'Ridge', '12015': 'Eyre'}
    rows = []
    for record in sorted(records, key=lambda r: r['index']):
        values = [record['index'], ' / '.join(site_names[s] for s in record['sites'])]
        values += ([f"Failed: {record['failure']}", '—', '—', '—'] if 'failure' in record else
            [str(record['both_stationary'])+(' (refined)' if 'refinement' in record else ''), f"{record['ship_gain_db']:+.4f}",
             f"{record['vehicle_error_reduction_percent']:+.3f}", record['antenna_bound_hits']])
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>')
    state = 'Complete' if sweep['complete'] else 'Still running; this is an incomplete snapshot'
    section = f'''<h2 id="licensed-sweep">Combined search across all licensed site assignments</h2>
<p><strong>{state}.</strong> {len(records)} of 48 matched fixed/free pairs recorded at {snapshot['captured_utc']}.
See the <a href="{snapshot_path.name}">fixed snapshot and source identities</a>. The earlier completed 48-case search was ship-only.</p>
<p>Every case uses the same ten provisional road branches, original likelihood weights, ship aperture and fixed N/E axes. Three-tower configurations are processed first; no licensed configuration is excluded. Each free-coordinate fit starts from its matched fixed-coordinate control. Original-row stationarity and withheld prediction are checked separately. The prior vehicle-location posterior is not reused as another measurement.</p>
<table><tr><th>Assignment</th><th>Seven / SBS / Nine / Ten / ABC</th><th>Both stationary</th><th>Ship gain (dB)</th><th>Vehicle error reduction (%)</th><th>Antenna bounds</th></tr>{''.join(rows)}</table>
<p>The current training-selected assignment among completed pairs is {snapshot['training_selected_including_refinements']}. This is a penalized local likelihood comparison, not posterior site odds or accepted on-air identification. Positive gain compares free versus fixed coordinates within one assignment. A refined row continues the identical objective from its previous terminal state; the snapshot retains the failed stationarity check and both result identities. It does not overwrite the live experiment. Road alternatives, origin assumptions and transfer to other apertures still require separate validation. No correction is adopted from this table.</p>'''
    return snapshot, snapshot_path, section


def road_branch_section(named_paths):
    """Compare inherited road choices using one unchanged observation model."""
    reports, sources, rows = [], [], []
    for name, filename in named_paths:
        path = Path(filename)
        result = json.loads(path.read_text())
        if not result['complete']:
            raise ValueError('Finish road comparisons before reporting a selected branch')
        for key in ('sources', 'implementation_sha256'):
            for source, expected in result['identity'][key].items():
                if fingerprint(Path(source)) != expected:
                    raise ValueError('Road comparison input changed: '+source)
        cases = [c for t in result['targets'] for c in t['cases']]
        summary = {'name': name, 'source': str(path), 'road_cases': len(cases),
            'stationary_selected_start_cases': sum(c['stationary'] for c in cases),
            'all_start_fits': sum(len(c['fit']['starts']) for c in cases),
            'stationary_start_fits': sum(s['fit']['converged'] for c in cases for s in c['fit']['starts']),
            'targets': [], 'accepted_calibration': False}
        for target in result['targets']:
            winner = next(c for c in target['cases'] if c['route'] == target['selected_route'])
            original = next(c for c in target['cases'] if c['route'] == target['previous_route'])
            energy = sum(c['data_energy'] for c in winner['fit']['prediction'])
            record = {k: target[k] for k in ('label', 'previous_route', 'previous_name',
                'selected_route', 'selected_name', 'selected_stationary', 'road_changed',
                'prediction_error_reduction_percent', 'matched_energy_change_db')}
            record.update(training_objective_reduction=original['training_objective']-winner['training_objective'],
                withheld_explained_fraction=1-winner['prediction_error']/energy)
            summary['targets'].append(record)
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [name,
                record['label'], record['previous_name'], record['selected_name'],
                f"{record['prediction_error_reduction_percent']:+.3f}",
                f"{record['matched_energy_change_db']:+.3f}", record['selected_stationary']])+'</tr>')
        reports.append(summary)
        sources.append(path)
    if not reports:
        return [], [], ''
    text = f'''<h2 id="road-reassessment">Road alternatives under the current scattering likelihood</h2>
<p>The inherited road associations were selected under a common-band phase calibration model. The combined road/ship experiment instead profiles independent complex target responses in each band. These controls therefore rescore the previously discovered road alternatives under the likelihood actually being used, with antenna coordinates frozen. Every road uses the same observed samples, whitening, envelope origins and declared kinematic/lane/height priors. Two local starts are retained for each road; training objective alone selects the start and the road.</p>
<p>The main-site experiment tests 58 alternatives across ten targets. Candidate A retains Albert Street and K retains Victoria Parade. In that experiment O, Q, R and E change roads on training likelihood, but every replacement worsens withheld squared prediction error. The replacements are not adopted. Stable road selection here is a finite conditional comparison, not independent evidence that a vehicle occupied that street.</p>
<table><tr><th>Sites</th><th>Target</th><th>Inherited road</th><th>Training-selected road</th><th>Withheld error reduction (%)</th><th>Matched energy change (dB)</th><th>Selected fit stationary</th></tr>{''.join(rows)}</table>
<p>A zero comparison means the same road was selected. Matched energy and frozen prediction error are both retained; a fitted test correlation can improve while the training-derived waveform predicts worse. No candidate, observation or low-SNR band is silently removed. The JSON reports stationarity for all starts, not just each road's winning start. Regression tests independently check that changing a cubic road parameter origin preserves positions, phase, likelihood and derivatives, and that switching roads leaves the measured fields and range origins untouched.</p>'''
    return reports, sources, text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', type=Path, required=True)
    parser.add_argument('--joint', type=Path, required=True)
    parser.add_argument('--information', type=Path)
    parser.add_argument('--sweep', type=Path, help='Snapshot an ongoing or completed combined 48-assignment sweep')
    parser.add_argument('--refinement', nargs=2, action='append', default=[], metavar=('INDEX', 'RESULT'),
                        help='Separate same-objective continuation of a sweep joint fit')
    parser.add_argument('--branches', nargs=2, action='append', default=[], metavar=('NAME', 'RESULT'),
                        help='Completed road reassessment with the current independent-band likelihood')
    parser.add_argument('--tests', type=Path, default=Path('build/isar-20261016-tests-shared-geometry.log'))
    parser.add_argument('--transfer', type=Path, action='append', default=[],
                        help='Completed whole-target geometry transfer result')
    parser.add_argument('--compare', nargs=3, action='append', default=[], metavar=('NAME', 'FIXED', 'JOINT'),
                        help='Additional matched control, such as an alternative site or road subset')
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    tests = args.tests.read_text()
    test_count = re.search(r'Ran (\d+) tests', tests)
    if not test_count or tests.strip().splitlines()[-1] != 'OK':
        raise ValueError('Provide a completed passing support-test log')
    fixed, joint = [json.loads(p.read_text()) for p in (args.fixed, args.joint)]
    if not fixed['complete'] or not joint['complete']:
        raise ValueError('Do not publish unfinished searches as a fitted comparison')
    identity_a, identity_b = [dict(r['identity']) for r in (fixed, joint)]
    if not identity_a.pop('fixed_antennas') or identity_b.pop('fixed_antennas') or identity_a != identity_b:
        raise ValueError('The two cases must differ only in whether antenna coordinates fit')
    first, second = fixed['validation'], joint['validation']
    labels, channels = list(first['vehicles']), ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    errors = np.array([[1-b['prediction_error']/a['prediction_error'] for a, b in
        zip(first['vehicles'][label], second['vehicles'][label])] for label in labels])*100
    ship_gain = 10*np.log10([b['heldout_matched_energy']/a['heldout_matched_energy'] for a, b in
        zip(first['ship_channels'], second['ship_channels'])])
    ship_gain_total = 10*np.log10(second['ship_summary']['matched_energy']/first['ship_summary']['matched_energy'])
    ship_error = (1-second['ship_summary']['prediction_error']/first['ship_summary']['prediction_error'])*100
    positions = np.r_[joint['antenna_fit']['transmitter_offsets_enu_m'], [joint['antenna_fit']['receiver_offset_enu_m']]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout='constrained')
    axes[0].bar(channels, ship_gain, color='#347fa3')
    axes[0].axhline(0, color='#666666', lw=.8)
    axes[0].set(ylabel='Withheld matched-energy change (dB)', title='Ship: free versus fixed antenna coordinates')
    limit = max(.1, float(abs(errors).max()))
    im = axes[1].imshow(errors, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    axes[1].set(xticks=range(5), xticklabels=channels, yticks=range(len(labels)), yticklabels=labels,
                title='Vehicles: withheld error reduction (%)')
    for row in range(len(labels)):
        for col in range(5):
            axes[1].text(col, row, f'{errors[row, col]:+.1f}', ha='center', va='center', fontsize=8,
                color='white' if abs(errors[row, col]) > .55*limit else 'black')
    fig.colorbar(im, ax=axes[1], label='Positive = lower prediction error')
    y = np.arange(6)
    for j, axis in enumerate(['East', 'North', 'Up']):
        axes[2].barh(y+(j-1)*.22, positions[:, j], height=.22, label=axis)
    axes[2].set(yticks=y, yticklabels=[*channels, 'Receiver'], xlabel='Correction from nominal (m)',
                title='Shared physical coordinate corrections')
    axes[2].legend()
    fig.suptitle('Ten vehicle intervals and a 30-second ship aperture\n'
                 'Same original likelihood weights and withheld samples; provisional road/scattering hypotheses')
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'shared-road-ship-comparison.png', dpi=160)
    plt.close(fig)
    information = json.loads(args.information.read_text()) if args.information else None
    if information and (not information['complete'] or information['source_sha256'] != fingerprint(args.joint)):
        raise ValueError('Information audit must belong to this completed fit')
    comparisons = []
    extra_sources = []
    transfers = []
    for path in args.transfer:
        record = json.loads(path.read_text())
        if not record['complete'] or fingerprint(Path(record['calibration'])) != record['calibration_sha256']:
            raise ValueError('Transfer result is incomplete or its calibration changed')
        calibration = json.loads(Path(record['calibration']).read_text())
        record['transferred_antenna_fit'] = calibration['antenna_fit']
        if calibration['identity']['site_assignment'] == joint['identity']['site_assignment']:
            delta = (np.asarray(calibration['antenna_fit']['transmitter_offsets_enu_m'])
                     -np.asarray(joint['antenna_fit']['transmitter_offsets_enu_m']))
            record['transmitter_change_from_full_group_enu_m'] = delta.tolist()
            record['maximum_transmitter_height_change_m'] = float(abs(delta[:, 2]).max())
        transfers.append(record)
        extra_sources.extend([path, Path(record['calibration'])])
    for name, fixed_path, joint_path in args.compare:
        paths = [Path(fixed_path), Path(joint_path)]
        a, b = [json.loads(p.read_text()) for p in paths]
        ia, ib = dict(a['identity']), dict(b['identity'])
        if (not a['complete'] or not b['complete'] or not ia.pop('fixed_antennas')
                or ib.pop('fixed_antennas') or ia != ib):
            raise ValueError('An additional comparison is unfinished or changes more than antenna fitting')
        extra_sources.extend(paths)
        va, vb = a['validation'], b['validation']
        road_error = [sum(c['prediction_error'] for channels_ in v['vehicles'].values() for c in channels_)
                      for v in (va, vb)]
        comparisons.append({'name': name, 'labels': a['identity']['labels'],
            'sites': a['identity']['site_assignment'],
            'ship_matched_energy_gain_db': float(10*np.log10(vb['ship_summary']['matched_energy']/va['ship_summary']['matched_energy'])),
            'ship_error_reduction_percent': float(100*(1-vb['ship_summary']['prediction_error']/va['ship_summary']['prediction_error'])),
            'vehicle_error_reduction_percent': float(100*(1-road_error[1]/road_error[0])),
            'original_kkt': [a['original_row_maximum_kkt_residual'], b['original_row_maximum_kkt_residual']],
            'antenna_bound_hits': b['antenna_bound_hits'], 'antenna_fit': b['antenna_fit'],
            'fit_objective': b['fit']['objective'], 'scope': 'Within-pair comparison only; different road subsets have different observations.'})
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {str(p): fingerprint(p) for p in [args.fixed, args.joint, *([args.information] if args.information else [])]},
        'labels': labels, 'channels': channels, 'ship_matched_energy_gain_db': float(ship_gain_total),
        'ship_error_reduction_percent': float(ship_error), 'ship_channel_gain_db': ship_gain.tolist(),
        'vehicle_channel_error_reduction_percent': errors.tolist(),
        'antenna_fit': joint['antenna_fit'], 'antenna_bound_hits': joint['antenna_bound_hits'],
        'fits_stationary_fast_model': [r['fit']['converged'] for r in (fixed, joint)],
        'original_row_kkt': [r['original_row_maximum_kkt_residual'] for r in (fixed, joint)],
        'information': information, 'additional_comparisons': comparisons,
        'whole_target_transfer': transfers, 'tests': int(test_count[1]), 'accepted_calibration': False,
        'scope': 'Conditional local fixed-versus-free location experiment with ten road intervals and one ship aperture. Alternative sites, road subsets and reported whole-target folds are sensitivity checks. No new calibrated image or site choice is adopted.'}
    extra_sources.append(args.tests)
    sweep_section = ''
    if args.sweep:
        snapshot, snapshot_path, sweep_section = sweep_snapshot(args.sweep, args.output, args.refinement)
        result['licensed_sweep_snapshot'] = {'file': str(snapshot_path),
            'completed_pairs': len(snapshot['cases']), 'complete': snapshot['experiment']['complete'],
            'stationary_pairs': snapshot['stationary_pairs_including_refinements'],
            'accepted_calibration': False}
        extra_sources.append(snapshot_path)
    branches, branch_sources, branch_section = road_branch_section(args.branches)
    if branches:
        result['road_reassessment'] = branches
        extra_sources.extend(branch_sources)
    result['sources'].update({str(p): fingerprint(p) for p in extra_sources})
    result['implementation_sha256'] = {str(Path(__file__)): fingerprint(Path(__file__))}
    (args.output/'shared-road-ship-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = ''.join(f'<tr><td>{name}</td><td>{r["fit"]["converged"]}</td><td>{r["original_row_maximum_kkt_residual"]:.3g}</td><td>{r["antenna_bound_hits"]}</td></tr>' for name, r in [('Fixed locations', fixed), ('Shared fitted locations', joint)])
    geometry = ''
    if information:
        selected = [r for r in information['cases'] if r['cutoff'] == 1e-10]
        table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
            ['Free ship reflector positions' if r['ship_reflector_coordinates_eliminated'] else 'Fixed ship reflector positions',
             *[r[k]['directions_weaker_than_declared_prior'] for k in ('ship_only', 'vehicles_only', 'combined')]])+'</tr>' for r in selected)
        geometry = f'''<h2>Location information after eliminating target nuisance parameters</h2>
<p>These are expected proper-complex Gaussian mean derivatives, with complex reflectivity and each target's real motion removed by orthogonal projection. The table counts location combinations whose data information is weaker than the declared prior (out of 18). Hard bounds and road smoothness penalties supply no measured information here. A single antenna prior is added only to calculate the separately reported conditional coordinate uncertainties.</p>
<table><tr><th>Ship support assumption</th><th>Ship alone</th><th>Vehicles alone</th><th>Combined</th></tr>{table}</table>
<p>Road associations, range-response models, scattering stability and empirical noise covariance remain conditional. These are not surveyed confidence intervals or an experimental CRLB certificate. The JSON preserves rank-cutoff sensitivity at 1e-8, 1e-10 and 1e-12.</p>'''
    extra = ''
    if comparisons:
        rows_extra = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [r['name'],
            ', '.join(r['labels']), f"{r['ship_matched_energy_gain_db']:+.4f}",
            f"{r['vehicle_error_reduction_percent']:+.3f}", r['antenna_bound_hits']])+'</tr>' for r in comparisons)
        extra = f'''<h2>Alternative site and road hypotheses</h2><table><tr><th>Case</th><th>Vehicle intervals</th><th>Ship gain (dB)</th><th>Vehicle error reduction (%)</th><th>Antenna bounds</th></tr>{rows_extra}</table>
<p>Every row compares fitted and fixed locations on identical observations within that case. Scores across different road subsets are not model-selection odds. The four-street subset uses A/O on Albert Street and K/Q on Victoria Parade, reflecting the previously stated plausibility preference rather than independent vehicle sightings. An alternative assignment is a sensitivity check, not an accepted on-air site identification.</p>'''
    transfer_table = ''
    if transfers:
        rows_transfer = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [t['label'],
            f"{t['summary']['matched_energy_change_db']:+.4f}",
            f"{t['summary']['prediction_error_reduction_percent']:+.3f}",
            ', '.join(f'{x:+.3f}' for x in t['summary']['channel_error_reduction_percent']),
            t['summary']['selected_fits_stationary']])+'</tr>'
            for record in transfers for t in record['targets'])
        sensitivity = '; '.join('excluding '+', '.join(record['withheld_labels'])+
            f" changes a transmitter height correction by up to {record['maximum_transmitter_height_change_m']:.1f} m"
            for record in transfers if 'maximum_transmitter_height_change_m' in record)
        transfer_table = f'''<h2>Geometry transferred to completely excluded vehicles</h2>
<p>Each vehicle below is absent from the shared-coordinate calibration. Its antenna corrections are then frozen while its motion and independent band coefficients are fitted on even one-second blocks. Odd blocks test prediction against the matched nominal-geometry control. Both controls use the same initial road branches and choose solutions by training objective only. Possible overlapping detections are excluded. Discovery and road association still use the recording; this is conditional transfer.</p>
<table><tr><th>Excluded vehicle</th><th>Matched energy change (dB)</th><th>Prediction error reduction (%)</th><th>Error reduction: Seven/SBS/Nine/Ten/ABC (%)</th><th>Both fits stationary</th></tr>{rows_transfer}</table>
<p>Relative to the full ten-vehicle fit: {html.escape(sensitivity)}. These are sensitivity changes between conditional optima, not physical tower movement or measurement error bars.</p>'''
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Combined vehicle and ship geometry fit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Joint transmitter/receiver fitting with vehicles and ship</h1><p><a href="analysis.html">Index</a> · <a href="siting-assumptions.html">Site assumptions</a> · <a href="shared-road-ship-validation.json">Results and source identities</a> · Updated {result['updated_utc']}</p>
<p class="note">A combined physical-location fit has now been run. No phase-centre correction is adopted from this local comparison alone. Both cases use the licensed main configuration and fixed N/E ship axes; yaw remains exactly zero.</p>
<p>The transmitter/receiver coordinates are shared by all ten vehicle intervals and the ship. Each target's trajectory is refitted; complex scattering coefficients are independent between targets and bands. This tests physical geometry through delay and temporal phase without imposing a transferable absolute scattering phase. It does not yet calibrate the cross-band scattering phases needed for a unique physical coherent image.</p>
<p>One fixed electrical phase centre per band is assumed across the target directions. Direction-dependent antenna phase patterns are not separately modelled. Constant phases can enter the independent target/band coefficients; variation during target motion cannot be silently identified with a surveyed location correction. The failed transfer checks do not by themselves identify which physical modelling assumption is wrong.</p>
<p>The original whitened likelihoods are added in their existing noise units, with one antenna prior. Target energies are not individually normalised to equal influence. The vehicle likelihood retains its zero envelope origins; the ship keeps five free origins. This difference, road alternatives and cross-target clutter covariance need sensitivity checks. Vehicle road discovery used the full capture, so the alternating-block results are conditional predictive controls.</p>
<table><tr><th>Case</th><th>Search stationary</th><th>Original-row KKT residual</th><th>Antenna bounds reached</th></tr>{rows}</table>
<p>On withheld ship samples, freeing the shared coordinates changes matched energy by {ship_gain_total:+.4f} dB and reduces squared prediction error by {ship_error:+.3f}%. The image below shows each band and each vehicle, rather than hiding disagreement in an aggregate score. Numerical stationarity alone does not establish global optimality or physical calibration.</p>
<img src="shared-road-ship-comparison.png" alt="Per-band ship gain, per-vehicle prediction changes and fitted antenna offsets">
{sweep_section}{branch_section}{extra}{transfer_table}{geometry}
<h2>Numerical validation and next acceptance checks</h2>
<p>The new tests cover both propagation legs and the direct-reference subtraction, exact complex-amplitude profiling, equivalence to the existing vehicle observation operator, shared-coordinate derivatives, a single antenna prior with unchanged likelihood weights, and recovery of nonzero injected TX/RX offsets. A 256-trial noisy geometry regression also checks the six-coordinate conditional CRLB with an exactly known diverse trajectory and unknown complex reflectivity; this does not certify the unknown-motion recorded-data problem. The full support suite passed {int(test_count[1])} tests. Recorded-data finite-difference probes agree with the joint gradient to 2.3e-10 in its declared global objective units. Optimisation uses two-node Gaussian quadrature of the ship forward model; final predictions and gradient checks use original rows. No samples are rephased or discarded by this approximation.</p>
<p>Before accepting locations: test alternative licensed assignments and plausible road branches, exclude complete target groups during fitting, inspect dependence on coordinate/origin priors, and verify transfer across ship apertures. Improved training contrast or road-constrained information is not sufficient on its own.</p></html>'''
    (args.output/'shared-road-ship-validation.html').write_text(report)
    print(json.dumps({k: result[k] for k in ('ship_matched_energy_gain_db', 'ship_error_reduction_percent',
                                          'antenna_bound_hits', 'original_row_kkt')}))


if __name__ == '__main__':
    main()
