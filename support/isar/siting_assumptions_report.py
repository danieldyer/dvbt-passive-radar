"""Audit the actual site priors and the vehicle/ship viewing directions.

This describes existing hypotheses. It does not fit new antenna coordinates,
promote road associations to observations, or turn angular coverage into a
calibration information bound.
"""
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from ais import utc_seconds
from imaging import motion_state, sites
from road_geometry import RoadCurve
from road_model import RoadTrajectory
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def angles(position, site):
    look, direct = np.asarray(position)-site, -site
    azimuth = np.arctan2(look[:, 1], look[:, 0])-np.arctan2(direct[1], direct[0])
    azimuth = np.angle(np.exp(1j*azimuth))
    # Independent dot/cross calculation catches bearing-wrap/sign mistakes.
    cross = direct[0]*look[:, 1]-direct[1]*look[:, 0]
    np.testing.assert_allclose(azimuth, np.arctan2(cross, look[:, :2]@direct[:2]), atol=1e-14)
    elevation = np.arctan2(look[:, 2], np.linalg.norm(look[:, :2], axis=1))
    elevation -= np.arctan2(direct[2], np.linalg.norm(direct[:2]))
    return np.degrees(np.column_stack((azimuth, elevation)))


def main():
    directory = Path('docs/data/calibration_20261013')
    output = Path('docs/data/isar_20261015')
    motion_path = Path('build/isar-20261016/shared-origin-fixed-motion-30s.json')
    road_path = Path('build/calibration-20261013/shared-seed-fit/00002-joint-refined.json')
    sweep_path = output/'joint-sites-validation.json'
    best_path = Path('build/isar-20261016/joint-sites-polished-fixed-ne/assignment-45.json')
    paths = [motion_path, road_path, sweep_path, best_path, directory/'geometry-prior.json',
             directory/'sites-main.json', directory/'acma-transmitters.json']
    motion, roads, sweep, best, prior, main_sites, licences = [json.loads(p.read_text()) for p in paths]
    transmitters, receiver, _ = sites(directory)
    rotation = np.asarray(prior['ecef_to_enu'])
    centres = np.asarray(main_sites['site_positions_enu_m'])
    curve = SplineMotion(motion)
    offsets = motion.get('antenna_fit', {})
    if (np.max(abs(np.asarray(offsets.get('transmitter_offsets_enu_m', [0.])))) > 1e-6 or
            np.max(abs(np.asarray(offsets.get('receiver_offset_enu_m', [0.])))) > 1e-6 or
            np.any(curve.yaw_coefficients)):
        raise ValueError('This report expects the stated nominal-site, fixed-N/E control')
    time = np.linspace(*curve.interval, 101)
    state = motion_state(None, utc_seconds(motion['nominal_start_utc'])+time, motion)
    ship = (curve.apply(time, state, transmitters[motion['reference_channel']], receiver)
            ['position_ecef_m']-receiver)@rotation.T
    route_path = Path(roads['input_routes'])
    paths.append(route_path)
    routes = json.loads(route_path.read_text())['routes']
    parameters = np.asarray(roads['reduced_state'])
    tracks = []
    for i, (label, route) in enumerate(zip(roads['labels'], roads['selected_roads'])):
        source = Path(roads['input_histories'])/(label+'.json')
        paths.append(source)
        seed = json.loads(source.read_text())['channels'][1]
        trajectory = RoadTrajectory(seed, RoadCurve(routes[route['route']]), route['coarse_state'],
            roads['assignment'], roads['channel_offsets_enu_m'])
        t = seed['epoch']+np.linspace(-trajectory.half, trajectory.half, 101)
        p = parameters[9+i*trajectory.count:9+(i+1)*trajectory.count]
        position = trajectory.positions(p, t, centres)[0]
        tracks.append({'label': label, 'road': route['name'], 'position': position})
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    coverage = []
    for axis, name, site in zip(axes, ['Ornata', 'Eyre', 'Ridge'], centres):
        vehicle = []
        for track in tracks:
            bearing = angles(track['position'], site)
            vehicle.append(bearing)
            axis.plot(*bearing.T, lw=1.5, label=track['label'])
            axis.annotate(track['label'], bearing[len(bearing)//2], fontsize=8)
        vehicle, vessel = np.concatenate(vehicle), angles(ship, site)
        axis.plot(*vessel.T, color='black', lw=4, marker='o', markevery=[0], label='Ship')
        axis.annotate('Ship, 30 s', vessel[0], xytext=(-65, -20), textcoords='offset points')
        axis.plot(0, 0, '+', color='#777777', ms=10)
        axis.set(title=name, xlabel='Azimuth from direct RX direction (degrees)',
                 ylabel='Elevation from direct RX direction (degrees)')
        coverage.append({'site': name, 'road_azimuth_range_deg': [float(vehicle[:, 0].min()), float(vehicle[:, 0].max())],
            'road_elevation_range_deg': [float(vehicle[:, 1].min()), float(vehicle[:, 1].max())],
            'ship_azimuth_range_deg': [float(vessel[:, 0].min()), float(vessel[:, 0].max())],
            'ship_elevation_range_deg': [float(vessel[:, 1].min()), float(vessel[:, 1].max())],
            'sector_separation_deg': float(abs(np.median(vehicle[:, 0])-np.median(vessel[:, 0])))})
    figure.suptitle('Provisional road trajectories and the ship as viewed from each transmitter\n'
                     'Receiver-local ENU angles; geometry diagnostic, not independently known target positions')
    figure.savefig(output/'road-ship-angular-coverage.png', dpi=160)
    plt.close(figure)
    shared_path = output/'shared-road-ship-validation.json'
    shared = json.loads(shared_path.read_text()) if shared_path.exists() else None
    if shared:
        paths.append(shared_path)
    shared_note = ('The <a href="shared-road-ship-validation.html">subsequent combined vehicle-and-ship '
        'fits and transfer checks</a> are now available; no location correction is adopted.' if shared
        else 'A combined vehicle-and-ship geometry fit has not yet been completed.')
    combined_sweep = shared.get('licensed_sweep_snapshot') if shared else None
    sweep_note = ''
    if combined_sweep:
        sweep_note = (f"The combined licensed-site search has {combined_sweep['completed_pairs']} of 48 "
            'matched fixed/free pairs in the <a href="shared-road-ship-validation.html#licensed-sweep">'
            f"published snapshot</a>; {combined_sweep['stationary_pairs']} pairs pass its numerical "
            'stationarity gate. This is separate from the completed ship-only sweep above. '
            'Finishing the enumeration does not mean that every local fit converged, or that '
            'the newer common-origin/reflecting-point model was tested across all 48 assignments.')
    licensed_options = []
    for channel in main_sites['channels']:
        main_device = next(d for d in licences['devices'] if d['SDD_ID'] == channel['device_id'])
        options = [{'site_id': d['SITE_ID'], 'station_type': d['STATION_TYPE'],
                    'mounting_height_agl_m': float(d['HEIGHT'])}
                   for d in licences['devices'] if d['CALL_SIGN'] == main_device['CALL_SIGN']]
        licensed_options.append({'channel': channel['channel'], 'options': options})
    transfer_checks = [{'label': target['label'], **target['summary']}
                       for fold in (shared or {}).get('whole_target_transfer', [])
                       for target in fold['targets']]
    origin_path = output/'shared-origin-validation.json'
    origin_control = json.loads(origin_path.read_text()) if origin_path.exists() else None
    origin_audit_path = Path('build/isar-20261016/shared-origins-audit.json')
    origin_audit = json.loads(origin_audit_path.read_text()) if origin_audit_path.exists() else None
    if origin_control:
        paths.append(origin_path)
    if origin_audit:
        paths.append(origin_audit_path)
    preferred = []
    for name in ('preferred-road-validation', 'preferred-road-alternative-validation',
                 'preferred-road-full-bin-validation', 'preferred-road-scattered-validation'):
        path = output/(name+'.json')
        if path.exists():
            value = json.loads(path.read_text())
            if not value['complete'] or value['accepted_calibration']:
                raise ValueError('This assumptions report requires completed unaccepted site controls')
            preferred.append({'report': name+'.html', 'identity': value['identity'],
                'comparison': value['comparison'],
                'antenna_bound_hits': int(np.sum(np.isclose(
                    np.abs(value['antenna_fit']['standardised_parameters']), 3., atol=1e-7, rtol=0)))})
            paths.append(path)
    information_path = Path('build/isar-20261016/preferred-road-scattered-information-fitted.json')
    information_summary = None
    if information_path.exists():
        information = json.loads(information_path.read_text())
        if not information['complete'] or information['accepted_calibration']:
            raise ValueError('Require a completed conditional information audit')
        for path, expected in information['source_sha256'].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Information audit input changed: '+path)
        paths.append(information_path)
        information_summary = {'source': str(information_path), 'fit': information['fit'],
            'scope': information['scope'], 'cases': []}
        for case in information['cases']:
            combined = case['combined']
            information_summary['cases'].append({
                'free_vehicle_reflectors': case['free_vehicle_reflectors'],
                'free_ship_reflector_positions': case['free_ship_reflector_positions'],
                'cutoff': case['cutoff'],
                'directions_weaker_than_declared_prior': combined['directions_weaker_than_declared_prior'],
                'largest_information_eigenvalue': max(combined['information_eigenvalues_before_prior'])})
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'nominal_channel_sites': main_sites['channels'], 'receiver_prior': prior['receiver'],
        'licensed_options': licensed_options,
        'nominal_motion_source': str(motion_path),
        'nominal_motion_uses_vehicle_refined_coordinates': False,
        'accepted_phase_centre_calibration': False,
        'joint_fit_priors_and_offsets': best['site_local_fit'],
        'joint_site_sweep': sweep, 'joint_vehicle_and_ship_fit_completed': shared is not None,
        'combined_licensed_sweep_snapshot': combined_sweep,
        'whole_target_transfer_checks': transfer_checks,
        'shared_fit_range_origin_assumption': 'Road envelope origins remain zero; each ship band has a fitted envelope origin. This is an unresolved sensitivity assumption, not known absolute timing.',
        'shared_origin_sensitivity': {'report': str(origin_path.with_suffix('.html')) if origin_control else None,
            'recorded_derivative_audit_completed': bool(origin_audit and origin_audit['complete']),
            'recorded_derivative_maximum_error': max((c['absolute_error'] for c in origin_audit['checks']), default=0.) if origin_audit else None,
            'scope': 'Separate matched control sharing five absolute per-band envelope offsets across every vehicle and the ship, with unchanged observations, likelihood weights and parameter count. Constancy across intervals and calibration transfer remain hypotheses.',
            'accepted_calibration': False},
        'common_origin_whole_target_transfer': origin_control.get('whole_target_transfer') if origin_control else None,
        'preferred_street_transfer_controls': preferred,
        'fitted_reflector_information': information_summary,
        'viewing_directions': coverage, 'ship_interval_s': curve.interval.tolist(),
        'road_hypotheses': [{k: t[k] for k in ('label', 'road')} for t in tracks],
        'horizontal_datum': 'Licence GDA94 coordinates are converted directly to ECEF; no explicit epoch/frame transformation is present. Broad declared location priors do not constitute a surveyed datum tie.',
        'source_sha256': {str(p): fingerprint(p) for p in paths},
        'implementation_sha256': {str(p): fingerprint(p) for p in [Path(__file__),
            Path('support/isar/imaging.py'), Path('support/isar/spline_motion.py'),
            Path('support/calibration/road_model.py'), Path('support/calibration/site_hypotheses.py')]}}
    (output/'siting-assumptions.json').write_text(json.dumps(report, indent=2)+'\n')
    names = {'12013': 'Ornata Road', '12015': 'Eyre Road', '12014': 'Ridge Road'}
    rows = ''
    for channel, licensed in zip(main_sites['channels'], licensed_options):
        alternatives = '; '.join(f"{names[o['site_id']]} ({o['mounting_height_agl_m']:g} m AGL)"
                                for o in licensed['options'] if o['site_id'] != channel['site_id'])
        values = [channel['channel'], names[channel['site_id']], channel['licensed_height_m'], alternatives]
        rows += '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>'
    coordinate_rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in
        [name, site_id, licences['sites'][site_id]['LATITUDE'], licences['sites'][site_id]['LONGITUDE']])+'</tr>'
        for site_id, name in names.items())
    transfer_note = ' '.join(f"Excluded target {c['label']}: {c['matched_energy_change_db']:+.4f} dB "
                            'withheld matched-energy change.' for c in transfer_checks)
    best_names = ' / '.join(names[s] for s in sweep['selected_sites'])
    origin_note = ('The <a href="shared-origin-validation.html">common-origin sensitivity control</a> '
        'now reports the completed fits and their withheld predictions. No correction is adopted.' if origin_control else
        'A separate common-origin control is in progress: the same five absolute per-band envelope offsets '
        'are shared by all vehicles and the ship, without adding parameters or changing observations. '
        'Synthetic offset recovery and the recorded-data derivative audit pass. This tests the origin '
        'assumption; it does not establish absolute timing or a physical location calibration.')
    if origin_control and origin_control['joint_fit_completed']:
        current = next(row for row in origin_control['records'] if row['name'] == 'Fitted antennas')
        origin_note += (f" The matched free-antenna fit now passes original-row stationarity and adds "
            f"{current['ship_matched_energy_change_db']:+.5f} dB of withheld ship matched energy, "
            f"with {current['antenna_bound_hits']} standardized antenna coordinates at their search bounds. "
            f"Aggregate withheld vehicle prediction error changes by "
            f"{-current['vehicle_prediction_error_reduction_percent']:+.4f}%. "
            'These results do not justify adopting its physical offsets. The paired independent-channel '
            '<a href="motion-origin-images.html">image check and sliding preview</a> keep nominal antennas.')
        if origin_control.get('whole_target_transfer'):
            transfer = origin_control['whole_target_transfer']
            origin_note += (f" With vehicle {transfer['identity']['excluded_label']} removed from calibration, "
                f"transferring the fitted coordinates and origins changes its withheld matched energy by "
                f"{transfer['summary']['matched_energy_change_db']:+.4f} dB and prediction error by "
                f"{-transfer['summary']['prediction_error_reduction_percent']:+.3f}%.")
    preferred_note = ''
    for trial in preferred:
        assignment = trial['identity'].get('site_assignment', ['12013']*4+['12014'])
        comparison = trial['comparison']
        observation_model = html.escape(trial['identity'].get('road_observation_model',
            'Legacy Doppler gate, bin-mean moving response, no explicit static nuisance.'))
        preferred_note += (f"<p><a href=\"{trial['report']}\">Preferred-street control: "
            f"{' / '.join(names[s] for s in assignment)}</a>. O/K/Q and the ship fit geometry and "
            'common origins; A is excluded entirely. Transferring the free-antenna fit changes A\'s '
            f"withheld prediction error by {comparison['prediction_error_change_percent']:+.3f}% and "
            f"matched energy by {comparison['matched_energy_change_db']:+.4f} dB. The ship's withheld "
            f"matched-energy change is {comparison['ship_matched_energy_change_db']:+.5f} dB; "
            f"{trial['antenna_bound_hits']} standardized antenna coordinates reach their bounds. "
            f"Vehicle observation model: {observation_model} This control does not support "
            'adopting its physical offsets.</p>')
    information_note = ''
    if information_summary:
        records = [case for case in information_summary['cases'] if case['cutoff'] == 1e-10]
        rows_ = ''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in [
            'Unknown independent per-band positions' if case['free_ship_reflector_positions'] else 'Held fixed',
            'Unknown' if case['free_vehicle_reflectors'] else 'Held known',
            str(case['directions_weaker_than_declared_prior'])+'/18',
            f"{case['largest_information_eigenvalue']:.4g}"])+ '</tr>' for case in records)
        information_note = f'''<h2>Which assumptions supply apparent location precision?</h2>
<p>The information audit was repeated at the converged free-antenna solution, with O/K/Q and the ship, shared envelope origins and per-band vehicle reflecting points. Complex gains, target motion and shared origins are eliminated before evaluating antenna information. Coordinates are scaled by the declared location prior; an information eigenvalue of one equals that prior's information in that direction. Motion penalties and search bounds are not counted as measured information.</p>
<table><tr><th>Ship reflecting points</th><th>Vehicle reflecting points</th><th>Directions weaker than prior</th><th>Largest information eigenvalue</th></tr>{rows_}</table>
<p>The table uses a relative numerical rank cutoff of 10⁻¹⁰; the JSON also retains 10⁻⁸ and 10⁻¹². Treating fitted reflector coordinates as known can overstate calibration precision. Conversely, allowing independent unknown three-dimensional ship points in every band is deliberately permissive; it does not represent all the structure of a shared hull. These are conditional local bounds under the stated noise model, not confidence intervals for the real antennas or a proof that joint calibration cannot work. Stable physical reflector correspondence across targets or apertures remains a substantive scene-model question.</p>'''
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Transmitter siting assumptions</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>What the ship fits assume about transmitter siting</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="siting-assumptions.json">Values and source identities</a> · Updated {report['updated_utc']}</p>
<p class="note">No transmitter phase-centre calibration has been accepted. The latest motion-only controls use licensed main-site coordinates, not the earlier vehicle-refined trial coordinates. {shared_note}</p>
<h2>The nominal geometry</h2><table><tr><th>Band</th><th>Main site</th><th>Main mounting height (m AGL)</th><th>Licensed alternatives tested</th></tr>{rows}</table>
<p>Thus the nominal model has two horizontal tower locations, with per-band mounting heights. Three candidate towers appear across the licensed alternatives; five RF bands are not assumed to provide five widely separated towers. Seven and Nine can use Eyre; SBS can use Ridge; Ten can use Eyre or Ridge; ABC can use Ornata. The 2 × 2 × 2 × 3 × 2 choices give 48 assignments. These describe the cached 13 October ACMA licence snapshot, not an observation of which installation was active.</p>
<table><tr><th>Site</th><th>ACMA site ID</th><th>Latitude</th><th>Longitude</th></tr>{coordinate_rows}</table>
<p>The receiver reference is {prior['receiver']['latitude_deg']:.6f}, {prior['receiver']['longitude_deg']:.6f}, with 15 m approximate antenna height above terrain. It is a property-map position, not a roof-antenna survey. Ground and geoid models provide the height datum; the inconsistent RRL ground elevations were not used. The site-to-ECEF conversion currently has no explicit GDA94 epoch/frame transformation, another reason to keep absolute coordinates provisional.</p>
<h2>What was free in the ship-only joint sweep?</h2>
<p>Each frequency has its own electrical phase-centre correction. Channels assigned to the same mast share a location-error covariance. Declared standard deviations are 30/30/75 m east/north/up for a common site error, plus 5/5/10 m for each band's offset. Receiver standard deviations are 15/15/7.5 m. These are modelling priors, not measured error bars. Standardised correlated coordinates have ±3 search bounds.</p>
<p>Each band is represented by one fixed Cartesian electrical phase centre over the fitted observations. There is no explicit direction-dependent antenna phase-pattern model. Independent complex target/band coefficients absorb constant response phases, but do not establish that a single physical phase centre explains phase variation across all target directions. This remains an assumption to test before interpreting fitted offsets as antenna locations.</p>
<p>All 48 assignments received a joint fit of translation, five transmitter positions, receiver position and five envelope origins over 150.772–180.772 s. Yaw, roll and pitch were fixed. Complex reflector amplitudes are independent per band; eight reflector positions per band and a coarse transverse anchor remain conditional. Envelope origins can move ±75 m around their assignment-specific initial values. This protects against interpreting unknown scattering phase or an assumed envelope origin as surveyed antenna geometry, but also removes potential calibration information.</p>
<p>The training-selected order Seven/SBS/Nine/Ten/ABC was {best_names}. All 48 fits reached their numerical stationarity criterion. Against the matched fixed-location control, the selected fit changes withheld matched energy by {sweep['selected_gain_db_vs_fixed_locations']:.5f} dB and reduces squared prediction error by {sweep['selected_heldout_error_reduction_percent']:.3f}%. It reaches seven antenna-coordinate bounds and one envelope-origin bound; its receiver correction is approximately −45 m east, +45 m north. This is a local constrained model comparison, not calibrated site probabilities or a trustworthy physical position estimate. See the <a href="joint-sites-validation.html">complete sweep and nuisance-eliminated information audit</a>.</p>
<h2>How the earlier vehicles can help</h2>
<p>The road analysis used connected OpenStreetMap trajectories with speed, smoothness, lane and height uncertainty. Albert Street and Victoria Parade were plausibility preferences, not independently observed vehicle positions. Phase-transfer failures and alternative road branches remain relevant. See the <a href="../calibration_20261013/site-refits.html">audited common-source vehicle refits</a>.</p>
<img src="road-ship-angular-coverage.png" alt="Vehicle and ship departure-angle hypotheses at the three licensed towers">
<p>The current ten-road main-site hypotheses span about {np.ptp(coverage[0]['road_azimuth_range_deg']):.2f}° in azimuth at Ornata. The ship is about {coverage[0]['sector_separation_deg']:.2f}° from their median direction, so combining the datasets supplies a substantially different direction. The earlier 0.36° figure described a narrower preferred-street hypothesis set; it must not be applied to this revised set. These are angular-coverage calculations, not a proof of calibration rank or accuracy.</p>
<p>The combined fit shares physical TX/RX coordinates and site assignment across both datasets while refitting each target's trajectory and scattering nuisance. {sweep_note} Road alternatives, whole-target/time-block transfer, and dependence on coordinate and range-origin assumptions remain explicit acceptance checks. The previous road fit is an initialization and comparison, not an independent measurement to add as a second prior. Temporal phase and envelope migration can constrain geometry without declaring arbitrary target-dependent constant phases to be antenna calibration. A cross-band coherent scattering model still needs separate predictive validation.</p>
<p>The older combined sweep retains zero envelope-origin offsets for the vehicles while allowing fitted offsets for the ship. The newer matched controls instead share one absolute envelope origin per band across every vehicle and the ship. These assumptions are kept separate; neither establishes absolute-delay calibration. The candidate roads were inferred from this recording; their coordinates do not make the vehicle locations independently known.</p>
<p>{origin_note}</p>
{preferred_note}
{information_note}
<p>The <a href="road-temporal-validation.html">Doppler-gate information audit</a> and
<a href="full-road-validation.html">full-bin stationary-nuisance controls</a> test the vehicle likelihood before repeating physical calibration. Removing unnecessary temporal projection is a modelling improvement to test, not evidence that a particular tower assignment is correct. The completed full-bin joint fit retains original-row moving/static integration; its excluded-target result above still does not support adopting physical offsets.</p>
<p>The <a href="road-model-validation.html">additional averaging, covariance and reflecting-point controls</a> separate processing loss from scene assumptions. Saved 20 ms averaging has less than 0.007 dB conditional matched-SNR loss for these trajectories. Pooled noise estimates have mixed real-data prediction and are not adopted. Allowing a fixed reflecting point per band gives small gains, but frequent offset-bound hits prevent interpreting those offsets as measured vehicle structure or antenna calibration.</p>
<p>The completed leave-one-target-out controls give: {transfer_note} These tests do not yet support transferring the fitted antenna correction to an unseen target. They also do not establish that the calibration problem is insoluble. Observation-model checks, road alternatives, physical scene correspondence and site assignments under the revised model remain open work.</p>
<p>The <a href="shared-reflector-validation.html">shared-reflector experiment</a> tests another material assumption before further site refinement: whether common physical locations with independent complex reflectivity per band predict the ship data. Its fixed-site controls keep the present translation and range origins. This separates the common-scene hypothesis from an antenna correction; a fit conditioned on known or independently movable reflector positions cannot by itself establish the accuracy of a joint geometry solution.</p></html>'''
    (output/'siting-assumptions.html').write_text(text)
    print(json.dumps({'report': str(output/'siting-assumptions.html'), 'viewing_directions': coverage}))


if __name__ == '__main__':
    main()
