"""Fit every configured assignment to the bright flash and expose its gauges.

No phase continuity is assumed between short arcs. This report compares actual
phase/range estimates, not image sharpness or a fitted brightness statistic.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from compare_ais_phase import NAMES, COLOURS
from flash_geometry import Flash, assignments, observables
from imaging import sites
from motion import C
from ship_site_signature import licensed_options

SITE_NAMES = {'12013': 'Ornata', '12015': 'Eyre', '12014': 'Ridge'}
SITE_LETTERS = {'12013': 'O', '12015': 'E', '12014': 'R'}
EPOCH = utc_seconds('2026-10-13T08:33:36.107619Z')


def ais_observables(track, tx, rx, midpoint=164.5):
    state = track.sample([EPOCH+midpoint], max_gap_s=180)
    pair = track.sample([EPOCH+midpoint-.05, EPOCH+midpoint+.05], max_gap_s=180)
    acceleration = np.diff(pair['velocity_ecef_mps'], axis=0)[0]/.1
    return observables(state['position_ecef_m'][0], state['velocity_ecef_mps'][0], acceleration, tx, rx)


def field_reference(fields, track):
    """Recover the immutable deramp/range reference from the data themselves."""
    tx, rx, stored_path = [], [], []
    for channel in range(5):
        with np.load(fields/f'channel-{channel}.npz') as data:
            tx.append(data['transmitter_ecef_m']);rx.append(data['receiver_ecef_m'])
            stored_path.append(np.interp(164.5, data['row_time'], data['centre_path_m']))
    tx, rx = np.array(tx), np.array(rx)
    if not np.all(rx == rx[0]):
        raise ValueError('Field references must share the same receiver')
    reference = ais_observables(track, tx, rx[0])
    error = float(np.max(abs(reference[:, 0]-stored_path)))
    if error > 1e-5:
        raise ValueError('AIS or epoch differs from the saved field reference')
    return tx, rx[0], error


def flash_measurements(source, track, reference_tx, reference_rx, width=5, midpoint=164.5):
    """Restore absolute observables with the saved, not newly fitted, reference."""
    rows = json.loads(source.read_text())
    predicted = ais_observables(track, reference_tx, reference_rx, midpoint)
    measured, covariance = [], []
    for i, row in enumerate(rows):
        chosen = next(v for v in [row, *row.get('range_width_controls', [])] if len(v['offset_samples']) == width)
        scale = chosen['scale_s'];u = (midpoint-chosen['epoch_s'])/scale
        transform = np.array([[C/8e6, 0, u, u*u], [0, 0, 1/scale, 2*u/scale], [0, 0, 0, 2/scale**2]])
        measured.append(predicted[i]+transform@chosen['parameters'])
        covariance.append(transform@chosen['conditional_parameter_covariance']@transform.T)
    return np.array(measured), np.array(covariance), predicted


def sweep(source, geometry, output, fields, fit_options=None):
    fit_options = {} if fit_options is None else fit_options
    tx, rx, main = sites(geometry)
    prior = json.loads((geometry/'geometry-prior.json').read_text())
    rotation = np.array(prior['ecef_to_enu'])
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    reference_tx, reference_rx, reference_error = field_reference(fields, track)
    nominal = ais_observables(track, tx, rx)
    options = assignments(licensed_options(geometry))
    nominal_sites = [c['site_id'] for c in main['channels']]
    result = {'epoch_recording_s': 164.5, 'nominal_start_utc': '2026-10-13T08:33:36.107619Z',
              'nominal_sites': nominal_sites, 'assignment_count': len(options), 'cases': [],
              'accepted_calibration': False,
              'observable_definition': 'Carrier phase origins independent by band. Joint envelope/rate/acceleration covariance from the physical cluster fit, with constant reflectivity phase eliminated.',
              'nuisance_prior_sigma': {'reflector_offset_enu_m': [150, 150, 20], 'velocity_correction_enu_mps': [.5, .5, .1],
                  'acceleration_correction_enu_mps2': [.1, .1, .05], 'relative_ais_time_s': 1.},
              'prior_scope': 'Explicit sensitivity assumptions, not measurements. Position offset includes uncertain AIS interpolation and unknown reflector lever arm. No heading or yaw is inferred.',
              'ranking_scope': 'Penalized conditional profile least squares, not site probabilities. Real-data point-model and covariance error remain separate from numerical convergence.',
              'receiver_prior': prior['receiver'],
              'fit_options':fit_options,
              'physical_geometry_directory': str(geometry), 'fields_directory': str(fields),
              'data_reference': {'receiver_ecef_m': reference_rx.tolist(),
                  'transmitter_ecef_m': reference_tx.tolist(),
                  'maximum_ais_path_check_error_m': reference_error,
                  'scope': 'Immutable saved range/phase reference, independent of the physical antenna priors'},
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (source, geometry/'geometry-prior.json', geometry/'sites-main.json',
                   geometry/'acma-transmitters.json', Path('docs/data/isar_20261013/rio-delta-ais.csv'))},
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__),Path(__file__).with_name('flash_geometry.py'),
                   Path(__file__).with_name('flash_common_phase.py'),
                   Path(__file__).with_name('flash_siting_validation.py'),
                   Path(__file__).with_name('ship_site_signature.py'))}}
    for width in (5, 3, 7):
        measured, covariance, reference = flash_measurements(source, track, reference_tx, reference_rx, width)
        for with_range, with_acceleration in ((False, True), (True, True), (False, False)):
            records = []
            for index, assignment in enumerate(options):
                problem = Flash(measured, covariance, track, EPOCH+164.5, rotation, rx, assignment,
                                include_range=with_range, include_acceleration=with_acceleration, free_antennas=True,
                                **fit_options)
                fit = problem.fit()
                record = problem.describe(fit);record['index'] = index
                if width == 5:
                    fixed = Flash(measured, covariance, track, EPOCH+164.5, rotation, rx, assignment,
                                  include_range=with_range, include_acceleration=with_acceleration, free_antennas=False,
                                  **fit_options)
                    fixed_fit = fixed.fit()
                    record['fixed_antennas'] = fixed.describe(fixed_fit)
                records.append(record)
                if index%16 == 15:
                    print('flash sweep', width, 'cells', 'range+phase' if with_range else 'phase' if with_acceleration else 'rate only', index+1, '/', len(options), flush=True)
            # Check local convergence independently of the original zero seed
            # for the leading contenders; keep this separate from evidence of
            # physical identifiability or correctness of the point model.
            for record in sorted(records, key=lambda r:r['penalized_squared_residual'])[:3]:
                problem = Flash(measured, covariance, track, EPOCH+164.5, rotation, rx, options[record['index']],
                    include_range=with_range, include_acceleration=with_acceleration, **fit_options)
                costs = []
                for sign in (-1., 1.):
                    seed = np.array(record['parameters'])
                    seed[18:21] += sign*np.array([.3, -.2, .1]);seed[21:24] += sign*np.array([.15, -.1, .05])
                    low,high = problem.bounds()
                    seed = np.clip(seed, low+1e-6, high-1e-6)
                    check = problem.fit(seed);costs.append(float(check.fun@check.fun))
                    if costs[-1] < record['penalized_squared_residual']:
                        record.update(problem.describe(check))
                record['multistart_check_squared_costs'] = costs
            selected = min(records, key=lambda r:r['penalized_squared_residual'])
            result['cases'].append({'width_cells': width, 'include_range': with_range,
                                    'include_acceleration': with_acceleration,
                                    'measurements': measured.tolist(), 'covariance': covariance.tolist(),
                                    'nominal_ais_prediction': nominal.tolist(), 'records': records,
                                    'data_reference_ais_prediction': reference.tolist(),
                                    'selected_index': selected['index']})
            (output/'flash-siting.json').write_text(json.dumps(result, indent=2))
    return result


def plots(result, geometry, output):
    tx, rx, _ = sites(geometry)
    prior = json.loads((geometry/'geometry-prior.json').read_text())
    rotation = np.array(prior['ecef_to_enu'])
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    options = assignments(licensed_options(geometry))
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    summaries = []
    for column, with_range in enumerate((False, True)):
        case = next(c for c in result['cases'] if c['width_cells']==5 and c['include_range']==with_range)
        records = case['records'];ordered = sorted(records, key=lambda r:r['penalized_squared_residual'])
        baseline = ordered[0]['penalized_squared_residual']
        axes[0, column].plot(np.arange(len(ordered)), [r['penalized_squared_residual']-baseline for r in ordered], '.-', label='TX/RX coordinates fitted')
        axes[0, column].plot(np.arange(len(ordered)), [r['fixed_antennas']['penalized_squared_residual']-baseline for r in ordered], '.', label='TX/RX coordinates fixed')
        labels = [''.join(SITE_LETTERS[s] for s in r['sites']) for r in ordered]
        axes[0, column].set_xticks(np.arange(len(ordered))[::3], labels[::3], rotation=75, fontsize=8)
        axes[0, column].set_ylabel('Penalized squared residual above best')
        axes[0, column].set_title('Phase rate + acceleration'+(' + absolute envelope' if with_range else '; range origins free'))
        axes[0, column].legend(fontsize=8)
        for width, colour in ((3, '#009e73'), (5, '#222222'), (7, '#d55e00')):
            sensitivity = next(c for c in result['cases'] if c['width_cells']==width and c['include_range']==with_range)
            r = min(sensitivity['records'], key=lambda r:r['penalized_squared_residual'])
            data = r['data_information_eigenvalues_after_free_reflector_nuisance']
            axes[1, column].semilogy(np.arange(1, 19), np.maximum(data, 1e-16), '.-', color=colour, label=f'{width} cells')
            summaries.append({'width': width, 'range': with_range, 'sites': r['sites'],
                'score': r['penalized_squared_residual'], 'data_score': r['data_squared_residual'],
                'prior_score': r['prior_squared_residual'], 'optimality': r['first_order_optimality'],
                'largest_information': max(data), 'directions_above_prior': int(np.count_nonzero(np.array(data)>1))})
        axes[1, column].axhline(1, color='#777777', ls='--', label='One prior unit')
        axes[1, column].set_xlabel('Antenna-coordinate information direction')
        axes[1, column].set_ylabel('Data information after reflector/motion nuisance')
        axes[1, column].legend(fontsize=8)
    fig.suptitle(f'Bright-flash siting test · all {len(options)} configured assignments fitted\n'
                 'O = Ornata, E = Eyre, R = Ridge; channel order Seven / SBS / Nine / Ten / ABC')
    fig.savefig(output/'01-assignment-comparison.png', dpi=170);plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    case = next(c for c in result['cases'] if c['width_cells']==5 and not c['include_range'])
    best = min(case['records'], key=lambda r:r['penalized_squared_residual'])
    nominal_record = next(r for r in case['records'] if r['sites']==result['nominal_sites'])
    observed = np.array(case['measurements']);nominal = np.array(case['nominal_ais_prediction'])
    sigma = np.sqrt(np.diagonal(case['covariance'], axis1=1, axis2=2))
    for col, (label, unit) in enumerate((('Envelope offset from nominal AIS', 'm'), ('Excess-path rate', 'm/s'), ('Excess-path acceleration', 'm/s²'))):
        subtract = nominal[:, 0] if col == 0 else np.zeros(5)
        axes[col].errorbar(np.arange(5), observed[:, col]-subtract, yerr=sigma[:, col], fmt='o', color='black', label='Measured; conditional 1σ')
        axes[col].plot(np.arange(5), nominal[:, col]-subtract, '--', color='#009e73', label='Uncorrected AIS + nominal sites')
        axes[col].plot(np.arange(5), np.array(nominal_record['predicted_observables'])[:, col]-subtract, 's-', color='#0072b2', label='Nominal assignment joint fit')
        axes[col].plot(np.arange(5), np.array(best['predicted_observables'])[:, col]-subtract, 'x-', color='#d55e00', label='Best assignment joint fit')
        axes[col].set_xticks(np.arange(5), NAMES, rotation=25);axes[col].set_ylabel(unit);axes[col].set_title(label)
    axes[0].legend(fontsize=7)
    fig.suptitle('Does one AIS-anchored reflector explain the five flash phase histories?\n'
                 'Shared point position, velocity and acceleration; TX/RX and timing jointly fitted. No per-band velocity correction.')
    fig.savefig(output/'02-measured-versus-geometry.png', dpi=170);plt.close(fig)

    assignment = options[best['index']]
    problem = Flash(case['measurements'], case['covariance'], track, EPOCH+164.5, rotation, rx, assignment,
                    include_range=False, **result.get('fit_options', {}))
    p, v, a, fitted_tx, fitted_rx = problem.state(np.array(best['parameters']))
    reference = observables(p, v, a, fitted_tx, fitted_rx)
    shifts = np.linspace(-45, 45, 91)
    errors = []
    for east in shifts:
        shift = np.array([east, 0., 0.])
        value = observables(p+shift, v, a, fitted_tx+shift, fitted_rx+shift)
        errors.append(np.max(abs(value-reference), axis=0))
    errors = np.array(errors)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    axes[0].plot(shifts, errors[:, 0]*1e9, label='Excess path (nm)')
    axes[0].plot(shifts, errors[:, 1]*1e9, label='Path rate (nm/s)')
    axes[0].set_xlabel('Common eastward displacement of TX, RX and reflector (m)')
    axes[0].set_ylabel('Change in predicted observable; numerical roundoff')
    axes[0].legend(fontsize=8);axes[0].set_title('Exact common-translation ambiguity of a single flash')
    rx_changes = [];point_changes = []
    for c in result['cases']:
        if c['width_cells'] != 5:
            continue
        chosen = min(c['records'], key=lambda r:r['penalized_squared_residual'])
        offset = np.array(chosen['antenna_fit']['receiver_offset_enu_m'])
        axes[1].plot(offset[0], offset[1], 'o', label='Range+phase' if c['include_range'] else 'Phase derivatives only')
        rx_changes.append(offset.tolist());point_changes.append(chosen['reflector_offset_enu_m'])
    axes[1].plot(0, 0, '+', color='black', ms=12, label='Receiver prior')
    theta = np.linspace(0, 2*np.pi, 201)
    axes[1].plot(15*np.cos(theta), 15*np.sin(theta), '--', color='#999999', label='Declared 15 m horizontal prior σ')
    axes[1].set_aspect('equal');axes[1].set_xlabel('Receiver east correction (m)');axes[1].set_ylabel('North correction (m)')
    axes[1].legend(fontsize=8);axes[1].set_title('Prior-conditioned estimates, not adopted coordinates')
    fig.suptitle('What the flash can and cannot locate\n'
                 'A free AIS-to-reflector offset can absorb a common antenna translation; priors, not phase coherence, fix that gauge')
    fig.savefig(output/'03-translation-ambiguity.png', dpi=170);plt.close(fig)
    result['summary'] = summaries
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    rate_summaries = []
    for width, colour in ((3, '#009e73'), (5, '#222222'), (7, '#d55e00')):
        case = next(c for c in result['cases'] if c['width_cells']==width and not c['include_acceleration'])
        ordered = sorted(case['records'], key=lambda r:r['penalized_squared_residual'])
        values = [r['penalized_squared_residual'] for r in ordered]
        axes[0].plot(np.arange(len(values)), np.array(values)-values[0], '.-', color=colour, label=f'{width} cells')
        best_rate = ordered[0]
        measured = np.array(case['measurements'])[:, 1]
        sigmas = np.sqrt(np.array(case['covariance'])[:, 1, 1])
        axes[1].errorbar(np.arange(5)+(width-5)*.04, measured, yerr=sigmas, fmt='o', color=colour, label=f'{width}-cell measurement')
        axes[1].plot(np.arange(5), np.array(best_rate['predicted_observables'])[:, 1], '--', color=colour)
        rate_summaries.append({'width': width, 'sites': best_rate['sites'], 'score': values[0],
                               'runner_up_gap': values[1]-values[0],
                               'data_score': best_rate['data_squared_residual'],
                               'prior_score': best_rate['prior_squared_residual']})
    axes[0].set_xlabel('Assignment rank within each extraction');axes[0].set_ylabel('Penalized rate-only residual² above best')
    axes[0].set_title('Control: omit both envelope and phase curvature');axes[0].legend()
    axes[1].set_xticks(np.arange(5), NAMES);axes[1].set_ylabel('Excess-path rate (m/s)')
    axes[1].set_title('Measurements and each extraction’s best joint fit');axes[1].legend(fontsize=8)
    fig.suptitle('Rate-only siting control · does the choice depend on phase curvature?')
    fig.savefig(output/'07-rate-only-control.png', dpi=170);plt.close(fig)
    result['rate_only_summary'] = rate_summaries
    fig, axes = plt.subplots(1, 5, figsize=(16, 5), layout='constrained')
    for ch, ax in enumerate(axes):
        available = sorted({a[ch]['site_id'] for a in options})
        for width, colour in ((3, '#009e73'), (5, '#222222'), (7, '#d55e00')):
            case = next(c for c in result['cases'] if c['width_cells']==width and not c['include_acceleration'])
            best_cost = min(r['penalized_squared_residual'] for r in case['records'])
            profiled = [min(r['penalized_squared_residual'] for r in case['records'] if r['sites'][ch]==site)-best_cost
                        for site in available]
            ax.plot(np.arange(len(available)), profiled, '.-', color=colour, label=f'{width} cells')
        ax.set_xticks(np.arange(len(available)), [SITE_NAMES[s] for s in available], rotation=25)
        ax.set_title(NAMES[ch]);ax.set_ylabel('Best conditional score penalty')
    axes[0].legend(fontsize=8)
    fig.suptitle('Rate-only profile by transmitter · all other site choices and nuisance variables refitted\n'
                 'Sensitivity comparison, not posterior probabilities; absolute range and phase curvature omitted')
    fig.savefig(output/'08-channel-site-profiles.png', dpi=170);plt.close(fig)
    from flash_siting_validation import validate
    result['withheld_complex_validation'] = validate(result, options, track, EPOCH, rotation, rx, output,
                                                    fields=Path(result['fields_directory']))
    result['translation_gauge_max_roundoff'] = errors.max(axis=0).tolist()
    (output/'flash-siting.json').write_text(json.dumps(result, indent=2))
    return summaries


def receiver_plot(result, geometry, output):
    """Display a changed physical prior separately from unchanged observations."""
    _, rx, _ = sites(geometry)
    prior = json.loads((geometry/'geometry-prior.json').read_text())
    rotation = np.array(prior['ecef_to_enu'])
    old_rx = np.array(result['data_reference']['receiver_ecef_m'])
    displacement = (rx-old_rx)@rotation.T
    case = result['cases'][0]
    change = np.array(case['nominal_ais_prediction'])-case['data_reference_ais_prediction']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    axes[0].plot(0, 0, 'x', ms=10, color='#777777', label='Previous property-map prior')
    axes[0].plot(*displacement[:2], '+', ms=13, color='#0072b2', label='User-supplied antenna point')
    axes[0].annotate('', displacement[:2], (0, 0), arrowprops={'arrowstyle': '->', 'color': '#0072b2'})
    axes[0].set_aspect('equal');axes[0].margins(.25);axes[0].legend(fontsize=8)
    axes[0].set_xlabel('East from previous prior (m)');axes[0].set_ylabel('North from previous prior (m)')
    axes[0].set_title(f'Physical receiver correction; vertical {displacement[2]:+.2f} m')
    for axis, column, factor, unit in ((axes[1], 0, 1., 'm'), (axes[2], 1, 1000., 'mm/s')):
        axis.bar(np.arange(5), factor*change[:, column], color=COLOURS)
        axis.set_xticks(np.arange(5), NAMES, rotation=25);axis.set_ylabel(unit)
    axes[1].set_title('Change in predicted AIS excess path at the flash')
    axes[2].set_title('Change in predicted AIS excess-path rate')
    fig.suptitle('Physical antenna priors compared with the immutable saved reference geometry\n'
                 'These are changes in the physical prediction; saved range/phase references and measured observables are unchanged')
    fig.savefig(output/'11-receiver-reference-check.png', dpi=170);plt.close(fig)


def report(result, output, piecewise):
    piecewise_path = os.path.relpath(piecewise, output)
    receiver = result['receiver_prior']
    fit_options = result.get('fit_options', {})
    receiver_sigma = fit_options.get('receiver_sigma', [15.,15.,7.5])
    up_bounds = fit_options.get('receiver_up_bounds_m', [-22.5,22.5])
    height_bounds = [receiver['antenna_agl_m']+v for v in up_bounds]
    common = result['common_phase_control']
    common_best = sorted(common['records'], key=lambda r:r['conditional_squared_cost'])[:2]
    common_table = ''.join('<tr>'+''.join(f'<td>{v}</td>' for v in
        [*[SITE_NAMES[s] for s in row['sites']], f"{row['phase_rms_deg']:.4f}°",
         f"{row['conditional_squared_cost']:.4f}"] )+'</tr>' for row in common_best)
    phase_data_cost = [row['conditional_squared_cost']-np.sum(np.array(row['parameters'][:-1])**2)
                       for row in common_best]
    range_case = next(c for c in result['cases'] if c['width_cells']==5 and c['include_range'])
    range_records = [next(r for r in range_case['records'] if r['index']==row['index']) for row in common_best]
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [
        r['width'], 'Range + phase' if r['range'] else 'Phase derivatives',
        '/'.join(SITE_NAMES[s] for s in r['sites']), f"{r['data_score']:.2f}", f"{r['prior_score']:.2f}",
        f"{r['optimality']:.2g}", str(r['directions_above_prior'])+'/18'])+'</tr>' for r in result['summary'])
    text = f'''<!doctype html><meta charset="utf-8"><title>Bright-flash TX/RX siting test</title>
<style>body{{font:17px/1.5 system-ui;max-width:1300px;margin:2em auto;padding:0 1em;color:#222}}img{{width:100%}}td,th{{padding:.4em;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff2cc;padding:1em}}</style>
<h1>Bright-flash TX/RX siting test</h1>
<p>All {result['assignment_count']} configured primary/backup assignments are fitted to the 164–165 s flash. Receiver and transmitter corrections, one shared reflector position, velocity, acceleration and relative AIS time are fitted jointly. Independent per-band velocity corrections are <b>not</b> allowed. This is a fresh flash-specific fit, not a reuse of the long-aperture image-sharpness fit.</p>
<p class="note">Receiver prior updated to {receiver['latitude_deg']}, {receiver['longitude_deg']} at {receiver['antenna_agl_m']:g} m above ground, as supplied by the user. The original saved range and phase references are preserved explicitly. The phase-derivative controls allow independent band phase origins; a separate common-phase control below uses the constant interband phases as well.</p>
<h2>Receiver position and reference bookkeeping</h2><img src="11-receiver-reference-check.png">
<p>The physical antenna coordinates come from the selected geometry snapshot. The extracted absolute paths, rates, accelerations and their covariances retain the original data reference; updating physical priors does not alter these observations. The AIS path reconstructed with the saved data-reference geometry agrees with the archived path to {result['data_reference']['maximum_ais_path_check_error_m']:.3g} m at the flash. The physical model uses the new receiver independently of that reversible data reference.</p>
<h2>Configured primary/backup alternatives</h2><p>O/E/R denote Ornata/Eyre/Ridge, in Seven/SBS/Nine/Ten/ABC order. The two controls either use the measured envelope origin as absolute excess path, or allow unknown per-band envelope origins and use phase derivatives alone. These are intentionally different assumptions. No local normalization or brightness maximization enters this comparison.</p>
<img src="01-assignment-comparison.png">
<table><tr><th>Cells</th><th>Observations</th><th>Selected assignment</th><th>Data residual²</th><th>Prior penalty²</th><th>Optimality</th><th>Information above prior</th></tr>{table}</table>
<p>The covariance is the full delay/rate/acceleration covariance from the joint PSF fit, including elimination of constant phase and amplitude. It is a conditional thermal calculation. The 3/5/7-cell sensitivity exposes model dependence; scores are not Bayesian probabilities or calibrated confidence levels. AIS offset uncertainty is not removed by calling the interpolation an exact position.</p>
<h2>Measured phase histories versus geometry</h2><img src="02-measured-versus-geometry.png">
<p>The flash supplies one location on the vessel and a very short trajectory. The plot shows whether a single physical reflector, with shared motion, accounts for all channels. A high phase correlation does not imply that the measured rate differences equal the small geometric differences between the towers.</p>
<h2>Rate-only sensitivity control</h2><img src="07-rate-only-control.png">
<p>This repeats all {result['assignment_count']} assignments using only phase rate, with neither envelope origins nor phase acceleration counted as location information. It checks whether uncertain short-arc curvature drives the site ranking. The likelihood and all motion/location nuisance variables are otherwise unchanged.</p>
<img src="08-channel-site-profiles.png">
<h2>Using the constant interband phases</h2><img src="10-common-phase-control.png">
<p class="note"><b>The phase RMS values below describe five fitted carrier-phase residuals at the single midpoint 164.5 s.</b> It is not phase accuracy throughout the flash, prediction accuracy, or agreement with all the observations. Absolute envelope range and phase curvature were excluded from this particular control.</p>
<table><tr><th>Seven</th><th>SBS</th><th>Nine</th><th>Ten</th><th>ABC</th><th>Fitted midpoint phase RMS</th><th>Conditional cost</th></tr>{common_table}</table>
<p>The two leading assignments are listed explicitly above. This control assumes one common reflector phase across the bands and calibrated relative instrumental phase. It fits phase and path rate at 164.5 s jointly, including their covariance, while allowing TX/RX positions, reflector offset, velocity and AIS timing to adjust within the declared priors. It enumerates nearby carrier integers for every configured assignment. Best-to-runner-up conditional cost difference: {common['runner_up_cost_gap']:.4f}; this is not a posterior odds ratio.</p>
<p>Which gives smaller measurement residuals after refinement depends on the observations retained. Both columns below use refitted transmitter/receiver and reflector parameters. Priors influence those fits but are excluded from the displayed measurement costs:</p>
<table><tr><th>Fit observations</th><th>{"/".join(SITE_LETTERS[s] for s in common_best[0]["sites"])}: measurement residual²</th><th>{"/".join(SITE_LETTERS[s] for s in common_best[1]["sites"])}: measurement residual²</th></tr>
<tr><td>Midpoint carrier phase + path rate</td><td>{phase_data_cost[0]:.3f}</td><td>{phase_data_cost[1]:.3f}</td></tr>
<tr><td>Envelope range + path rate + curvature; independent band phase origins</td><td>{range_records[0]['data_squared_residual']:.3f}</td><td>{range_records[1]['data_squared_residual']:.3f}</td></tr></table>
<p>The two objectives use different observations; compare their residuals separately. Bound contacts in the range-inclusive refits: {range_records[0]['bound_hits']} and {range_records[1]['bound_hits']}. A low midpoint phase residual alone does not establish agreement with the full envelope/rate/curvature data.</p>
<p>Carrier phase constrains excess path modulo a wavelength, approximately 1.3–1.7 m across these bands. A different site changes the integer cycle count; coordinate corrections can then align the fractional cycle. These fits allow shared site uncertainty and separate per-band phase-centre offsets, so the good midpoint agreement is a fitted result with substantial freedom. The search covers nearest-cycle patterns around each rate-only solution, not every branch throughout the spatial priors.</p>
<img src="12-common-phase-validation.png">
<p>The left panel above evaluates the same two common-phase fits on unused samples across 164–165 s, holding the fitted common phase fixed. The right panel shows the physical range mismatch that their fitting objective excluded. The independent range origins are retained only for the left panel, consistently with the control. Neither solution is a demonstrated good fit to the complete range and phase data. The ambiguity shown here belongs to this restricted phase-plus-rate test; it does <b>not</b> prove that the full joint range/phase/motion problem is ambiguous. <a href="common-phase-control.json">All {result['assignment_count']} controls and excluded-observable errors</a>.</p>
<h2>Separate independent-phase fits: original-sample prediction check</h2><img src="09-withheld-complex-validation.png">
<p>This plot checks the earlier independent-band-phase rate/range controls, not the two common-phase fits immediately above. It uses the untouched complex observations through the exact original row epochs, both OFDM moments and recorded binning. Constant band phases and amplitudes are learned on training samples only. The test samples fit neither geometry nor phase. The rate-only controls retain the free envelope origin they assumed; the absolute-range control does not receive that extra freedom.</p>
<h2>Piecewise fits to many candidates</h2><p>Twelve one-second candidate arcs span three range regions and four time regions. Discovery uses only the training half of the samples and a multiband coherent delay/rate search with unknown constant band gains. The other half tests each local fit. There is no phase stitching across flashes, fades or different candidates. These are candidate observations, not twelve established persistent physical points; neighbouring regions may include sidelobes of the same echo.</p>
<img src="{piecewise_path}/04-piecewise-candidates.png"><img src="{piecewise_path}/05-piecewise-phase-fits.png"><img src="{piecewise_path}/06-piecewise-validation.png">
<p>Low phase error alone does not identify a distinct reflector: a candidate can follow leaked energy from a nearby bright return while explaining little of its own range cluster. The complex-prediction fraction and range-bound flags are retained alongside phase error. Weak candidates are shown rather than silently promoted to calibration points.</p>
<p><a href="{piecewise_path}/piecewise-fits.json">All 60 per-band local fits and prediction statistics</a> · <a href="{piecewise_path}/piecewise-discovery.json">Training-only discovery surface and candidate definitions</a>. These are fits to the unchanged archived data reference; changing the physical receiver prior does not alter the extracted observations.</p>
<h2>An explicit ambiguity check</h2><img src="03-translation-ambiguity.png">
<p>For excess path L = |x − T| + |x − R| − |T − R|, replacing all three positions by x+d, T+d and R+d leaves L and every time derivative unchanged. During one short flash, an unknown constant AIS-to-reflector offset permits exactly this transformation. The plotted calculation repeats it numerically with the fitted geometry. Receiver/TX absolute corrections along these directions are supplied by location and reflector priors, not measured by this flash.</p>
<p>A common translation is harmless to relative coherent image formation: a reference position can fix that gauge. It is <b>not</b> the reason to reject a useful relative calibration. The separate issues here are the changing licensed-site ranking and the remaining measured interband mismatch. Fixing a coordinate origin cannot remove those discrepancies.</p>
<p>At a single epoch, z_c = a_c exp(−ik_c L_c). An alternative path L′_c gives exactly the same sample with a′_c = a_c exp[ik_c(L′_c−L_c)]. Consequently, a stable unknown band phase cannot be counted as an additional absolute-range equation. Temporal derivatives and range envelopes still provide useful information, as tested above. This does not say that many independent known reflectors or additional geometric constraints cannot resolve the siting.</p>
<h2>Inputs and assumptions</h2><p>Nominal UTC origin 2026-10-13T08:33:36.107619Z. The flash is between AIS reports separated by 56 s. Receiver: {receiver['latitude_deg']}, {receiver['longitude_deg']}, approximately {receiver['antenna_agl_m']:g} m AGL. Terrain height {receiver['terrain_egm96_m']:.3f} m EGM96 plus geoid undulation {receiver['geoid_undulation_m']:.3f} m gives the ellipsoid height used in ECEF conversion. Receiver σ is {receiver_sigma} m ENU (zero fixes that coordinate); receiver height is bounded to {height_bounds} m AGL. Transmitter covariance uses shared tower σ 30/30/75 m plus per-channel σ 5/5/10 m. Reflector/AIS-position offset σ 150/150/20 m, velocity correction σ 0.5/0.5/0.1 m/s, acceleration σ 0.1/0.1/0.05 m/s², relative time σ 1 s; other bounds ±3σ. These are sensitivity assumptions, not survey measurements. Fixed horizontal reflector offsets over this one-second flash do not assert that AIS course equals vessel heading.</p>
<p><a href="flash-siting.json">All numerical fits, priors, covariance, information spectra and source identity</a> · <a href="../phase-cluster-refinement/index.html">Phase-extraction controls and plots</a></p>
<p>The live and file-input metadata now carry the user-supplied receiver position for future use. No fitted transmitter correction was adopted. Historical capture sidecars, archived field references and earlier analysis files are unchanged.</p>'''
    (output/'index.html').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-report', type=Path, default=Path('docs/data/isar_20261016/phase-cluster-refinement/02-bright-cluster.json'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261016_sites'))
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261016/flash-siting-sites-updated'))
    parser.add_argument('--piecewise-report', type=Path, default=Path('docs/data/isar_20261016/flash-siting'))
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--receiver-horizontal-sigma',type=float,default=0.,help='Metres; zero fixes the supplied latitude/longitude')
    parser.add_argument('--receiver-height-agl',type=float,nargs=2,default=[0.,20.],metavar=('MIN','MAX'))
    args = parser.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    if args.receiver_horizontal_sigma < 0 or not 0 <= args.receiver_height_agl[0] < args.receiver_height_agl[1]:
        parser.error('Use nonnegative receiver uncertainty and ordered above-ground height bounds')
    prior = json.loads((args.geometry/'geometry-prior.json').read_text())
    fit_options = {'receiver_sigma':[args.receiver_horizontal_sigma,args.receiver_horizontal_sigma,7.5],
                   'receiver_up_bounds_m':[h-prior['receiver']['antenna_agl_m'] for h in args.receiver_height_agl]}
    path = args.output/'flash-siting.json'
    if path.exists() and not args.reuse:
        parser.error('Output already contains a fit; use --reuse or choose a fresh output directory')
    result = json.loads(path.read_text()) if args.reuse else sweep(args.phase_report, args.geometry, args.output, args.fields, fit_options)
    if result.get('fit_options') != fit_options:
        raise ValueError('Cached receiver constraints differ; choose a fresh output directory')
    expected = [[o['site_id'] for o in a] for a in assignments(licensed_options(args.geometry))]
    if any([r['sites'] for r in case['records']] != expected for case in result['cases']):
        raise ValueError('Cached assignments differ from the configured candidate set; use a fresh output directory')
    for source, digest in result['source_sha256'].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError('Cached siting inputs changed: '+source)
    plots(result, args.geometry, args.output)
    receiver_plot(result, args.geometry, args.output)
    from flash_common_phase import run
    _, rx, _ = sites(args.geometry)
    prior = json.loads((args.geometry/'geometry-prior.json').read_text())
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    result['common_phase_control'] = run(result, assignments(licensed_options(args.geometry)), track, EPOCH,
        np.array(prior['ecef_to_enu']), rx, args.phase_report, args.output)
    from flash_siting_validation import validate_common
    result['common_phase_validation'] = validate_common(result, assignments(licensed_options(args.geometry)), track, EPOCH,
        np.array(prior['ecef_to_enu']), rx, args.output, fields=args.fields)
    path.write_text(json.dumps(result, indent=2))
    report(result, args.output, args.piecewise_report)
    print(args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
