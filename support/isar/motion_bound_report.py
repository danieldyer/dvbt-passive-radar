"""Compare declared motion limits without changing the fitted scene or data split."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from kinematics import constraints
from motion import enu_basis, excess_path
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def trajectory_check(model, track, transmitter, receiver, limits):
    """Check the sampled linear constraints and exact trajectory independently."""
    curve = SplineMotion(model)
    parameters, active = curve.gauge_fixed_parameters()
    matrix, lower, upper, description = constraints(track, utc_seconds(model['nominal_start_utc']),
        model, transmitter, receiver, active, len(active), path_acceleration_mps2=limits[0],
        heading_rate_degrees_ps=limits[1], heading_acceleration_degrees_ps2=limits[2])
    centre = np.concatenate([description[name] for name in ('prior_reference_path_acceleration_mps2',
        'prior_heading_rate_degrees_ps', 'prior_heading_acceleration_degrees_ps2')])
    sampled = (matrix@parameters[active]+centre).reshape(3, -1)
    # A very small second-difference step amplifies rounding in absolute UTC
    # timestamps. Use the same 50 ms scale as the declared constraints and
    # record the interior interval rather than silently extrapolating splines.
    time = np.linspace(curve.interval[0]+.06, curve.interval[1]-.06, 3001)
    step = .05
    def state_at(t):
        prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+t, model['attitude_prior'])
        return curve.apply(t, prior, transmitter, receiver)
    def observables(state):
        _, gradient = excess_path(state['position_ecef_m'], transmitter, receiver)
        path_rate = np.sum(gradient*state['velocity_ecef_mps'], axis=1)
        local = np.einsum('nji,nj->ni', enu_basis(state['latitude'], state['longitude']),
                          state['body_to_ecef'][:, :, 0])
        heading = np.degrees(np.unwrap(np.arctan2(local[:, 0], local[:, 1])))
        return path_rate, heading
    before, middle, after = [observables(state_at(time+offset)) for offset in (-step, 0., step)]
    plus = (after[1]-middle[1]+180)%360-180
    minus = (before[1]-middle[1]+180)%360-180
    exact = np.array([(after[0]-before[0])/(2*step), (plus-minus)/(2*step),
                      (plus+minus)/step**2])
    return {'limits': limits, 'sampled_minimum': sampled.min(axis=1).tolist(),
        'sampled_maximum': sampled.max(axis=1).tolist(),
        'sampled_active_counts': np.sum(np.abs(np.abs(sampled)-np.array(limits)[:, None]) < 1e-7, axis=1).tolist(),
        'maximum_sampled_violation': float(max(0., np.max(lower-matrix@parameters[active]),
                                              np.max(matrix@parameters[active]-upper))),
        'exact_dense_minimum': exact.min(axis=1).tolist(), 'exact_dense_maximum': exact.max(axis=1).tolist(),
        'exact_epochs': len(time), 'exact_interval_s': [float(time[0]), float(time[-1])], 'derivative_step_s': step,
        'scope': 'Exact ECEF path-rate derivative and body-frame heading relative to the prior local tangent frame; finite-difference derivatives. A dense sampled check is not a continuous-time constraint proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bounded', 'relaxed', 'gauges', 'ais', 'geometry-directory', 'test-log', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    bounded, relaxed, gauges = [read(p) for p in (args.bounded, args.relaxed, args.gauges)]
    seed_path = Path(bounded['identity']['source_motion'])
    seed = read(seed_path)
    if not gauges['complete'] or not all(m['physical_profile']['converged'] for m in (bounded, relaxed)):
        raise ValueError('Finish both constrained solves and the reference check')
    if bounded['identity']['source_sha256'] != relaxed['identity']['source_sha256']:
        raise ValueError('Control fits must use the same inputs')
    for path, digest in bounded['identity']['source_sha256'].items():
        if fingerprint(Path(path)) != digest:
            raise ValueError(f'Control input changed: {path}')
    np.testing.assert_allclose(bounded['points_body_m'], relaxed['points_body_m'], rtol=0, atol=0)
    before = bounded['physical_profile']['before']
    np.testing.assert_allclose([b['prediction_error'] for b in before],
        [b['prediction_error'] for b in relaxed['physical_profile']['before']], rtol=1e-12)
    log = args.test_log.read_text()
    tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical test log is not successful')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=seed['reference_height_ellipsoid_m'], mmsi=563052200)
    reference = seed['reference_channel']
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'interval_s': seed['interval_s'], 'numerical_tests_passed': int(tests[1]),
        'roll_pitch_fitted': False, 'correction_adopted': False,
        'seed_constraints': trajectory_check(seed, track, tx[reference], rx, [.1, .5, .01]),
        'controls': {}, 'gauges': gauges,
        'source_sha256': {str(p): fingerprint(p) for p in (args.bounded, args.relaxed, args.gauges,
            args.ais, seed_path, args.test_log, args.geometry_directory/'geometry-prior.json',
            args.geometry_directory/'sites-main.json', Path(__file__))},
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in (
            'fit_physical_profile.py', 'physical_profile.py', 'full_motion.py', 'field.py',
            'kinematics.py', 'spline_motion.py', 'stationarity.py', 'validate_motion.py', 'test_field.py')}}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout='constrained')
    time = np.linspace(*seed['interval_s'], 601)
    p0, y0 = SplineMotion(seed).curves(time)
    fig_motion, motion_axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    for i, (name, model) in enumerate((('Original limits', bounded), ('Doubled accelerations', relaxed))):
        fit = model['physical_profile']
        limits = fit['kinematic_constraints']
        radii = [limits[key] for key in ('path_acceleration_bound_mps2', 'heading_rate_bound_degrees_ps',
                                       'heading_acceleration_bound_degrees_ps2')]
        after = fit['after']
        error = [100*(b['prediction_error']/a['prediction_error']-1) for a, b in zip(before, after)]
        phases = [b['prediction_phase_rms_degrees'] for b in after]
        pooled = 100*(sum(b['prediction_error'] for b in after)/sum(a['prediction_error'] for a in before)-1)
        summary['controls'][name] = {'objective': fit['objective'], 'kkt_residual': fit['maximum_kkt_residual'],
            'error_change_percent': pooled, 'channel_error_change_percent': error,
            'phase_rms_degrees': phases,
            'matched_energy_change_db': [float(10*np.log10(b['heldout_matched_energy']/a['heldout_matched_energy']))
                                        for a, b in zip(before, after)],
            'range_origins_m': model['range_bias_m'], 'stationarity_polish': fit['stationarity_polish'],
            'constraints': trajectory_check(model, track, tx[reference], rx, radii)}
        axes[0].bar(np.arange(5)+(i-.5)*.35, error, width=.35, label=name)
        axes[1].plot(names, phases, 'o-', label=name)
        path, yaw = SplineMotion(model).curves(time)
        motion_axes[0].plot(time, path-p0, label=name)
        motion_axes[1].plot(time, -np.degrees(yaw-y0), label=name)
        motion_axes[2].bar(np.arange(5)+(i-.5)*.35, np.array(model['range_bias_m'])-seed['range_bias_m'],
                           width=.35, label=name)
    axes[0].axhline(0, color='black', lw=.7)
    axes[0].set(xticks=np.arange(5), xticklabels=names, ylabel='Withheld squared prediction error change (%)',
                title='Relative to the unchanged motion seed\nNegative is better')
    axes[1].plot(names, [x['prediction_phase_rms_degrees'] for x in before], 'o--', color='black', label='Seed')
    axes[1].set(ylabel='One-second prediction-phase RMS (degrees)',
                title='Includes noise, glint and scene mismatch\nNot a pure motion-error measurement')
    for axis in axes: axis.legend(fontsize=9)
    fig.savefig(args.output/'motion-bounds-validation.png', dpi=160)
    plt.close(fig)
    motion_axes[0].set(xlabel='Capture time (s)', ylabel='Path change (m)', title='Additional physical path correction')
    motion_axes[1].set(xlabel='Capture time (s)', ylabel='Heading change (degrees)', title='Conditional body-frame heading')
    motion_axes[2].set(xticks=np.arange(5), xticklabels=names, ylabel='Envelope-origin change (m)',
                       title='Envelope correction only')
    for axis in motion_axes: axis.legend(fontsize=9)
    fig_motion.savefig(args.output/'motion-bounds-corrections.png', dpi=160)
    plt.close(fig_motion)
    comparison = 100*(relaxed['physical_profile']['after_summary']['prediction_error']/
                      bounded['physical_profile']['after_summary']['prediction_error']-1)
    summary['relaxed_error_change_from_bounded_percent'] = comparison
    summary['conclusion'] = ('No correction accepted. Numerical stationarity is established, but these conditional '
        'motion models do not establish absolute heading, hull centroid or transferable cross-band phase calibration.')
    (args.output/'motion-bounds-validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    def link(path, title):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(title)}</a>'
    rows = ''.join(f'<tr><td>{name}</td><td>{item["objective"]:.9f}</td><td>{item["kkt_residual"]:.3g}</td>'
        f'<td>{item["error_change_percent"]:+.3f}%</td><td>{item["constraints"]["sampled_active_counts"]}</td></tr>'
        for name, item in summary['controls'].items())
    detail = ''.join(f'<tr><td>{name}</td><td>{before[c]["prediction_phase_rms_degrees"]:.2f}°</td>'+
        ''.join(f'<td>{item["channel_error_change_percent"][c]:+.3f}% / {item["phase_rms_degrees"][c]:.2f}°</td>'
                for item in summary['controls'].values())+'</tr>' for c, name in enumerate(names))
    exact_rows = ''.join(f'<tr><td>{name}</td><td>{item["constraints"]["limits"]}</td>'
        f'<td>{[round(max(abs(a), abs(b)), 6) for a, b in zip(item["constraints"]["exact_dense_minimum"], item["constraints"]["exact_dense_maximum"])]}</td></tr>'
        for name, item in summary['controls'].items())
    max_geometry = max(c['maximum_point_position_difference_m'] for c in gauges['channels'])
    max_dictionary = max(c['body_relabelling_dictionary_relative_error'] for c in gauges['channels'])
    max_phase = max(c['band_relabelling_prediction_relative_error'] for c in gauges['channels'])
    polish = relaxed['physical_profile']['stationarity_polish']
    curvature = ('No separate positive-curvature certificate is available.' if not polish else
        f'The relaxed solution has tangent-curvature eigenvalues '
        f'{min(polish["curvature_eigenvalues"]):.4f}–{max(polish["curvature_eigenvalues"]):.4f} '
        f'in {polish["tangent_dimensions"]} directions on the active constraint face, '
        'using the declared information-scaled coordinates.')
    methods = [(args.bounded, 'Original-limit fit'), (args.relaxed, 'Relaxed-acceleration fit'),
        (args.gauges, 'Recorded-geometry gauge check'), (args.test_log, 'Numerical test log'),
        (Path(__file__), 'Report and exact-trajectory checks'),
        (Path('support/isar/check_reference_gauges.py'), 'Reference-gauge validation code')]
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship motion bounds and reference ambiguities</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 22px;color:#20242a}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border:1px solid #cbd2d8;text-align:left}}.note{{background:#fff1ce;padding:16px}}code{{background:#eef1f3;padding:2px 4px}}</style>
<h1>Ship autofocus: are the motion bounds hiding a useful correction?</h1>
<p>Updated {summary['updated_utc']}. Capture <code>live-vhf-20261013-193334</code>, 142–172 seconds.</p>
<p class="note"><strong>Roll and pitch remain fixed.</strong> Both physical motion fits converge. The relaxed fit changes withheld error by {summary['controls']['Doubled accelerations']['error_change_percent']:+.3f}% relative to the unchanged seed, and by {comparison:+.3f}% relative to the original-limit refit. The tables and phase plots below distinguish numerical convergence from predictive improvement. No new calibration or replacement video is adopted.</p>
<h2>Controlled physical fit</h2>
<p>The original fit touches both path-acceleration and heading-acceleration limits; its heading-rate limit is inactive. This control doubles only the acceleration limits: 0.1 → 0.2 m/s² and 0.01 → 0.02°/s². Heading rate remains bounded by 0.5°/s. The 64 continuously refined locations, observations, row operators, independent complex band coefficients, stationary-nuisance treatment and withheld blocks are unchanged. Each trial refits the scene coefficients. Range envelope, carrier phase and symbol Doppler change together before row averaging.</p>
<p>Odd interior one-second blocks are withheld. Both aperture edges are observed. The initial motion and support were discovered on this aperture, so these are conditional predictive tests, not an independent target-identification experiment.</p>
<table><tr><th>Fit</th><th>Normalized training error</th><th>KKT residual</th><th>Withheld error vs seed</th><th>Active samples: path accel., heading rate, heading accel.</th></tr>{rows}</table>
<p>{curvature} These are local numerical diagnostics of this fixed-support model; they do not establish a globally unique physical trajectory.</p>
<img src="motion-bounds-validation.png" alt="Withheld error and residual phase under original and relaxed physical-motion bounds">
<table><tr><th>Channel</th><th>Seed phase RMS</th><th>Original limits: error change / phase RMS</th><th>Doubled accelerations: error change / phase RMS</th></tr>{detail}</table>
<p>Prediction-phase RMS includes noise, glint and scattering-model mismatch. It is not a direct heading-error or path-error measurement. Decreasing a training objective alone does not establish a physical correction.</p>
<img src="motion-bounds-corrections.png" alt="Physical path, heading and envelope-origin changes">
<p>The constraint implementation samples every 0.25 s and linearizes the reference path correction. An independent check samples the actual corrected ECEF trajectory 3,001 times over 142.06–171.94 s, using exact path gradients and 50 ms finite differences of path rate. Heading is measured in the prior local tangent frame. The following maxima document numerical and between-sample differences; this is not a continuous-time constraint certificate.</p>
<table><tr><th>Fit</th><th>Declared limits: m/s², °/s, °/s²</th><th>Dense exact maximum magnitudes</th></tr>{exact_rows}</table>
<h2>What the measurements can identify without an extra reference</h2>
<p>For an unknown rigid scattering model, a body-frame relabelling preserves every physical point: <code>x′=x+R d</code>, <code>R′=R Q</code>, <code>p′=Qᵀ(p−d)</code>. The corresponding antenna coordinate transforms in the same way, so the AIS antenna trajectory is unchanged too. A constant heading offset and the hull-origin convention therefore cannot be inferred from freely located scatterers alone. Known hull shape or identified physical features can break this ambiguity; this calculation does not claim that every relabelling is compatible with the known hull.</p>
<p>The recorded-geometry test shifts the body origin by [80, 5, 0] m and rotates its axes by 15°. It transforms all 64 points and their velocities consistently. Maximum physical point difference is {max_geometry:.3g} m; the largest relative complex-template change is {max_dictionary:.3g}. This verifies the reference ambiguity in the actual forward operator. It does not imply that time-varying yaw is wholly unobservable.</p>
<p>With independent complex band reflectivities, <code>g[c] → g[c] exp(iδ[c])</code> and <code>a[c,j] → a[c,j] exp(−iδ[c])</code> leave the prediction invariant. The five recorded dictionaries reproduce this to {max_phase:.3g} relative error. This concerns unknown calibration versus unknown scattering phase, not a loss of acquisition coherence. Geometric variations with time or position are not removed by this constant-phase gauge.</p>
<p>The <a href="joint-point-validation.html">shared-phase and range-origin tests</a> tried to provide a common scattering reference, but their fitted channel phases did not transfer reliably between aperture halves or anchor subsets. The <a href="refined-phase-validation.html">refined-support phase autofocus tests</a> likewise failed complementary-range transfer. The supplied AIS contains course over ground, not true heading or antenna-to-hull offsets. Those additional measurements would anchor the body frame, but would not by themselves establish a cross-band scattering-phase reference.</p>
<h2>Validation and status</h2>
<p>{int(tests[1])} numerical support tests pass, including known-motion recovery, exact profiled residual derivatives and physical delay/phase/Doppler consistency. The separate gauge check uses the recorded epochs and geometry. All new work is in offline support code. Existing videos remain conditional trial images. The requested accurately calibrated, hull-centred five-channel reconstruction remains unfinished.</p>
<ul>{''.join('<li>'+link(p, title)+'</li>' for p, title in methods)}</ul>
<p><a href="analysis.html">Analysis index</a></p></html>'''
    (args.output/'motion-bounds-validation.html').write_text(page)
    print(json.dumps({key: summary[key] for key in ('controls', 'relaxed_error_change_from_bounded_percent')}, indent=2))


if __name__ == '__main__':
    main()
