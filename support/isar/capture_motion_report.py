"""Publish full-capture motion checks and the north-up diagnostic videos."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from imaging import sites, motion_state
from motion import excess_path
from spline_motion import SplineMotion

ROOT = Path('build/isar-20261015')
OUTPUT = Path('docs/data/isar_20261015')
CAPTURE = Path('captures/live-vhf-20261013-193334')


def completed(directory):
    return [json.loads(p.read_text()) for p in sorted((ROOT/directory).glob('window-???.json'))]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = json.loads((ROOT/'full-capture-radar-motion.json').read_text())
    groups = {name: completed(name) for name in ('full-response-radar-motion', 'common-path-radar-motion',
        'two-dimensional-radar-motion', 'two-dimensional-anchored-radar-motion',
        'zero-start-radar-motion', 'fixed-reference-radar-motion', 'linear-reference-radar-motion')}
    report = {'motion_scope': 'Radar path/yaw initialization in the videos; full-response one- and two-dimensional refinements remain separately tested controls.',
        'no_new_phase_centre_calibration': True, 'groups': {}}
    for name, records in groups.items():
        report['groups'][name] = []
        for record in records:
            item = {key: record[key] for key in ('interval_s', 'before_summary', 'after_summary')}
            item.update(converged=record['fit']['converged'], validation_version=record.get('validation_version', 1),
                        training_objective=record['fit']['objective'],
                        error_change_percent=100*(record['after_summary']['prediction_error']/record['before_summary']['prediction_error']-1))
            if record.get('transverse_spacing_seconds') is not None:
                curve = SplineMotion(record)
                time = np.linspace(*record['interval_s'], 361)
                transverse = curve.components(time)[:, -1]
                item['transverse_extent_m'] = [float(transverse.min()), float(transverse.max())]
            report['groups'][name].append(item)
    # Geometry diagnostic only: equal-precision path-rate measurements with
    # known positions. This is not an empirical uncertainty on vessel motion.
    time = np.linspace(*model['interval_s'], 301)
    geo_dir = Path(model['geometry_directory'])
    tx, rx, config = sites(geo_dir)
    geo = json.loads((geo_dir/'geometry-prior.json').read_text())
    enu = np.asarray(geo['ecef_to_enu'])
    track = read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'), reference_height_ellipsoid_m=5., mmsi=563052200)
    state = track.sample(utc_seconds(model['nominal_start_utc'])+time, max_gap_s=400.)
    gradient = np.array([excess_path(state['position_ecef_m'], antenna, rx)[1]@enu.T for antenna in tx])[:, :, :2]
    singular = np.linalg.svd(gradient.transpose(1, 0, 2), compute_uv=False)
    ratio = singular[:, 0]/singular[:, 1]
    report['geometry'] = {'channel_sites': config['channel_sites'],
        'maximum_transmitter_baseline_m': float(np.linalg.norm(tx[:, None]-tx[None, :], axis=2).max()),
        'horizontal_path_rate_singular_ratio': [float(ratio.min()), float(ratio.max())],
        'scope': 'Known nominal geometry and equal-precision instantaneous path-rate measurements; no temporal continuity, scattering, noise or location uncertainty included. Not a full-aperture CRLB or impossibility result.'}
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    global_curve = SplineMotion(model)
    axes[0].plot(time, global_curve.curves(time)[0], color='0.45', label='Video initializer')
    for i, record in enumerate(groups['common-path-radar-motion']):
        t = np.linspace(*record['interval_s'], 181)
        curve = SplineMotion(record)
        axes[0].plot(t, curve.curves(t)[0], alpha=.65, label='Local full-response fit' if i == 0 else None)
    axes[0].set_ylabel('Path correction (m)'); axes[0].legend()
    axes[0].set_title('Radar motion controls; local curves have not been accepted as one global trajectory')
    for name, label, color in [('common-path-radar-motion', 'Fixed hardware offsets', 'C0'),
                              ('full-response-radar-motion', 'Free local hardware offsets (control)', 'C1')]:
        valid = [r for r in report['groups'][name] if r['validation_version'] == 2]
        if valid:
            axes[1].plot([np.mean(r['interval_s']) for r in valid], [r['error_change_percent'] for r in valid], '.-', label=label, color=color)
    axes[1].axhline(0., color='0.5', lw=.8)
    axes[1].set_ylabel('Held-out error change (%)'); axes[1].legend()
    axes[2].plot(time, ratio)
    axes[2].set_ylabel('Path-rate geometry\nsingular-value ratio'); axes[2].set_xlabel('Capture time (s)')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); fig.savefig(OUTPUT/'full-capture-motion-checks.png', dpi=150); plt.close(fig)
    if groups['linear-reference-radar-motion']:
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        prior = motion_state(track, utc_seconds(model['nominal_start_utc'])+time, model)
        for name, state, colour in [('AIS coarse trajectory', prior, '0.5'),
            ('Video initializer', global_curve.apply(time, prior, tx[3], rx), 'C0')]:
            path, g = excess_path(state['position_ecef_m'], tx[3], rx)
            axes[0].plot(time, path/1000, color=colour, label=name)
            axes[1].plot(time, np.sum(g*state['velocity_ecef_mps'], axis=1), color=colour)
        for i, record in enumerate(groups['linear-reference-radar-motion']):
            t = np.linspace(*record['interval_s'], 181)
            local_model = {**model, **record}
            coarse = motion_state(None, utc_seconds(model['nominal_start_utc'])+t, local_model)
            physical = SplineMotion(local_model).apply(t, coarse, tx[3], rx)
            path, g = excess_path(physical['position_ecef_m'], tx[3], rx)
            axes[0].plot(t, path/1000, label='Radar fit; one coarse anchor' if i == 0 else None, color='C1')
            axes[1].plot(t, np.sum(g*physical['velocity_ecef_mps'], axis=1), color='C1')
        axes[0].set_ylabel('Excess path (km)'); axes[0].legend()
        axes[1].set_ylabel('Physical path rate (m/s)'); axes[1].set_xlabel('Capture time (s)')
        for ax in axes: ax.spines[['top', 'right']].set_visible(False)
        fig.suptitle('Local radar trajectory fits without interpolated AIS motion; validation controls')
        fig.tight_layout(); fig.savefig(OUTPUT/'radar-only-motion-controls.png', dpi=150); plt.close(fig)
    midpoint = json.loads((ROOT/'midpoint-model-audit.json').read_text())
    checks = [point for r in midpoint['results'] for point in r['points']]
    report['midpoint_forward_model'] = {
        'tested_points': len(checks), 'maximum_matched_loss_db': max(p['matched_snr_loss_db'] for p in checks),
        'maximum_constant_phase_error_degrees': max(abs(p['constant_phase_difference_degrees']) for p in checks),
        'scope': 'Forward approximation only; observations unchanged. Original rows used for heldout validation and final video backprojection.'}
    monte_carlo = ROOT/'transverse-motion-efficiency.json'
    report['transverse_efficiency'] = json.loads(monte_carlo.read_text()) if monte_carlo.exists() else None
    point_information = OUTPUT/'aperture-point-information.json'
    report['conditional_aperture_point_information'] = (json.loads(point_information.read_text())
        if point_information.exists() else None)
    starts = sorted(ROOT.glob('linear-reference-yaw-*/window-007.json'))
    zero = ROOT/'linear-reference-radar-motion/window-007.json'
    report['rotation_start_controls'] = []
    if starts and zero.exists():
        baseline = json.loads(zero.read_text())
        for path in [zero, *starts]:
            trial = json.loads(path.read_text())
            for key in ('interval_s', 'selected_points_body_m', 'linear_reference', 'range_bias_m'):
                if trial[key] != baseline[key]:
                    raise ValueError('Rotation starts must fit the same model and observations')
            report['rotation_start_controls'].append({'source': str(path),
                'initial_yaw_rate_deg_ps': trial.get('initial_yaw_rate_deg_ps', 0.),
                'training_objective': trial['fit']['objective'], 'converged': trial['fit']['converged'],
                'heldout_error_change_percent_vs_zero_start': 100*(trial['after_summary']['prediction_error']/baseline['after_summary']['prediction_error']-1)})
    if report['transverse_efficiency'] is not None and report['transverse_efficiency']['cases']:
        fig, axes = plt.subplots(1, len(report['transverse_efficiency']['cases']), figsize=(11, 4), squeeze=False)
        for ax, case in zip(axes[0], report['transverse_efficiency']['cases']):
            mse = np.array(case['mse_over_crlb_by_parameter'])
            ci = np.array(case['variance_ratio_95_percent_intervals'])
            midpoint = ci.mean(axis=1)
            ax.errorbar(np.arange(len(mse)), midpoint, yerr=np.diff(ci, axis=1).ravel()/2, fmt='none', color='C0', label='Variance 95% interval')
            ax.plot(mse, 'o', color='C1', ms=4, label='MSE / CRLB')
            ax.axhline(1., color='0.4', lw=.8)
            ax.set_title(f'{case["snr_db"]:g} dB; {case["trials"]} trials; {len(case["bound_hit_trials"])} bound hits')
            ax.set_xlabel('Physical motion coefficient'); ax.set_ylabel('Normalized error')
            ax.legend(fontsize=8); ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(OUTPUT/'transverse-motion-efficiency.png', dpi=150); plt.close(fig)
    final = CAPTURE/'isar-north-up-30s'
    final_summary = json.loads((final/'videos.json').read_text()) if (final/'videos.json').exists() else {}
    videos = final if final_summary.get('complete') else CAPTURE/'isar-north-up-30s-preview'
    report['videos'] = json.loads((videos/'videos.json').read_text())
    report['videos_complete_capture'] = not report['videos']['partial_preview'] and report['videos']['complete']
    (OUTPUT/'full-capture-motion.json').write_text(json.dumps(report, indent=2)+'\n')
    relative_video = '../../../'+str(videos)
    rows = []
    for name, records in report['groups'].items():
        valid = [r for r in records if r['validation_version'] == 2]
        changes = [r['error_change_percent'] for r in valid]
        bounds = 'pending' if not changes else f'{min(changes):+.3f} to {max(changes):+.3f}%'
        rows.append(f'<tr><td>{name}</td><td>{len(records)}</td><td>{sum(r["converged"] for r in records)}</td><td>{len(valid)}</td><td>{bounds}</td></tr>')
    direct_rows = []
    reference = {tuple(r['interval_s']): r for r in report['groups']['common-path-radar-motion']}
    for name in ('zero-start-radar-motion', 'fixed-reference-radar-motion', 'linear-reference-radar-motion'):
        for r in report['groups'][name]:
            baseline = reference.get(tuple(r['interval_s']))
            if baseline is None or r['validation_version'] != 2:
                continue
            change = 100*(r['after_summary']['prediction_error']/baseline['after_summary']['prediction_error']-1)
            direct_rows.append(f'<tr><td>{name}</td><td>{r["interval_s"][0]:.1f}–{r["interval_s"][1]:.1f}</td><td>{r["training_objective"]:.6f}</td><td>{change:+.3f}%</td><td>{r["converged"]}</td></tr>')
    efficiency = report['transverse_efficiency']
    mc_rows = '' if efficiency is None else ''.join(f'<tr><td>{c["snr_db"]:g}</td><td>{c["trials"]}</td><td>{c["mean_mahalanobis_error_per_parameter"]:.4f}</td><td>{len(c["nonconverged_trials"])}</td><td>{len(c["bound_hit_trials"])}</td></tr>' for c in efficiency['cases'])
    aperture_section = ''
    point = report['conditional_aperture_point_information']
    if point is not None:
        old, new = point['results']
        assumption = 'known_relative_band_calibration'
        old_bound = old['with_stationary_nuisance'][assumption]
        new_bound = new['with_stationary_nuisance'][assumption]
        ratio_bound = new_bound['principal_axis_standard_deviation_m'][0]/old_bound['principal_axis_standard_deviation_m'][0]
        aperture_section = f'''<h2>Better prediction does not automatically mean a sharper image</h2>
<p>For the 143.27–188.27 s local fit, the single-anchor radar model reduces withheld complex prediction error by 29.61% relative to the completed phase-initialized, fixed-hardware fit. Its conditional loss relative to refitting the same point dictionary on the withheld samples is 0.149 dB instead of 3.490 dB. These are model-prediction diagnostics, not a full ship-imaging optimality certificate.</p>
<img src="radar-motion-north-up-comparison.png" alt="Same original-row aperture with the video initializer and single-anchor radar motion">
<p>The 30-second images above use original rows, identical calibration and the same colour scale. Each camera follows its own model's centre. The alternative image has long ridges; it has not demonstrated improved hull focusing. The new motion infers {abs(new['net_heading_change_degrees']):.2f}° of net heading change, compared with {abs(old['net_heading_change_degrees']):.2f}° in the initializer.</p>
<p>An original-row point-template derivative calculation confirms the resulting loss of angular sensitivity. At equal imposed integrated SNR, with relative band calibration and motion treated as known, the weaker principal-axis position bound becomes {ratio_bound:.2f} times larger; the information condition number changes from {old_bound['information_condition_number']:.2f} to {new_bound['information_condition_number']:.2f}. Unknown complex point amplitude and stationary clutter amplitudes are profiled out. <a href="aperture-point-information.json">Results also include independent complex band responses and the unprojected case.</a> This is local point information under each proposed motion, not an empirical motion error bar or a measured image width. Ambiguous distant peaks are not represented by a local CRLB.</p>
<p>Thus apparent sharpness is not enough to choose the physical trajectory. The fitted angular aperture, scattering support and cross-band calibration must also predict held-out observations. No rotation is imposed merely to make the image look more ship-like.</p>'''
    starts_section = ''
    if report['rotation_start_controls']:
        candidates = report['rotation_start_controls']
        best = min(candidates, key=lambda r: r['training_objective'])
        starts_rows = ''.join(f'<tr><td>{r["initial_yaw_rate_deg_ps"]:+.2f}</td><td>{r["training_objective"]:.6f}</td><td>{r["heldout_error_change_percent_vs_zero_start"]:+.3f}%</td><td>{r["converged"]}</td></tr>' for r in sorted(candidates, key=lambda r: r['initial_yaw_rate_deg_ps']))
        starts_section = f'''<h2>Turn-rate initialization check</h2><p>The same 143.27–188.27 s radar likelihood is fitted from different initial yaw rates. These values perturb the starting curve only; they impose no motion penalty, add no AIS dynamics and do not change the observations, reflector support or nuisance model.</p>
<table><tr><th>Initial body yaw rate (°/s)</th><th>Training objective</th><th>Held-out error versus zero start</th><th>Converged</th></tr>{starts_rows}</table>
<p>Of the {len(candidates)} completed starts, the best training likelihood comes from {best['initial_yaw_rate_deg_ps']:+.2f}°/s. Withheld data are reported separately and do not select the start. This tests sensitivity to several plausible local initializations; it does not certify the global optimum of the motion/scattering problem.</p>'''
    status = 'Complete capture, wherever a full 30-second aperture is available' if report['videos_complete_capture'] else 'Two-minute preview; full-capture render continues'
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Full-capture ship motion and videos</title>
<style>body{{font:16px/1.55 system-ui;max-width:1150px;margin:35px auto;padding:0 24px;color:#20242a}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.status{{padding:18px;background:#fff2d1}}video{{width:100%;max-height:850px;background:#111}}code{{background:#eee}}</style>
<h1>Radar-derived ship motion and full-capture north-up videos</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="information-autofocus-audit.html">Stage-by-stage information audit and literature review</a> · <a href="full-capture-motion.json">Numerical results</a></p>
<p class="status">{status}. These are actual complex five-channel sums, with a sliding 30-second aperture, not a claim that phase centres or the hull reconstruction are calibrated. Focusing and calibration uncertainty do not prevent viewing the diagnostic result. No new transmitter or receiver coordinates have been adopted.</p>
<h2>Videos</h2><p><a href="{relative_video}/ship-following-north-up-30s-30fps.mp4">Ship-following, north up</a> · <a href="{relative_video}/ship-geographic-north-up-30s-30fps.mp4">Fixed geographic viewport, north up</a>.</p>
<p>30 fps, H.264 with every frame intra-coded, no B frames and faststart for QuickTime scrubbing. Both views use one fixed colour scale, with its lower limit 1.5 dB below the dominant displayed background and its upper limit above the measured maximum. There is no aperture DC notch. Every pixel in both viewports is backprojected under the fitted rigid ship-motion hypothesis. Static or differently moving objects need their own motion hypothesis to focus.</p>
<p>The displayed coordinate transformation is applied to power after coherent formation. It does not rotate or blend complex images from different times. The original recorded row epochs and complex range profiles are used throughout backprojection. The first and last half-window are excluded rather than shortening the coherent interval.</p>
<h2>What comes from radar, and what still comes from AIS</h2>
<p>The video initializer estimates physical path and yaw corrections from radar. Its transverse coordinate still follows the coarse AIS track. The new two-dimensional model additionally fits horizontal transverse motion; it moves delay, carrier phase and within-symbol Doppler together. Roll and pitch stay fixed. The new fits contain no AIS residual penalty. A separate control fixes only the coarse transverse position at the aperture midpoint while estimating transverse rate and curvature from radar.</p>
<p>AIS has a 392-second reporting gap spanning the final part of this capture. Interpolation across it supplies a starting frame, not a precise phase trajectory. A new linear-reference control uses only one coarse position, velocity and heading anchor: it fits subsequent motion directly from radar, without inheriting any AIS position or heading evolution. Local motion fits are not silently relabelled as a globally fitted trajectory, and a fitted body-frame origin is not necessarily the hull centre.</p>
<p>The fixed calibration observations were originally cropped and deramped around the coarse AIS track. That sets the searched support; it is not a trajectory residual in the objective. The forward model includes exactly the same fixed operator, and the original profiles remain available. Consequently this is radar-driven motion refinement within a coarse search region, not yet AIS-free discovery or a globally fitted capture-long track.</p>
<h2>Recorded-data comparisons</h2><img src="full-capture-motion-checks.png" alt="Motion, withheld prediction and instantaneous geometry checks">
<table><tr><th>Control</th><th>Completed windows</th><th>Numerically converged</th><th>Comparable validation</th><th>Held-out error change</th></tr>{''.join(rows)}</table>
<p>The preceding changes compare each fit to its own starting model. For a direct comparison against the completed fixed-hardware, phase-initialized fit on the same samples:</p>
<table><tr><th>Alternative</th><th>Aperture (s)</th><th>Training objective</th><th>Held-out error versus phase-initialized fit</th><th>Converged</th></tr>{''.join(direct_rows)}</table>
<p>Negative error change means better prediction. Only validation version 2 is plotted: the same stationary-nuisance projection is applied to data and templates before and after fitting. An earlier version compared nuisance prediction before with nuisance projection after; those differences were not comparable and have been superseded. The source results retain that audit trail. Target parameters and complex responses are estimated on training samples and held fixed on withheld samples.</p>
<p>Independent hardware delay offsets in each window are a diagnostic control, not a physical calibration. They disagree between windows. The free absolute-transverse trial also reaches its search boundary. Neither result is adopted. The single-anchor two-dimensional fits improve prediction in some intervals and worsen it in others; they have not established one globally consistent physical trajectory.</p>
<p><a href="full-capture-rate-observations.png">Full-capture measured path-rate spectra</a> show both the moving return and strong stationary contamination inside the range crop. The phase-only initializer does not separately profile stationary nuisance. The full-response controls do. Alternative initializations are compared using training likelihood, with withheld prediction reported separately; the strongest raw spectral peak is not automatically labelled as vessel motion.</p>
{('<img src="radar-only-motion-controls.png" alt="Radar motion from a single coarse anchor">' if groups['linear-reference-radar-motion'] else '')}
{aperture_section}
{starts_section}
<h2>Numerical checks</h2><p>The spline tests now cover transverse position/velocity/range-rate consistency, complete profiled derivatives, legacy model equivalence and fractional-epoch endpoint construction. The latter exposed and removed a redundant endpoint coefficient in certain local fits. Existing interior trajectories are preserved when reading those files.</p>
<p>The faster forward model changes no observations. Across {len(checks)} tested point/aperture/channel combinations, its worst matched-template loss is {report['midpoint_forward_model']['maximum_matched_loss_db']:.3g} dB and its largest constant phase difference is {report['midpoint_forward_model']['maximum_constant_phase_error_degrees']:.4g} degrees. Final predictions and backprojection use original rows. The GPU interpolation test differs from the direct kernel by 0.0000161 degrees on strong cells; the independent double-precision phase check is 0.000618 degrees.</p>
<table><tr><th>Synthetic per-observation SNR (dB)</th><th>Trials</th><th>Mean normalized MSE / CRLB</th><th>Nonconverged</th><th>Bound hits</th></tr>{mc_rows}</table>
{('<img src="transverse-motion-efficiency.png" alt="Two-dimensional physical motion efficiency">' if mc_rows else '<p>The new Monte Carlo is still running.</p>')}
<p>The new five-channel Monte Carlo contains unknown complex reflector responses and stationary nuisance, fits both physical translation coordinates and yaw, and uses no AIS. A coarse midpoint position is fixed explicitly. This is a conditional numerical-efficiency test with known point support and Gaussian noise; it does not certify the real ship model or resolve global phase ambiguities.</p>
<p>Any bound hits are reported, not discarded: those trials do not certify unbiased unconstrained CRLB efficiency, even when normalized MSE is close to one. The 95% intervals shown concern variance under the Gaussian error approximation; they are not simultaneous confidence bands.</p>
<h2>What spatial diversity currently provides</h2>
<p>The working licensed-main hypothesis has a maximum transmitter baseline of {report['geometry']['maximum_transmitter_baseline_m']:.1f} m and channel/site assignments {config['channel_sites']}. In a simple known-position, equal-precision instantaneous path-rate calculation, the horizontal geometry has singular-value ratio {ratio.min():.0f}–{ratio.max():.0f}. Thus it has two-dimensional sensitivity, but much less sensitivity in one horizontal direction. Temporal continuity, angular aperture and shared target structure can add information; this instantaneous diagnostic is not a verdict on the full estimation problem. The alternative licensed/backup hypotheses and antenna uncertainty remain separate calibration questions.</p>
<p>The remaining work is a globally consistent radar motion fit and a scattering/calibration model whose predictions transfer across withheld time and frequency observations. The videos are available while that work continues.</p></html>'''
    (OUTPUT/'full-capture-motion.html').write_text(text)
    print(OUTPUT/'full-capture-motion.html')


if __name__ == '__main__':
    main()
