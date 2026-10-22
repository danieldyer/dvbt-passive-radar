"""Publish the radar motion acquisition controls and geographic video audit."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds
from geographic_movie import state_at
from imaging import sites
from spline_motion import SplineMotion


ROOT = Path('build/isar-20261015')
OUTPUT = Path('docs/data/isar_20261015')
CAPTURE = '../../../captures/live-vhf-20261013-193334/'


def heading(state, frame):
    forward = state['body_to_ecef'][:, :, 0]@frame
    derivative = state['frame_velocity'][:, :, 0]@frame
    angle = np.degrees(np.unwrap(np.arctan2(forward[:, 0], forward[:, 1])))+360
    rate = np.degrees((forward[:, 1]*derivative[:, 0]-forward[:, 0]*derivative[:, 1])/
                      np.sum(forward[:, :2]**2, axis=1))
    return angle, rate


def main():
    description = json.loads((ROOT/'full-north-up-movie/movie.json').read_text())
    original = json.loads(Path(description['motion']).read_text())
    track = read_csv('docs/data/isar_20261013/rio-delta-ais.csv',
        reference_height_ellipsoid_m=original['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(Path(original['geometry_directory']))
    frame = np.array(description['camera']['enu_to_ecef'])
    epoch = utc_seconds(original['nominal_start_utc'])
    time = np.linspace(*original['interval_s'], 6001)
    angles, rates = heading(state_at(track, original, SplineMotion(original), tx, rx, time), frame)
    velocity = track.sample(epoch+time, max_gap_s=400.)['velocity_ecef_mps']@frame
    course = np.degrees(np.unwrap(np.arctan2(velocity[:, 0], velocity[:, 1])))+360
    seed = json.loads((ROOT/'full-capture-radar-motion.json').read_text())
    records, curves = [], []
    for path in sorted((ROOT/'radar-rate-initialized-motion').glob('window-???.json')):
        old = json.loads((ROOT/'linear-reference-radar-motion'/path.name).read_text())
        polished = ROOT/'radar-rate-refined-motion'/path.name
        new = json.loads((polished if polished.exists() else path).read_text())
        selected = new['fit']['objective'] < old['fit']['objective']
        model = {**seed, **new}
        model.pop('path_reference_time_s', None)
        t = np.linspace(*new['interval_s'], 451)
        h, r = heading(state_at(None, model, SplineMotion(model), tx, rx, t), frame)
        curves.append((t, h, r))
        records.append({'window': int(path.stem[-3:]), 'interval_s': new['interval_s'],
            'new_training_objective': new['fit']['objective'], 'old_training_objective': old['fit']['objective'],
            'radar_start_selected_by_training': selected, 'converged': new['fit']['converged'],
            'maximum_kkt_residual': new['fit']['maximum_kkt_residual'],
            'heldout_error_change_percent': 100*(new['after_summary']['prediction_error']/old['after_summary']['prediction_error']-1),
            'conditional_dictionary_loss_db': new['after_summary']['conditional_coherent_loss_db'],
            'net_fitted_heading_change_deg': float(h[-1]-h[0]),
            'maximum_absolute_heading_rate_deg_ps': float(abs(r).max())})
    audit = {'full_video_motion': description['motion'],
        'full_video_heading_range_deg': [float(angles.min()), float(angles.max())],
        'full_video_heading_rate_range_deg_ps': [float(rates.min()), float(rates.max())],
        'full_video_integrated_absolute_heading_change_deg': float(np.trapezoid(abs(rates), time)),
        'full_video_net_heading_change_deg': float(angles[-1]-angles[0]),
        'turn_direction_reversals_above_003_deg_ps': int(np.sum(np.diff(np.sign(rates[abs(rates)>.03])) != 0)),
        'interpretation': 'The earlier fitted heading has implausible oscillations. A fixed north-up camera does not remove errors in the imaging trajectory. AIS course is not measured true heading.',
        'local_fits': records}
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time, angles, label='Earlier video: fitted body heading', color='C1')
    axes[0].plot(time, course, label='Interpolated AIS course (not heading)', color='C0')
    known = (track.seconds >= epoch+time[0])&(track.seconds <= epoch+time[-1])&np.isfinite(track.course)
    axes[0].scatter(track.seconds[known]-epoch, np.degrees(track.course[known]), s=14, color='C0', label='Reported AIS course')
    axes[1].plot(time, rates, color='C1', label='Earlier fitted body turn rate')
    for i in (6, 7):
        t, h, r = curves[i]
        axes[0].plot(t, h, color='C2', lw=2, label='New physical radar fit (two strong intervals)' if i == 6 else None)
        axes[1].plot(t, r, color='C2', lw=2, label='New physical radar fit' if i == 6 else None)
    axes[0].set_ylabel('Compass degrees'); axes[1].set_ylabel('Turn rate, degrees/s')
    centres = np.array([np.mean(v['interval_s']) for v in records])
    improvement = np.array([v['heldout_error_change_percent'] for v in records])
    selected = np.array([v['radar_start_selected_by_training'] for v in records])
    axes[2].scatter(centres[~selected], improvement[~selected], marker='x', color='grey', label='Earlier zero start wins training')
    axes[2].scatter(centres[selected], improvement[selected], color='C2', label='New radar start wins training')
    axes[2].axhline(0, color='black', lw=.7)
    axes[2].set_ylabel('Held-out error change, %'); axes[2].set_xlabel('Capture time, seconds')
    for ax in axes:
        ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle('Motion audit: improved local prediction does not certify a capture-long physical heading')
    fig.tight_layout(); fig.savefig(OUTPUT/'radar-acquisition-heading.png', dpi=160); plt.close(fig)
    efficiency = json.loads((ROOT/'radar-rate-acquisition-efficiency.json').read_text())
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mc = []
    for case in efficiency['cases']:
        method = case['methods']['radar_rate_start']
        ax.plot(np.arange(1, 7), method['mse_over_crlb_by_parameter'], 'o-', label=f"{case['snr_db']:g} dB, {method['trials']} trials")
        mc.append({'snr_db': case['snr_db'], 'trials': method['trials'],
            'joint_normalized_mse': method['mean_mahalanobis_error_per_parameter'],
            'failed_fits': len(method['failures']), 'bound_hits': len(method['bound_hits']),
            'zero_start_joint_normalized_mse': case['methods']['zero_start']['mean_mahalanobis_error_per_parameter'],
            'zero_start_trials': case['methods']['zero_start']['trials']})
    ax.axhline(1, color='black', ls='--', label='Conditional CRLB')
    ax.set(xlabel='Physical path spline coefficient', ylabel='Mean squared error / CRLB', ylim=(.8, 1.2),
        title='Large-error acquisition followed by physical likelihood refinement; known point support')
    ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT/'radar-acquisition-efficiency.png', dpi=160); plt.close(fig)
    gains = json.loads((ROOT/'radar-motion-complex-gains.json').read_text())
    audit['conditional_monte_carlo'] = mc
    audit['constant_complex_gain_test'] = {
        'converged': gains['complex_gain_fit']['converged'],
        'error_change_percent': 100*(gains['trials'][2]['test_error']/gains['trials'][0]['test_error']-1),
        'penalty_vs_independent_percent': 100*(gains['trials'][2]['test_error']/gains['independent']['prediction_error']-1),
        'correction_adopted': False}
    (OUTPUT/'radar-acquisition.json').write_text(json.dumps(audit, indent=2)+'\n')
    rows = ''.join(f"<tr><td>{v['window']}</td><td>{v['interval_s'][0]:.1f}–{v['interval_s'][1]:.1f}</td><td>{v['radar_start_selected_by_training']}</td><td>{v['heldout_error_change_percent']:+.3f}%</td><td>{v['converged']}</td></tr>" for v in records)
    links = []
    for directory, label in [('isar-north-up-30s-ais', 'Full capture, earlier motion, with AIS'),
                             ('isar-radar-motion-comparison-ais', 'New radar motion, 158.33–173.20 s, with AIS')]:
        for name in ('following', 'geographic'):
            links.append(f'<li>{label}: <a href="{CAPTURE}{directory}/ship-{name}-north-up-30s-30fps.mp4">{name} view</a></li>')
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Radar motion acquisition and video audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1180px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Radar-derived motion, heading audit and AIS video overlays</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="radar-acquisition.json">Measurements</a> · <a href="information-autofocus-audit.html">Earlier stage-by-stage numerical and information audit</a></p>
<p class="note">The earlier video motion has implausible heading oscillations, including turn rates from {rates.min():.3f} to {rates.max():.3f}°/s and {audit['turn_direction_reversals_above_003_deg_ps']} direction reversals above 0.03°/s. This is a motion-model issue; the geographic camera is fixed north-up. The full-length overlay videos retain that earlier motion explicitly. The new short render uses the improved physical fit, but has not demonstrated a resolved hull or validated cross-band calibration.</p>
<h2>Videos and geographic reference</h2><ul>{''.join(links)}</ul>
<p>All videos use a full sliding 30-second aperture, 30 fps, the same fixed 0.875–48.8 dB scale, and all-intra H.264 for scrubbing. Cyan shows the interpolated AIS antenna track; squares are actual reports and the moving circle is the current interpolated position. Gaps longer than 60 seconds are dashed, including the 392-second gap. This overlay is applied after image formation. It cannot steer motion fitting, rotate with the fitted body, or change the radar pixels underneath it. The timestamps retain the capture's nominal UTC reference.</p>
<p><img src="{CAPTURE}isar-radar-motion-comparison-ais/following-frame-00223.png" alt="New motion comparison with an independent cyan AIS reference"></p>
<p>The new image is elongated rather than a verified sharp hull. Better withheld prediction is useful evidence about the motion/scattering model, but it is not interchangeable with improved spatial resolution.</p>
<h2>Radar initialization and recorded-data checks</h2>
<p>The new initializer uses adjacent complex phase products after projecting the stationary response space from training observations. It accounts for the different RF wavenumbers and physical path sensitivities. A bounded smooth path-rate estimate supplies an initial physical curve; it does not modify the data. The subsequent full likelihood moves delay, carrier phase and symbol Doppler together and profiles the complex scene coefficients. Roll and pitch remain fixed.</p>
<p>AIS supplies one coarse pose and velocity per local interval. Its interpolated evolution does not enter these new likelihood fits. The original observation crop still uses the coarse AIS search region. This is radar motion refinement within that region, not yet AIS-free discovery or a globally estimated capture-long track.</p>
<p>Of 29 intervals, the new start wins training likelihood in {int(selected.sum())}; {sum(v['radar_start_selected_by_training'] and v['heldout_error_change_percent'] < 0 for v in records)} of those also improve held-out prediction. It is not a universal replacement for the zero-motion start. The strongest changes are −47.558% and −5.738% in intervals 6 and 7; both now pass the numerical stationarity check. Early intervals 0 and 1 have lower training error but worse prediction, so that mismatch remains visible.</p>
<img src="radar-acquisition-heading.png" alt="Full heading and turn-rate audit with independent prediction comparisons">
<p>AIS course over ground is shown as a reference, not true heading: the supplied file has no heading or rate-of-turn reports. The new strong-interval body-heading changes are {records[6]['net_fitted_heading_change_deg']:.3f}° and {records[7]['net_fitted_heading_change_deg']:.3f}° over 45 seconds. Local fits have not been joined and relabelled as a globally optimized trajectory.</p>
<details><summary>Every recorded-data interval</summary><table><tr><th>Window</th><th>Capture seconds</th><th>New start wins training</th><th>Held-out error change</th><th>Numerically converged</th></tr>{rows}</table></details>
<h2>Estimator efficiency under a declared model</h2>
<img src="radar-acquisition-efficiency.png" alt="Path estimator normalized mean squared error">
<p>Five channels observe a known point with unknown complex band amplitudes, strong stationary clutter and proper complex Gaussian noise. Physical path motion is unknown and begins with a large phase error. With 500 trials at each of 10 and 20 dB, the joint normalized MSE is {mc[0]['joint_normalized_mse']:.4f} and {mc[1]['joint_normalized_mse']:.4f}; no initialized trial fails or reaches a bound. Expected Fisher information comes from noiseless truth after nuisance elimination. The 30 zero-start controls at each SNR mostly reach incorrect local minima. No trial is discarded. Point support, orientation and antenna positions are known here: this does not certify real ship motion or joint geolocation.</p>
<p>The fixed-orientation audit exposed redundant zero rows in the optimizer constraints after fixed coordinates were eliminated. Those constraints are now removed, with a regression. The existing all-free real-data controls were unaffected. Analytic gain gradients, ill-conditioned scene solves and geographic overlay registration have separate tests.</p>
<h2>Coherent calibration and the requested joint location fit</h2>
<p>At the improved motion, constant complex gain adjustment lowers withheld error by {abs(audit['constant_complex_gain_test']['error_change_percent']):.3f}% relative to the original shared-scene calibration. Its error remains {audit['constant_complex_gain_test']['penalty_vs_independent_percent']:.2f}% higher than independent band responses. Constant phase adjustment alone is worse. These corrections are not adopted in the comparison video; calibration is held fixed to isolate the motion change.</p>
<p>The earlier <a href="geometry-refinement.html">joint motion/TX/RX fit</a> adds only 0.076 dB of held-out matched energy, reaches location bounds and leaves 14 of 18 location combinations weaker than their prior after unknown motion and point coordinates are eliminated. It is a completed conditional fit, not accepted antenna calibration. The <a href="transmitter-phase-controls.html">48 licensed/backup assignment screen</a> used fixed motion. A full joint trajectory/location re-optimization under every assignment is still outstanding.</p>
<p>Nothing here proves that transmitter diversity plus motion continuity cannot locate the ship. It also does not establish a unique geographic solution. A capture-long joint likelihood, shared antenna positions, explicit coordinate gauges, competing geographic starts and held-out predictions are the next required checks. The current video position remains coarsely anchored by AIS.</p>
<h2>Why acquisition is separate from phase-accurate refinement</h2>
<p><a href="https://link.springer.com/article/10.1155/ASP/2006/90716">Yau, Berry and Haywood (2006)</a> describe a multimodal motion-compensation objective and staged initialization before parametric refinement. Their particular range-alignment/subspace procedure is not implemented verbatim here. It supports testing acquisition separately from local estimator efficiency.</p>
<p><a href="https://www.sandia.gov/research/publications/details/the-phase-gradient-autofocus-algorithm-an-optimal-estimator-of-the-phase-de-1989-09-01/">Eichel's Sandia PGA report</a> gives phase-gradient optimality within its signal assumptions. The adjacent-product initializer here is only a starting estimate; the subsequent physical likelihood carries the efficiency claim under the controlled model.</p>
<p><a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith, Passive ISAR Part I</a> separates coarse motion compensation, phase refinement and image formation, and discusses narrowband centroid/glint limitations. The present multi-transmitter extended-ship fit needs its own transfer tests; these references do not certify it.</p></html>'''
    (OUTPUT/'radar-acquisition.html').write_text(text)
    print(OUTPUT/'radar-acquisition.html')


if __name__ == '__main__':
    main()
