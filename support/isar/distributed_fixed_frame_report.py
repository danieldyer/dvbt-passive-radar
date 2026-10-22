"""Publish fixed-N/E distributed-scene and physical-translation controls."""
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

from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from fit_distributed_motion import aggregate


NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def read(path):
    return json.loads(path.read_text())


def collect(experiment):
    if not experiment['complete'] or [c['channel'] for c in experiment['channels']] != list(range(5)):
        raise ValueError('Finish all five channels before publishing')
    result = []
    for channel in experiment['channels']:
        for trial in channel['trials']:
            tuning = trial.get('tuning', trial)
            result.append({'channel': channel['channel'], 'spacing_m': trial['spacing_m'],
                'points': len(trial['points_enu_m']),
                'rank': tuning['effective_scene_rank'],
                'validation': trial['validation'],
                'matched_gain_db': trial['heldout_matched_energy_gain_db'],
                'prediction_error_change_percent': trial['heldout_prediction_error_change_percent']})
    return result


def compare(before, after):
    gain = None if before['heldout_matched_energy'] == 0 else float(10*np.log10(
        after['heldout_matched_energy']/before['heldout_matched_energy']))
    return {'matched_energy_change_db': gain,
        'prediction_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1),
        'explained_before_after': [before['heldout_explained_fraction'], after['heldout_explained_fraction']]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('build/isar-20261016'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    p.add_argument('--motion', type=Path)
    args = p.parse_args()
    motion_path = args.motion or args.work/'distributed-ne-motion-30s.json'
    paths = [args.work/'distributed-ne-30s/experiment.json',
        args.work/'distributed-ne-cv-30s/experiment.json', motion_path,
        args.work/'distributed-motion-gradient.json']
    gaussian, cv, motion, gradient = [read(path) for path in paths]
    if not motion['complete'] or not motion['fixed_ne_frame'] or np.any(SplineMotion(motion).yaw_coefficients):
        raise ValueError('Require a completed motion result with yaw exactly zero')
    source = read(Path(motion['source_motion']))
    if any(e['source_sha256'] != fingerprint(Path(motion['source_motion'])) for e in (gaussian, cv)):
        raise ValueError('Scene controls used different physical motion')
    if motion['training_sha256'] != fingerprint(paths[1]):
        raise ValueError('Penalty evidence changed after motion fitting')
    log_path = args.work.parent/'isar-20261016-tests-distributed-final.log'
    log = log_path.read_text()
    tests = re.search(r'Ran (\d+) tests', log)
    if tests is None or not log.rstrip().endswith('OK'):
        raise ValueError('Complete the support tests before publishing')
    args.output.mkdir(parents=True, exist_ok=True)
    gaussian_summary, cv_summary = collect(gaussian), collect(cv)
    changes = [compare(b, a) for b, a in zip(motion['before'], motion['after'])]
    sparse_summary = aggregate([c['control']['validation'] for c in cv['channels']])
    sparse_comparison = compare(sparse_summary, motion['after_distributed_summary'])
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(Path(path).resolve(), args.output.resolve()))}">{escape(label)}</a>'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax in axes:
        ax.axhline(0., color='#444', lw=.8)
        ax.set(xticks=np.arange(5), xticklabels=NAMES, ylabel='Withheld explained energy (%)')
        ax.plot(range(5), [100*c['control']['validation']['heldout_explained_fraction']
            for c in cv['channels']], 'o--', color='#333', label='Original eight-point model')
    for spacing, marker in [(20., 'o'), (10., 'x')]:
        axes[0].plot(range(5), [100*t['validation']['heldout_explained_fraction']
            for t in gaussian_summary if t['spacing_m'] == spacing], marker+'-', label=f'Gaussian variance fit, {spacing:g} m')
    for spacing, marker in [(20., 'o'), (10., 'x'), (5., '+')]:
        axes[1].plot(range(5), [100*t['validation']['heldout_explained_fraction']
            for t in cv_summary if t['spacing_m'] == spacing], marker+'-', label=f'Blocked CV ridge, {spacing:g} m')
    axes[0].set_title('Marginal Gaussian scene fit is not grid-stable')
    axes[1].set_title('Training-only temporal selection is grid-stable')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle('Same 30 s aperture and fixed translation; yaw exactly zero\nComplex prediction error, not image SNR')
    fig.savefig(args.output/'distributed-ne-scene-controls.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for c, (channel, null) in enumerate(zip(cv['channels'], motion['null_checks'])):
        trial = next(t for t in channel['trials'] if t['spacing_m'] == motion['spacing_m'])
        tuning = trial['tuning']
        relative = np.array(tuning['strengths'])/tuning['singular_values'][0]**2
        score = np.array(tuning['inner_prediction_errors'])/null['zero_prediction_inner_error']
        axes[0].semilogx(relative, score, label=NAMES[c])
        chosen = int(np.argmin(score))
        axes[0].plot(relative[chosen], score[chosen], 'o')
        trials = sorted(channel['trials'], key=lambda t:t['spacing_m'], reverse=True)
        axes[1].loglog([t['spacing_m'] for t in trials],
            [t['tuning']['effective_scene_rank'] for t in trials], 'o-', label=NAMES[c])
    axes[0].axhline(1., color='#333', ls='--', lw=.8)
    axes[0].set(xlabel='Penalty / largest squared training singular value',
        ylabel='Inner error / zero-prediction error', ylim=(.65, 1.15),
        title='Inner selection (large unstable errors extend above view)')
    axes[1].set(xlabel='Spatial grid interval (m)', ylabel='Effective scene rank', title='Stable complexity under grid refinement')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(args.output/'distributed-ne-regularization.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for values, label, marker in [(motion['before'], 'Before translation refit', 'o--'),
                                   (motion['after'], 'After translation refit', 'x-')]:
        axes[0].plot(NAMES, [100*v['heldout_explained_fraction'] for v in values], marker, label=label)
        axes[1].plot(NAMES, [np.nan if v['phase_rms_degrees'] is None else v['phase_rms_degrees']
            for v in values], marker, label=label)
    axes[0].set(ylabel='Withheld explained energy (%)', title='Fixed scene penalties; original samples')
    axes[1].set(ylabel='One-second prediction phase RMS (degrees)', title='Includes scattering mismatch and noise')
    times = np.linspace(*motion['interval_s'], 601)
    old, new = [SplineMotion(m).components(times) for m in (source, motion)]
    axes[2].plot(times, new[:, 0]-old[:, 0], label='Reference path coordinate')
    if new.shape[1] > 2:
        axes[2].plot(times, new[:, -1]-old[:, -1], label='Transverse coordinate')
    axes[2].set(xlabel='Capture time (s)', ylabel='Correction from previous fit (m)',
        title='Physical translation update; no yaw')
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.savefig(args.output/'distributed-ne-motion.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, layout='constrained')
    for c, ax in enumerate(axes):
        for stage, label, style in [('before', 'Before translation refit', 'o--'), ('after', 'After translation refit', 'x-')]:
            data = motion[stage][c]
            if data['prediction_energy']:
                phase = data['phase_by_one_second']
                ax.plot([v['time_s'] for v in phase], [v['phase_degrees'] for v in phase], style, label=label)
        ax.set(ylabel=NAMES[c]+' phase (°)', ylim=(-180, 180))
        ax.axhline(0., color='#444', lw=.7)
        if not motion['after'][c]['prediction_energy']:
            ax.text(.5, .5, 'Training selected zero prediction: phase undefined', transform=ax.transAxes, ha='center')
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Capture time (s)')
    fig.suptitle('Unrefitted withheld complex phase, at one-second intervals\nNo phase screen, no amplitude normalization, yaw exactly zero')
    fig.savefig(args.output/'distributed-ne-prediction-phase.png', dpi=150)
    plt.close(fig)
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'sources': [{'path': str(p), 'sha256': fingerprint(p)} for p in paths],
        'gaussian_controls': gaussian_summary, 'cross_validated_controls': cv_summary,
        'motion_changes_by_channel': changes, 'motion_fit': motion['fit'],
        'heldout_matched_energy_gain_db': motion['heldout_matched_energy_gain_db'],
        'heldout_prediction_error_change_percent': motion['heldout_prediction_error_change_percent'],
        'before_summary': motion['before_distributed_summary'], 'after_summary': motion['after_distributed_summary'],
        'original_sparse_summary': sparse_summary, 'final_comparison_with_original_sparse_model': sparse_comparison,
        'search_approximation_check': motion['search_approximation_check'], 'gradient_audit': gradient,
        'null_checks': motion['null_checks'], 'support_tests_passed': int(tests[1]), 'test_log': str(log_path),
        'production_changed': False, 'phase_centres_validated': False, 'cross_band_phase_validated': False,
        'yaw_fitted': False, 'literature': [{'title': 'Zhang et al., High-Resolution ISAR Imaging by Exploiting Sparse Apertures (2012)',
            'url': 'https://faculty.utrgv.edu/zhijun.qiao/Qiao-IEEE-TAP-06058607.pdf',
            'scope': 'Joint scene/autofocus motivation only. Our exact bistatic ridge/physical-motion model is different from their sparse Fourier and independent phase-error formulation.'}]}
    grid_html, image_html = '', ''
    audit_path = args.work/'distributed-ne-motion-grid-audit.json'
    if audit_path.exists():
        audit = read(audit_path)
        if not audit['complete'] or audit['motion_sha256'] != fingerprint(motion_path):
            raise ValueError('Complete the final-motion grid audit before publishing')
        result['final_motion_grid_audit'] = audit
        result['sources'].append({'path': str(audit_path), 'sha256': fingerprint(audit_path)})
        measured = [v for v in audit['channels'] if not v.get('zero_prediction_selected')]
        largest = max(abs(v['matched_energy_change_db']) for v in measured)
        discrepancy_db = max(v['prediction_discrepancy']['matched_snr_loss_db'] for v in measured)
        physical = audit['physical_checks']
        acceleration = physical['physical_acceleration_step_checks'][-1]
        grid_html = f'''<p>After refitting motion, the unchanged per-area penalties were repeated on a 10 m grid. The largest per-band matched-energy change is {largest:.5f} dB; the largest discrepancy between the 20 m and 10 m withheld prediction vectors is {discrepancy_db:.5f} dB. The original 20 m scores reproduce to {max(v['original_score_reproduction_relative_error'] for v in measured):.3g} relative error. This is a numerical quadrature check, not an absolute image-resolution claim.</p>
<p>A separate 6,001-epoch physical-motion check confirms exactly zero yaw. The largest total acceleration is {acceleration['maximum_physical_acceleration_mps2']:.6f} m/s². Total bistatic path acceleration reaches {max(acceleration['maximum_absolute_bistatic_path_acceleration_mps2_by_channel']):.6f} m/s²; the optimizer's 0.15 m/s² bound applies to the spline correction coordinate, not the total physical path. The audit retains the two finite-difference step sizes, component limits and full geometry separately. {link(audit_path, 'Grid and physical-motion audit')}</p>'''
    image_path = args.output/'distributed-ne-image-comparison.json'
    if image_path.exists():
        images = read(image_path)
        if images['records'][1]['model_sha256'] != fingerprint(motion_path):
            raise ValueError('Rendered images use different physical motion')
        result['original_row_backprojection'] = images
        result['sources'].append({'path': str(image_path), 'sha256': fingerprint(image_path)})
        image_html = f'''<h2>Original-data backprojection comparison</h2>
<p>Both panels backproject the same complete five-channel original profiles for 29.997248 seconds. The fitted distributed reflectivities do not replace the data. In particular, SBS remains present in both images. Yaw is zero in both; only the physical translation differs. Calibration, range origins, N/E viewport, 2 m scientific image spacing and fixed colour scale are shared. Cyan is the interpolated AIS antenna track. The provisional coherent-sum peak changes by {images['peak_change_db']:+.3f} dB; neither that peak nor its shape validates cross-band calibration.</p>
<img src="distributed-ne-image-comparison.png" alt="Paired original-row fixed-N/E backprojections before and after distributed-scene translation refitting">
<p>The distributed model still predicts less withheld signal than the original eight-point baseline overall. This remains a diagnostic image comparison; it does not replace the existing preview videos or production motion model.</p>'''
    (args.output/'distributed-ne-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = []
    for c, change in enumerate(changes):
        gain = 'undefined (zero prediction)' if change['matched_energy_change_db'] is None else f"{change['matched_energy_change_db']:+.3f} dB"
        rows.append('<tr><td>'+NAMES[c]+'</td>'+''.join(f'<td>{100*v:.3f}%</td>' for v in change['explained_before_after'])+
            f'<td>{gain}</td><td>{change["prediction_error_change_percent"]:+.3f}%</td></tr>')
    max_model_error = max(v['same_coefficient_prediction']['matched_snr_loss_db'] for v in motion['search_approximation_check']
        if not v.get('zero_prediction_selected'))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed N/E distributed-scene motion validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}pre{{padding:16px;background:#f4f6f8;overflow:auto}}.note{{padding:18px;background:#fff0ce}}</style>
<h1>Translation-only autofocus with a distributed fixed-N/E scene</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="distributed-ne-validation.json">Measurements and source identities</a> · <a href="fixed-ne-duration-validation.html">Paired yaw-removal tests</a> · <a href="matrix-covariance-validation.html">Prior covariance controls</a></p>
<p class="note">Yaw remains exactly zero. The distributed-scene translation fit is numerically converged and improves on its own frozen-motion starting point by {motion['heldout_matched_energy_gain_db']:+.3f} dB of withheld matched energy ({motion['heldout_prediction_error_change_percent']:+.3f}% prediction error). It still performs worse than the original eight-point fixed-N/E baseline: {sparse_comparison['matched_energy_change_db']:+.3f} dB matched energy and {sparse_comparison['prediction_error_change_percent']:+.3f}% error. The new model is not adopted. These are conditional prediction statistics, not calibrated image SNR or proof of a unique vessel trajectory.</p>
<h2>Fixed-scene comparison before changing motion</h2>
<p>The same 30-second aperture ({motion['interval_s'][0]:.6f}–{motion['interval_s'][1]:.6f} s) and original complex observations are used throughout. Reflectors cover east −120 to +120 m and north −180 to +200 m at 20 m height relative to the moving prior, with axes fixed in the geographic frame. Cell-centred quadrature at 20, 10 and 5 m contains 228, 912 and 3,648 points. Each cell represents its area, so refinement approximates the same spatially white reflectivity model, rather than increasing prior energy by adding pixels. This rectangle is experimental support, not a measured hull outline.</p>
<p>Every channel has independent complex reflectivity. Constant band phases are therefore absorbed by the scenes: this experiment cannot calibrate inter-band phase or physical antenna locations. All delay, carrier phase and within-symbol Doppler use the same physical trajectory. Stationary nuisance is projected in both training and test subsets. The test prediction is not rescaled, rephased or selected using the withheld observations.</p>
<p>Training and test alternate in 0.5-second blocks. The first control fits one Gaussian scene variance per band by marginal likelihood, including its determinant. Its 20 m fit predicts more of the withheld signal, but activates almost every scene mode; the 10 m fit becomes unstable and ABC's squared error is worse than predicting no moving signal. A better-looking coarse result therefore does not justify adoption.</p>
<img src="distributed-ne-scene-controls.png" alt="All five channels, eight-point and distributed-scene prediction controls at successive spatial resolutions">
<p>The second control selects one fixed Tikhonov penalty using three folds wholly within the outer training blocks. It minimizes complex prediction error in complete inner blocks; it does not use ordinary sample-wise GCV or the outer test scores. Its effective rank and scores converge under grid refinement, but most channels are strongly suppressed at the existing motion estimate. That result motivates the subsequent physical translation fit, rather than treating the initial motion as known.</p>
<img src="distributed-ne-regularization.png" alt="Training-only regularization selection and effective rank under spatial refinement">
<p>The finite penalty sweep omitted the exact infinite-penalty limit. Before motion fitting, this zero-prediction candidate is evaluated on the same inner folds, without reading outer test samples. It wins for SBS. This band supplies no motion information in this specific experiment; its original observations remain available for all later imaging. A null prediction has undefined phase and is not reported as zero phase error.</p>
<h2>Joint translation and distributed reflectivity</h2>
<p>The penalty for each band is fixed before the motion search. At every trial, its complex scene coefficients are solved exactly by SVD. The minimized objective is:</p>
<pre>F(p) = Σc min_ac [ ||Pc (yc − Ac(p) Dc ac)||² + λc ||ac||² ]
Dc = diagonal(square root of cell area)
Pc = training stationary-nuisance projector
p  = the same physical translation spline in delay, carrier and Doppler</pre>
<p>The full augmented-residual derivative includes the scene-coefficient refit. It is evaluated by streaming the original-row derivatives one point at a time, avoiding a points × samples × motion-parameters array. The representative-epoch operator accelerates search only; original-row values and gradients govern final feasibility and stationarity. Yaw, roll, pitch, range origins, antenna locations and channel gains remain fixed. The original motion and support choices already used this aperture, so the outer blocks provide conditional transfer evidence rather than independent target discovery.</p>
<table><tr><th>Channel</th><th>Explained before</th><th>Explained after</th><th>Matched-energy change</th><th>Error change</th></tr>{''.join(rows)}</table>
<p>Explained energy is <code>1 − ||y − prediction||² / ||y||²</code>. Matched energy is <code>|predictionᴴ y|² / ||prediction||²</code>; its magnitude alone does not penalize every phase or amplitude error, which is why the unrefitted prediction error is reported alongside it. The combined statistic concatenates the five band predictions and observations; it does not assume a common reflectivity or establish coherent spatial calibration.</p>
<img src="distributed-ne-motion.png" alt="Withheld prediction scores and physical translation updates with yaw fixed at zero">
<img src="distributed-ne-prediction-phase.png" alt="Withheld one-second phase errors, showing undefined SBS phase rather than a false zero">
{image_html}
<h2>Numerical evidence and limits</h2>
<p>The support suite passes {int(tests[1])} tests. New checks compare the regularized solution with independent augmented least squares, differentiate both the observation and penalty residuals, compare streamed derivatives against the existing original-row point operator, test the zero-prediction limit, and verify area-weighted covariance convergence. On the actual 228-point Ten field, an independent directional finite difference agrees with the normalized analytic gradient to {min(t['absolute_error'] for t in gradient['checks']):.3g}; the larger-step error falls quadratically before rounding dominates.</p>
<p>The final original-row constrained gradient residual is {motion['fit']['maximum_kkt_residual']:.3g}, with feasibility error {motion['fit']['maximum_constraint_violation']:.3g}. The largest same-coefficient composite-template mismatch from the accelerated search operator is {max_model_error:.3g} dB. These checks concern numerical correctness. Regularization deliberately introduces bias, the conditional thermal covariance omits unexplained interference, and independent scenes cannot certify cross-band phase. The optimizer curvature is not asserted to be the recording's CRLB.</p>
{grid_html}
<p>There is a precise conditional optimum for the linear step: if the declared scene really is proper-complex Gaussian with covariance <code>λ⁻¹I</code>, the motion is known, and the projected noise has unit covariance, then this ridge solution is its posterior mean and minimizes mean squared scene error. Its posterior covariance is <code>(AᴴA + λI)⁻¹</code> in the scaled scene coordinates. The existing 6,000-trial Gaussian-scene test checks that identity. Choosing λ by validation on this recording does not establish those assumptions, eliminate motion bias, or turn that conditional MMSE statement into an empirical CRLB certificate.</p>
<p>Joint estimation of image reflectivity and autofocus parameters is motivated by <a href="https://faculty.utrgv.edu/zhijun.qiao/Qiao-IEEE-TAP-06058607.pdf">Zhang et al., High-Resolution ISAR Imaging by Exploiting Sparse Apertures</a>. That work uses a sparse Fourier image and phase-error model. Here the scene is distributed, the penalty is chosen by blocked prediction, and only a physical translation can change phase. Its numerical or imaging gains do not transfer automatically to this bistatic recording.</p>
<p>{link(paths[0], 'Gaussian controls')} · {link(paths[1], 'Nested block controls')} · {link(motion_path, 'Original-row motion result')} · {link(paths[3], 'Real-data gradient audit')} · {link(log_path, 'Test log')}</p>
<p>No production data, antenna coordinates or phase calibration have been changed by this experiment. The broader goal of independently validated full-capture coherent ISAR remains open.</p></html>'''
    (args.output/'distributed-ne-validation.html').write_text(page)
    print(json.dumps({'report': str(args.output/'distributed-ne-validation.html'),
        'gain_db': result['heldout_matched_energy_gain_db'], 'error_change_percent': result['heldout_prediction_error_change_percent'],
        'tests': int(tests[1]), 'converged': motion['fit']['converged']}))


if __name__ == '__main__':
    main()
