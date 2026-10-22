"""Publish physical autofocus and channel-phase transfer controls."""
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

from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=Path('build/isar-20261015'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text())
    motion_path = args.work/'physical-profile-64-interior.json'
    model = read(motion_path)
    seed_path = Path(model['identity']['source_motion'])
    seed = read(seed_path)
    fit = model['physical_profile']
    coupling_paths = {key: args.work/f'phase-coupling-{key}-64.json' for key in ('seed', 'physical')}
    coupling = {key: read(path) for key, path in coupling_paths.items()}
    grid_path, pulse_path = args.work/'phase-coupling-grid-control.json', args.work/'pulse-time-variation.json'
    grid, pulse = read(grid_path), read(pulse_path)
    amplitude_path = args.work/'phase-coupling-amplitude-3s-16.json'
    amplitude = read(amplitude_path)
    test_log = Path('build/isar-tests-20261015-physical-profile.log')
    log = test_log.read_text()
    passed = re.search(r'Ran (\d+) tests', log)
    if not passed or not log.rstrip().endswith('OK') or not all(x['complete'] for x in [grid, amplitude, *coupling.values()]):
        raise ValueError('Finish the numerical tests and declared controls first')
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    def link(path, label):
        return f'<a href="{escape(os.path.relpath(path.resolve(), args.output.resolve()))}">{escape(label)}</a>'
    before, after = fit['before'], fit['after']
    error_change = [100*(b['prediction_error']/a['prediction_error']-1) for a, b in zip(before, after)]
    gain = [10*np.log10(b['heldout_matched_energy']/a['heldout_matched_energy']) for a, b in zip(before, after)]
    pooled_error = 100*(fit['after_summary']['prediction_error']/fit['before_summary']['prediction_error']-1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    axes[0].bar(names, error_change, color=['#287544' if d < 0 else '#b33d37' for d in error_change])
    axes[0].axhline(0, color='black', lw=.7)
    axes[0].set(ylabel='Change in withheld prediction error (%)', title='Physical path / yaw / range-origin refit\nNegative is better')
    axes[1].plot(names, [x['prediction_phase_rms_degrees'] for x in before], 'o-', label='Before')
    axes[1].plot(names, [x['prediction_phase_rms_degrees'] for x in after], 'o-', label='After')
    axes[1].set(ylabel='One-second prediction-phase RMS (degrees)', title='Includes scene mismatch, noise and glint\nNot an isolated motion error')
    axes[1].legend()
    fig.savefig(args.output/'physical-profile-validation.png', dpi=160)
    plt.close(fig)
    time = np.linspace(*model['interval_s'], 601)
    original, updated = SplineMotion(seed), SplineMotion(model)
    p0, y0 = original.curves(time)
    p1, y1 = updated.curves(time)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), layout='constrained')
    axes[0].plot(time, p1-p0)
    axes[0].set(xlabel='Capture time (s)', ylabel='Additional path correction (m)', title='Small path changes')
    axes[1].plot(time, -np.degrees(y1-y0))
    axes[1].set(xlabel='Capture time (s)', ylabel='Additional heading correction (degrees)', title='Conditional body-coordinate heading')
    axes[2].bar(names, np.array(model['range_bias_m'])-seed['range_bias_m'])
    axes[2].set(ylabel='Envelope-origin change (m)', title='Envelope changes only\nNo extra carrier-phase rotation')
    fig.savefig(args.output/'physical-profile-corrections.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    phase_summary = []
    for row, (key, results) in enumerate(coupling.items()):
        selected = [t for t in results['trials'] if t['count'] == 64]
        difference = np.degrees(np.angle(np.exp(1j*(np.array(selected[2]['fit']['channel_phase_radians'])-
            selected[1]['fit']['channel_phase_radians']))))
        axes[row, 0].bar(names, difference)
        axes[row, 0].set(ylabel='Second-half minus first-half phase (degrees)', ylim=(-180, 180),
            title=('Earlier motion' if key == 'seed' else 'Rejected physical refit')+' / 64 reflectors')
        axes[row, 0].axhline(0, color='black', lw=.7)
        for span in [model['interval_s'], [142., 157.], [157., 172.]]:
            trials = [t for t in results['trials'] if t['interval_s'] == span]
            differences = [100*(t['validation']['coupled']['summary']['prediction_error']/
                t['validation']['independent']['summary']['prediction_error']-1) for t in trials]
            axes[row, 1].plot([t['count'] for t in trials], differences, 'o-', label=f'{span[0]:g}–{span[1]:g} s')
        axes[row, 1].axhline(0, color='black', lw=.7)
        axes[row, 1].set(xlabel='Shared reflector locations', ylabel='Withheld error change (%)',
            title='Common phases versus independent complex band responses')
        axes[row, 1].legend(fontsize=9)
        phase_summary.append({'motion': key, 'half_phase_difference_degrees': difference.tolist(),
            'full_64_error_change_percent': 100*(selected[0]['validation']['coupled']['summary']['prediction_error']/
                selected[0]['validation']['independent']['summary']['prediction_error']-1)})
    fig.savefig(args.output/'channel-phase-transfer.png', dpi=160)
    plt.close(fig)
    coarse = [t for t in grid['trials'] if t['model'] == 'quantized']
    exact = [t for t in grid['trials'] if t['model'] == 'correct_locations']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    distance = [t['maximum_coordinate_offset_m'] for t in coarse]
    axes[0].plot(distance, [max(abs(np.array(t['phase_error_degrees']))) for t in coarse], 'o-')
    axes[0].set(xlabel='Maximum offset on each horizontal axis (m)', ylabel='Largest channel-phase bias (degrees)',
        title='Known-truth, off-grid control\nOne declared synthetic scene')
    axes[1].plot(distance, [100*t['heldout_error_fraction'] for t in coarse], 'o-')
    axes[1].set(xlabel='Maximum offset on each horizontal axis (m)', ylabel='Withheld error / true signal energy (%)',
        title='Grid mismatch matters even with exact motion')
    axes[2].bar(names, [c['worst_point_loss_old_mean_db'] for c in pulse['channels']])
    axes[2].set(ylabel='Conditional point-template loss (dB)', title='Measured pulse variation\nTime-averaged versus 30-symbol replayed PSF')
    fig.savefig(args.output/'physical-model-controls.png', dpi=160)
    plt.close(fig)
    rows = ''.join(f'<tr><td>{name}</td><td>{d:+.2f}%</td><td>{g:+.3f} dB</td>'
        f'<td>{a["prediction_phase_rms_degrees"]:.2f}° → {b["prediction_phase_rms_degrees"]:.2f}°</td></tr>'
        for name, d, g, a, b in zip(names, error_change, gain, before, after))
    all_fits = [t['fit'] for x in coupling.values() for t in x['trials']]
    amplitude_rows = []
    amplitude_checks = []
    for trial in amplitude['trials']:
        relative = {key: value['summary']['prediction_error']/value['summary']['stationary_only_error']
            for key, value in trial['validation'].items()}
        amplitude_checks.append({'interval_s': trial['interval_s'], 'prediction_error_ratios': relative,
            'converged': trial['fit']['converged']})
        amplitude_rows.append(f'<tr><td>{trial["interval_s"][0]:g}–{trial["interval_s"][1]:g} s</td>'
            f'<td>{relative["independent"]:.2f}</td><td>{relative["coupled"]:.2f}</td></tr>')
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'numerical_tests_passed': int(passed[1]),
        'physical_fit_converged': fit['converged'], 'physical_fit_kkt_residual': fit['maximum_kkt_residual'],
        'physical_fit_heldout_error_change_percent': pooled_error, 'physical_fit_error_change_by_channel_percent': error_change,
        'channel_phase_fits': len(all_fits), 'all_channel_phase_fits_stationary': all(f['converged'] for f in all_fits),
        'maximum_channel_phase_gradient': max(f['maximum_gradient'] for f in all_fits),
        'minimum_channel_phase_curvature': min(min(f['curvature_eigenvalues']) for f in all_fits),
        'phase_transfer': phase_summary, 'roll_pitch_fitted': False, 'correction_adopted': False,
        'three_second_strength_controls': amplitude_checks,
        'worst_pulse_variation_loss_db': max(c['worst_point_loss_old_mean_db'] for c in pulse['channels']),
        'known_truth_exact_maximum_phase_error_degrees': max(max(abs(np.array(t['phase_error_degrees']))) for t in exact),
        'known_truth_one_metre_maximum_phase_error_degrees': max(abs(np.array(coarse[-1]['phase_error_degrees']))),
        'remaining': 'No accepted absolute heading, hull centroid, phase centres or coherent five-band calibration. Numerical convergence is established for these conditional models; predictive transfer is not.'}
    (args.output/'physical-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    methods = [(motion_path, 'Physical fit, derivatives, convergence trace and withheld scores'),
        (seed_path, 'Unchanged earlier motion control'), *[(path, f'Channel-phase tests: {key} motion') for key, path in coupling_paths.items()],
        (grid_path, 'Known-truth grid test'), (pulse_path, 'Pulse-variation measurements'),
        (amplitude_path, 'Three-second positive-strength controls'), (test_log, 'Numerical test log')]
    links = '<ul>'+''.join(f'<li>{link(path, label)}</li>' for path, label in methods)+'</ul>'
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: physical and channel-phase validation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd4d9;padding:8px;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff1cf;padding:16px}}code{{background:#eff2f4;padding:2px 4px}}</style>
<h1>Ship autofocus: physical corrections converge, but do not transfer reliably</h1>
<p>Updated {result['updated_utc']}. Capture <code>live-vhf-20261013-193334</code>; ship aperture 142–172 s.</p>
<p class="note"><strong>Roll and pitch remain fixed.</strong> The new physical path, yaw and five range-origin fit is numerically converged, with positive local curvature, but increases withheld prediction error by {pooled_error:.2f}%. It is rejected. Constant channel-phase corrections also fail stability and prediction checks. No production settings, transmitter coordinates, recorded samples or existing videos were changed.</p>
<p>This follows the <a href="blocked-validation.html">whole-block validation audit</a>. The next tests use both aperture edges for training and withhold only interior one-second blocks. They therefore test interpolation without the previous terminal extrapolation problem. The original motion seed and point-support discovery used this aperture; all results remain conditional on them.</p>
<h2>1. Physical motion and envelope origins</h2>
<p>The fit uses 64 shared physical locations with independent complex responses in each band. Every candidate moves the range envelope, RF phase and within-symbol Doppler together, before the original row averaging and range whitening. Complex reflector amplitudes and stationary nuisance are re-solved at every candidate. The exact variable-projection residual Jacobian includes the amplitude-refitting term, not just a frozen-scene derivative. One anchored path gauge is removed explicitly.</p>
<p>The normalized training error falls from {fit['initial_objective']:.9f} to {fit['objective']:.9f}. The maximum KKT residual is {fit['maximum_kkt_residual']:.3g}; the 22-dimensional feasible tangent Hessian has strictly positive eigenvalues {min(fit['stationarity_polish']['curvature_eigenvalues']):.4f}–{max(fit['stationarity_polish']['curvature_eigenvalues']):.4f} in the declared scaled coordinates. This establishes a conditional local constrained minimum, not global optimality or physical calibration.</p>
<img src="physical-profile-corrections.png" alt="Additional physical path, heading and envelope-origin corrections">
<p>Range-origin shifts affect the compressed envelope only. A free constant band phase is a separate observable; multiplying it by the range-origin shift again would introduce a redundant phase parameter. Nominal antenna locations and zero relative channel path rates remain fixed in this control.</p>
<img src="physical-profile-validation.png" alt="Withheld prediction errors and phase residuals before and after the physical fit">
<table><tr><th>Channel</th><th>Withheld error change</th><th>Matched-energy change</th><th>Prediction-phase RMS</th></tr>{rows}</table>
<p>Prediction-phase RMS measures the phase of withheld data relative to the fitted multi-point template. It includes noise, glint, unresolved scattering and model mismatch; it is not a direct measurement of motion error. The total prediction error rises despite better training fit, so this correction is not carried into imaging.</p>
<h2>2. Are constant transmitter/channel phase corrections sufficient?</h2>
<p>At fixed physical motion and envelope origins, the nested model is <code>a[c,j] = m[c,j] exp(i(φ[j] + θ[c]))</code>, with nonnegative independent magnitudes <code>m</code>, common reflector phases <code>φ</code>, and one constant phase <code>θ</code> per band. Ten fixes the common phase gauge. QR compression preserves the full complex likelihood; nonnegative magnitudes are profiled exactly. Five data-derived starts are compared using training likelihood only.</p>
<p>Both the earlier motion and the rejected physical refit are tested with 16, 32 and 64 reflectors, over the full aperture and each half. All {len(all_fits)} fits reach the gradient gate and positive local curvature. This is a numerically well-behaved test whose real-data assumption does not transfer reliably.</p>
<img src="channel-phase-transfer.png" alt="Phase differences between halves and withheld error penalty for coupled phases">
<p>The 64-reflector common-phase model increases full-aperture withheld error by {phase_summary[0]['full_64_error_change_percent']:.2f}% at the earlier motion and {phase_summary[1]['full_64_error_change_percent']:.2f}% at the physical refit. Half-aperture channel phases differ by tens to more than 100 degrees. These are incompatible with claiming a stable constant channel calibration from this model. Independent band scattering is a control; its free complex coefficients cannot themselves identify arbitrary band phase offsets.</p>
<p>This does not rule out geometric phase-centre errors, which can vary across the ship and over time, or frequency-dependent extended scattering. It rejects the simpler claim that one constant phase per channel adequately explains the present fixed-support scene. The coarse point support is not a surveyed set of physical reflectors.</p>
<h3>Changing brightness is not sufficient in this model</h3>
<p>A separate 16-reflector control permits independent nonnegative strengths in three-second blocks while keeping reflector and band phases constant. A new numerical test verifies that such brightness changes do not secretly add block-dependent phase parameters. All three real-data solves converge, but withheld prediction becomes unstable. The table shows error divided by the stationary-only baseline; a useful prediction must be below one. Neither this positive-strength model nor its unconstrained complex-amplitude control meets that requirement. This failure does not say that the real ship has constant reflectivity; it says this extra freedom is not justified by predictive performance.</p>
<table><tr><th>Aperture</th><th>Independent complex blocks</th><th>Positive strengths, shared phases</th></tr>{''.join(amplitude_rows)}</table>
<h2>3. Numerical and physical-model controls</h2>
<img src="physical-model-controls.png" alt="Known-truth grid phase bias, prediction errors and pulse-variation loss">
<p>A noiseless known-truth scene uses the real epochs, geometry, weights and nuisance operator, with declared shared reflector phases and known band phases. With correct locations, phase recovery is accurate to {result['known_truth_exact_maximum_phase_error_degrees']:.2g} degrees and relative withheld errors are below 10<sup>−27</sup>. Deliberate offsets of up to 1 m on each horizontal axis produce a maximum band-phase bias of {result['known_truth_one_metre_maximum_phase_error_degrees']:.2f}° and {100*coarse[-1]['heldout_error_fraction']:.2f}% prediction error in this particular synthetic scene. This quantifies one discretization effect; it is not a universal bound on all possible ship scattering.</p>
<p>The replayed 30-symbol range pulses vary in time, and archived filter weights vary too. In the conditional point-template check, using the existing average pulse costs at most {result['worst_pulse_variation_loss_db']:.5f} dB over the tested crop. That is too small to motivate a new time-varying pulse stage here. These replayed circular PSFs are not the exact original finite-waveform ambiguity function, so the test is labelled accordingly.</p>
<p>{int(passed[1])} numerical support tests pass. New checks cover exact physical variable-projection residual derivatives, envelope/range/phase/Doppler consistency, known-motion recovery across interior gaps, QR likelihood equivalence, phase-profile gradients, known channel-phase recovery with differing band magnitudes, and rejection of incompatible reflector phases. The existing production GPU forward tests were not changed or rerun for these offline-only changes.</p>
<h2>What follows from this</h2>
<p>Roll and pitch remain fixed. The unresolved issue is identifying a physical scattering/motion model whose corrections transfer across time and reflectors. Releasing new transmitter coordinates or a newly labelled calibrated video from the present fits would overstate the evidence.</p>
<p>The goal remains active. Absolute ship heading and hull centroid are still conditional on the body-shape reference, and the supplied AIS file contains course over ground rather than true heading. Constant body rotation can be counteracted by rotating unknown reflector coordinates; unknown independent band reflectivities likewise absorb constant band phases. Those gauges must remain explicit in any subsequent joint geometry refinement.</p>
<h2>Reproduction</h2>{links}
<p><a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust, Variable Projection for Nonlinear Least Squares Problems</a> provides the separable least-squares framework and residual-Jacobian treatment. The withheld tests and acceptance decisions here are measured implementation controls, not guarantees from that reference.</p>
</html>'''
    (args.output/'physical-validation.html').write_text(page)
    index = args.output/'analysis.html'
    page = index.read_text()
    page = re.sub(r'<p><strong>Latest work:</strong>.*?</p>',
        '<p><strong>Latest work:</strong> <a href="physical-validation.html">Physical range/heading refinement and channel-phase transfer tests</a>. '
        'Roll and pitch remain fixed. The new fits converge numerically but fail withheld prediction or calibration stability; '
        'no correction is adopted. Earlier trial videos remain conditional.</p>', page, count=1)
    index.write_text(page)
    session_path = args.output/'session.json'
    session = read(session_path)
    session.update(updated_utc=result['updated_utc'],
        status='Active. Physical motion/range and channel-phase fits pass numerical checks but fail predictive transfer; no new correction accepted.',
        physical_validation_report=str(args.output/'physical-validation.html'),
        physical_validation_results=str(args.output/'physical-validation.json'),
        physical_profile_trial=str(motion_path), physical_profile_trial_accepted=False,
        channel_phase_trials={key: str(path) for key, path in coupling_paths.items()},
        pulse_variation_control=str(pulse_path), point_grid_control=str(grid_path),
        positive_strength_control=str(amplitude_path), numerical_support_tests=int(passed[1]), pending=[],
        remaining=[
            'Keep roll and pitch fixed. Absolute heading, hull centroid and transmitter calibration remain unvalidated.',
            'The 64-point physical path/yaw/range refit is a positive-curvature local constrained minimum, but worsens withheld error by 4.535%; reject it.',
            'Constant band-phase/shared-reflector-phase fits are unstable across time and support. The three-second positive-strength extension also fails prediction; neither is an accepted calibration.',
            'Known-truth phase recovery passes. In the declared off-grid control, one-metre point offsets bias band phases by at most 4.153 degrees; this does not explain the large real-data half-aperture differences by itself.',
            'Measured average-pulse mismatch is only 0.0213 dB in the conditional point-template check; no dynamic pulse stage is justified by this control.',
            'Next investigate point-dependent geometric phase and extended-scattering mismatch with explicit predictive controls, before accepting further antenna or motion corrections.',
            'The earlier frozen-template dense-phase results and alternating-bin validation remain superseded. Do not reuse their small phase residuals as accuracy evidence.',
            'Existing videos use the earlier conditional motion and gains. The calibrated hull-centred coherent five-band imaging goal remains unfinished.',
            'True AIS heading and antenna-to-hull offsets remain an unanswered optional data request.'])
    session_path.write_text(json.dumps(session, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
