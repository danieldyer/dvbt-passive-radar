"""Publish convergence and prediction checks for the joint point/phase control."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'check', 'halves', 'baseline', 'test-log', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--signed', type=Path, help='Optional same-support sign-changing scattering control')
    p.add_argument('--ranges', type=Path, help='Optional joint range-origin and phase control')
    p.add_argument('--range-check', type=Path, help='Recreated range likelihood, derivatives and phase-escape checks')
    p.add_argument('--anchors', type=Path, help='Optional partial shared-phase anchor control')
    p.add_argument('--anchor-sensitivity', type=Path, help='Optional leave-one-anchor-out control')
    args = p.parse_args()
    fitted, check, halves, baseline = [json.loads(path.read_text()) for path in
        (args.fit, args.check, args.halves, args.baseline)]
    if not (fitted['complete'] and fitted['fit']['converged'] and check['complete']
            and check['source_converged'] and not check['restart_needed'] and halves['complete']):
        raise ValueError('Complete convergence, phase-escape and aperture-half controls first')
    if check['source_sha256'] != fingerprint(args.fit):
        raise ValueError('Phase/information check refers to a different fit')
    if Path(halves['source_support']).resolve() != args.fit.resolve():
        raise ValueError('Aperture-half control uses different physical points')
    if not all(t['fit']['converged'] for t in halves['trials']):
        raise ValueError('An aperture-half phase solve did not converge')
    log = args.test_log.read_text()
    tests = re.search(r'Ran (\d+) tests', log)
    if not tests or not log.rstrip().endswith('OK'):
        raise ValueError('Numerical regression tests did not pass')
    full, first, second = halves['trials']
    if abs(full['fit']['objective']-fitted['fit']['objective']) > 1e-9:
        raise ValueError('Independent phase audit did not reproduce the joint likelihood')
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    others = [0, 1, 2, 4]
    phase = np.array([t['fit']['channel_phase_radians'] for t in halves['trials']])
    difference = np.degrees(np.angle(np.exp(1j*(phase[2]-phase[1]))))
    old = baseline['trials'][0]['validation']
    rows = []
    for label, variants in [('Fixed support', old), ('Joint coordinates and phases', full['validation']),
                            ('First half', first['validation']), ('Second half', second['validation'])]:
        independent = variants['independent']['summary']['prediction_error']
        coupled = variants['coupled']['summary']['prediction_error']
        rows.append({'label': label, 'independent_error': independent, 'coupled_error': coupled,
                     'coupled_penalty_percent': 100*(coupled/independent-1)})
    information = []
    for row in check['information']:
        fixed, free = [np.asarray(row[key+'_coordinate_information']) for key in ('fixed', 'free')]
        information.append({'cutoff': row['cutoff'],
            'fixed_sigma_degrees': np.degrees(np.sqrt(np.diag(np.linalg.inv(fixed)))).tolist(),
            'free_sigma_degrees': np.degrees(np.sqrt(np.diag(np.linalg.inv(free)))).tolist(),
            'retained_information_fractions': row['retained_information_fractions']})
    old_error = old['coupled']['summary']['prediction_error']
    new_error = full['validation']['coupled']['summary']['prediction_error']
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'source_sha256': {str(path): fingerprint(path) for path in
            (args.fit, args.check, args.halves, args.baseline, args.test_log)},
        'numerical_tests_passed': int(tests[1]), 'fit': {key: fitted['fit'][key] for key in
            ('converged', 'objective', 'maximum_kkt_residual', 'iterations', 'evaluations', 'termination')},
        'coordinate_bound_hits': fitted['coordinate_bound_hits'],
        'prediction_comparisons': rows,
        'coupled_error_change_percent': 100*(new_error/old_error-1),
        'half_phase_difference_degrees': difference.tolist(),
        'half_phase_difference_rms_degrees': float(np.sqrt(np.mean(difference[others]**2))),
        'conditional_information': information,
        'phase_escape_training_improvement': check['relative_training_improvement'],
        'roll_pitch_fitted': False, 'correction_adopted': False}
    args.output.mkdir(parents=True, exist_ok=True)
    range_section, anchor_section = '', ''
    if args.ranges:
        ranges = json.loads(args.ranges.read_text())
        if (not ranges['complete'] or Path(ranges['source_support']).resolve() != args.fit.resolve()
                or len(ranges['trials']) != 3 or not all(t['fit']['converged'] for t in ranges['trials'])):
            raise ValueError('Complete the full and half-aperture range/phase fits on the same support')
        if args.range_check is None:
            raise ValueError('Independently check the range/phase fits before publishing them')
        range_check = json.loads(args.range_check.read_text())
        if (not range_check['complete'] or range_check['source_sha256'] != fingerprint(args.ranges)
                or len(range_check['trials']) != len(ranges['trials'])
                or any(t['restart_needed'] for t in range_check['trials'])
                or [t['interval_s'] for t in range_check['trials']] != [t['interval_s'] for t in ranges['trials']]):
            raise ValueError('Range/phase audit is incomplete, mismatched or requires another fit')
        gradient_error = max(t['maximum_gradient_error'] for t in range_check['trials'])
        if gradient_error > 1e-7:
            raise ValueError('Range/phase derivatives did not pass the numerical check')
        result['source_sha256'][str(args.range_check)] = fingerprint(args.range_check)
        result['range_phase_audit'] = {
            'maximum_directional_gradient_error': gradient_error,
            'maximum_phase_escape_improvement': max(t['phase_escape_improvement'] for t in range_check['trials']),
            'restart_needed': False}
        result['source_sha256'][str(args.ranges)] = fingerprint(args.ranges)
        trials = ranges['trials']
        origins = np.array([t['range_origins_m'] for t in trials])
        fitted_phase = np.array([t['band_phase_radians'] for t in trials])
        phase_difference = np.degrees(np.angle(np.exp(1j*(fitted_phase[2]-fitted_phase[1]))))
        range_rows = []
        for label, t in zip(['Full 30 s', 'First 15 s', 'Second 15 s'], trials):
            before = t['before']['coupled']['summary']['prediction_error']
            after = t['after']['coupled']['summary']['prediction_error']
            independent = t['after']['independent']['summary']['prediction_error']
            range_rows.append({'label': label, 'prediction_error_change_percent': 100*(after/before-1),
                'coupled_penalty_percent': 100*(after/independent-1),
                'kkt_residual': t['fit']['maximum_kkt_residual'], 'range_origins_m': t['range_origins_m']})
        result['range_phase_control'] = {'trials': range_rows,
            'half_phase_difference_degrees': phase_difference.tolist(),
            'half_phase_difference_rms_degrees': float(np.sqrt(np.mean(phase_difference[others]**2)))}
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        for row, label, marker in zip(origins, ['Full 30 s', 'First 15 s', 'Second 15 s'], ['o', 's', '^']):
            axes[0].plot(names, row, marker+'-', label=label)
        axes[0].set(ylabel='Fitted compressed-envelope origin (m)', title='Range origins and band phases fitted together')
        axes[0].legend()
        axes[1].plot([names[c] for c in others], phase_difference[others], 'o-')
        axes[1].axhline(0, color='black', linewidth=.7)
        axes[1].set(ylim=(-185, 185), ylabel='Second-half minus first-half phase (degrees)', title='Constant-phase transfer after range refinement')
        fig.savefig(args.output/'joint-range-phase.png', dpi=160); plt.close(fig)
        table_rows = ''.join(f'<tr><td>{r["label"]}</td><td>{r["kkt_residual"]:.3g}</td>'
            f'<td>{r["prediction_error_change_percent"]:+.2f}%</td><td>{r["coupled_penalty_percent"]:+.2f}%</td></tr>' for r in range_rows)
        range_section = f'''<h2>Joint range-origin and phase refinement</h2>
<p>This control jointly fits all five envelope origins, 64 point phases and four relative band phases. Only the nonnegative band strengths are eliminated at each trial, by exact nonnegative least squares. The range origin changes the compressed envelope; a separate constant phase represents its carrier convention. Together they describe an affine phase correction in range frequency. The physical ship motion, 64 point coordinates and all antenna positions remain fixed. These are effective channel corrections, not separate measurements of propagation delay, clock drift or transmitter coordinates.</p>
<p>The analytic profiled gradient, uncompressed likelihood and recovery of known range offsets and phases pass a synthetic test. The real-data fits retain the declared ±75 m range-origin search bound and the same KKT acceptance threshold. The revised solver stops on that independently checked condition, avoiding extra objective comparisons near geometry round-off.</p>
<p>A separate audit recreates each recorded-data likelihood and checks eight derivative directions per aperture away from stationarity. The largest absolute derivative discrepancy is {gradient_error:.3g}. A new phase search at each final origin finds no improvement above 10⁻⁹ in normalized error. These checks establish local numerical stationarity, not a globally optimal solution. Documented initializations are compared using training error only. An earlier nested phase/range search was stopped because its finite inner phase search introduced discontinuities; its incomplete trials are not accepted results. <a href="../../../{args.range_check}">Independent recorded-data audit</a>.</p>
<img src="joint-range-phase.png" alt="Fitted range origins and channel phase transfer">
<table><tr><th>Aperture</th><th>Maximum KKT residual</th><th>Withheld error change from fixed origins</th><th>Penalty versus independent responses at the same origins</th></tr>{table_rows}</table>
<p>The fitted phase offsets differ by {result['range_phase_control']['half_phase_difference_rms_degrees']:.1f}° RMS between halves. These results do not establish a stable calibration. <a href="../../../{args.ranges}">All fits, derivatives and withheld scores</a>.</p>'''
    if args.anchors:
        anchors = json.loads(args.anchors.read_text())
        if (not anchors['complete'] or Path(anchors['source_support']).resolve() != args.fit.resolve()
                or not all(t['fit']['converged'] for t in anchors['trials'])):
            raise ValueError('Complete the anchor control on the same physical support')
        result['source_sha256'][str(args.anchors)] = fingerprint(args.anchors)
        anchor_rows = []
        for count in sorted(set(t['count'] for t in anchors['trials'])):
            trials = [t for t in anchors['trials'] if t['count'] == count]
            phases = np.array([t['fit']['channel_phase_radians'] for t in trials])
            delta = np.degrees(np.angle(np.exp(1j*(phases[2]-phases[1]))))
            anchor_rows.append({'count': count,
                'full_prediction_penalty_percent': trials[0]['coupled_prediction_penalty_percent'],
                'half_phase_difference_degrees': delta.tolist(),
                'half_phase_difference_rms_degrees': float(np.sqrt(np.mean(delta[others]**2)))})
        result['anchor_control'] = anchor_rows
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        counts = [r['count'] for r in anchor_rows]
        axes[0].plot(counts, [r['full_prediction_penalty_percent'] for r in anchor_rows], 'o-')
        axes[0].set(xlabel='Number of shared anchors', ylabel='Extra withheld squared error (%)', title='Other points remain independent complex responses')
        axes[1].plot(counts, [r['half_phase_difference_rms_degrees'] for r in anchor_rows], 'o-')
        axes[1].set(xlabel='Number of shared anchors', ylabel='Half-to-half phase RMS (degrees)', title='Conditional stability across the declared subsets')
        fig.savefig(args.output/'joint-anchor-control.png', dpi=160); plt.close(fig)
        anchor_section = f'''<h2>Calibrate from a few common points</h2>
<p>For each declared subset size, only the selected points share phase across bands. All other points have independent complex band responses and are eliminated exactly before fitting the anchors. Points are ranked using the weakest-band conditional information on full-aperture training data. The same coordinates and ordering are used in both halves. No point is selected for looking phase-consistent on withheld data.</p>
<img src="joint-anchor-control.png" alt="Prediction and phase stability versus shared anchor count">
<p>The four-anchor result is comparatively stable between halves, but it was noticed among several tested subset sizes. It is an exploratory candidate, not an independently validated selection. The known-truth numerical test recovers common phases in the presence of much stronger independent scattering; that checks the estimator, not this ship model. <a href="../../../{args.anchors}">All anchor results</a>.</p>'''
    if args.anchor_sensitivity:
        sensitivity = json.loads(args.anchor_sensitivity.read_text())
        if (not sensitivity['complete'] or sensitivity['leave_one_out_count'] != 4
                or not all(t['fit']['converged'] for t in sensitivity['trials'])):
            raise ValueError('Complete the four-anchor sensitivity controls')
        result['source_sha256'][str(args.anchor_sensitivity)] = fingerprint(args.anchor_sensitivity)
        sensitivity_rows = []
        for interval in (full['interval_s'], first['interval_s'], second['interval_s']):
            trials = [t for t in sensitivity['trials'] if t['interval_s'] == interval]
            reference = next(t for t in trials if t['omitted_anchor'] is None)['fit']['channel_phase_radians']
            for t in trials:
                if t['omitted_anchor'] is None:
                    continue
                change = np.degrees(np.angle(np.exp(1j*(np.asarray(t['fit']['channel_phase_radians'])-reference))))
                sensitivity_rows.append({'interval_s': interval, 'omitted_anchor': t['omitted_anchor'],
                    'phase_change_degrees': change.tolist(),
                    'phase_change_rms_degrees': float(np.sqrt(np.mean(change[others]**2)))})
        result['four_anchor_sensitivity'] = sensitivity_rows
        fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
        values = np.array([r['phase_change_degrees'] for r in sensitivity_rows])[:, others]
        im = ax.imshow(values, aspect='auto', cmap='RdBu_r', vmin=-180, vmax=180)
        ax.set_xticks(range(4), [names[c] for c in others])
        ax.set_yticks(range(len(sensitivity_rows)),
            [f'{r["interval_s"][0]:g}–{r["interval_s"][1]:g} s; omit point {r["omitted_anchor"]}' for r in sensitivity_rows])
        ax.set_title('Phase change when one of the four anchors is omitted')
        fig.colorbar(im, ax=ax, label='Wrapped phase change (degrees)')
        fig.savefig(args.output/'joint-anchor-sensitivity.png', dpi=160); plt.close(fig)
        anchor_section += f'''<p>Omitting each of the four anchors in turn exposes large changes, up to {np.max(abs(values)):.1f}° in a channel. The four-point result is not robust enough to adopt as a calibration. This is a sensitivity check on the already examined aperture, not a new validation sample.</p>
<img src="joint-anchor-sensitivity.png" alt="Channel phases change substantially when individual anchors are omitted">
<p><a href="../../../{args.anchor_sensitivity}">Leave-one-anchor-out results</a>. No phase branch or anchor subset is chosen to sharpen a displayed image.</p>'''
    signed_section = ''
    if args.signed:
        signed = json.loads(args.signed.read_text())
        if (not signed['complete'] or signed['strengths'] != 'unrestricted real'
                or Path(signed['source_support']).resolve() != args.fit.resolve()
                or not all(t['fit']['converged'] for t in signed['trials'])):
            raise ValueError('Complete the real-strength control on the same physical support')
        signed_phase = np.array([t['fit']['channel_phase_radians'] for t in signed['trials']])
        signed_difference = np.degrees(np.angle(np.exp(2j*(signed_phase[2]-signed_phase[1]))))/2
        signed_error = signed['trials'][0]['validation']['coupled']['summary']['prediction_error']
        result['source_sha256'][str(args.signed)] = fingerprint(args.signed)
        result['signed_strength_control'] = {'prediction_error': signed_error,
            'change_from_nonnegative_percent': 100*(signed_error/new_error-1),
            'penalty_over_independent_percent': 100*(signed_error/rows[1]['independent_error']-1),
            'half_phase_difference_modulo_pi_degrees': signed_difference.tolist(),
            'half_phase_difference_modulo_pi_rms_degrees': float(np.sqrt(np.mean(signed_difference[others]**2))),
            'calibration_pi_ambiguities_resolved': False}
        control = result['signed_strength_control']
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
        axes[0].bar(['Nonnegative strengths', 'Real strengths, either sign'],
            [rows[1]['coupled_penalty_percent'], control['penalty_over_independent_percent']], color=['#d55e00', '#0072b2'])
        axes[0].set(ylabel='Extra withheld squared error (%)', title='Same joint physical point coordinates')
        axes[1].plot([names[c] for c in others], signed_difference[others], 'o-')
        axes[1].axhline(0, color='black', linewidth=.7)
        axes[1].set(ylim=(-95, 95), ylabel='Phase difference modulo 180 degrees', title='Sign-changing model: aperture-half transfer')
        fig.savefig(args.output/'joint-point-signed-control.png', dpi=160); plt.close(fig)
        signed_section = f'''<h2>Sign-changing scattering control</h2>
<p>Two equal unresolved paths at L ± d/2 sum to <code>2 cos(kd/2) exp(−ikL)</code>. The cosine can change sign with frequency or aspect, even when calibration is perfect. An independent two-ray synthesis with 0.8 m excess-path separation demonstrates a sign change within the five recorded carrier frequencies. This is an example, not a measured ship path separation.</p>
<p>The control therefore replaces nonnegative point strengths with unrestricted real strengths, retaining the same locations, motion, shared phases and constant band offsets. Its exact phase-coordinate update and profiled gradient pass independent numerical tests. Adding π to any band phase and negating that band's strengths leaves the predicted data unchanged: the model identifies band phases only modulo π.</p>
<img src="joint-point-signed-control.png" alt="Signed scattering prediction and aperture-half phase check">
<p>Withheld squared error changes by {control['change_from_nonnegative_percent']:+.2f}% versus nonnegative strengths, but remains {control['penalty_over_independent_percent']:.2f}% above independent complex band responses. The two halves disagree by {control['half_phase_difference_modulo_pi_rms_degrees']:.1f}° RMS even modulo π. Sign reversals do not yield a transferable calibration here. No branch is chosen to make the image look sharper, and no correction is adopted. <a href="../../../{args.signed}">Full signed-strength results</a>.</p>'''
    trace = fitted.get('previous_trace', [])+fitted['fit']['trace']
    callback = np.arange(1, len(trace)+1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    axes[0].plot(callback, [t['objective'] for t in trace], '.-')
    axes[0].set(xlabel='Solver callback', ylabel='Training squared error / data energy',
                title='Exact physical coordinate and phase likelihood')
    axes[1].semilogy(callback, [t['kkt_residual'] for t in trace], '.-')
    axes[1].axhline(1e-7, color='black', linestyle='--', label='Acceptance threshold')
    axes[1].set(xlabel='Solver callback', ylabel='Maximum coordinate KKT residual', title='Numerical stationarity')
    axes[1].legend()
    fig.savefig(args.output/'joint-point-convergence.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    x = np.arange(len(rows))
    axes[0].bar(x, [row['coupled_penalty_percent'] for row in rows], color='#d55e00')
    axes[0].set_xticks(x, [row['label'] for row in rows], rotation=20, ha='right')
    axes[0].set(ylabel='Extra withheld squared error (%)', title='Shared phases versus independent band responses')
    axes[1].plot([names[c] for c in others], difference[others], 'o-', color='#0072b2')
    axes[1].axhline(0, color='black', linewidth=.7)
    axes[1].set(ylabel='Second-half minus first-half phase (degrees)', ylim=(-185, 185),
                title='Same physical points; Ten is the reference')
    fig.savefig(args.output/'joint-point-validation.png', dpi=160); plt.close(fig)
    (args.output/'joint-point-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    table = ''.join(f'<tr><td>{r["label"]}</td><td>{r["independent_error"]:.1f}</td>'
        f'<td>{r["coupled_error"]:.1f}</td><td>{r["coupled_penalty_percent"]:+.2f}%</td></tr>' for r in rows)
    precision = information[1]['free_sigma_degrees']
    precision_rows = ''.join(f'<tr><td>{names[c]}</td><td>{precision[j]:.2f}°</td>'
        f'<td>{difference[c]:+.1f}°</td></tr>' for j, c in enumerate(others))
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship ISAR: joint coordinate and phase validation</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccd4d9;text-align:left}}th{{background:#eef2f5}}.note{{background:#fff2cf;padding:16px}}</style>
<h1>Joint reflector coordinates and channel-phase residuals</h1>
<p class="note">The physical fit passes the numerical stationarity check, with maximum KKT residual {result['fit']['maximum_kkt_residual']:.3g}. Allowing coordinates to move changes withheld shared-phase prediction error by {result['coupled_error_change_percent']:+.2f}% relative to the fixed-support control. Calibration transfer is assessed separately below. Roll and pitch remain fixed; no correction from this test has been adopted.</p>
<h2>What is fitted</h2><p>64 reflector locations move horizontally within their original two-metre grid cells. Their geometric delay, carrier phase and Doppler all change together in the original-symbol forward operator. Each point has a common scattering phase across the five bands, and each band has its own nonnegative point strengths and constant phase offset. Ten fixes the common phase convention. Ship motion, reflector heights, antenna locations and range-envelope origins stay fixed.</p>
<p>The nonlinear channel/point phases are solved inside each coordinate trial using a deterministic family of initializations. Linear strengths are profiled by nonnegative least squares. QR compression and a reference-epoch phase reparameterization preserve the likelihood. No fitted image normalization, extra propagation model or fitted roll/pitch is introduced.</p>
<h2>Numerical convergence</h2><img src="joint-point-convergence.png" alt="Objective and KKT residual across solver callbacks">
<p>{len(trace)} callbacks including the retained continuation history; {result['coordinate_bound_hits']} of 128 coordinate bounds active. A callback may contain a line-search trial; the plotted objective is not necessarily a sequence of accepted iterates. The independent fixed-coordinate phase audit reproduces the final objective, and the phase-escape test finds no improvement exceeding 10⁻⁹ in normalized squared error. This establishes first-order constrained stationarity for the tested start, not global optimality.</p>
<p>{result['numerical_tests_passed']} numerical support tests pass. They cover the physical observation operator, exact profiled residual derivatives, inactive nonnegative strengths, nuisance-projected information, nested phase gradients, known-truth recovery and repeatable geometry caching. The implementation retains the residual-dependent term in the variable-projection Jacobian described by <a href="https://www.cs.umd.edu/~oleary/software/varpro/varpro.pdf">O’Leary and Rust, §2.3</a>.</p>
<h2>Prediction and phase transfer</h2><img src="joint-point-validation.png" alt="Withheld prediction penalty and aperture-half phase differences">
<table><tr><th>Control</th><th>Independent squared error</th><th>Shared-phase squared error</th><th>Shared-phase penalty</th></tr>{table}</table>
<p>Odd interior one-second blocks are withheld. Both data and prediction undergo the identical stationary-nuisance projection. Support discovery and the motion seed used this overall aperture, so these are conditional checks of the stated scattering model, not independent discovery validation. Squared prediction error is not target SNR or a pure motion-phase metric.</p>
<p>The two 15 s halves retain the same point coordinates. Their band phases differ by {result['half_phase_difference_rms_degrees']:.1f}° RMS over the four nonreference channels. The ideal-noise information calculation below profiles out all point coordinates and point phases, including coordinates at search bounds. Motion, sites and range origins remain fixed.</p>
<table><tr><th>Channel</th><th>Conditional ideal-noise phase standard deviation</th><th>Measured half-to-half phase difference</th></tr>{precision_rows}</table>
<p>These conditional standard deviations assume the scattering and thermal-covariance models are correct. They are not empirical calibration confidence intervals. Rank cutoffs 10⁻⁸, 10⁻¹⁰ and 10⁻¹² are included in the JSON. Large empirical differences cannot be explained merely by pointing to a small retained-information fraction.</p>
<h2>Interpretation</h2><p>Coordinate errors can bias multiband phase calibration, so jointly fitting them is a necessary control. The multiband method of <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/rsn2.12030">Zhu et al. (2021), §2</a> assumes shared frequency-independent point scattering. That assumption must be checked for this ship; numerical convergence does not establish it. Arbitrary independent complex reflectivities have an exact ambiguity with constant channel gains, so allowing unrestricted scattering and calling the resulting gains calibrated would not resolve the problem.</p>
<p>The existing videos retain their stated conditional motion and phase choices. This report does not establish absolute hull heading, hull centroid or transmitter phase centres.</p>
{signed_section}
{range_section}
{anchor_section}
<h2>Evidence</h2><ul><li><a href="joint-point-validation.json">Summary JSON</a></li>
<li><a href="../../../{args.fit}">Completed joint fit</a> and <a href="../../../{args.check}">phase/information audit</a></li>
<li><a href="../../../{args.halves}">Full and half-aperture phase fits</a></li>
<li><a href="../../../{args.test_log}">Numerical test log</a></li>
<li><a href="scattering-validation.html">Earlier fixed-coordinate analysis</a> and <a href="sea-validation.html">curved-water controls</a></li></ul></html>'''
    (args.output/'joint-point-validation.html').write_text(page)
    print(json.dumps(result['fit']))
    print(json.dumps({k: result[k] for k in ('coupled_error_change_percent', 'half_phase_difference_rms_degrees')}))


if __name__ == '__main__':
    main()
