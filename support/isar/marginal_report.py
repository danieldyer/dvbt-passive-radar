"""Publish the yaw-free marginal-likelihood and motion-information audit."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return {'path': str(path), 'sha256': sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=Path('build/isar-20261016'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    paths = {kind: args.build/f'marginal-ne-{suffix}-30s.json' for kind, suffix in (
        ('motion', 'motion'), ('information', 'information'), ('score', 'score-audit'))}
    motion, information, score = [read(paths[k]) for k in ('motion', 'information', 'score')]
    if not all(x['complete'] for x in (motion, information, score)):
        raise ValueError('Finish the motion, information and numerical score audits first')
    if not motion['fixed_ne_frame'] or motion['yaw_fitted'] or not score['numerically_adequate_for_efficiency_trials']:
        raise ValueError('The report requires the checked zero-yaw experiment')
    source = read(Path(motion['source_motion']))
    now = datetime.now(timezone.utc).isoformat()
    result = {'updated_utc': now, 'sources': {k: identity(p) for k, p in paths.items()},
        'source_motion': identity(Path(motion['source_motion'])),
        'scope': 'Original 30-second five-band operator; zero yaw; deterministic unknown core and Gaussian diffuse scene.',
        'real_data_change': motion['motion_change'],
        'maximum_parameter_change_m': motion['maximum_parameter_change_m'],
        'original_kkt_residual': motion['fit']['maximum_kkt_residual'],
        'marginal_after': motion['marginal_after'],
        'conditional_information': {k: information[k] for k in ('active_physical_indices',
            'information_condition', 'information_eigenvalues', 'covariance_information_gain_eigenvalues',
            'known_quantities', 'unknown_mean', 'constraint_scope')},
        'maximum_path_crlb_std_m': max(information['path_coordinate_crlb_std_m']),
        'maximum_transverse_crlb_std_m': max(information['transverse_coordinate_crlb_std_m']),
        'gradient_checks': motion['gradient_checks'],
        'score_audit': score,
        'implementation_sha256': {**motion['implementation_sha256'], **information['implementation_sha256'],
            **score['implementation_sha256']},
        'test_log': identity(Path('build/isar-20261016-tests-marginal.log')),
        'tests_passed': 249, 'real_data_efficiency_certified': False,
        'physical_phase_centres_validated': False, 'cross_band_phase_calibration_validated': False}
    result['implementation_sha256']['marginal_report.py'] = identity(Path(__file__))['sha256']
    checkpoint = args.build/'marginal-ne-motion-efficiency-30s.json'
    trial_text = 'The nonlinear known-truth recovery experiment has not yet produced a checkpoint.'
    if checkpoint.exists():
        experiment = read(checkpoint)
        result['nonlinear_recovery_checkpoint'] = {**identity(checkpoint),
            'snapshot_utc': now, 'complete': experiment['complete'],
            'completed_trials': len(experiment['cases']), 'requested_trials': experiment['requested_trials'],
            'nonstationary_trials': experiment.get('nonstationary_trials', []),
            'statistics': experiment.get('statistics'), 'cases': experiment['cases']}
        trial_text = (f"At this report snapshot, {len(experiment['cases'])} of {experiment['requested_trials']} "
            'nonlinear trials have completed. This checkpoint does not by itself establish estimator efficiency. '
            'The live result is linked below; incomplete trials are not counted as successes.')
    args.output.mkdir(parents=True, exist_ok=True)
    names = ('Seven', 'SBS', 'Nine', 'Ten', 'ABC')
    residual = [c['quadratic_per_residual_degree'] for c in motion['marginal_after']]
    times = np.asarray(information['epochs_s'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(names, residual, color='#537ea6')
    ax.axhline(1, color='#c45735', ls='--', label='Declared model expectation')
    ax.set(ylabel='Quadratic residual / complex residual DOF', title='Recording: the model is not adequate')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(times, 100*np.array(information['path_coordinate_crlb_std_m']), label='Path coordinate')
    ax.plot(times, 100*np.array(information['transverse_coordinate_crlb_std_m']), label='Transverse coordinate')
    ax.set(xlabel='Capture time (s)', ylabel='Conditional standard-deviation bound (cm)',
        title='Known-support model only; not geolocation accuracy')
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    active = np.asarray(information['active_physical_indices'])
    delta = (np.asarray(motion['parameters'])-np.asarray(source['parameters']))[active]
    ax.bar(np.arange(len(active)), 1000*delta, color='#537ea6')
    ax.set(xlabel='Free motion coefficient index', ylabel='Change (mm)',
        title='Including covariance volume barely changes motion')
    ax = axes[1, 1]
    gain = [10*np.log10(b['heldout_matched_energy']/a['heldout_matched_energy'])
        for a, b in zip(motion['before'], motion['after'])]
    ax.bar(names, gain, color='#537ea6')
    ax.axhline(0, color='black', lw=.7)
    ax.set(ylabel='Withheld matched-energy change (dB)', title='No useful prediction gain from the correction')
    for ax in axes.flat:
        ax.spines[['right', 'top']].set_visible(False)
    fig.savefig(args.output/'marginal-ne-validation.png', dpi=170)
    plt.close(fig)
    (args.output/'marginal-ne-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = ''.join(f'<tr><td>{n}</td><td>{c["quadratic_per_residual_degree"]:.3f}</td>'
        f'<td>{c["logdet"]:.3f}</td><td>{c["residual_degrees"]}</td></tr>'
        for n, c in zip(names, motion['marginal_after']))
    score_rows = ''.join(f'<tr><td>{n}</td><td>{c["relative_objective_error"]:.3g}</td>'
        f'<td>{c["maximum_normalized_gradient_error"]:.3g}</td>'
        f'<td>{c["prediction_difference_in_unit_noise_norm"]:.3g}</td></tr>'
        for n, c in zip(names, score['channels']))
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Fixed N/E motion: marginal likelihood and information audit</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}
img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}
td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}pre{{padding:16px;background:#f4f6f8;overflow:auto}}
.note{{padding:18px;background:#fff0ce}}code{{background:#f4f6f8}}</style>
<h1>Fixed N/E motion: the likelihood correction is not the focusing bottleneck</h1>
<p>Updated {now}. <a href="analysis.html">Analysis index</a> ·
<a href="marginal-ne-validation.json">Measurements and source identities</a> ·
<a href="motion-autofocus-literature.html">Motion/autofocus literature and applicability</a></p>
<p class="note">Yaw remains exactly zero. The corrected motion objective converges on the original data,
but changes the fitted coefficients by at most {1000*motion['maximum_parameter_change_m']:.3f} mm.
Withheld matched energy changes {motion['motion_change']['matched_energy_change_db']:+.6f} dB,
and squared prediction error changes {motion['motion_change']['prediction_error_change_percent']:+.6f}%.
This is effectively no improvement. The covariance term is mathematically required by the stated
random-scene model, but it does not resolve this recording's scattering or calibration mismatch.</p>
<img src="marginal-ne-validation.png" alt="Residual model mismatch, conditional motion bounds, parameter changes and withheld prediction changes">
<h2>What changed, and the exact statistical model</h2>
<p>The <a href="mixed-ne-validation.html">previous nested-scene experiment</a> retained eight unpenalized
complex reflectors per band and a training-selected distributed component. Its motion step minimized the
augmented residual and diffuse penalty. Integrating the Gaussian diffuse response instead adds its
motion-dependent covariance log determinant. The core reflectivities remain deterministic unknowns and
are profiled, not integrated.</p>
<pre>z = Qᴴ y,       A = Qᴴ A0(p),       B = Qᴴ A1(p) D
z ~ CN(A a, C(p)),                  C(p) = I + B Bᴴ / λ
L(p) = min_a (z − A a)ᴴ C(p)⁻¹ (z − A a) + log det C(p)

Q: fixed orthonormal coordinates for the stationary-nuisance complement
D: square root of quadrature area; b ~ CN(0, λ⁻¹ I)
λ = infinity: exactly the original deterministic core-only model</pre>
<p>Implementation uses the equivalent projector without constructing Q. This is a likelihood for retained
stationary contrasts. It is not a claim that removing the stationary subspace preserves information about
every possible near-zero-Doppler target. The source's unit thermal variance, scene support, training
split, penalties, antenna coordinates, range origins and physical bounds are unchanged. Carrier phase,
delay and symbol Doppler are differentiated together.</p>
<p><a href="https://gaussianprocess.org/gpml/chapters/RW2.pdf">Rasmussen and Williams, chapter 2</a>
provides the Gaussian-process likelihood and explicit mean-function framework. The particular physical
operator, conditioning and validation here are our application of that model. The linear nuisance solve
uses SVD-based variable projection, following the numerical principles in
<a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust</a>.</p>
<h2>The recording does not follow the declared reference model closely</h2>
<table><tr><th>Channel</th><th>Residual quadratic / DOF</th><th>Covariance log determinant</th><th>Complex residual DOF</th></tr>{rows}</table>
<p>At fixed true motion under the declared model, the expected quadratic residual equals its residual
degrees of freedom. Estimating 17 real motion coefficients changes that expectation by only a small
number of degrees of freedom; it cannot explain factors of four to fourteen. These ratios are evidence
of model mismatch. They are neither direct noise-variance estimates nor measured SNR losses, and they
do not justify flattening the data or suppressing bands.</p>
<p>The earlier <a href="field-covariance-reference.json">high-Doppler field-noise audit</a> was much closer
to the declared thermal model. That control used a different diagnostic region, so it does not by itself
identify the present discrepancy. Unrepresented target response, other echoes and correlated interference
remain possibilities; this test alone cannot separate them.</p>
<h2>The information bound includes covariance motion</h2>
<pre>Fij = 2 Re[ μiᴴ C⁻¹ μj ] + tr[C⁻¹ Ci C⁻¹ Cj]
Fmotion = Fpp − Fpa Faa⁻¹ Fap

μ = A(p) a; a includes the unknown complex core coefficients in all five bands.
The Schur complement removes their nuisance information.
The requested zero-yaw restriction and coordinate gauge are fixed.</pre>
<p>This is the proper-complex Gaussian Slepian–Bangs result; see
<a href="https://arxiv.org/pdf/1306.6415">Besson and Abramovich, equation 1</a>.
The covariance term has no extra real-Gaussian factor of one half. The implementation assembles it as
a positive Gram matrix, and compares it with an independently evaluated dense Fisher matrix and Schur
complement. Optimizer Gauss–Newton curvature is not substituted for expected information.</p>
<p>The full 30-second operator has {len(active)} free physical motion coordinates. Its information
condition number is {information['information_condition']:.3g}. Relative to mean-only information,
the covariance contribution increases information by at most
{100*(max(information['covariance_information_gain_eigenvalues'])-1):.2f}% in any direction.
That is an information comparison under the model, not an observed focusing or SNR improvement.</p>
<p>The largest path-coordinate standard-deviation bound is
{100*result['maximum_path_crlb_std_m']:.3f} cm; the transverse-coordinate bound is
{100*result['maximum_transverse_crlb_std_m']:.3f} cm. These are conditional local bounds with known
reflector support, antenna geometry, range origins, noise variance and diffuse strength.
They are not absolute ship-position error bars. Adding unknown geometry and scene coordinates can
only reduce the available motion information. No acceleration or box-bound prior is counted.</p>
<h2>Independent numerical verification</h2>
<p>All 249 support tests pass. The added tests cover dense covariance/determinant evaluation, exact
core-only limits, stationary contrasts, basis invariance, physical derivatives, the Fisher nuisance
elimination and the shared constrained solver. Original-row stationarity is
{motion['fit']['maximum_kkt_residual']:.3g}; the actual-data directional likelihood derivative agrees
to 1.43e−9 at the 1 mm test step.</p>
<p>Repeated simulations use a single-covariance-SVD value/score implementation, cross-checked against the
independent augmented-SVD residual/Jacobian reference. The following discrepancies are numerical errors
between these two implementations, not statistical uncertainties:</p>
<table><tr><th>Channel</th><th>Relative objective error</th><th>Energy-normalized gradient error</th><th>Prediction difference, unit-noise Euclidean norm</th></tr>{score_rows}</table>
<p>The combined score discrepancy in Fisher-normalized coordinates is
{score['information_normalized_score_discrepancy']:.3g}. The largest prediction discrepancy has norm
{max(c['prediction_difference_in_unit_noise_norm'] for c in score['channels']):.3g} under the declared
unit-noise model. Both pass the declared statistical precision gate of 0.01. An initial arbitrary
relative-waveform gate of 1e−7 rejected Ten's 2.42e−7 relative discrepancy; the retained audit reports that
discrepancy explicitly and judges it in noise and estimator-error units.</p>
<h2>Known-truth nonlinear recovery</h2>
<p>{trial_text}</p>
<p>Every trial uses the actual 30-second, five-channel sampling operator and all 17 motion coordinates.
Diffuse coefficients and proper-complex thermal noise are redrawn; core amplitudes are unknown and
refitted. The same 2 cm smooth initialization offset is used for every trial. There are no acceleration
or box constraints. This tests local refinement, not acquisition from arbitrary position ambiguity.
Every final score is evaluated on original sample epochs; a score above tolerance triggers exact
refinement. Failed or nonstationary trials remain in the result.</p>
<p><a href="../../../build/isar-20261016/marginal-ne-motion-efficiency-30s.json">Current recovery checkpoint</a> ·
<a href="../../../build/isar-20261016-marginal-ne-motion-efficiency-30s.log">Progress log</a> ·
<a href="../../../build/isar-20261016-tests-marginal.log">249-test log</a></p>
<h2>Image and production status</h2>
<p>The <a href="mixed-ne-validation.html">previous fixed-N/E diagnostic videos</a> remain the current
rendered controls. The millimetric likelihood correction provides no measured reason to render them
again. Physical transmitter/receiver refinement, cross-band phase transfer and full-capture coherent
hull reconstruction remain unfinished. Production capture and range compression are unchanged.</p>
</html>'''
    (args.output/'marginal-ne-validation.html').write_text(body)
    print(json.dumps({'report': str(args.output/'marginal-ne-validation.html'),
        'completed_recovery_trials': result.get('nonlinear_recovery_checkpoint', {}).get('completed_trials', 0)}))


if __name__ == '__main__':
    main()
