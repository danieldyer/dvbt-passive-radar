"""Publish shared-reflector component efficiency and recorded fit status."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from scene_dictionary import fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('efficiency', 'experiment', 'tests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--refinement', type=Path, action='append', default=[])
    p.add_argument('--information', type=Path)
    p.add_argument('--arithmetic', type=Path)
    p.add_argument('--marginal-experiment', type=Path)
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    efficiency = json.loads(args.efficiency.read_text())
    if not efficiency['complete'] or not efficiency['all_stationary']:
        raise ValueError('Finish all requested efficiency trials')
    if (len(efficiency['cases']) != efficiency['trials_requested']
            or any(not case['converged'] or case['bound_hits'] or case['kkt'] > 1e-7
                   for case in efficiency['cases'])):
        raise ValueError('Require all unconstrained stationary efficiency results')
    for path, expected in efficiency['source_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Efficiency implementation changed: '+path)
    log = args.tests.read_text(); count = re.search(r'Ran (\d+) tests in [\d.]+s\s+OK\s*$', log)
    if not count:
        raise ValueError('Require a completed passing support-test log')
    manifest = args.experiment/'experiment.json'; raw = manifest.read_bytes(); experiment = json.loads(raw)
    for path, expected in experiment['source_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Recorded fit input changed: '+path)
    errors = np.asarray([case['error_m'] for case in efficiency['cases']])
    information = np.asarray(efficiency['expected_information_per_m2'])
    z = np.einsum('ni,ij,nj->n', errors, information, errors)
    ratio = float(z.mean()/errors.shape[1])
    np.testing.assert_allclose(ratio, efficiency['statistics']['joint_mse_over_crlb'], atol=1e-12, rtol=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    axes[0].hist(z, bins=18, density=True, color='#377e9c', alpha=.75, label='Independent noise trials')
    x = np.linspace(0, max(24., z.max()), 400)
    axes[0].plot(x, chi2.pdf(x, errors.shape[1]), color='#d07d26', label='Efficient Gaussian reference')
    axes[0].set(xlabel='Squared error measured with Fisher information', ylabel='Density',
                title=f'{len(errors)} trials; joint MSE / CRLB = {ratio:.4f}'); axes[0].legend()
    axes[1].bar(range(errors.shape[1]), efficiency['statistics']['per_axis_mse_over_crlb'], color='#377e9c')
    axes[1].axhline(1., color='#d07d26')
    axes[1].set(xticks=range(errors.shape[1]), xticklabels=['P1 E', 'P1 N', 'P2 E', 'P2 N', 'P3 E', 'P3 N'],
                ylabel='Empirical MSE / scalar CRLB', ylim=(0, 1.4), title='Coordinates fitted jointly')
    fig.suptitle('Known motion and geometry; unknown independent complex reflectivity in each band')
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'shared-reflector-efficiency.png', dpi=160); plt.close(fig)
    trials = [dict(trial) for trial in experiment['trials']]
    refinements = []
    for path in args.refinement:
        refined = json.loads(path.read_text())
        if not refined['complete']:
            raise ValueError('Only include terminal refinements')
        for source, expected in refined['source_sha256'].items():
            if fingerprint(Path(source)) != expected:
                raise ValueError('Refinement input changed: '+source)
        original = next(t for t in trials if t['source_band'] == refined['trial']['source_band'])
        if original != refined['source_trial']:
            raise ValueError('Refinement is not from this completed start')
        refinements.append(refined); trials.append({**refined['trial'], 'refinement': str(path)})
    info = json.loads(args.information.read_text()) if args.information else None
    if info:
        if not info['complete']:
            raise ValueError('Require a completed information calculation')
        for path, expected in info['source_sha256'].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Information input changed: '+path)
    arithmetic = json.loads(args.arithmetic.read_text()) if args.arithmetic else None
    if arithmetic:
        if not arithmetic['complete']:
            raise ValueError('Require a completed arithmetic audit')
        for path, expected in arithmetic['source_sha256'].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Arithmetic audit input changed: '+path)
    marginal = None
    marginal_snapshot = None
    if args.marginal_experiment:
        marginal_raw = (args.marginal_experiment/'experiment.json').read_bytes()
        marginal = json.loads(marginal_raw)
        for path, expected in marginal['source_sha256'].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Marginal experiment input changed: '+path)
        marginal_snapshot = args.output/'shared-marginal-fit-snapshot.json'
        marginal_snapshot.write_bytes(marginal_raw)
    baseline = experiment['independent_band_support_prediction']['summary']
    rows, records = [], []
    for trial in trials:
        value = trial['after']['summary']; fit = trial['fit']
        record = {'source_band': trial['source_band'], 'refinement': trial.get('refinement'),
            'converged': fit['converged'], 'objective': fit['objective'],
            'kkt': fit['maximum_kkt_residual'], 'bounds_hit': trial['bound_hits'],
            'prediction_error_change_percent': 100*(value['prediction_error']/baseline['prediction_error']-1),
            'matched_energy_change_db': float(10*np.log10(value['matched_energy']/baseline['matched_energy'])),
            'maximum_scene_condition': max(fit['scene_conditions'])}
        records.append(record)
        values = [str(record['source_band'])+(' refined' if record['refinement'] else ''),
            record['converged'], f"{record['kkt']:.3g}", f"{record['objective']:.8f}",
            f"{record['prediction_error_change_percent']:+.4f}", f"{record['matched_energy_change_db']:+.4f}",
            record['bounds_hit'], f"{record['maximum_scene_condition']:.3g}"]
        rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    snapshot = args.output/'shared-reflector-fit-snapshot.json'; snapshot.write_bytes(raw)
    sources = [args.efficiency, args.tests, Path(__file__), *args.refinement,
               *([args.information] if args.information else []),
               *([args.arithmetic] if args.arithmetic else []),
               *([marginal_snapshot] if marginal_snapshot else [])]
    report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'efficiency': efficiency, 'recorded_trials': records, 'information': info,
        'experiment_complete': experiment['complete'], 'requested_starts': experiment['initial_support_bands'],
        'fit_snapshot': str(snapshot), 'fit_snapshot_sha256': sha256(raw).hexdigest(),
        'live_manifest': str(manifest), 'tests_passed': int(count[1]),
        'source_sha256': {str(path): fingerprint(path) for path in sources},
        'arithmetic_audit': arithmetic, 'marginal_scene_snapshot': marginal,
        'accepted_calibration': False, 'production_changed': False}
    lo, hi = efficiency['statistics']['efficient_normal_reference_interval_95']
    status = 'Finished' if experiment['complete'] else 'Running; the table is a saved partial snapshot'
    info_text = ''
    if info:
        info_rows = ''
        for case in info['cases']:
            for name in ('known', 'shared', 'independent'):
                data = case[name]
                values = [name, f"{case['cutoff']:.0e}",
                    str(data['directions_weaker_than_declared_prior'])+'/18',
                    f"{max(data['information_eigenvalues_before_prior']):.4g}"]
                info_rows += '<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>'
        info_text = '<h2>Antenna information at the fitted common scene</h2><table><tr><th>Point nuisance</th><th>Cutoff</th><th>Directions weaker than prior</th><th>Largest information eigenvalue</th></tr>'+info_rows+'</table>'
    arithmetic_text = ''
    if arithmetic:
        maximum_change = max(max(case['template_relative_errors_by_band']) for case in arithmetic['cases'])
        maximum_condition = max(case['maximum_scene_condition'] for case in arithmetic['cases'])
        arithmetic_text = f'''<h2>Arithmetic versus scene conditioning</h2>
<p>A separate audit expresses reflector displacements directly in local coordinates, avoiding subtraction of two Earth-sized positions. On the identical recorded likelihood, the largest relative template change is {maximum_change:.3g}. This is a useful precision check, but it does not explain the difficult fits. One support has a response-matrix condition number of {maximum_condition:.3g}; almost dependent columns permit large cancelling reflection coefficients and highly sensitive position derivatives. The full finite-difference step sweeps are retained in the JSON, including the poorly resolved derivative check at this ill-conditioned point. No correction or successful convergence is inferred from that audit.</p>'''
    marginal_text = ''
    if marginal:
        marginal_rows = ''
        for trial in marginal['trials']:
            fit, predicted = trial['fit'], trial['after']['summary']
            values = [trial['source_band'], fit['converged'], f"{fit['maximum_kkt_residual']:.3g}",
                f"{100*(predicted['prediction_error']/baseline['prediction_error']-1):+.4f}",
                f"{10*np.log10(predicted['matched_energy']/baseline['matched_energy']):+.4f}",
                trial['bound_hits']]
            marginal_rows += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>'
        marginal_status = 'Finished' if marginal['complete'] else 'Running; this is a saved partial snapshot'
        marginal_text = f'''<h2>Shared-position Gaussian scene control</h2>
<p>{marginal_status}. {len(marginal['trials'])} of {len(marginal['initial_support_bands'])} requested starts are terminal. This reuses the existing proper-complex Gaussian scene likelihood with independent reflection coefficients in each band. Its covariance determinant is included, and scene strength is estimated from training samples. Physical responses, observations, thermal weights, point count and bounds are unchanged. It is an explicit additional statistical assumption, not a numerical substitute for the unregularized model. Withheld waveform prediction tests whether it is useful on this recording. The known-truth efficiency result above does not certify this additional model.</p>
<table><tr><th>Start band</th><th>Stationary</th><th>KKT residual</th><th>Withheld error change (%)</th><th>Matched energy change (dB)</th><th>Bounds</th></tr>{marginal_rows}</table>
<p>Neither model fits antenna coordinates in this experiment. Common physical reflector locations are being tested before another joint motion/site refinement.</p>'''
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Shared physical reflectors</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}</style>
<h1>Shared physical reflector locations: efficiency and recorded-data checks</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="shared-reflector-validation.json">Values and source identities</a> · {report['updated_utc']}</p>
<p class="note">The component test supports the numerical position estimator under its declared Gaussian model. It does not certify the real ship's motion, scattering model, antenna locations or coherent combination across frequencies. No production processing or calibrated coordinates have changed.</p>
<h2>Known-truth efficiency</h2>
<p>Three moving physical reflectors are synthesized independently using time-domain symbol quadrature, with different complex scattering coefficients in each of three bands and strong stationary clutter. Every noisy realization starts from the same unshifted point coordinates; the true offsets include 0.25 m. Motion, antennas and band range origins are known in this test. Unknown independent complex amplitudes and stationary clutter are profiled, and their information cost is included in the point-coordinate CRLB.</p>
<p>All {len(errors)} noise trials reach the original physical-coordinate stationarity gate without hitting the ±1 m search bounds. Joint mean squared error divided by its Fisher bound is <strong>{ratio:.5f}</strong> ({10*np.log10(ratio):+.4f} dB). The efficient-normal reference interval for this finite ensemble is [{lo:.4f}, {hi:.4f}]. This is a reference-distribution comparison, not a confidence interval proving unknown real-data efficiency. Independent synthesis/model disagreement is {efficiency['independent_mean_mismatch_in_noise_units']:.3g} squared-noise units in total.</p>
<img src="shared-reflector-efficiency.png" alt="Known-scene error distribution and coordinate MSE relative to the Fisher bound">
<h2>One common scene on the recorded fixed-N/E aperture</h2>
<p>{status}. The experiment holds the current 30-second physical motion, nominal antennas, common per-band envelope origins and point heights fixed. Eight shared horizontal positions move within ±24 m of each inherited support. Five source-band supports supply different starts. Each band retains independent complex amplitudes; no equal-phase assumption, yaw fit or gain normalization is added. Original-row responses, existing noise weights and complementary half-second test blocks are unchanged.</p>
<p>The table compares against the original eight independently located points per band at the same frozen motion. This is a conditional model comparison: motion and support discovery already used this recording. A positive prediction-error change is worse. Nonstationary rows are diagnostics, never accepted calibrations or selected results.</p>
<table><tr><th>Start band</th><th>Stationary</th><th>KKT residual</th><th>Training objective</th><th>Withheld error change (%)</th><th>Matched energy change (dB)</th><th>Bounds</th><th>Largest scene condition</th></tr>{''.join(rows)}</table>
<p>The first transformed-constraint solver left its physical point box during a trial and was stopped by the retained pulse-support guard. Its failed log and source snapshot are preserved. The replacement uses native bounds, the same exact variable-projection derivatives and the unchanged 10⁻⁷ KKT threshold. Any separate quasi-Newton continuation retains the same scientific model and preserves its original terminal result.</p>
{arithmetic_text}
{marginal_text}
{info_text}
<h2>What establishes physical correspondence?</h2>
<p>The new information operator compares known points, shared unknown three-dimensional points, and independent unknown points per band at one identical mean. It removes complex coefficients, motion and shared range origins without counting priors or optimizer bounds as measurements. Nested projections ensure that extra unknowns cannot increase information. Tests compare against a direct dense Gaussian calculation and independent finite differences through range, carrier phase and Doppler; a one-band control shows no spurious information gain from correspondence.</p>
<p>Sharing point labels across the distant apertures would be unsupported: their selected points differ, and their windows span several minutes. Shared geometry within one fixed-N/E aperture is being checked first. Neither good optimizer convergence nor a larger information eigenvalue demonstrates that the physical correspondence holds.</p>
<p>The implementation retains both terms of the exact variable-projection residual derivative described by <a href="https://www.cs.umd.edu/users/oleary/software/varpro.pdf">O’Leary and Rust, §2.3</a>. Profiling independent linear coefficients is a numerical construction; physical equality of scattering locations remains an additional hypothesis tested here. The support suite passes {int(count[1])} tests.</p></html>'''
    (args.output/'shared-reflector-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'shared-reflector-validation.html').write_text(text+'\n')
    print(json.dumps({'report': str(args.output/'shared-reflector-validation.html'),
        'joint_mse_over_crlb': ratio, 'recorded_completed_starts': len(experiment['trials']),
        'recorded_experiment_complete': experiment['complete']}))


if __name__ == '__main__':
    main()
