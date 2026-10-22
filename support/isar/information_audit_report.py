"""Publish the stage audit, explicit theoretical scope and literature decisions."""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']


def table(headers, rows):
    return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('build/isar-20261015'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = ['input-information-audit', 'scene-information-audit', 'motion-efficiency',
        'field-covariance-audit', 'field-covariance-reference', 'quadratic-phase-audit',
        'production-information-probe', 'backprojection-information-roundtrip']
    sources = {}
    for name in files:
        path = args.results/f'{name}.json'
        sources[name] = json.loads(path.read_text())
        if sources[name].get('complete') is False:
            raise ValueError(f'Unfinished audit: {name}')
        shutil.copy2(path, args.output/path.name)
    for name in ('production-stage-tests-gpu-access.log', 'backprojection-information-self-test.log'):
        shutil.copy2(args.results/name, args.output/name)
    stages, scene = [sources[k] for k in ('input-information-audit', 'scene-information-audit')]
    monte = sources['motion-efficiency']
    covariance, covariance_null = [sources[k] for k in ('field-covariance-audit', 'field-covariance-reference')]
    phases = sources['quadratic-phase-audit']
    rows = []
    for i in range(5):
        a, b = stages['channels'][i], scene['channels'][i]
        rows.append([NAMES[i], f"{a['worst_motion_information_loss_db']:.6f}",
            f"{b['joint_scene_temporal_information']['worst_information_loss_db']:.6f}",
            f"{b['worst_range_snr_loss_db']:.7f}", f"{b['worst_range_delay_information_loss_db']:.7f}"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    x = np.arange(5)
    axes[0].bar(x-.18, [c['worst_motion_information_loss_db'] for c in stages['channels']], .36, label='Path/rate/acceleration Fisher loss')
    axes[0].bar(x+.18, [c['joint_scene_temporal_information']['worst_information_loss_db'] for c in scene['channels']], .36, label='Worst 64-point mixture')
    axes[0].set(xticks=x, xticklabels=NAMES, ylabel='Information loss (dB)', title='16-symbol averaging, exact recorded epochs')
    axes[0].legend(fontsize=8)
    for c, name in zip(scene['channels'], NAMES):
        a = c['range_crop']
        axes[1].plot([v['delay_samples'] for v in a], [v['worst_delay_information_loss_db'] for v in a], label=name)
    axes[1].set(xlabel='Point offset from tracked centre (samples)', ylabel='Delay-information loss (dB)', title='17-sample range crop; every archived spectrum')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'information-retention.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    mcrows, bias_tests = [], []
    for case in monte['cases']:
        errors_path = args.results/f"motion-efficiency-{case['snr_db']:g}dB.npz"
        with np.load(errors_path) as f:
            errors = f['errors']; bound = f['covariance']
        normalized = np.linalg.solve(np.linalg.cholesky(bound), errors.T).T
        n, p = normalized.shape
        mean = normalized.mean(axis=0)
        bias_stat = float(n*mean@np.linalg.solve(np.cov(normalized.T), mean))
        bias_tests.append({'snr_db': case['snr_db'], 'joint_bias_statistic': bias_stat,
            'large_sample_chi_square_tail_probability': float(chi2.sf(bias_stat, p)),
            'scope': 'Approximate joint bias diagnostic; correlated parameters are not independent bias tests.'})
        axes[0].plot(np.arange(p), case['mse_over_crlb_by_parameter'], 'o-', ms=3, label=f"{case['snr_db']:g} dB")
        axes[1].plot(np.arange(p), case['bias_in_crlb_standard_deviations'], 'o-', ms=3, label=f"{case['snr_db']:g} dB")
        mcrows.append([f"{case['snr_db']:g}", case['trials'],
            f"{case['mean_mahalanobis_error_per_parameter']:.4f}",
            f"{max(case['mse_over_crlb_by_parameter']):.4f}", len(case['nonconverged_trials']), len(case['bound_hit_trials'])])
    axes[0].axhline(1, color='black', lw=1); axes[1].axhline(0, color='black', lw=1)
    axes[0].set(xlabel='Active physical parameter index (11 motion + 1 range)', ylabel='MSE / conditional CRLB', title='Actual physical variable-projection solver')
    axes[1].set(xlabel='Active physical parameter index', ylabel='Bias / CRLB standard deviation', title='Bias reported separately; no trials removed')
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(args.output/'motion-estimator-efficiency.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for c, name in zip(covariance['channels'], NAMES):
        record = c['crops'][1]
        freq = np.array(record['frequency_hz']); order = np.argsort(freq)
        axes[0].plot(freq[order], 10*np.log10(np.array(record['mean_spectrum'])[order]), label=name)
        cs = covariance_null['channels'][c['channel']]['crops']
        axes[1].plot([r['range_shift_samples'] for r in cs],
            [r['metrics']['normalized_eigenvalue_spread']['value'] for r in cs], 'o-', label=name)
    interval = covariance_null['finite_sample_null_95_intervals']['normalized_eigenvalue_spread']
    axes[1].axhspan(*interval, color='grey', alpha=.2, label='95% white-noise reference')
    axes[0].set(xlim=(-28, 28), ylim=(-2, 30), xlabel='Residual Doppler after AIS deramp (Hz)',
        ylabel='Power relative to conditional noise (dB)', title='Measured field spectrum; no rescaling')
    axes[1].set(xlabel='Range-crop displacement (samples)', ylabel='Trace-normalized eigenvalue spread',
        title='Range covariance versus 4,000 tapered-noise trials')
    for ax in axes: ax.legend(fontsize=8)
    fig.savefig(args.output/'field-covariance-reference.png', dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    for trial in phases['trials']:
        ax.plot(x, np.degrees(trial['phase_increment_radians']), 'o-', label=trial['training_split'].replace('_', ' '))
    ax.set(xticks=x, xticklabels=NAMES, ylabel='Phase increment from existing gains (degrees)',
        title='Globally bounded constant-phase fits; same physical scene support')
    ax.legend(fontsize=9)
    fig.savefig(args.output/'globally-bounded-phase-control.png', dpi=160); plt.close(fig)

    literature = [
        ('Garry & Smith, 2019, Passive ISAR I', 'https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233',
         'Full framework and bistatic geometry. Separates translation, rotation and formation, and motivates a narrow stationary null for near-tangential targets.',
         'Keep exact bistatic BP and shared physical delay/phase/rate. A broad DC notch is inappropriate for this ship.'),
        ('Garry & Smith, 2019, Passive ISAR II', 'https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5236',
         'Narrowband examples show PGA can focus incorrectly, and image contrast/entropy can favor the wrong result. Parametric centroid compensation plus tracking performs better in their tested scenes.',
         'Use physical likelihood and withheld complex prediction. The ship is larger than their aircraft examples, so measure its support rather than assume the same failure regime.'),
        ('She, Gray & Bogner, 2001', 'https://wrexham.repository.guildhe.ac.uk/211/1/fulltext.pdf',
         'Complex-vector beamforming/subspace autofocus avoids phase unwrapping. Sections 5–6 relate the estimators to likelihood and conditional CRLBs. The simplified model assumes aligned range cells and negligible rotation within its phase-estimation segment.',
         'Use complex phases directly and test expected Fisher information. An unrestricted common phase screen can remove real rotational structure if that assumption fails.'),
        ('Evers, 2019, GPGA dissertation', 'https://scholar.afit.edu/etd/2240/',
         'Repository abstract describes GPGA with BP/PFA, approximate marginal likelihood, and Max-SDR phase estimation with an approximation bound. The full dissertation download was unavailable in this review.',
         'New implementation uses an independently derived semidefinite dual bound for the exact shared-scene phase likelihood. It is not represented as a line-for-line reproduction of the unavailable dissertation.'),
        ('Xiong et al., 2019', 'https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-rsn.2018.5400',
         'Uses U-ESPRIT and matched dominant poles for inter-band linear phase, followed by common-rotation-centre phase compensation. Pole correspondence and a suitable shared scattering model are essential.',
         'Continuous dominant-reflector registration is a candidate. Preserve the actual bistatic rays; do not assume a simple global phase ramp relates two extended-target band responses.'),
        ('Zhu et al., 2021', 'https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/rsn2.12030',
         'Joint multiband phase compensation and complex Gaussian scale-mixture sparse imaging, with off-grid dictionary refinement. Its model explicitly assumes adjacent radars with the same view and frequency-independent point scattering.',
         'Sparse evidence-based support selection is worth testing, but sparsity or a sharp image is not a calibration certificate. Use independent complex band responses as a comparison and withhold observations.'),
        ('van Dorp et al., 2010', 'https://publications.tno.nl/publication/105287/GOTIrj/dorp-2010-high.pdf',
         'Coherent multiband spectral modelling and bandwidth interpolation. Its discussion explicitly notes that interpolation errors enter later image formation.',
         'No invented data between the VHF bands. Exact BP of measured bands already provides the appropriate adjoint; spectral extrapolation needs a validated scattering prior.'),
        ('Santi et al., 2013', 'https://iris.uniroma1.it/handle/11573/543328',
         'The institutional abstract describes multisensor roll/pitch/yaw estimation and theoretical CRB analysis. Only the abstract was accessible here.',
         'Extra rotational parameters need observable geometric information. Keep roll and pitch fixed, as requested, while testing the higher-value range/phase variables.'),
        ('O’Leary & Rust, variable projection', 'https://www.cs.umd.edu/~oleary/software/varpro.pdf',
         'Separable nonlinear least squares eliminates linear coefficients at each nonlinear trial. The residual derivative must account for their refit; numerical convergence still need not identify a global minimum.',
         'Retain QR/SVD elimination and the complete residual Jacobian. The Monte Carlo exercises this implemented solver, not a proxy estimator.'),
        ('Scharf & Friedlander, 1994', 'https://api.mountainscholar.org/server/api/core/bitstreams/0bd25191-6792-419a-b526-2b4914839e17/content',
         'Matched-subspace GLRT with structured nuisance and known covariance; its optimality statement is within the declared invariant model family.',
         'Project data and templates identically and use projected information. Finite calibration uncertainty is a covariance term, not automatically an infinite-variance nuisance.'),
        ('Pakrooh et al., Fisher information after compression', 'https://arxiv.org/abs/1504.01081',
         'Analyzes Fisher information under linear compression for nonlinear parameter estimation.',
         'Compare nuisance-aware expected information before and after each fixed linear operation. A small pointwise energy loss alone does not certify arbitrary interfering mixtures.')]
    review = ''.join(f'<tr><td><a href="{url}">{html.escape(title)}</a></td><td>{html.escape(method)}</td><td>{html.escape(decision)}</td></tr>' for title, url, method, decision in literature)
    phase_rows = [[t['training_split'], t['certificate']['globally_certified'],
        f"{t['certificate']['relative_gap_to_phase_dependent_scale']:.3g}",
        f"{t['heldout_error_change_percent']:+.3f}%"] for t in phases['trials']]
    stamp = datetime.now(timezone.utc).isoformat()
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>ISAR information and autofocus audit</title>
<style>body{{font:16px/1.55 system-ui;max-width:1180px;margin:35px auto;padding:0 24px;color:#20242a}}h1,h2{{line-height:1.2}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d7dce2;text-align:left;vertical-align:top}}.status{{padding:18px;background:#fff2d1;border-left:5px solid #ba7600}}code,pre{{background:#f1f3f5}}pre{{padding:16px;overflow:auto}}figure{{margin:28px 0}}figcaption{{color:#555}}</style>
<h1>Ship ISAR: information retention, estimator efficiency and bounded phase fitting</h1>
<p>{stamp}. Capture live-vhf-20261013-193334; working interval 142–172 s. <a href="analysis.html">Main analysis index</a>.</p>
<p class="status">The goal remains active. The fixed numerical transformations pass their conditional checks, and the physical motion solver is close to its conditional CRLB at 0–20 dB in the specified test. This is not an end-to-end CRLB certificate for the ship recording: SBS covariance and the shared scattering/calibration model still need work. Roll and pitch remain fixed. No new physical phase-centre calibration is adopted.</p>
<h2>1. What “optimal” means in this audit</h2>
<p>For proper complex Gaussian data with mean μ(θ) and known covariance R, the real-parameter expected information is <code>J = 2 Re(Dᴴ R⁻¹ D)</code>, D = ∂μ/∂θ. Unknown complex scene and clutter amplitudes must be eliminated by a Schur complement. A residual Hessian at a noisy fit is not a substitute for expected information at a known truth.</p>
<p>For a fixed compression L, compare <code>J_L = 2 Re(Dᴴ Lᴴ (L R Lᴴ)⁻¹ L D)</code> with J, after the same nuisance elimination. The eigenvalues of the whitened pair measure the worst loss over parameter combinations. Compression can be sufficient if its retained whitened row space contains the signal and score directions. A small error for one isolated point does not establish this for all mixtures; the new audit tests all 64 complex scene modes as well.</p>
<p>For a known candidate a and unknown deterministic stationary coefficients Bβ, whiten first, then use <code>P = I − Q_B Q_Bᴴ</code>, <code>z = aᴴ P y</code> and <code>I = aᴴ P a</code>. The amplitude estimate z/I has complex variance 1/I under that model. The matched statistic |z|²/I is a likelihood quantity, not empirical image normalization. If cancellation uncertainty instead has a finite independently measured covariance V, the appropriate model is <code>R_eff = R + B V Bᴴ</code>; projection is its infinite-uncertainty limit. The current nuisance model must not be described as universally optimal for both cases.</p>
<h2>2. Recorded-geometry compression bounds</h2>
{table(['Channel', 'Worst motion Fisher loss (dB)', 'Worst joint scene loss (dB)', 'Worst range SNR loss (dB)', 'Worst range-delay Fisher loss (dB)'], rows)}
<figure><img src="information-retention.png" alt="Measured compression information losses"><figcaption>Temporal tests use original row times, weights, within-symbol combination noise and the exact forward operator. Range tests independently evaluate Fourier sums using every archived response/noise spectrum in the interval.</figcaption></figure>
<p>All 64 scene modes were resolved in every channel at the declared 10⁻¹⁰ relative information floor: none were silently omitted. The independent point tests remove the unknown complex amplitude before comparing path, path-rate and acceleration information. The actual point support lies between −1.38 and +4.28 samples; the range test covers −1.5 to +4.5, inside the retained −8…+8 crop.</p>
<p>These results certify the stated Gaussian compression model, not arbitrary waveforms or an unknown real interference covariance. Range weights include recorded hardware/noise spectra but do not replace each finite random DVB-T waveform by a newly measured ambiguity function. The separate production finite-waveform tests cover prefix context, source-clock drift, Doppler moments and beyond-guard echoes. Losses from different stages should not simply be added as independent worst-case guarantees.</p>
<h2>3. Physical estimator Monte Carlo</h2>
{table(['Per-observation SNR (dB)', 'Trials', 'Joint normalized MSE / parameter', 'Largest parameter MSE / CRLB', 'Nonconverged', 'Bound hits'], mcrows)}
<figure><img src="motion-estimator-efficiency.png" alt="Physical estimator efficiency and bias"><figcaption>8,000 trials; all start at zero corrections. Three unknown complex point amplitudes and stationary nuisance are refitted. Expected Fisher information is evaluated at noiseless truth. No failed or bound-hit trial is removed.</figcaption></figure>
<p>At 0–20 dB the joint normalized error is 0.996–1.023. The worst individual parameter ratio is 1.109 (0.45 dB), so “close” does not mean exactly efficient for every finite-SNR parameter. At −10 dB the joint ratio rises to 1.290 and 98/2,000 trials hit the declared box. That regime fails a tight efficiency claim. The complete JSON includes bias, parameter covariance, variance confidence intervals and convergence diagnostics. This isolated three-reflector test does not certify the much denser real scattering model.</p>
<h2>4. Real-data covariance is an explicit remaining issue</h2>
<figure><img src="field-covariance-reference.png" alt="Measured spectra and finite-sample covariance comparison"><figcaption>The frequency plots are measured power, with no local or per-channel rescaling. The covariance reference includes the frequency correlation induced by the diagnostic Hann taper. The grey interval is a pointwise 95% reference, not a simultaneous confidence band.</figcaption></figure>
<p>In the diagnostic 15–25 Hz residual-Doppler band, mean variances at the ship crop are 1.086, 1.203, 1.084, 1.032 and 1.040 versus a nominal value of one. The 4,000-trial unit-Gaussian reference gives a 95% mean-variance interval 0.974–1.028. Scalar excess power is distinct from correlation. After dividing covariance by its own trace for this diagnostic only, SBS has eigenvalue spreads 1.185–1.283 across all four crops, outside every simulated null realization. Its range correlation also exceeds the reference. Most other spread/correlation measurements are much closer to the finite-sample reference.</p>
<p>This establishes a model-check failure worth investigating, not permission to flatten the ship spectrum. The crops overlap or share underlying samples; their tests are not independent. Neighboring returns can contaminate secondary data, and the high-Doppler band does not establish the covariance near the ship's low residual Doppler. No whitening correction has been applied. Next: test secondary-data covariance transfer at the relevant Doppler and use injected known echoes to measure any net sensitivity gain before adopting a correction.</p>
<h2>5. Fresh upstream and backprojection checks</h2>
<p><a href="production-stage-tests-gpu-access.log">All 37 production tests pass</a> (13 CPU, 24 GPU). The initial sandbox-only attempt could not expose Metal; the successful run used the existing SDK and GPU access. The raw-IQ-to-multitransmitter test retains coherent amplitude fraction 1, with per-channel phase RMS 0.065–0.086°. Finite reference correlation with fractional source-clock wraps has maximum phase error 0.00300°. The three independent diagonal-covariance matched-filter controls lose 0.00824–0.00863 dB.</p>
<p>The <a href="backprojection-information-roundtrip.json">independent five-channel Vulkan round trip</a>, including nonzero physical TX/RX offsets, envelope biases and complex channel gains, has combined phase error 0.000255°. The separate GPU/double self-test has maximum coherent phase error 0.000618°. These are numerical checks with known truth, not a fit to real vessel motion.</p>
<p>The fresh 6,000-trial-per-condition <a href="production-information-probe.json">pilot timing probe</a> gives MSE/CRLB 1.003–1.027 for its declared pilot-only likelihood, including colored noise and estimated weights. Decoded data carriers contain additional timing information; this is not the full-waveform timing CRLB. The existing tight-clock stress test gives 0.374° RMS over three seconds, as previously discussed. The magnitude-only minimum-phase estimator is also not a universally efficient joint scene/hardware estimator. Neither of these limits is hidden by the new imaging-stage results.</p>
<h2>6. A globally bounded channel-phase control</h2>
<p>For the fixed physical dictionaries and band magnitudes, write <code>y_c = exp(iφ_c) A_c a + n_c</code>. QR eliminates the common complex scene a and reduces the phase fit to <code>max zᴴ C z, |z_c| = 1</code>, with <code>z_c = exp(−iφ_c)</code>. The semidefinite dual <code>min Σμ_c, diag(μ) − C ⪰ 0</code> gives an upper bound. A feasible phase vector attaining that bound certifies the optimum of this stated phase problem. The new implementation checks feasibility explicitly and reports a gap when the relaxation is not tight.</p>
{table(['Training split', 'Global numerical certificate', 'Gap / phase-dependent scale', 'Holdout error change'], phase_rows)}
<figure><img src="globally-bounded-phase-control.png" alt="Globally bounded phase fits across splits"><figcaption>Ten fixes the common phase gauge. Independent fits on aperture halves still differ; the largest difference is 36.96° in ABC. Conditional noise-only phase CRLBs are much smaller, and are not valid real-data error bars under model mismatch.</figcaption></figure>
<p>All three fixed-support phase problems reach their upper bound to numerical tolerance. The alternate-second control nevertheless increases heldout error by 4.543% and is 48.43% worse than independent complex band responses. The two half-aperture prediction scores are strongly affected by extrapolation of the ill-conditioned scene; they are reported but not treated as equally reliable acceptance tests. This rules out a missed phase optimum within this restricted control, not better geometry, support, scattering models or motion. No correction is adopted from it.</p>
<h2>7. Literature review and implementation decisions</h2>
<p>The review distinguishes a sharper image, a numerically optimized likelihood, and a validated physical reconstruction. These are different tests. It covers primary papers and institutional copies; access limits are stated explicitly. Newer sparse/Bayesian approaches do not remove their scattering-model assumptions.</p>
<table><tr><th>Primary source</th><th>Method and assumptions</th><th>Decision for this recording</th></tr>{review}</table>
<h2>8. Work still required for the requested deliverable</h2>
<p>The useful progress is a quantified sub-0.011 dB conditional compression loss, a statistically tested motion solver, fresh GPU phase verification, and a global phase-fit bound. The real covariance check and shared-scattering prediction remain open. The next implementation should address those observed gaps, then re-estimate range/phase residuals jointly with physical motion and compare withheld predictions. Roll/pitch fitting is not the next step.</p>
<p>A new accepted autofocus video remains pending. Existing movies are conditional trials, and a movie of the rejected phase control would not constitute the requested improved result. The goal stays active; no external-information blocker is declared by this audit.</p>
<details><summary>Reproducible outputs and commands</summary><p>{' · '.join(f'<a href="{name}.json">{name}</a>' for name in files)}</p>
<pre>env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 build/validation-venv/bin/python support/isar/audit_motion_efficiency.py --trials 2000 --snr-db -10 0 10 20 --output build/isar-20261015/motion-efficiency.json
env OPENBLAS_NUM_THREADS=2 build/validation-venv/bin/python support/isar/audit_phase_quadratic.py --dictionary build/isar-20261015/subgrid-dictionary-64 --output build/isar-20261015/quadratic-phase-audit.json
build/validation-venv/bin/python support/isar/information_audit_report.py</pre></details></html>'''
    (args.output/'information-autofocus-audit.html').write_text(content)
    report = {'updated_utc': stamp, 'status': 'Active; conditional stages tested, full real-data certificate incomplete',
        'report': str(args.output/'information-autofocus-audit.html'), 'bias_diagnostics': bias_tests,
        'roll_pitch_fitted': False, 'new_calibration_adopted': False,
        'production_tests': {'passed': 37, 'cpu': 13, 'gpu': 24},
        'new_method': 'QR-profiled constant channel phase likelihood with semidefinite global bound',
        'remaining': ['Real covariance at ship Doppler, especially SBS', 'Transferable physical scattering/range/phase calibration',
            'New validated independent and coherent autofocus video']}
    (args.output/'information-autofocus-audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
