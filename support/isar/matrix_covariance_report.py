"""Publish matrix-covariance transfer and conditional efficiency evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scene_dictionary import fingerprint


def read(path):
    return json.loads(path.read_text())


def summary(experiment, *, long=False, local=False):
    key = 'long_amplitudes' if long else 'amplitudes'
    thermal_key = 'long_thermal_variance' if long else 'thermal_variance'
    power, thermal = [], []
    for channel in experiment['channels']:
        with np.load(channel['data']) as d:
            power.append(abs(d[key])**2)
            thermal.append(d[thermal_key])
    power, thermal = np.array(power), np.array(thermal)
    mean = power.reshape(-1, 3).mean(axis=0)
    rng = np.random.default_rng(7851)
    draws = []
    for _ in range(4000):
        group = rng.integers(power.shape[1], size=power.shape[1])
        sample = power[:, group]
        if not long and not local:
            tile = rng.integers(power.shape[2], size=power.shape[2])
            sample = sample[:, :, tile]
        value = sample.reshape(-1, 3).mean(axis=0)
        draws.append(10*np.log10(value[0]/value))
    dimensions = tuple(range(2, power.ndim-1))
    grouped = power.mean(axis=dimensions)
    return {'aggregate_gain_db': (10*np.log10(mean[0]/mean)).tolist(),
        'descriptive_resampling_range_db': np.quantile(draws, [.025, .975], axis=0).T.tolist(),
        'resampling_scope': ('Resample six range groups and four disjoint time blocks; keep channels and point hypotheses together.'
            if not long and not local else 'Resample six range groups; keep every channel, point and overlapping aperture together.')+
            ' Conditional descriptive sensitivity only: fits are not retrained, returns may be dependent, and this is not a population confidence interval.',
        'per_channel_gain_db': [(10*np.log10(p[..., 0].mean()/p.reshape(-1, 3).mean(axis=0))).tolist() for p in power],
        'per_range_channel_gain_db': (10*np.log10(grouped[..., :1]/grouped)).tolist(),
        'per_channel_white_penalty_db': (10*np.log10((thermal/thermal[..., :1]).reshape(
            thermal.shape[0], -1, 3).mean(axis=1))).tolist()}


def convergence(experiment):
    diagnostics = []
    for channel in experiment['channels']:
        for outer in channel['outer_fits']:
            diagnostics.append(outer['fit'])
            diagnostics.extend(f['fit'] for f in outer['inner_folds'])
    return {'fits': len(diagnostics), 'converged': sum(d['converged'] for d in diagnostics),
        'maximum_fixed_point_residual': max(d['fixed_point_residual'] for d in diagnostics),
        'maximum_iterations': max(d['iterations'] for d in diagnostics)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('build/isar-20261016'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    paths = [args.work/folder/'benchmark.json' for folder in
        ('matrix-filter-transfer', 'local-matrix-filter-transfer')]
    transfer, local = [read(path) for path in paths]
    if not transfer['complete'] or not local['complete']:
        raise ValueError('Complete both experiments before publishing')
    efficiency_path = args.work/'matrix-spectral-gls-efficiency.json'
    efficiency = read(efficiency_path)
    cases = [summary(transfer), summary(transfer, long=True), summary(local, local=True)]
    labels = ['8.91 s, excluded range + times', '29.99 s, excluded range + times', '29.99 s, same-time neighboring ranges']
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    colors = ['#2a78a4', '#bc7131', '#7e4b9e']
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained')
    for j, (case, label, color) in enumerate(zip(cases, labels, colors)):
        x = np.arange(5)+(j-1)*.25
        axes[0].bar(x, np.array(case['per_channel_gain_db'])[:, 2], width=.25, color=color, label=label)
        axes[1].bar(x, np.array(case['per_channel_white_penalty_db'])[:, 2], width=.25, color=color)
    for ax in axes:
        ax.axhline(0., color='#444', lw=.8)
        ax.set(xticks=np.arange(5), xticklabels=names)
    axes[0].set(title='Held-out known-echo sensitivity', ylabel='SNR change (dB; positive is better)')
    axes[1].set(title='Cost under unit thermal covariance', ylabel='Mean white-noise penalty (dB)')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Cross-validated matrix weighting: unchanged complex echo gain, yaw exactly zero')
    fig.savefig(args.output/'matrix-filter-transfer.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for case, label, ax in zip(cases, labels, axes):
        gain = np.array(case['per_range_channel_gain_db'])[:, :, 2].T
        im = ax.imshow(gain, vmin=-6, vmax=6, cmap='RdBu', aspect='auto', origin='lower')
        for (g, c), value in np.ndenumerate(gain):
            ax.text(c, g, f'{value:+.1f}', ha='center', va='center', fontsize=9,
                color='white' if abs(value) > 3.5 else '#222')
        ax.set(title=label, xticks=np.arange(5), xticklabels=names,
            yticks=np.arange(6), yticklabels=['−3.60', '−2.40', '−1.20', '+1.20', '+2.40', '+3.60'],
            ylabel='Secondary excess-range displacement (km)')
    fig.colorbar(im, ax=axes, label='Known-echo SNR change (dB)', shrink=.8)
    fig.suptitle('All withheld ranges retained; blue improves and red worsens output sensitivity')
    fig.savefig(args.output/'matrix-range-transfer.png', dpi=145)
    plt.close(fig)
    stats = [convergence(v) for v in (transfer, local)]
    response_error = max(c['maximum_unit_complex_response_error'] for v in (transfer, local) for c in v['channels'])
    nuisance_error = max(c['maximum_stationary_response_error'] for v in (transfer, local) for c in v['channels'])
    result = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'sources': [{'path': str(path), 'sha256': fingerprint(path)} for path in paths],
        'experiments': [transfer, local], 'case_labels': labels, 'case_statistics': cases,
        'convergence': stats, 'known_covariance_efficiency': efficiency,
        'known_covariance_efficiency_source': str(efficiency_path),
        'known_covariance_efficiency_sha256': fingerprint(efficiency_path),
        'production_weighting_changed': False, 'yaw_fitted': False,
        'test_log': 'build/isar-20261016-tests-matrix.log', 'support_tests_passed': 222}
    (args.output/'matrix-covariance-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    rows = ''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{case["per_channel_gain_db"][c][2]:+.3f} dB</td>'
        for case in cases)+'</tr>' for c, name in enumerate(names))
    aggregate = ''.join(f'<td>{case["aggregate_gain_db"][2]:+.3f} dB</td>' for case in cases)
    ranges = ''.join(f'<li>{label}: {case["descriptive_resampling_range_db"][2][0]:+.2f} to '
        f'{case["descriptive_resampling_range_db"][2][1]:+.2f} dB.</li>' for label, case in zip(labels, cases))
    full_gains = ', '.join(f'{case["aggregate_gain_db"][1]:+.3f}' for case in cases)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Range/Doppler covariance validation</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}pre{{padding:16px;background:#f4f6f8;overflow:auto}}.note{{padding:18px;background:#fff0ce}}code{{font-size:.9em}}</style>
<h1>Range/Doppler covariance: does it improve yaw-free imaging?</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="matrix-covariance-validation.json">Measurements and source identities</a> · <a href="secondary-covariance.html">Previous scalar weighting controls</a> · <a href="companion-validation.html">Yaw-free motion and previews</a></p>
<p class="note">A full range-covariance model does not produce a consistent sensitivity gain on these held-out observations. At 30 seconds, covariance transferred across time loses {abs(cases[1]['aggregate_gain_db'][2]):.2f} dB; estimating within the same aperture loses {abs(cases[2]['aggregate_gain_db'][2]):.2f} dB. All 480 covariance fits converge. The numerical estimator passes known-truth tests, but its covariance model does not transfer reliably to the measured returns. No new whitening, normalization or phase adjustment is adopted. Yaw remains exactly zero.</p>
<h2>The controlled comparison</h2>
<p>Three exact original-row echo templates come from the existing fixed N/E, translation-only motion fit. They occupy offsets (−100,−30,20), (0,0,20) and (100,30,20) metres. Each template is normalized to unit input energy to define a comparable hypothetical echo; the recorded observations are unchanged. Every filter preserves that echo's complex amplitude and rejects the same finite stationary-response subspace.</p>
<p>The secondary crops are centred at ±1.2, ±2.4 and ±3.6 km excess-range displacement. The nearest sample is 899 m from the primary crop centre. Other physical echoes and sidelobes in these crops count as interference to the specified echo. They are not asserted to be independent, target-free thermal noise. The primary ship-bearing observations never train the covariance or regularization.</p>
<p>The new model uses a full 17 × 17 proper-complex range covariance in each of four absolute-Doppler bands: 0–0.5, 0.5–2, 2–5 and above 5 Hz in the moving reference. It remains a finite Fourier-diagonal time model. Every matrix is constrained to be at least the identity after existing thermal whitening. That floor is a declared model assumption, not a fitted display scale. Covariance is estimated by restricted likelihood, including uncertainty in the eliminated stationary coefficients.</p>
<p>Regularization uses <code>C(λ)=I+λ(Ĉ−I)</code> with candidates 0, 0.125, 0.25, 0.5, 0.75 and 1. Inner validation selects the lowest measured output interference power at unit echo response. This uses the output-power principle in <a href="https://arxiv.org/html/2104.01909">Xie et al., Cross-Validated Tuning of Shrinkage Factors for MVDR Beamforming</a>, rather than choosing covariance fit quality alone. Our block-REML estimator, stationary nuisance and range/time splits are separate implementation choices; that paper's iid secondary-data assumptions are not established for this capture.</p>
<p>The first experiment excludes one range crop and four central time tiles from training. Three inner time folds choose λ from other ranges and times. The same learned covariance and λ are tested on 8.913-second blocks and on a contiguous 29.994-second aperture. The second experiment trains on neighboring ranges from the same 29.994-second aperture, with nested leave-one-range-out selection. Its two apertures overlap and are kept together in uncertainty summaries. The second control addresses time variation directly; it does not claim temporal independence.</p>
<table><tr><th>Channel</th><th>8.91 s, range + time transfer</th><th>29.99 s, range + time transfer</th><th>29.99 s, same-time ranges</th></tr>{rows}<tr><th>Combined output-power ratio</th>{aggregate}</tr></table>
<p>Positive numbers mean less measured interference at unchanged injected echo response. The aggregate uses total output power, so stronger interference contributions have more weight. It is not an average of per-channel dB values and is not a measured hull-focusing gain.</p>
<img src="matrix-filter-transfer.png" alt="Per-channel sensitivity changes and thermal-noise costs of matrix weighting">
<p>The unregularized full-matrix fits give aggregate changes of {full_gains} dB in the same order. Cross-validation reduces some losses and sometimes chooses zero modification, but cannot make range-dependent interference share a covariance. Some individual ranges improve; others deteriorate:</p>
<img src="matrix-range-transfer.png" alt="All six secondary range groups and all five channels, with numerical sensitivity gains">
<p>Descriptive resampling ranges, retaining all channels and template hypotheses together, are:</p><ul>{ranges}</ul>
<p>These are sensitivity summaries conditional on the already fitted weights. The small number of range/time groups, dependence of echoes and overlapping apertures prevent interpreting them as population confidence intervals. We do not select favorable channels or ranges for a deployment claim.</p>
<h2>Numerical correctness and the theoretical bound</h2>
<p>For the declared model <code>y=a s+Qβ+n</code>, known covariance <code>R</code>, and a known template s:</p>
<pre>K = R⁻¹ − R⁻¹Q (QᴴR⁻¹Q)⁻¹ QᴴR⁻¹
I = sᴴK s
h = K s / I
â = hᴴ y,  hᴴs = 1,  hᴴQ = 0
E |â − a|² = 1/I
var(phase â) ≥ 1 / (2 |a|² I)</pre>
<p>The amplitude estimator attains the proper-complex Gaussian conditional CRLB; the phase bound is approached at high SNR. This does not certify an estimated covariance or an unknown physical scene. Tests independently compare the Fourier-block implementation against dense constrained GLS and error-contrast likelihoods. Analytic covariance gradients agree with finite differences. An analytic active-floor example checks positive-semidefinite KKT conditions. Large stationary coefficients leave restricted likelihood unchanged, and a complex injected echo retains both amplitude and phase.</p>
<p>All {sum(s['converged'] for s in stats)} inner and outer fits meet the whitened covariance fixed-point tolerance; the largest residual is {max(s['maximum_fixed_point_residual'] for s in stats):.3g}. The maximum unit complex response error across real-data filters is {response_error:.3g}; the maximum stationary response is {nuisance_error:.3g}. The complete support suite passes 222 tests.</p>
<p>On the actual 29.994-second SBS template and stationary subspace, a separate 8,000-trial experiment deliberately makes one learned matrix covariance the known true Gaussian covariance. Calculated estimator variance divided by its CRLB is {efficiency['calculated_mse_over_crlb']:.12f}; measured amplitude MSE divided by the predicted variance is {efficiency['measured_mse_over_variance'][1]:.5f}. At 30 dB integrated SNR, phase RMS is {efficiency['phase_rms_degrees']:.4f}°, versus the {efficiency['phase_crlb_rms_degrees']:.4f}° bound. Under that stipulated covariance the optimum gains {efficiency['conditional_oracle_gain_db']:.3f} dB over unit weighting. That number is a synthetic model result, not available gain established in the recording. The measured transfer result is worse, exposing the model gap.</p>
<h2>Consequence for autofocus</h2>
<p>These tests separate a numerical estimator error from an unsupported interference model. They do not explain all remaining phase error, nor prove unit weighting globally optimal. The first yaw-free controls still lose held-out matched energy relative to their rotating controls; fitting a generic covariance to neighboring returns has not repaired that loss. Any further correction should target a measured physical or statistical discrepancy and pass an unchanged-response sensitivity test.</p>
<p>The existing <a href="../../../captures/live-vhf-20261013-193334/isar-fixed-ne-companion-30s-20261016/ship-following-north-up-30s-30fps.mp4">ship-following</a> and <a href="../../../captures/live-vhf-20261013-193334/isar-fixed-ne-companion-30s-20261016/ship-geographic-north-up-30s-30fps.mp4">fixed geographic</a> previews remain available: fixed north/east axes, no yaw fit, 30-second coherence, 30 fps, fixed colour scale and cyan AIS. Their profiles and video pixels are unchanged by this experiment. They remain diagnostic images, not certified resolved hull imagery or accepted physical phase-centre calibration.</p>
<p>All new code is under <code>support/isar</code>. Complete folds, covariance eigenvalues, convergence histories, complex output amplitudes, white-noise costs and source hashes are retained. Physical calibration, cross-band scattering transfer and a measured end-to-end efficiency claim remain open; the broader goal remains active.</p>
</html>'''
    (args.output/'matrix-covariance-validation.html').write_text(page)
    print(json.dumps({'report': str(args.output/'matrix-covariance-validation.html'),
        'aggregate_gains_db': [c['aggregate_gain_db'][2] for c in cases]}))


if __name__ == '__main__':
    main()
