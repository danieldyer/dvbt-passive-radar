"""Publish the guarded secondary-covariance and known-echo transfer tests."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from motion import C
from scene_dictionary import fingerprint


def read(path):
    return json.loads(path.read_text())


def summarize_benchmark(path):
    record = read(path)
    if not record['complete']:
        raise ValueError('Finish the benchmark before publishing it')
    power = np.array([abs(np.load(c['data'])['amplitudes'])**2 for c in record['channels']])
    gain = float(10*np.log10(power[..., 0].mean()/power[..., 2].mean()))
    rng = np.random.default_rng(4552)
    resampled = []
    for _ in range(4000):
        group = rng.integers(power.shape[1], size=power.shape[1])
        tile = rng.integers(power.shape[2], size=power.shape[2])
        value = power[:, group][:, :, tile]
        resampled.append(10*np.log10(value[..., 0].mean()/value[..., 2].mean()))
    return {'source': str(path), 'source_sha256': fingerprint(path), 'experiment': record,
        'aggregate_known_echo_gain_db': gain,
        'two_way_resampling_interval_db': np.quantile(resampled, [.025, .975]).tolist(),
        'resampling_scope': 'Descriptive range-group/time-tile resampling, preserving channels and point hypotheses together. Four test tiles and heterogeneous dependent returns do not establish a population confidence interval.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=Path('build/isar-20261016'))
    p.add_argument('--output', type=Path, default=Path('docs/data/isar_20261015'))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
    source = args.work/'secondary-spectra-120-210'
    spectrum_meta = read(source/'spectra.json')
    if not spectrum_meta['complete']:
        raise ValueError('Finish the spectra first')
    benchmarks = [summarize_benchmark(args.work/folder/'benchmark.json') for folder in
        ('secondary-filter-transfer', 'secondary-filter-signed-converged')]
    efficiency = read(args.work/'spectral-gls-efficiency.json')
    measured = []
    fig, axes = plt.subplots(5, 2, figsize=(14, 15), layout='constrained')
    overlap_fig, overlap_axes = plt.subplots(1, 5, figsize=(16, 3.8), layout='constrained')
    for c in range(5):
        with np.load(source/f'channel-{c}.npz') as d:
            freq = d['frequency_hz']
            order = np.argsort(freq)
            shifts = d['shifts']
            secondary = np.flatnonzero(shifts != 0)
            spectrum = d['residual_spectrum'].mean(axis=1)
            nominal = d['nominal_spectrum'].mean(axis=(0, 1))
            for chosen, color, label in ((shifts < 0, '#2677a8', 'Lower excess ranges'),
                    (shifts > 0, '#c67a22', 'Higher excess ranges'),
                    (shifts == 0, '#874ca0', 'Ship crop')):
                axes[c, 0].plot(freq[order], 10*np.log10(spectrum[chosen].mean(axis=0))[order],
                    color=color, lw=1.2, label=label)
            axes[c, 0].plot(freq[order], 10*np.log10(nominal)[order], color='#333', lw=.9,
                ls='--', label='Propagated unit-noise model')
            axes[c, 0].set(title=names[c], xlim=(-3, 3), ylim=(-3, 35),
                xlabel='Residual Doppler in the moving reference (Hz)', ylabel='Spectral power (dB / unit noise)')
            if c == 0:
                axes[c, 0].legend(fontsize=8)
            band_power = np.trace(d['covariance'], axis1=-2, axis2=-1).real/17
            tiles = d['tile_centres_s']
            half_tile = (tiles[1]-tiles[0])/2
            heat = axes[c, 1].imshow(10*np.log10(band_power[secondary, :, 0]),
                aspect='auto', origin='lower', vmin=0, vmax=28, cmap='viridis',
                extent=(tiles[0]-half_tile, tiles[-1]+half_tile, -.5, len(secondary)-.5))
            axes[c, 1].set(title=names[c]+': secondary |Doppler| < 0.5 Hz',
                xlabel='Capture time (s)', ylabel='Range shift from ship (km)',
                yticks=np.arange(len(secondary)),
                yticklabels=[f'{s*C/8e9:+.2f}' for s in shifts[secondary]])
            vectors = []
            for group in secondary:
                _, u = np.linalg.eigh(d['covariance'][group, :, 0].mean(axis=0))
                vectors.append(u[:, -1])
            vectors = np.array(vectors).T
            overlap = abs(vectors.conj().T@vectors)**2
            im = overlap_axes[c].imshow(overlap, origin='lower', vmin=0, vmax=1, cmap='viridis')
            overlap_axes[c].set(title=names[c], xticks=np.arange(len(secondary)),
                yticks=np.arange(len(secondary)), xticklabels=shifts[secondary], yticklabels=shifts[secondary],
                xlabel='Range shift (samples)', ylabel='Range shift (samples)')
            measured.append({'channel': c, 'secondary_shifts_samples': shifts[secondary].tolist(),
                'secondary_mean_band_power': band_power[secondary].mean(axis=1).tolist(),
                'secondary_low_band_tile_power': band_power[secondary, :, 0].tolist(),
                'leading_low_band_range_mode_overlap': overlap.tolist()})
    fig.colorbar(heat, ax=axes[:, 1], label='Mean band power (dB; unit noise = 0 dB)', shrink=.6)
    fig.suptitle('Guarded secondary observations: structured power is range- and time-dependent\nFixed units; exact stationary-projector noise propagation; no spectrum flattening')
    fig.savefig(args.output/'secondary-spectra.png', dpi=135)
    plt.close(fig)
    overlap_fig.colorbar(im, ax=overlap_axes, label='Squared leading-mode overlap', shrink=.7)
    overlap_fig.suptitle('Range covariance also varies: dominant modes are not shared by all secondary crops')
    overlap_fig.savefig(args.output/'secondary-range-modes.png', dpi=140)
    plt.close(overlap_fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for index, (b, label) in enumerate(zip(benchmarks, ('4 symmetric bands', '13 signed bands'))):
        x = np.arange(5)+(index-.5)*.34
        records = b['experiment']['channels']
        axes[0].bar(x, [r['heldout_injection_snr_gain_db'][2] for r in records], width=.34, label=label)
        axes[1].bar(x, [r['mean_white_noise_penalty_db'][2] for r in records], width=.34, label=label)
    axes[0].set(title='Known-echo gain on withheld secondary data', ylabel='SNR change (dB; positive is better)')
    axes[1].set(title='Cost if the residual were actually white', ylabel='Mean thermal-noise penalty (dB)')
    for ax in axes:
        ax.set(xticks=np.arange(5), xticklabels=names)
        ax.axhline(0., color='#444', lw=.7)
        ax.legend(fontsize=9)
    fig.suptitle('Same yaw-free point templates and unit complex gain; no weights trained on the test cells')
    fig.savefig(args.output/'secondary-filter-transfer.png', dpi=155)
    plt.close(fig)

    output = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'scope': __doc__,
        'secondary_source': str(args.work/'secondary-120-210/secondary.json'),
        'spectrum_manifest': spectrum_meta, 'spectral_measurements': measured,
        'benchmarks': benchmarks, 'known_covariance_efficiency': efficiency,
        'production_weighting_changed': False, 'yaw_fitted': False,
        'support_test_log': 'build/isar-20261016-tests-secondary.log'}
    (args.output/'secondary-covariance.json').write_text(json.dumps(output, indent=2)+'\n')
    rows = ''.join(f'<tr><td>{name}</td><td>{benchmarks[0]["experiment"]["channels"][c]["heldout_injection_snr_gain_db"][2]:+.3f} dB</td>'
        f'<td>{benchmarks[1]["experiment"]["channels"][c]["heldout_injection_snr_gain_db"][2]:+.3f} dB</td></tr>'
        for c, name in enumerate(names))
    response_error = max(c['maximum_unit_complex_response_error'] for b in benchmarks for c in b['experiment']['channels'])
    phase_error = max(c['maximum_nyquist_phase_error_degrees'] for c in spectrum_meta['channels'])
    maximum_gradient = max(f['fit']['maximum_gradient'] for b in benchmarks
        for c in b['experiment']['channels'] for f in c['outer_fits'])
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Secondary covariance and ISAR sensitivity</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#243040}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#fff0ce;padding:18px}}pre{{overflow:auto;background:#f4f6f8;padding:15px}}code{{font-size:.9em}}</style>
<h1>Secondary covariance: a measured check on yaw-free autofocus</h1>
<p><a href="analysis.html">Analysis index</a> · <a href="fixed-ne-followup.html">Yaw-free fit, point-response plots and videos</a> · <a href="secondary-covariance.json">Measurements and identities</a></p>
<p class="note">Yaw remains exactly zero. Two statistically defined spectral-weighting candidates fail to improve known-echo sensitivity consistently on withheld neighboring data. Their aggregate changes are {benchmarks[0]['aggregate_known_echo_gain_db']:+.3f} dB and {benchmarks[1]['aggregate_known_echo_gain_db']:+.3f} dB. Neither is adopted in the production pipeline or the videos. This rules out these candidates on this benchmark; it does not establish that correlated interference can never be handled better.</p>
<h2>What the recording says</h2>
<p>I re-extracted 120–210 seconds through the identical range interpolation, original-row phase response, information-weighted 16-row averaging and declared range whitening used by autofocus. The ship crop reproduces the existing complex field to an absolute tolerance of 10⁻¹¹. Six other crops are centred at ±1.20, ±2.40 and ±3.60 km excess-range displacement. Their nearest sample is 899 m from the ship-crop centre. These are secondary observations; they can contain other echoes or sidelobes and are not labelled pure noise.</p>
<p>Each spectral diagnostic uses ten disjoint 512-sample tiles, about 8.91 seconds each, and three unit-energy DPSS tapers with time-half-bandwidth two. I propagated the saved stationary projector through the taper and Fourier transform to obtain the white-noise reference. In particular, subtracting a fitted stationary response does not leave unit variance in every Doppler bin. The maximum difference between actual sample epochs and the diagnostic FFT grid corresponds to {phase_error:.5f}° at Nyquist.</p>
<img src="secondary-spectra.png" alt="Per-channel spectra and time/range dependence of low-Doppler secondary power">
<p>The near-thermal high-Doppler floor is not representative of the low residual-Doppler region occupied by the ship. The low-frequency power also changes substantially across neighboring ranges and over time. Some dominant range-covariance modes agree between lower-range crops, especially for Ten and ABC, but they do not transfer uniformly to the higher-range crops:</p>
<img src="secondary-range-modes.png" alt="Squared overlap of dominant range covariance modes between secondary crops">
<p>This is a concrete obstacle to treating one neighboring covariance as truth for the ship. It is not evidence that those returns should be erased. The source could be other scattering, interference or residual model error; this analysis does not identify it.</p>
<h2>A controlled filter test, not a flatter-looking image</h2>
<p>The test uses three exact physical point templates in the fixed N/E frame, with translation from the existing no-yaw fit. A template is normalized to unit input energy solely to define a comparable injected echo. The recording itself is never power-normalized. All filters have unit complex response to the template and jointly eliminate constant-delay nuisance amplitudes.</p>
<p>Four complete tiles inside the fitted 45-second interval are withheld for testing. Each outer test also withholds one secondary range crop. Covariance training uses only the other five crops and the six non-test tiles. Three inner time folds select the amount of covariance regularization using restricted likelihood, with zero weighting change among the candidates. Every inner selection chose the full fitted weighting. The hypotheses and motion were selected earlier; this is a conditional experiment, not a new independent capture.</p>
<p>The first covariance has four bands indexed by |Doppler|. The second uses thirteen signed bands with finer resolution near zero, so it does not impose positive/negative spectral symmetry on complex data. Both are finite-block circulant time models with independent already-whitened range coordinates. The experiment tests that assumption instead of calling it optimal for arbitrary clutter. All 60 outer covariance fits pass stationarity, with maximum normalized gradient {maximum_gradient:.3g}; the signed model's initial iteration-limited run was retained separately and superseded by the converged run.</p>
<table><tr><th>Channel</th><th>Four symmetric bands</th><th>Thirteen signed bands</th></tr>{rows}</table>
<img src="secondary-filter-transfer.png" alt="Heldout sensitivity changes and white-noise costs of spectral weighting">
<p>Scores compare measured output interference power at exactly the same known echo response, including any other real returns in the secondary data. They are not reductions of a differently scaled likelihood. The largest complex response error is {response_error:.3g}. The small ABC improvements are not consistent across the other channels. A descriptive two-way resampling range for the aggregate change is {benchmarks[0]['two_way_resampling_interval_db'][0]:+.2f} to {benchmarks[0]['two_way_resampling_interval_db'][1]:+.2f} dB for the first model, and {benchmarks[1]['two_way_resampling_interval_db'][0]:+.2f} to {benchmarks[1]['two_way_resampling_interval_db'][1]:+.2f} dB for the signed model. With only four time tiles and dependent heterogeneous returns, these are not population confidence intervals.</p>
<h2>The conditional optimum and its limits</h2>
<p>For <code>y = a s + Dβ + n</code>, with a known positive-definite covariance <code>R</code>, define:</p>
<pre>K = R⁻¹ − R⁻¹D (DᴴR⁻¹D)⁻¹ DᴴR⁻¹
h = K s / (sᴴK s)
â = hᴴy
E[|â − a|²] = 1 / (sᴴK s)</pre>
<p>This is the minimum-variance linear unbiased amplitude estimator. For proper-complex Gaussian noise it attains the complex-amplitude CRLB, conditional on the known signal, nuisance subspace and covariance. It preserves the target's complex amplitude and phase while annihilating the declared nuisance. It is the generalized matched-subspace construction; <a href="https://api.mountainscholar.org/server/api/core/bitstreams/0bd25191-6792-419a-b526-2b4914839e17/content">Scharf and Friedlander's matched-subspace analysis</a> supplies the corresponding known-model detection framework. Its optimality assumptions do not certify our real ship model.</p>
<p>Covariance powers are fitted to secondary data using the proper-complex restricted negative log likelihood <code>log det R + log det(DᴴR⁻¹D) + yᴴK y</code>. The nuisance-information determinant is retained. Direct tests compare this expression with a dense likelihood of independent error contrasts and check every analytic derivative. <a href="https://arxiv.org/html/1504.00302">Castrillón-Candás, Genton and Yokota, equations 3–6</a>, describe the Gaussian contrast/REML construction; this code uses its proper-complex finite-block form, not their spatial approximation algorithm.</p>
<p>On the actual SBS yaw-free point template and stationary subspace, I deliberately made the fitted spectral model the true covariance and generated {efficiency['trials']:,} independent complex Gaussian trials. Amplitude MSE divided by the CRLB is {efficiency['mse_over_known_variance'][1]:.5f}. At 30 dB integrated SNR, phase RMS is {efficiency['measured_phase_rms_degrees']:.5f}° against {efficiency['phase_crlb_rms_degrees']:.5f}° predicted. Under that stipulated covariance the filter gains {efficiency['conditional_optimal_gain_db']:.3f} dB; the real heldout SBS data instead lose sensitivity. The discrepancy is evidence against covariance transfer, not a numerical failure or proof that real processing is already CRLB-tight.</p>
<p><a href="https://arxiv.org/html/2103.09628">Xie et al.'s robust sea-clutter covariance work</a> addresses heavy tails, regularization and training limitations, but explicitly assumes independent, identically distributed signal-free secondary observations and a justified covariance structure. Those assumptions are not established by our neighboring crops. A robust estimator may help a demonstrated shared structure; citing that literature does not justify flattening every low-Doppler feature.</p>
<h2>Implementation and next boundary</h2>
<p>The new support-only GLS/REML reference has exact unit-gain, nuisance-annihilation, derivative, covariance-propagation and known-truth tests. The complete support suite passes 204 tests. All code is under <code>support/isar</code>; production weighting and the fixed-N/E renderer are unchanged. The source identities, complete per-crop fits, raw statistics and synthetic efficiency result are linked in the JSON report.</p>
<p>The no-yaw video and its measured losses remain the current controlled result. Real residual covariance, frequency-dependent ship scattering and phase-centre transfer still need validation before a measured full-pipeline efficiency or calibrated coherent ship-image claim is justified. These experiments specifically prevent a visually flatter spectrum or a better training likelihood from being presented as an SNR gain.</p>
</html>'''
    (args.output/'secondary-covariance.html').write_text(page)
    print(json.dumps({'report': str(args.output/'secondary-covariance.html'),
        'aggregate_gain_db': [b['aggregate_known_echo_gain_db'] for b in benchmarks]}))


if __name__ == '__main__':
    main()
