"""Review constant-velocity, finite-facet and stationary-ridge controls.

The ridge diagnostic uses exact physical zero-Doppler contours predicted from
the saved motion, then compares two linear native-data backprojections with
identical coordinates, weights and band coefficients. No image registration,
local contrast normalization or geometry fitting is used in this comparison.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from finite_facet import C
from recovery_experiment import NAMES
from render_injected_point import fitted_state
from report_measured_scatterer_phases import serial


def main():
    root = Path('docs/data/isar_20261018/native-spatial-validation')
    long = root/'ais-near-three-point-30s'
    out = root/'motion-controls-review'; out.mkdir(parents=True, exist_ok=True)
    with np.load(long/'native-image/images.npz') as saved: before = dict(saved)
    with np.load(long/'stationary-control-image/images.npz') as saved: after = dict(saved)
    for key in ['east_m', 'north_m', 'information', 'gains']:
        np.testing.assert_array_equal(before[key], after[key])
    fit = json.loads((long/'render-fit.json').read_text())
    plan = json.loads((root/'baseline-plan.json').read_text())
    rx, tx, rotation = map(np.asarray, (plan['receiver_ecef_m'], plan['transmitters_ecef_m'], plan['ecef_to_enu']))
    state = np.asarray(fit['coarse_state_enu']); mid = fit['midpoint_s']
    frame = fitted_state(np.array([mid]), state, mid, rotation, rx, fit)
    east, north = before['east_m'], before['north_m']
    e, n = np.meshgrid(east, north); pixels = np.stack([e, n, np.zeros_like(e)], axis=-1)
    position = frame['position_ecef_m'][0]+pixels@frame['body_to_ecef'][0].T
    velocity = frame['velocity_ecef_mps'][0]+pixels@frame['frame_velocity'][0].T
    doppler = []
    for antenna, rf in zip(tx, plan['rf_hz']):
        g = sum((position-a)/np.linalg.norm(position-a, axis=-1)[..., None] for a in (antenna, rx))
        doppler.append(-rf/C*np.sum(g*velocity, axis=-1))
    doppler = np.asarray(doppler)
    # This is the linearly removed component, not a fitted image difference.
    component = before['complex_amplitude']-after['complex_amplitude']
    removed_individual = 8192*abs(component)**2*before['information'][:, None, None]
    removed_combined = 8192*abs(before['combined_amplitude']-after['combined_amplitude'])**2*sum(abs(before['gains'])**2*before['information'])
    removed = np.r_[removed_individual, removed_combined[None]]
    result = dict(complete=True, scope=__doc__, focused_ship_certified=False,
        original_profile_data_modified=False, original_information_denominator_retained=True,
        projected_template_information_correction_applied=False, stationary_ridge_metrics=[])
    width = 1/30
    for c in range(6):
        fd = doppler[c if c < 5 else 2]
        mask = abs(fd) < width
        a, b = before['power'][c][mask].sum(), after['power'][c][mask].sum()
        result['stationary_ridge_metrics'].append(dict(name=(NAMES+['Combined'])[c],
            predicted_midpoint_doppler_halfwidth_hz=width, pixels=int(mask.sum()),
            original_power_sum=float(a), projected_power_sum=float(b),
            projected_to_original_power_db=float(10*np.log10(b/a))))
    extent = [east[0], east[-1], north[0], north[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 9), sharex=True, sharey=True, layout='constrained')
    high = 10*np.log10(before['power'].max()); low = high-45
    for ax, power, title in zip(axes, [before['power'][5], removed[5], after['power'][5]],
            ['Observed 30 s backprojection', 'Linearly removed constant-profile component', 'After constant-profile projection']):
        im = ax.imshow(10*np.log10(np.maximum(power, 1e-30)), origin='lower', extent=extent,
            vmin=low, vmax=high, cmap='magma', interpolation='nearest')
        ax.contour(east, north, doppler[2], levels=[0.], colors=['#66ffdd'], linewidths=.8)
        ax.contour(east, north, doppler[4], levels=[0.], colors=['white'], linewidths=.8, linestyles='dashed')
        ax.set(title=title, xlabel='East from fitted point (m)', ylabel='North (m)', aspect='equal')
    fig.colorbar(im, ax=axes.tolist(), label='Coherent power / unchanged original information (dB)', shrink=.6)
    fig.suptitle('The dominant northeast ridge lies on the predicted stationary-response locus\nSolid mint: Nine zero Doppler; dashed white: ABC zero Doppler; identical fixed colour scale')
    fig.savefig(out/'stationary-ridge.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout='constrained')
    edges = np.linspace(-.4, .4, 81)
    for c, ax in enumerate(axes.ravel()):
        fd = doppler[c if c < 5 else 2].ravel()
        count = np.histogram(fd, edges)[0]
        for power, label in [(before['power'][c], 'Before'), (after['power'][c], 'After projection')]:
            summed = np.histogram(fd, edges, weights=power.ravel())[0]
            mean = np.divide(summed, count, out=np.full_like(summed, np.nan), where=count > 0)
            ax.plot((edges[:-1]+edges[1:])/2, 10*np.log10(mean), label=label)
        ax.axvline(0., color='#777', lw=.6)
        ax.set(title=(NAMES+['Combined'])[c], xlabel='Predicted pixel Doppler at aperture midpoint (Hz)', ylabel='Mean image-cell power (dB)')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('Power by physical Doppler locus: same measured data, fixed motion and coefficients\nThis is a clutter diagnostic; the unchanged denominator is not a post-projection SNR calibration')
    fig.savefig(out/'stationary-doppler-locus.png', dpi=150); plt.close(fig)
    trials = []
    for name, label in [('ais-near-constant-velocity-4s', 'Constant translation; no rotation'),
            ('ais-near-constant-translation-4s-native', 'Constant translation + constant rigid rotation'),
            ('ais-near-free-motion-4s', 'Cubic radial motion + angular acceleration')]:
        p = root/name/'analysis.json'
        if not p.exists(): continue
        d = json.loads(p.read_text())
        if d.get('complete'): trials.append((name, label, d))
    result['short_arc_models'] = {name:dict(fit=d['fit'], prediction=d['prediction']) for name, _, d in trials}
    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    x, width = np.arange(5), .24
    for j, (_, label, d) in enumerate(trials):
        ax.bar(x+(j-1)*width, [p['explained_fraction'] for p in d['prediction']], width, label=label)
    ax.set(xticks=x, xticklabels=NAMES, ylabel='Excluded complex-error reduction vs stationary-only fit', ylim=(0, .75))
    ax.legend(fontsize=9); ax.set_title('Four-second flash: original rows and frozen band calibration\nA candidate reaches its spatial bound in the simpler models; this is not unique geographic recovery')
    fig.savefig(out/'constant-velocity-models.png', dpi=160); plt.close(fig)
    facet = json.loads((root/'ais-near-native-facet-12s/analysis.json').read_text())
    result['finite_facet_fit'] = dict(fit=facet['fit'], prediction=facet['prediction'],
        accepted=False, reason='Velocity correction reaches its declared bound; systematic phase residuals remain, and the native numerical convergence gate is not passed')
    result['numerical_regression_log'] = '../regressions.log'
    (out/'analysis.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n')
    rows = ''.join(f'<tr><td>{label}</td><td>{d["fit"]["objective"]:.6f}</td><td>{d["fit"]["maximum_scaled_projected_gradient"]:.3g}</td><td>{d["fit"]["active_bounds"]}</td><td>'+
        ', '.join(f'{p["explained_fraction"]:.1%}' for p in d['prediction'])+'</td></tr>' for _, label, d in trials)
    ridge = result['stationary_ridge_metrics'][-1]['projected_to_original_power_db']
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native motion controls</title>
<style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Constant velocity, finite-surface phase and stationary leakage</h1>
<p class="note">The strong northeast ridge in the new 30-second image is predominantly stationary leakage: it follows the zero-Doppler locus predicted from the fitted physical motion. Removing only the constant native profile reduces combined image power within ±1/30 Hz of that locus by {abs(ridge):.1f} dB. A narrower north-northwest feature remains near the AIS antenna. Its identification as resolved hull structure, rather than a mixture of point-spread responses, is still unproved. The full-duration coherent ship videos remain unfinished.</p>
<h2>Constant velocity is a useful simplification on the bright short arc</h2>
<img src="constant-velocity-models.png"><table><tr><th>Four-second hypothesis</th><th>Training residual</th><th>Scaled projected gradient</th><th>Active bounds</th><th>Excluded reduction: Seven / SBS / Nine / Ten / ABC</th></tr>{rows}</table>
<p>Constant translation plus a constant physical angular rate retains almost all of the excluded-data prediction of the acceleration/jerk model on 158–162 s. These fits hold the weak transverse motion at its AIS prior and use a training-selected support; they do not establish globally unique position or motion. The simpler fits move one candidate to its spatial bound. The frozen band calibration, original symbol times, waveform responses and positive real brightness convention are unchanged. Angular rate is a physical rotation of material offsets; all image axes remain north/east.</p>
<p>The earlier <a href="../../../injection_20261018/constant-velocity-iq/index.html">recorded-IQ constant-velocity injection</a> remains the independent receiver-path control: 7.09 m position error, 3.44 mm/s velocity error and 0.00188 dB native point-focus loss at the stronger test level. The original-strength case remains about 68 m from truth, with a conditional coloured-noise bound of about 41 m. The strong control is not a substitute for weak-signal sensitivity.</p>
<h2>Do not mistake a surface's optical glint for material-point acceleration</h2>
<p>An independently integrated 300 m material line, with a constant centre velocity and 0.227°/s rotation, produces an optical stationary point sweeping at about 65 m/s and roughly −0.51 m/s² of additional optical path curvature. Exact paths, quadrature convergence, native profile derivatives and unknown-motion synthetic recovery are tested. See the <a href="../finite-facet-control/index.html">finite-surface mechanism control</a>, including known-motion backprojection and deliberately wrong point-phase flattening.</p>
<p>That mechanism is physically plausible but the particular uniform line is <strong>not accepted as a fit to the real ship</strong>. Its tested native fit reaches the +1 m/s radial-rate bound and retains systematic phase residuals. It explains only 2.5%, 0.05%, 6.1%, 23.7% and 18.8% of excluded complex error over 154–166 s. No multipath model or time-dependent phase correction has been adopted.</p>
<img src="../ais-near-native-facet-12s/prediction.png">
<h2>Why the bright image ridge had an implausible orientation</h2>
<img src="stationary-ridge.png"><img src="stationary-doppler-locus.png">
<p>For a trial material pixel, its midpoint velocity is v + ω Jx. A fixed background echo can add coherently where ∇L(x)·(v+ω Jx) ≈ 0. This generally produces a ridge across the image, not along the ship's heading. The solid and dashed curves above are computed from exact two-leg geometry and the saved motion, without fitting them to the ridge. The removed image component is the complex difference of two linear backprojections with identical pixel coordinates, weights and frozen band coefficients.</p>
<p>The control subtracts the information-weighted constant C0/C1 profile. It does not notch nearby moving-target Dopplers. It can nevertheless remove part of a slow target's waveform: a production nuisance-projected matched filter must project its template too and use the resulting template information. The images above retain the original denominator and are explicitly labelled diagnostic; their powers are not calibrated post-projection SNR or detection statistics.</p>
<p><a href="../ais-near-three-point-30s/native-image/point-images.png">All five original images and coherent sum</a> · <a href="../ais-near-three-point-30s/stationary-control-image/point-images.png">All five stationary-projected controls and coherent sum</a> · <a href="../ais-near-three-point-30s/phase-histories/phase-histories.png">Frozen-component phase histories</a>. The 30-second motion is conditionally fitted; the native checkpoint is an evaluated representative-epoch solution, not a fully polished native optimum.</p>
<p>The next image-formation check is to compare the remaining feature against native point-spread controls with the same visibility, motion and stationary projection, using the correct projected information. That must precede claiming resolved ship structure or transferring the motion across the full capture.</p>
<p><a href="analysis.json">Numerical review</a> · <a href="../regressions.log">Numerical regressions</a> · <a href="../index.html">Full spatial-validation history</a>. Source tools are under <code>support/isar</code>; captured data and production processing are unchanged by these controls.</p></html>''')
    print(json.dumps(serial(result['stationary_ridge_metrics'])), flush=True)


if __name__ == '__main__':
    main()
