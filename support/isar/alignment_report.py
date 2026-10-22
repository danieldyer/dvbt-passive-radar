"""Publish the numerical gates and the measured channel-alignment experiment."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    trial = json.loads((args.trial/'alignment.json').read_text())
    audit = json.loads(args.audit.read_text())
    stability = json.loads((args.trial/'stability.json').read_text())
    refined = json.loads((args.trial/'subpixel.json').read_text()) if (args.trial/'subpixel.json').exists() else None
    rate = json.loads((args.trial/'path-rate.json').read_text()) if (args.trial/'path-rate.json').exists() else None
    rate_stability = (json.loads((args.trial/'path-rate-stability.json').read_text())
                      if (args.trial/'path-rate-stability.json').exists() else None)
    motion_trials = []
    for name in ('coherent-motion.json', 'profiled-coherent-motion-refined.json'):
        path = args.trial/name
        if path.exists():
            motion_trials.append((name, json.loads(path.read_text())))
    args.output.mkdir(parents=True, exist_ok=True)
    stages = [(s['name'], s['fit'], s['validation']) for s in trial['stages']]
    if refined:
        stages.append(('Shared subpixel positions + alignment', refined['fit'], refined['after']))
    if rate:
        stages.append(('Constant path-rate trial (not adopted)', rate['fit'], rate['after']))
    names = ['Seven', 'SBS', 'Nine', 'Ten (reference)', 'ABC']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    labels = ['Nominal', 'Phase / gain', 'Range + phase']+(['Subpixel'] if refined else [])+(['Path-rate trial'] if rate else [])
    energy0 = stages[0][2]['matched_energy']
    gains = [10*np.log10(v['matched_energy']/energy0) for _, _, v in stages]
    axes[0].bar(labels, gains, color=['#929ba8', '#25659b', '#4b8fbd', '#df7924', '#9c6a96'][:len(labels)])
    axes[0].set(ylabel='Held-out matched-energy gain vs nominal (dB)', title='Conditional shared-scene coherent fit')
    axes[0].tick_params(axis='x', rotation=15)
    x = np.arange(5)
    for i, (label, fit, validation) in enumerate(stages):
        axes[1].plot(x, [c['prediction_phase_rms_degrees'] for c in validation['channels']], 'o-', label=label)
    axes[1].set(xticks=x, xticklabels=names, ylabel='One-second prediction phase RMS (degrees)',
                title='Residuals include unresolved scattering and noise')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'channel-alignment-metrics.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for half in stability['trials']:
        f = half['fit']
        axes[0].plot(x, f['gain_phase_degrees'], 'o-', label='Fit '+['first', 'second'][half['training_half']]+' 15 s')
    axes[0].set(xticks=x, xticklabels=names, ylim=(-190, 190), ylabel='Wrapped channel phase (degrees)', title='Constant phase correction is not yet transferable')
    axes[0].legend(fontsize=9)
    with np.load(args.trial/'training-support.npz') as f:
        z = 10*np.log10(np.maximum(f['power'], 1e-20))
        axes[1].imshow(z, extent=[f['along_m'][0], f['along_m'][-1], f['port_m'][0], f['port_m'][-1]],
            origin='lower', aspect='equal', cmap='viridis', vmin=z.max()-30, vmax=z.max())
        points = f['points_body_m']
    axes[1].scatter(points[:, 0], points[:, 1], marker='+', color='white', s=60, label='Training-selected grid points')
    if refined:
        adjusted = np.array(refined['fit']['points_body_m'])
        axes[1].scatter(adjusted[:, 0], adjusted[:, 1], facecolors='none', edgecolors='#ffae4c', s=45, label='Shared subpixel fit')
    axes[1].set(xlabel='Forward coordinate (m)', ylabel='Port coordinate (m)', title='Training-only support; body origin is AIS antenna prior')
    axes[1].legend(fontsize=8, loc='upper left')
    fig.savefig(args.output/'channel-alignment-stability.png', dpi=170)
    plt.close(fig)
    rows = ''.join(f'<tr><td>{escape(label)}</td><td>{fit.get("converged", "fixed")}</td>'
        f'<td>{fit.get("maximum_projected_gradient", 0):.3g}</td><td>{validation["conditional_coherent_loss_db"]:.3f}</td>'
        f'<td>{100*(1-validation["prediction_error"]/validation["stationary_only_error"]):.2f}%</td></tr>'
        for label, fit, validation in stages)
    numerical = audit['known_range_phase_recovery']
    forward = audit['known_three_dimensional_scene']
    probes = ''.join(f'<tr><td>{escape(r["support"])}</td><td>{r["rotation"]}</td>'
        f'<td>{r["phase_rms_degrees_before_common_path_yaw"]:.3f}°</td>'
        f'<td>{r["phase_rms_degrees_after_common_path_yaw"]:.3f}°</td></tr>' for r in audit['actual_geometry_roll_pitch_sensitivity'])
    def link(path, text):
        return f'<a href="{escape(os.path.relpath(Path(path).resolve(), args.output.resolve()))}">{escape(text)}</a>'
    refinement_text = ('<p>The subpixel fit is still running.</p>' if refined is None else
        f'<p>The shared subpixel fit converged numerically: {refined["fit"]["converged"]}; maximum projected gradient '
        f'{refined["fit"]["maximum_projected_gradient"]:.3g}; coordinate-bound hit: {refined["fit"]["position_bound_hit"]}. '
        'This is a fixed-height, fixed-motion local fit. It does not establish true heading or hull centroid.</p>')
    rate_text = ''
    if rate:
        rate_text = (f'<p>The subsequent constant path-rate trial reaches conditional loss '
            f'{rate["after"]["conditional_coherent_loss_db"]:.3f} dB. Its range, RF phase and symbol Doppler '
            'are driven by the same linear excess-path perturbation; the isolated recovery and derivative tests pass. '
            'Its small interleaved improvement does not establish a physical transmitter drift.</p>')
        if rate_stability:
            rate_text += ('<p>The historical half-aperture path-rate scores were '+
                ', '.join(f'{t["validation"]["conditional_coherent_loss_db"]:.2f}' for t in rate_stability['trials'])+
                ' dB. These used the stationary-coefficient extrapolation now identified as unreliable; '
                'they are not sufficient evidence to reject a physical rate correction. No rate calibration is adopted. '
                'The newer geometry report contains the corrected validation method and motion/location controls.</p>')
    motion_text = ''
    if motion_trials:
        lines = []
        for name, model in motion_trials:
            fitted = model['coherent_motion']
            phases = ', '.join(f'{v["prediction_phase_rms_degrees"]:.1f}°' for v in fitted['after'])
            label = 'Channel gains profiled at each motion candidate' if model.get('channel_gains_profiled') else 'Channel gains held fixed during motion step'
            loss = ''
            if 'heldout_cross_real_imag' in fitted['after'][0]:
                scores = fitted['after']
                cross = sum(complex(*s['heldout_cross_real_imag']) for s in scores)
                energy = sum(s['heldout_prediction_energy'] for s in scores)
                upper = sum(s['independently_fitted_test_energy'] for s in scores)
                loss = f'{-10*np.log10(abs(cross)**2/energy/upper):.3f} dB'
            lines.append(f'<tr><td>{label}</td><td>{fitted["objective"]:.8f}</td>'
                f'<td>{fitted["converged"]}</td><td>{fitted["maximum_kkt_residual"]:.3g}</td><td>{loss}</td><td>{phases}</td></tr>')
        motion_text = ('<h2>5. Shared physical motion and coherent scene</h2>'
            '<p>The next block refits path and yaw against one common complex scene, with roll/pitch and physical antenna locations fixed. '
            'The second variant also optimizes channel gains at each motion candidate and eliminates their nuisance directions from the motion information matrix. '
            'The nested likelihood gradient and information projection pass independent synthetic checks.</p>'
            '<table><tr><th>Variant</th><th>Objective</th><th>Strict numerical gate</th><th>KKT residual</th><th>Conditional coherent loss</th><th>Held-out phase RMS: Seven, SBS, Nine, Ten, ABC</th></tr>'+
            ''.join(lines)+'</table>'
            '<p>The fully profiled fit remains above the predeclared 10⁻⁷ KKT threshold even after tighter inner-gain convergence and a restart. '
            'Its residual phases also remain substantial. This is not an accepted physical autofocus result. The conditional loss does not bound total image SNR.</p>'
            '<figure><img src="registered-coherent-image.png" alt="Nominal versus registered five-channel coherent backprojection">'
            '<figcaption>Actual complex five-channel backprojection using the numerically converged fixed-gain motion block. '
            'Both panels share samples, motion and colour scale. Channel range offsets change the sampled envelope; the complex gains enter the matched-filter sum. '
            'Peak matched power increases by 2.53 dB, which is different from the fixed-dictionary prediction gain. This remains a trial image.</figcaption></figure>')
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship autofocus: numerical audit and channel alignment</title>
<style>body{{max-width:1150px;margin:35px auto;padding:0 20px;font:17px/1.55 system-ui;color:#233143;background:#fafbfc}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:9px;border-bottom:1px solid #ccd3dd;text-align:left}}h1,h2{{line-height:1.2}}.status{{padding:18px;background:#e8eef7;border-left:4px solid #25659b}}code{{font-size:14px}}a{{color:#175f9a}}</style>
<h1>Ship autofocus: numerical audit and channel alignment</h1>
<p>Updated {datetime.now(timezone.utc).isoformat()}. Recording 13 October 2026, working aperture 142–172 s.</p>
<p class="status">The numerical phase/range operator and independent delay/phase recovery checks pass. Constant channel gains improve the conditional coherent fit. <a href="geometry-refinement.html">The newer joint antenna/motion analysis</a> adds converged controls, actual corrected-geometry backprojection, and an important correction to the stationary-background extrapolation in the older transfer tests below. No physical calibration is accepted.</p>
<h2>1. Numerical checks before interpreting real data</h2>
<p>The independent forward generator uses direct Euclidean path lengths, Fourier pulse synthesis and Gauss–Legendre symbol integration. Its relative complex difference from the fitted model is {forward['relative_complex_forward_error']:.3g}; the finite interpolation model explains this small floor. At the correct three-dimensional motion, held-out phase RMS is {forward['heldout_phase_rms_degrees']:.3g}°.</p>
<p>With injected range offsets, channel phases and strong stationary clutter, the optimizer recovers range within {numerical['maximum_range_error_m']:.3g} m and relative phase within {numerical['maximum_relative_phase_error_degrees']:.3g}°. Untouched two-second blocks have residual error {numerical['heldout_error_fraction']:.3g} of the stationary-only error. These are deterministic synthetic checks, not real-data accuracy claims.</p>
<p>The Vulkan self-test was rerun: maximum GPU/double complex error per unit information 3.48419×10⁻⁵; independent moving/static point error 1.85604×10⁻⁵; maximum coherent phase error 0.000617562°. The support test suite also checks exact averaging/whitening adjoints, analytic motion derivatives, phase/Doppler consistency, gain/range derivatives, and held-out subpixel recovery.</p>
<p>The current support suite has 61 passing tests. The original five-channel Vulkan round trip injected nonzero envelope offsets and complex channel gains: combined phase error was 0.0002915°, with complex error 1.2145×10⁻⁵. The newer geometry report adds displaced TX/RX coordinates and another independently synthesized GPU check.</p>
<h2>2. Constant transmitter-channel range and phase residuals</h2>
<p>Ten fixes the common gain, phase and range-origin gauges. The other four channels each have one complex gain and one envelope range offset. A common complex reflectivity is fitted for every physical point. Range bias moves the envelope; carrier phase is represented by the gain, avoiding an exact duplicate parameter. Stationary nuisance responses and all gains are estimated using training bins only. The data are not normalized.</p>
<table><tr><th>Stage</th><th>Numerically converged</th><th>Projected gradient</th><th>Conditional coherent loss</th><th>Held-out error reduction vs stationary-only</th></tr>{rows}</table>
<img src="channel-alignment-metrics.png" alt="Held-out coherent energy and prediction phase residuals">
<p>The dB comparisons refer to prediction in this specific scattering dictionary, not the full image's theoretical SNR. Independent per-band reflectivities provide the conditional comparison. The phase RMS includes noise, unresolved reflectors, glint and motion error; it is not an isolated motion-error measurement.</p>
{refinement_text}
{rate_text}
<h2>3. Calibration stability and support</h2>
<img src="channel-alignment-stability.png" alt="Phase transfer and scattering support">
<p>Historical diagnostic: these half-aperture scores extrapolated poorly constrained stationary coefficients. They must not alone be interpreted as channel-calibration failure. The <a href="geometry-refinement.html">corrected transfer test</a> profiles only the stationary nuisance in validation, with identical projection of the frozen target prediction, and reports information loss. Broad interleaved improvement still does not establish a unique physical calibration.</p>
<p>Support selection for the new alignment trial uses training-only backprojection. The older input motion used support selected from full-aperture images; consequently the overall chain is not yet a completely untouched end-to-end holdout. Further motion changes require a fresh outer validation split.</p>
<h2>4. Why roll and pitch remain fixed</h2>
<p>The probe below injects a 0.1° sinusoidal rotation with a ten-second period into the actual geometry. It removes constant point phases and then the best local common-path/yaw perturbation. A flat 20 m plane is almost degenerate with those existing parameters. Mixed, independently known reflector heights would carry more information, but those heights are not measured here. No roll/pitch fit is applied to the recording.</p>
<table><tr><th>Declared support</th><th>Probe</th><th>Phase RMS before path/yaw</th><th>Unabsorbed RMS</th></tr>{probes}</table>
{motion_text}
<h2>Reproducible data</h2><p>{link(args.audit, 'Numerical audit JSON')} · {link(args.trial/'alignment.json', 'Alignment fits and one-second residuals')} · {link(args.trial/'stability.json', 'First/second-half transfer')} · {link('support/isar/channel_alignment.py', 'Alignment implementation')} · {link('support/isar/test_channel_alignment.py', 'Independent recovery tests')}</p>
<p>{link(args.output/'ais-ground-track.png', 'North-up AIS track')} · {link(args.output/'analysis.html', 'Earlier motion experiments and trial videos')}</p></html>'''
    (args.output/'numerical-audit.html').write_text(page)


if __name__ == '__main__':
    main()
