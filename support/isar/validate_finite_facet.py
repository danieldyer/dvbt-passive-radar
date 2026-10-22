"""Compare material-point motion with a finite coherent facet, independently.

The geometry comes from the declared AIS-near prior. A facet tangent to an
equal-path contour at 160 s is an explicit synthetic hypothesis, not a fitted
hull or a claim that the actual flash is a specular surface. The calculation
checks whether such a surface can produce the measured phase-curvature scale
without fast translation or independent band phase histories.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from finite_facet import Facet, C, geometry, field, local_signature, stationary_coordinates, path, backproject
from recovery_experiment import NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('docs/data/isar_20261018/native-spatial-validation'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    prior = json.loads((root/'ais-near-shared-point-4s/analysis.json').read_text())
    plan = json.loads((root/'baseline-plan.json').read_text())
    state = np.asarray(prior['prior_mean_state_enu'])
    rotation, rx = np.asarray(plan['ecef_to_enu']), np.asarray(plan['receiver_ecef_m'])
    tx = (np.asarray(plan['transmitters_ecef_m'])-rx)@rotation.T
    rf = np.asarray(plan['rf_hz']); rx = np.zeros(3)
    g, _ = geometry(state[0], tx[2], rx)
    tangent = np.array([g[1], -g[0], 0.]); tangent /= np.linalg.norm(tangent)
    spin = .22678571428571473
    facet = Facet(state[0], state[1], tangent, np.radians(spin), 300.)
    times, offsets = np.linspace(-6., 6., 601), np.arange(-120., 120.1, 2.)
    profiles, signatures, histories, convergence = [], [], [], []
    observed = json.loads((root/'ais-near-brightness-4s/analysis.json').read_text())
    for c in range(5):
        values = field(facet, times, offsets, tx[c], rx, rf[c])
        finer = field(facet, times[::10], offsets, tx[c], rx, rf[c], quadrature=256)
        relative = np.linalg.norm(values[::10]-finer)/np.linalg.norm(finer)
        if relative > 1e-9: raise AssertionError('Finite-line quadrature did not converge')
        profiles.append(values); convergence.append(float(relative))
        signature = local_signature(facet, tx[c], rx); signatures.append(signature)
        q = stationary_coordinates(facet, times, tx[c], rx)
        centre = facet.points(times, [0.])[:, 0]
        optical = path(centre+q[:, None]*facet.frame(times), tx[c], rx)
        centre_path = path(centre, tx[c], rx)
        # The existing fixed-range chirp fit measures absolute optical path;
        # its final parameter multiplies ((t-epoch)/half_duration)^2.
        record = observed['channels'][c]
        with np.load(root/'ais-near-brightness-4s'/f'channel-{c}-histories.npz') as saved:
            duration = saved['time_s'][-1]-saved['time_s'][0]
        measured = 2*record['fading']['parameters'][3]/(duration/2)**2
        centre_index = len(offsets)//2
        complex_history = values[:, centre_index]
        path_residual = -np.unwrap(np.angle(complex_history))*C/(2*np.pi*rf[c])
        near = abs(times) < 1.25
        finite_curvature = 2*np.polynomial.polynomial.polyfit(times[near], path_residual[near], 2)[2]
        exact_curvature = 2*np.polynomial.polynomial.polyfit(times[near], (optical-centre_path)[near], 2)[2]
        signature.update(name=NAMES[c], rf_hz=float(rf[c]),
            measured_independent_flash_curvature_mps2=float(measured),
            finite_facet_fixed_cell_phase_curvature_mps2=float(finite_curvature),
            exact_stationary_path_extra_curvature_mps2=float(exact_curvature))
        histories.append(dict(q=q, path_residual=path_residual, optical_residual=optical-centre_path,
            strength=abs(complex_history)**2))
        print(NAMES[c], signature, 'quadrature error', relative, flush=True)
    profiles = np.asarray(profiles)
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout='constrained')
    for c, history in enumerate(histories):
        color = f'C{c}'
        axes[0, 0].plot(times+160, history['q'], color=color, label=NAMES[c])
        axes[0, 1].plot(times+160, 10*np.log10(np.maximum(history['strength'], 1e-12)), color=color)
        relative = history['path_residual']-np.interp(0., times, history['path_residual'])
        optical = history['optical_residual']-np.interp(0., times, history['optical_residual'])
        axes[1, 0].plot(times+160, relative, color=color)
        axes[1, 0].plot(times+160, optical, '--', color=color, alpha=.5)
    axes[0, 0].axhspan(-150, 150, color='#ddd', alpha=.4)
    axes[0, 0].set(ylabel='Stationary coordinate along material facet (m)', ylim=(-450, 450))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(ylabel='Fixed-cell synthetic power (dB, total facet amplitude = 1)', ylim=(-50, 0))
    axes[1, 0].set(ylabel='Optical path relative to translating origin (m)', ylim=(-6, 2),
        title='Solid: integrated finite line; dashed: exact stationary point')
    x = np.arange(5)
    axes[1, 1].bar(x-.2, [s['measured_independent_flash_curvature_mps2'] for s in signatures], .4, label='Measured independent chirp, 158–162 s')
    axes[1, 1].bar(x+.2, [s['finite_facet_fixed_cell_phase_curvature_mps2'] for s in signatures], .4, label='Synthetic finite facet, central 2.5 s')
    axes[1, 1].set(xticks=x, xticklabels=NAMES, ylabel='Optical phase curvature (m/s²)')
    axes[1, 1].legend(fontsize=8)
    for ax, c in zip(axes[2], [2, 4]):
        power = abs(profiles[c])**2
        ax.imshow(10*np.log10(np.maximum(power/power.max(), 1e-6)), origin='lower', aspect='auto',
            extent=[offsets[0], offsets[-1], 154., 166.], vmin=-35, vmax=0, cmap='magma')
        ax.set(title=f'{NAMES[c]}: independent scalar simulation', xlabel='Excess path relative to material origin (m)', ylabel='Capture time (s)')
    for ax in axes[:2].ravel():
        if ax is not axes[1, 1]: ax.set_xlabel('Capture time (s)')
    fig.suptitle('A material surface can produce a fast-moving optical glint\nConstant centre velocity; one slow rigid rotation; no multipath or fitted phase screen')
    fig.savefig(out/'facet-phase-control.png', dpi=160); plt.close(fig)
    east, north = np.arange(-130., 130.1, 3.), np.arange(-180., 180.1, 3.)
    e, n = np.meshgrid(east, north); pixels = np.c_[e.ravel(), n.ravel(), np.zeros(e.size)]
    images = backproject(facet, times, offsets, profiles, pixels, tx, rx, rf)
    wrong = backproject(facet, times, offsets, profiles, pixels, tx, rx, rf, flatten_glint=True)
    still = backproject(facet, times, offsets, profiles, pixels, tx, rx, rf, spin_radps=0.)
    fig, axes = plt.subplots(2, 3, figsize=(13, 12), layout='constrained')
    all_images = [images[2], images[4], images.sum(axis=0), wrong.sum(axis=0), still.sum(axis=0)]
    titles = ['Nine: correct material motion', 'ABC: correct material motion', 'Five bands: correct material motion',
        'Five bands: flatten glint phase, then same motion', 'Five bands: omit physical rotation']
    upper = max(np.max(abs(a)**2) for a in all_images)
    for ax, value, title in zip(axes.ravel(), all_images, titles):
        power = abs(value.reshape(e.shape))**2
        im = ax.imshow(10*np.log10(np.maximum(power/upper, 1e-8)), origin='lower',
            extent=[east[0], east[-1], north[0], north[-1]], cmap='magma', vmin=-35, vmax=0)
        hull = np.array([-150., 150.])[:, None]*tangent
        ax.plot(hull[:, 0], hull[:, 1], '-', color='cyan', lw=.8)
        ax.set(title=title, xlabel='East at midpoint (m)', ylabel='North at midpoint (m)', aspect='equal')
    axes[1, 2].axis('off')
    axes[1, 2].text(0, .95, 'All panels use the same fixed colour scale.\nCyan: actual synthetic material line.\n\nThe surface is deliberately coherent and uniform.\nNo real radar image is reconstructed here.\n\nThe wrong controls alter only the adjoint or phase.\nThey are not processing recommendations.', va='top', fontsize=11, linespacing=1.6)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Power relative to common maximum (dB)', shrink=.6)
    fig.suptitle('Independent exact-path backprojection of a continuous finite surface\nKnown-motion control: not a blind motion-recovery or geographic-localisation test')
    fig.savefig(out/'facet-image-control.png', dpi=150); plt.close(fig)
    result = dict(scope=__doc__, complete=True, actual_ship_focus_proved=False,
        midpoint_s=160., centre_state_enu=state.tolist(), tangent_enu=tangent.tolist(),
        tangent_heading_deg=float(np.degrees(np.arctan2(tangent[0], tangent[1]))%360),
        ais_course_deg=float(np.degrees(np.arctan2(state[1, 0], state[1, 1]))%360),
        assumed_spin_counterclockwise_degps=spin, length_m=facet.length_m,
        scalar_scattering_approximation='Uniform horizontal material line, exact spherical propagation, ideal rectangular occupied spectrum; not full electromagnetic physical optics',
        covariance_or_noise_added=False, quadrature_relative_errors=convergence, channels=signatures,
        combined_peak_power_by_control={name:float(np.max(abs(a)**2)) for name, a in
            [('correct_material_motion', images.sum(axis=0)), ('flatten_glint', wrong.sum(axis=0)), ('omit_rotation', still.sum(axis=0))]})
    (out/'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez(out/'synthetic-profiles.npz', time_s=times+160, range_offsets_m=offsets, profiles=profiles,
        pixels_east_north_up_m=pixels, correct_images=images, glint_flattened_images=wrong,
        omitted_rotation_images=still)
    (out/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Finite-facet physical control</title>
<style>body{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:0 24px}img{width:100%}.note{padding:16px;background:#eef6fa}code{white-space:normal}</style>
<h1>Can an extended surface explain the flash's apparent acceleration?</h1>
<p class="note">A 300 m coherent material line, translating at the AIS velocity and rotating at 0.227 degrees/s, produces a stationary optical reflection point sweeping at about 65 m/s and about −0.51 m/s² of apparent path curvature. This is a falsifiable physical explanation for the scale of the measured flash curvature. It does not establish that the real flash is this surface, recover unknown motion, or prove a focused ship.</p>
<p>The line is tangent to the Ornata equal-path contour at 160 s, about five degrees from the AIS course. Course is not heading. Length, orientation and rotation here are declared synthetic hypotheses. Exact two-leg spherical paths are integrated coherently over the material line. Its centre has zero acceleration. No multipath, per-time complex reflectivity, phase fitting, noise or instrument correction is involved. Range compression uses an independently evaluated ideal 6.65625 MHz sinc response, rather than the production waveform kernels.</p>
<img src="facet-phase-control.png" alt="Finite-surface glint motion, brightness and phase compared with measured chirp curvature">
<p>For local tangent coordinate q, write L(q,t) ≈ Lc(t)+a(t)q+Hq²/2. Stationary phase gives qg ≈ −a/H and Lg ≈ Lc−a²/(2H). Near a=0, the extra acceleration is −(da/dt)²/H. Here H ≈ 0.0001203/m; finite-path differentiation and independent quadrature verify the signs and scale. The Ridge transmitter shifts the stationary coordinate by about 56 m despite illuminating the same material surface. Finite edges and frequency-dependent diffraction make the integrated field depart from the infinite-surface stationary-point approximation.</p>
<img src="facet-image-control.png" alt="Known-motion exact backprojection and deliberately wrong phase/motion controls">
<p>A point-target autofocus that flattens this integrated optical phase would remove part of the surface's physical scattering history. It must not automatically be interpreted as a common ship-translation correction. The comparison above uses known synthetic motion and the observed synthetic profiles, not an image of the assumed line drawn as a substitute for backprojection. Cyan marks truth for validation only.</p>
<p>The scalar line omits vector polarisation, shadowing, hull curvature, vertical extent and frequency-dependent material response. It is a mechanism test, not a fitted electromagnetic ship model. The <a href="analysis.json">numerical results</a> and <a href="synthetic-profiles.npz">synthetic profiles/images</a> are retained. Tools: <code>support/isar/finite_facet.py</code>, <code>validate_finite_facet.py</code> and <code>test_finite_facet.py</code>.</p>
<p>Relevant prior art: <a href="https://repository.gatech.edu/bitstreams/f34aefd8-589b-4fdc-a8e6-63f5d49355b9/download">Georgia Tech near-field scattering experiments</a> discuss range-dependent interference and phase-centre glint from extended targets. The specific stationary-phase calculation and numerical checks above are our own control.</p></html>''')


if __name__ == '__main__':
    main()
