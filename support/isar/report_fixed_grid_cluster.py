"""Plot the selected returns and conditional measured phases across channels."""
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from component_phase import component_measurements
from compare_ais_phase import NAMES, COLOURS
from fit_fixed_grid_phases import prepare_fixed
from measured_phase import local_average, anchored_history, interpolate_connected, range_rate_map
from motion import C
from report_measured_scatterer_phases import load_fields, serial
from shared_echo_cluster import SharedEchoCluster


def main():
    root = Path('docs/data/isar_20261017')
    folder = root/'fixed-grid-multiple-phases'
    source = folder/'fits.json'
    fit = json.loads(source.read_text())
    if not fit['complete']:
        raise ValueError('Wait for every declared channel/start')
    first, last = fit['interval_s']
    geometry = Path('docs/data/calibration_20261016_sites')
    fields = load_fields(Path('build/isar-20261015/full-fields'), geometry)
    bands, partitions, _ = prepare_fixed(fields, geometry, first, last, stationary=True)
    grid = np.arange(first+.2, last-.199, .05)
    labels = fit['labels']
    points = [dict(name=name, channels=[]) for name in labels]
    diagnostics = []
    errors = []
    for c, (band, entry, part) in enumerate(zip(bands, fit['independent'], partitions)):
        with np.load(folder/'observations'/f'channel-{c}.npz') as saved:
            for key in ('value', 'row_time', 'row_information', 'information'):
                np.testing.assert_array_equal(saved[key], band.field.data[key])
        model = SharedEchoCluster([band], fit['selected_origins_samples'], origin_radius=fit['origin_radius_samples'])
        for trial in entry['trials']:
            r, _ = model.evaluate(np.asarray(trial['parameters']))
            errors.append(abs(r@r*model.energy-trial['cost'])/trial['cost'])
        best = entry['trials'][entry['best_index']]
        local = np.asarray(best['parameters'])[model.indices[0]]
        cluster = model.clusters[0]
        measured = component_measurements(cluster, local)
        information = cluster.evaluate(local)['information']
        scale = np.sqrt(np.diag(information))
        normalized = information/scale[:, None]/scale[None, :]
        eigen = np.linalg.eigvalsh(normalized)
        covariance = None
        if eigen[0]>eigen[-1]*1e-10:
            covariance = np.linalg.solve(normalized, np.eye(len(scale)))/scale[:, None]/scale[None, :]
        diagnostic = dict(name=NAMES[c], bounds=best['bound_indices'],
            projected_gradient=best['maximum_projected_gradient'],
            brightness_condition=best['active_brightness_condition'][0],
            component_to_sum_energy_ratio=best['component_to_sum_energy_ratio'][0],
            normalized_information_eigenvalues=eigen, components=[])
        value = cluster.evaluate(local)
        k = len(band.knots)
        parts = np.stack([value['design'][:, s*k:(s+1)*k]@value['amplitude'][s*k:(s+1)*k]
            for s in range(len(labels))])[:, band.selected]
        beta = band.nuisance.solve(band.field.y.ravel()[band.selected]-parts.sum(axis=0))
        static = (band.design@beta)[band.selected]
        diagnostic['moving_and_stationary_to_sum_energy_ratio'] = float((np.sum(abs(parts)**2)+np.sum(abs(static)**2))
            /np.sum(abs(parts.sum(axis=0)+static)**2))
        for s, record in enumerate(measured):
            p = local.reshape(len(labels), 4)[s]
            time = record['time_s']
            path = record['physical_path_change_m']
            rate = p[2]/band.scale + np.interp(band.epoch, time, np.gradient(
                np.interp(time, band.field.data['row_time'], band.known_path), time))
            epoch, z, variance = local_average(time, record['complex_coefficient'],
                record['conditional_information'], grid, .25, band.field.k, rate)
            points[s]['channels'].append(dict(name=NAMES[c], k=band.field.k,
                time_s=epoch, coefficient=z, variance=variance,
                averaged_real=z.real, averaged_imag=z.imag,
                power_to_noise=abs(z)**2/variance, fitted_path_at_grid_m=np.interp(grid, time, path),
                nominal_interband_path_m=np.interp(grid, band.field.data['row_time'], band.known_path),
                fitted_origin_samples=p[0], fitted_rate_mps=rate))
            # Per-bin innovations on unused rows remain measurements. No
            # unwrapping across a fade, or fitted noise rescaling, is used.
            innovation = np.angle(record['complex_coefficient']*np.exp(1j*band.field.k*path))
            strength = abs(record['frozen_component_prediction'])**2
            weight = strength.sum(axis=1)
            test = (part==3)&(weight>0)
            rms = np.degrees(np.sqrt(np.average(innovation[test]**2, weights=weight[test])))
            sigma = None if covariance is None else np.sqrt(covariance[4*s+2, 4*s+2])/band.scale
            h = parts[s]
            retained = float(np.vdot(band.nuisance.remove(h), band.nuisance.remove(h)).real/np.vdot(h, h).real)
            diagnostic['components'].append(dict(name=labels[s], rate_mps=rate,
                stationary_projected_signal_energy_retained=retained,
                conditional_rate_standard_deviation_mps=sigma,
                unused_conditional_phase_rms_deg=rms))
        diagnostics.append(diagnostic)
    if max(errors)>1e-9:
        raise ValueError('Saved fit objective reproduction failed')
    for point in points:
        middle = (grid>=first+(last-first)*.25)&(grid<=last-(last-first)*.25)
        strength = np.min([ch['power_to_noise'] for ch in point['channels']], axis=0)
        anchor = int(np.argmax(np.where(middle, strength, -np.inf)))
        point['anchor_time_s'] = grid[anchor]
        for ch in point['channels']:
            zero = np.zeros(len(grid))
            h = anchored_history(ch['time_s'], ch['coefficient'], ch['variance'], ch['k'], zero, zero, anchor)
            for key in ('phase_rad', 'measured_path_m'):
                values = interpolate_connected(ch['time_s'], h[key], grid)
                if np.isfinite(values[anchor]):
                    values -= values[anchor]
                ch[key] = values
            curve = ch['fitted_path_at_grid_m']
            ch['fitted_path_at_grid_m'] = curve-curve[anchor]
            del ch['coefficient']
    out = root/'multiple-scatterer-phase-plots'
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5.4), layout='constrained', sharex=True, sharey=True)
    rates = np.linspace(-3, 3, 301)
    maps = [range_rate_map(b.field.time, b.field.data['value'], b.field.data['information'], b.field.k, rates) for b in bands]
    upper = float(np.ceil(10*np.log10(max(np.max(m) for m in maps))))
    offsets = bands[0].field.offsets*C/8e6
    for c, ax in enumerate(axes):
        image = ax.imshow(10*np.log10(np.maximum(maps[c], 1e-6)), origin='lower', aspect='auto',
            extent=[offsets[0]-C/16e6, offsets[-1]+C/16e6, -3, 3], cmap='magma', vmin=0, vmax=upper,
            interpolation='nearest')
        for point in points:
            ch = point['channels'][c]
            x, y = ch['fitted_origin_samples']*C/8e6, ch['fitted_rate_mps']
            ax.plot(x, y, 's', mfc='none', mec='#00e5ff', ms=10, mew=1.2)
            ax.annotate(point['name'], (x, y), xytext=(7, 6), textcoords='offset points', color='#00e5ff', weight='bold')
        ax.axhline(0, color='#ccc', ls='--', lw=.7)
        ax.set(title=NAMES[c], xlabel='Range offset from fixed reference (m)')
    axes[0].set_ylabel('Physical excess-path rate (m/s)')
    fig.colorbar(image, ax=axes, label='Measured coherent cell power / conditional thermal noise (dB)', shrink=.75)
    fig.suptitle('163–167 s: the four selected range/rate features in every channel\n'
        'Original fixed-grid observations, including stationary energy. Boxes: independently fitted component centres; no across-band phase constraint.', fontsize=12)
    fig.savefig(out/'01-selected-points.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(points), 4, figsize=(20, 14), layout='constrained')
    for s, point in enumerate(points):
        nine = point['channels'][2]
        for c, (ch, colour) in enumerate(zip(point['channels'], COLOURS)):
            axes[s, 0].plot(grid, np.degrees(ch['phase_rad']), '.-', color=colour, ms=2, lw=1, label=NAMES[c])
            axes[s, 1].plot(grid, ch['measured_path_m'], '.-', color=colour, ms=2, lw=1)
            axes[s, 1].plot(grid, ch['fitted_path_at_grid_m'], color=colour, ls=':', lw=.8, alpha=.7)
            if c!=2:
                axes[s, 2].plot(grid, 100*(ch['measured_path_m']-nine['measured_path_m']), color=colour, lw=1.2)
                nominal = ch['nominal_interband_path_m']-nine['nominal_interband_path_m']
                nominal -= np.interp(point['anchor_time_s'], grid, nominal)
                axes[s, 2].plot(grid, 100*nominal, color=colour, ls='--', lw=.8)
            axes[s, 3].plot(ch['time_s'], 10*np.log10(np.maximum(ch['power_to_noise'], 1e-9)), color=colour, lw=1)
        axes[s, 0].set_ylabel(point['name']+' · phase change (degrees)')
        axes[s, 1].set_ylabel('Measured path change (m)')
        axes[s, 2].set_ylabel('Band minus Nine (cm)')
        axes[s, 3].set_ylabel('Conditional power / noise (dB)')
        axes[s, 3].axhline(10*np.log10(4), color='#777', ls=':', lw=.8)
        for ax in axes[s]:
            ax.axvline(point['anchor_time_s'], color='#aaa', ls=':', lw=.6)
            ax.set_xlim(first, last)
            ax.set_xlabel('Capture time (s)')
    for ax, title in zip(axes[0], ['Measured RF phase, separate wavelengths', 'Solid: measurements; dotted: own quadratic fit',
            'Dashed: nominal interband geometric difference', 'Fades are not phase connected']):
        ax.set_title(title, fontsize=10)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[2, 1].text(.03, .08, 'Ten C: only 3.2% survives stationary projection;\nnot accepted as a moving-point association',
        transform=axes[2, 1].transAxes, fontsize=8, bbox=dict(facecolor='white', edgecolor='#aaa', alpha=.9))
    fig.suptitle('Independent multiple-return phase measurements across all five channels\n'
        '250 ms complex averages, every 50 ms. Other returns and a constant stationary profile are subtracted using frozen training coefficients.\n'
        'Constant phase origins only; no measured slope/curvature is removed. These conditional diagnostics accompany the unchanged raw-phase plots.', fontsize=12)
    fig.savefig(out/'02-measured-trajectories.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), layout='constrained')
    differences = []
    for ax, s in zip(axes, (0, 2, 3)):
        d = np.array([a['measured_path_m']-b['measured_path_m'] for a, b in zip(points[s]['channels'], points[1]['channels'])])
        finite = np.isfinite(d)
        good = np.all(finite, axis=0)
        available = finite.sum(axis=0)
        possible = (available>=2)&finite[2]
        if possible.any():
            best = possible&(available==available[possible].max())
            anchor = int(np.flatnonzero(best)[len(np.flatnonzero(best))//2])
            d -= d[:, anchor, None]
            for v, colour, name in zip(d, COLOURS, NAMES):
                ax.plot(grid, 100*v, color=colour, label=name)
            differences.append(dict(pair=labels[s]+'-B', anchor_time_s=grid[anchor],
                common_duration_s=good.sum()*.05,
                phase_connected_at_origin=finite[:, anchor],
                interband_rms_against_nine_cm=100*np.sqrt(np.mean((d[:, good]-d[2, good])**2, axis=1))) if good.any()
                else dict(pair=labels[s]+'-B', anchor_time_s=grid[anchor], common_duration_s=0.,
                    phase_connected_at_origin=finite[:, anchor]))
            if not finite[:, anchor].all():
                missing = ', '.join(name for name, valid in zip(NAMES, finite[:, anchor]) if not valid)
                ax.text(.03, .95, 'No connected phase for '+missing, va='top', transform=ax.transAxes, fontsize=9)
        else:
            ax.text(.1, .5, 'No common five-band connected segment', transform=ax.transAxes)
        ax.set(title=labels[s]+' minus B', xlabel='Capture time (s)', ylabel='Within-band measured path-change difference (cm)', xlim=(first, last))
        ax.axhline(0, color='#aaa', lw=.6)
    axes[1].legend(ncol=3, fontsize=8)
    fig.suptitle('Translation-suppressed differences between selected returns\n'
        'A slowly changing rigid orientation would favour similar changes across bands for correctly associated material points. These data do not force that agreement.', fontsize=12)
    fig.savefig(out/'03-between-points.png', dpi=160)
    plt.close(fig)
    source_paths = [source, Path(__file__), Path('support/isar/component_phase.py'), Path('support/isar/test_component_phase.py')]
    snapshots = out/'sources'
    snapshots.mkdir(exist_ok=True)
    for path in source_paths:
        if path.suffix == '.py':
            shutil.copyfile(path, snapshots/path.name)
    result = dict(accepted_calibration=False, production_changed=False, interval_s=[first, last],
        time_s=grid, points=points, diagnostics=diagnostics, between_point_differences=differences,
        reproduced_objectives=len(errors), maximum_relative_cost_error=max(errors),
        original_observations_reproduced=True,
        uncertainty_scope='Conditional on the four-component model, stationary profile and known averaged range covariance. Thermal Fisher rate errors marginalize all four component envelope/constant-phase/path coordinates and active real brightness. These are not model-error or association confidence intervals. Overlapping 250 ms averages are correlated. Selected cells are not identified material hull points; D is retained as an independent neighbouring return, not a multipath model.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (out/'measurements.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False))
    rows = ''
    for d in diagnostics:
        for component in d['components']:
            sigma = component['conditional_rate_standard_deviation_mps']
            retained = component['stationary_projected_signal_energy_retained']
            rows += f'<tr><td>{d["name"]}</td><td>{component["name"]}</td><td>{component["rate_mps"]:+.4f}</td><td>{"unidentified" if sigma is None else f"{sigma:.4f}"}</td><td>{component["unused_conditional_phase_rms_deg"]:.1f}</td><td>{100*retained:.2f}%</td></tr>'
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Multiple-scatterer phase plots</title>
<style>body{{font:16px/1.55 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{background:#eef4f8;padding:16px}}td,th{{border:1px solid #ddd;padding:7px}}table{{border-collapse:collapse}}</style>
<h1>Selected points and their measured phase trajectories</h1>
<p class="note">Multipath modelling remains stopped. Start with the <a href="../phase-trajectories-review/index.html">unchanged raw measurements: ten selections in four matching windows</a>. This follow-up compares four range-resolved returns during 163–167 s, separating a time-constant stationary profile and the other selected returns before interrogating each phase. Every channel and every return has independent rate and curvature; similarity is tested, never imposed.</p>
<p class="note"><b>What the plots show:</b> B retains the clearest common bulk motion. A is too intermittent for a connected five-channel comparison. C is especially inconsistent in Ten: its selected component is almost stationary, with only 3.2% of its fitted energy independent of the static profile. That bright conditional curve is not accepted as a moving material point. D has a shared trend in the first four channels and a different ABC trend. These are measurements and association diagnostics, not a conclusive antenna or ship-motion solution.</p>
<h2>Which cells are being compared?</h2><a href="01-selected-points.png"><img src="01-selected-points.png" alt="Four selected returns marked in all five original range/rate maps"></a>
<p>The boxes mark independently fitted envelope/rate centres initialized at the existing selected cells A/B/C/D. They are local return labels, not proven material hull points. D is an independent neighbouring return needed to avoid contaminating the other measurements; no interpretation of its physical path is fitted. The dashed line is stationary zero path rate. This selection panel uses the original fixed-grid observations before the new stationary or neighbouring-return subtraction.</p>
<h2>Measured trajectories across channels</h2><a href="02-measured-trajectories.png"><img src="02-measured-trajectories.png" alt="Measured phase, physical path change, interband differences and fade strength for four selected returns"></a>
<p>RF phases have different slopes simply because the wavelengths differ; physical path change in metres is the appropriate interband comparison. Solid curves come from complex measurements, dotted curves from each channel's own quadratic fit. Only constant phase origins are removed. Lines stop at low-SNR phase breaks. Nominal geometric interband differences are dashed references, not slope corrections fitted to align the data.</p>
<h2>Do different selected points have the same relative change?</h2><a href="03-between-points.png"><img src="03-between-points.png" alt="Measured A minus B, C minus B and D minus B path-change differences in all five bands"></a>
<p>Subtracting B within each band suppresses shared translation. For consistently associated material points with slowly changing orientation, the remaining interband differences should be small. Significant residual disagreements are retained. Shared bulk motion alone does not establish that two resolution cells observe the same material scatterer. No fitted transmitter locations or phase screens explain away those disagreements.</p>
<h2>Independent rate measurements</h2><table><tr><th>Channel</th><th>Return</th><th>Physical midpoint path rate (m/s)</th><th>Conditional thermal σ (m/s)</th><th>Unused 17 ms phase innovation RMS (degrees)</th><th>Fitted signal energy surviving static projection</th></tr>{rows}</table>
<p>All ten declared optimization starts are complete and reproduce exactly from original observations (maximum relative cost error {max(errors):.3g}). No selected fit contacts its parameter bounds. Brightness conditioning and component-cancellation diagnostics are retained in the numerical file. The stationary design has condition one, and is defined only over actual observed cells, avoiding the earlier unobserved-delay extrapolation failure.</p>
<p>The deblended phase is conditional: neighbours, brightness and stationary coefficients are estimated using training rows; all are frozen when evaluating unused rows. A synthetic three-return test recovers the phases against a 100-times-stronger arbitrary static profile. Another test rotates only an unused component by 0.31 rad and verifies that the measured phase changes by exactly that amount while the fitted predictions remain bit-identical. This guards against inadvertently plotting a fitted curve as a measurement. It does not prove that four components completely describe the real ship.</p>
<p>The displayed conditional power/noise does not include uncertainty in the frozen subtraction. A large fitted amplitude can oppose a static coefficient when its phase rate is nearly zero. The retained-information column exposes that ambiguity; such a component is not a precise moving-target phase observation just because its conditional power is high. The full moving-plus-stationary cancellation ratio is also stored, in addition to the moving-only condition diagnostics.</p>
<p><a href="measurements.json">All complex averages, phase/path samples, fit diagnostics and provenance</a> · <a href="../fixed-grid-phase-review/tests.log">156 operator regression tests</a> · <a href="component-tests.log">Two additional measurement-isolation tests</a> · <a href="../fixed-grid-phase-review/index.html">Stationary separation and noise controls at other epochs</a> · <a href="../index.html">Analysis archive</a></p></html>''')
    print(out/'index.html', flush=True)


if __name__ == '__main__':
    main()
