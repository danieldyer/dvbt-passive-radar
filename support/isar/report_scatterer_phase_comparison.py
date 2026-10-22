"""A compact, reproducible review of measured multichannel scatterer phases.

Re-extract the declared raw-cell measurements, check the saved complex values,
and present physical RF phase, path change and within-band point differences.
No propagation, antenna, heading or common-trajectory parameters are fitted.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import COLOURS, NAMES
from motion import C
from report_measured_scatterer_phases import (
    analyse_window, load_fields, plot_selection, plot_increments, serial)


ROOT = Path('docs/data/isar_20261017')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reproduce(saved, fields):
    """Check every complex average, variance, path, phase and fade flag."""
    windows = []
    errors = {}
    checked = 0
    for old in saved['windows']:
        selections = [(p['name'], p['offset_samples']) for p in old['points']]
        fresh = analyse_window(fields, old['label'], *old['interval_s'],
            width=saved['window_width_s'], step=saved['output_step_s'], selections=selections)
        for before, after in zip(old['points'], fresh['points']):
            assert before['name'] == after['name']
            assert before['anchor_index'] == after['anchor_index']
            for a, b in zip(before['channels'], after['channels']):
                for key in ('effective_time_s', 'averaged_real', 'averaged_imag',
                        'conditional_variance', 'grid_phase_rad', 'grid_measured_path_m',
                        'grid_nominal_path_m', 'grid_short_measured_path_m',
                        'measured_increment_m', 'connected', 'increment_good'):
                    x, y = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
                    np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True)
                    finite = np.isfinite(x) & np.isfinite(y)
                    error = float(np.max(abs(x[finite]-y[finite]))) if finite.any() else 0.
                    errors[key] = max(errors.get(key, 0.), error)
                checked += 1
        windows.append(fresh)
        print(f"{old['label']}: original complex measurements reproduced", flush=True)
    return windows, dict(channel_candidate_series=checked, maximum_absolute_errors=errors)


def reproduce_conditional(saved, fields):
    """Re-evaluate saved fits and measure phases again; run no optimizer."""
    from component_phase import component_measurements
    from fit_fixed_grid_phases import prepare_fixed
    from measured_phase import local_average
    from shared_echo_cluster import SharedEchoCluster

    folder = ROOT/'fixed-grid-multiple-phases'
    fit = json.loads((folder/'fits.json').read_text())
    if not fit['complete']:
        raise ValueError('Conditional measurement fits are incomplete')
    bands, _, _ = prepare_fixed(fields, Path('docs/data/calibration_20261016_sites'),
        *fit['interval_s'], stationary=True)
    objective_errors = []
    value_error = 0.
    for c, (band, entry) in enumerate(zip(bands, fit['independent'])):
        with np.load(folder/'observations'/f'channel-{c}.npz') as archived:
            for key in ('value', 'row_time', 'row_information', 'information'):
                np.testing.assert_array_equal(archived[key], band.field.data[key])
        model = SharedEchoCluster([band], fit['selected_origins_samples'],
            origin_radius=fit['origin_radius_samples'])
        for trial in entry['trials']:
            residual, _ = model.evaluate(np.asarray(trial['parameters']))
            error = abs(residual@residual*model.energy-trial['cost'])/trial['cost']
            if error > 1e-10:
                raise ValueError('Conditional fit objective changed')
            objective_errors.append(float(error))
        best = entry['trials'][entry['best_index']]
        local = np.asarray(best['parameters'])[model.indices[0]]
        records = component_measurements(model.clusters[0], local)
        for point, record in zip(saved['points'], records):
            old = point['channels'][c]
            epoch, value, variance = local_average(record['time_s'], record['complex_coefficient'],
                record['conditional_information'], saved['time_s'], .25, band.field.k, old['fitted_rate_mps'])
            for expected, actual in ((old['time_s'], epoch), (old['averaged_real'], value.real),
                    (old['averaged_imag'], value.imag), (old['variance'], variance)):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                value_error = max(value_error, float(np.max(abs(actual-np.asarray(expected)))))
    print('Conditional flash: all 10 objectives and 20 complex histories reproduced', flush=True)
    return dict(original_observations_exact=True, objectives_reproduced=len(objective_errors),
        maximum_relative_objective_error=max(objective_errors), complex_histories_reproduced=20,
        maximum_absolute_measurement_error=value_error)


def trajectories(windows, output, *, phase):
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 12.5), layout='constrained')
    for row, window in enumerate(windows):
        points = {p['name']:p for p in window['points']}
        for col, name in enumerate(('A', 'B', 'C')):
            ax = axes[row, col]
            if name not in points:
                ax.text(.5, .5, 'No candidate selected', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            p = points[name]
            for ch, colour in zip(p['channels'], COLOURS):
                value = np.asarray(ch['grid_measured_path_m'])
                if phase:
                    value = -360*ch['rf_hz']/C*value
                ax.plot(window['time_s'], value, color=colour, lw=1.25, label=ch['name'])
            ax.axvline(p['anchor_time_s'], color='#999', ls=':', lw=.65)
            ax.axhline(0, color='#aaa', lw=.5)
            ax.set(title=f"{window['label']} · {name} · {p['offset_m']:+.0f} m",
                xlabel='Capture time (s)', ylabel='Physical RF phase change (°)' if phase else 'Excess-path change (m)',
                xlim=window['interval_s'])
    axes[0, 1].legend(ncol=3, fontsize=8)
    quantity = 'Physical RF phase' if phase else 'Measured excess-path change'
    fig.suptitle(f'{quantity} in the same capture windows, all five channels\n'
        'Saved carrier deramp restored; constant origins only. Lines stop at fades. '
        'Candidate labels are local to each window.', fontsize=12)
    name = '03-physical-phase.png' if phase else '02-path-trajectories.png'
    fig.savefig(output/name, dpi=150)
    plt.close(fig)


def differences(windows, output):
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), layout='constrained')
    comparisons = []
    for row, window in enumerate(windows):
        points = {p['name']:p for p in window['points']}
        t = np.asarray(window['time_s'])
        reference = points['B']
        for col, name in enumerate(('A', 'C')):
            ax = axes[row, col]
            ax.set(title=f"{window['label']} · {name} − B", xlabel='Capture time (s)',
                ylabel='Within-band path-change difference (cm)', xlim=window['interval_s'])
            if name not in points:
                ax.text(.5, .5, 'No candidate selected', ha='center', va='center', transform=ax.transAxes)
                continue
            delta = np.array([a['grid_measured_path_m']-b['grid_measured_path_m']
                for a,b in zip(points[name]['channels'], reference['channels'])])
            good = np.all(np.isfinite(delta), axis=0)
            entry = dict(window=window['label'], pair=name+'-B', common_phase_connection=False)
            if not good.any():
                ax.text(.5, .5, 'No shared five-band phase-connected epoch',
                    ha='center', va='center', transform=ax.transAxes)
            else:
                index = np.flatnonzero(good)
                anchor = index[len(index)//2]
                delta -= delta[:, anchor, None]
                for values, colour, band in zip(delta, COLOURS, NAMES):
                    ax.plot(t, 100*values, color=colour, lw=1.25, label=band)
                ax.axvline(t[anchor], color='#999', ls=':', lw=.65)
                disagreement = delta[:, good]-delta[2, good]
                entry.update(common_phase_connection=True, anchor_time_s=t[anchor],
                    common_first_s=t[index[0]], common_last_s=t[index[-1]], common_samples=len(index),
                    rms_difference_from_nine_cm=100*np.sqrt(np.mean(disagreement**2, axis=1)))
            ax.axhline(0, color='#aaa', lw=.5)
            comparisons.append(entry)
    axes[2, 1].legend(ncol=3, fontsize=8)
    fig.suptitle('Do different selected cells have the same relative motion across bands?\n'
        'Subtract B within each band; remove one constant at a common epoch. '
        'No slopes, curvature or phase screens fitted.', fontsize=12)
    fig.savefig(output/'04-between-scatterers.png', dpi=150)
    plt.close(fig)
    return comparisons


def conditional_flash(data, output):
    """Show the existing deblended measurements separately from raw cells."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), layout='constrained')
    t = data['time_s']
    for row, p in enumerate(data['points']):
        nine = np.asarray(p['channels'][2]['measured_path_m'], float)
        for ch, colour in zip(p['channels'], COLOURS):
            path = np.asarray(ch['measured_path_m'], float)
            axes[row, 0].plot(t, path, color=colour, lw=1.25, label=ch['name'])
            axes[row, 1].plot(t, 100*(path-nine), color=colour, lw=1.25)
        for ax in axes[row]:
            ax.axvline(p['anchor_time_s'], color='#999', ls=':', lw=.65)
            ax.set(xlabel='Capture time (s)', xlim=data['interval_s'])
        axes[row, 0].set(title=p['name']+' · measured separated component', ylabel='Excess-path change (m)')
        axes[row, 1].set(title=p['name']+' · channel minus Nine', ylabel='Interband path-change difference (cm)')
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[2, 0].text(.03, .06, 'Ten C is nearly stationary; only 3.2% survives static projection.',
        transform=axes[2, 0].transAxes, fontsize=8, bbox=dict(facecolor='white', edgecolor='#aaa'))
    fig.suptitle('163–167 s · conditional separation of four neighbouring returns\n'
        '250 ms complex averages; neighbours/static profile use frozen training coefficients. '
        'No common trajectory imposed.', fontsize=11)
    fig.savefig(output/'06-separated-flash.png', dpi=150)
    plt.close(fig)


def write_csv(windows, output):
    with (output/'measured-trajectories.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window','candidate','offset_m','channel','rf_hz','time_s',
            'phase_origin_s','physical_phase_change_deg','excess_path_change_m',
            'nominal_excess_path_change_m'])
        for w in windows:
            for p in w['points']:
                for ch in p['channels']:
                    for time, path, nominal in zip(w['time_s'], ch['grid_measured_path_m'], ch['grid_nominal_path_m']):
                        writer.writerow([w['label'],p['name'],p['offset_m'],ch['name'],ch['rf_hz'],time,
                            p['anchor_time_s'], -360*ch['rf_hz']/C*path if np.isfinite(path) else '',
                            path if np.isfinite(path) else '',nominal])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'scatterer-phase-comparison')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT/'measured-scatterer-phases/measurements.json'
    conditional_path = ROOT/'multiple-scatterer-phase-plots/measurements.json'
    saved = json.loads(raw_path.read_text())
    replay_source_changes = []
    for source, expected in saved['source_sha256'].items():
        path = Path(source)
        current = digest(path)
        if current != expected:
            snapshot = raw_path.parent/'sources'/path.name
            if path.suffix != '.py' or not snapshot.exists() or digest(snapshot) != expected:
                raise ValueError('Raw measurement provenance changed: '+source)
            # Preserve the earlier source identity and demand numerical replay
            # below rather than pretending an evolved plotting helper is old.
            replay_source_changes.append(dict(source=source, original_sha256=expected,
                original_snapshot=str(snapshot), current_sha256=current))
    fields = load_fields(Path('build/isar-20261015/full-fields'), Path('docs/data/calibration_20261016_sites'))
    windows, check = reproduce(saved, fields)
    plot_selection(windows, fields, output)
    trajectories(windows, output, phase=False)
    trajectories(windows, output, phase=True)
    comparison = differences(windows, output)
    increments = plot_increments(windows, output)
    (output/'04-local-increments.png').rename(output/'05-local-increments.png')
    conditional = json.loads(conditional_path.read_text())
    conditional_check = reproduce_conditional(conditional, fields)
    conditional_flash(conditional, output)
    shutil.copyfile(ROOT/'multiple-scatterer-phase-plots/01-selected-points.png', output/'06-separated-selected-points.png')
    write_csv(windows, output)
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
        w['label'], f"{w['interval_s'][0]:g}–{w['interval_s'][1]:g} s", p['name'], f"{p['offset_m']:+.1f} m",
        f"{p['anchor_time_s']:.2f} s"))+'</tr>' for w in windows for p in w['points'])
    sections = ''.join(f'<h2>{title}</h2><p>{caption}</p><a href="{file}"><img src="{file}" alt="{title}"></a>'
        for title, file, caption in [
            ('Selected cells in all five channels', '01-selected-points.png',
                'Cyan boxes identify the cells used below. Rows are channels; columns are matching capture windows. '
                'The range coordinate is relative to the saved AIS reference. The dashed ridge is stationary clutter in that moving reference.'),
            ('Measured physical path changes', '02-path-trajectories.png',
                'Each panel overlays all five channels for one local candidate and one window. Dividing by carrier wavenumber '
                'allows a fair comparison across RF bands. Vertical dotted lines mark the removed constant origin.'),
            ('Measured RF phase changes', '03-physical-phase.png',
                'The saved carrier deramp is restored. Physical phase is −2π ΔL/λ. Different carrier frequencies '
                'give different phase slopes for the same path change; no fitted rate or curvature is removed.'),
            ('Relative trajectories between cells', '04-between-scatterers.png',
                'A−B and C−B remove the common translation to first order within each band. Similar curves would support '
                'consistent material-point associations with slow orientation change. No common trajectory or heading is fitted. '
                'These independently selected cells are not yet identified hull points.'),
            ('Short increments through intermittent returns', '05-local-increments.png',
                'Direct 0.25 s phase differences retain useful pieces without connecting cycles through a fade. '
                'Solid curves are measurements; dashed curves are nominal AIS/site predictions, not fitted corrections. '
                'An increment retains a one-wavelength ambiguity.'),
            ('Flash: selected components in the fixed observer grid', '06-separated-selected-points.png',
                'This separate diagnostic uses the same four-second flash interval and four independently fitted return centres. '
                'Its fixed-grid range origin differs from the moving raw-cell reference above; raw A/B/C labels are not an exact reassociation.'),
            ('Flash: separating neighbours and stationary energy', '06-separated-flash.png',
                'These are conditional measurements after subtracting predicted neighbouring returns and a constant static profile. '
                'Coefficients use training rows and stay frozen on unused rows. They accompany the raw plots; they are not independent '
                'observations or evidence of a unique physical decomposition. No multipath propagation model is used.')])
    (output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Selected scatterers: five-channel measured phases</title>
<style>body{{font:16px/1.6 system-ui;max-width:1350px;margin:30px auto;padding:0 20px;color:#222}}img{{width:100%}}
.note{{background:#edf3f7;padding:16px}}td,th{{padding:6px 12px;border:1px solid #ddd}}table{{border-collapse:collapse}}h2{{margin-top:2em}}</style>
<h1>Selected scatterers and their measured phase trajectories</h1>
<p class="note">Ten local selections, four matching windows, all five channels. All 50 raw complex histories have been
re-extracted and reproduced to floating-point precision. The separate flash diagnostic also reproduces 10 saved objectives and 20 complex histories.
Multipath modelling is stopped. No antenna, heading or cross-channel phase screen is fitted here.</p>
<p><b>Result:</b> the main flash return B has the clearest shared bulk path change. For raw C−B in the flash window,
the disagreement with Nine is approximately 9–12 cm RMS in Seven/SBS/Ten and 42 cm in ABC over the common connected interval.
Other selections have stronger disagreements or fades. After conditional separation, Ten's C component is almost stationary:
only 3.2% of its fitted energy survives static projection. The plots therefore support some shared motion, but do not establish
that every selected resolution cell is the same material scatterer across bands.</p>
<p>The raw diagnostic uses 0.5 s rectangular complex averages every 0.05 s. The acquisition deramp is restored at each measurement epoch.
Each point and channel loses only one constant phase origin. No slopes or curvature are fitted to align the curves.
Lines stop at fades rather than bridging unknown cycles. These overlapping measurements are correlated; observed RMS differences
are descriptive, not statistical confidence intervals. The recording already contains capture-time cancellation; this raw diagnostic adds none.</p>
<p><a href="02-path-trajectories.png">Path overview</a> · <a href="03-physical-phase.png">Phase overview</a> ·
<a href="04-between-scatterers.png">Between-point differences</a> · <a href="measured-trajectories.csv">CSV samples</a> ·
<a href="validation.json">Reproduction/provenance</a> · <a href="tests.log">Numerical checks</a></p>
{sections}
<h2>Selections and phase origins</h2><p>Labels are local to each window. A/B/C are magnitude-selected features;
they are not assumed to persist as identical material points between windows.</p>
<table><tr><th>Window</th><th>Interval</th><th>Cell</th><th>Range offset</th><th>Constant origin</th></tr>{rows}</table>
<p>Production processing and approved transmitter/receiver metadata are unchanged. No new location calibration or focused video is accepted from this phase diagnostic.</p>
</html>''')
    sources = [Path(__file__), raw_path, conditional_path, *[Path('support/isar')/(x+'.py') for x in
        ('measured_phase','report_measured_scatterer_phases','component_phase','test_measured_phase','test_component_phase')]]
    result = dict(scope=__doc__, complete=True, production_changed=False, multipath_model=False,
        reproduction=check, conditional_reproduction=conditional_check,
        raw_between_point_comparisons=comparison, local_increments=increments,
        replay_source_changes=replay_source_changes,
        conditional_scope=conditional['uncertainty_scope'],
        source_sha256={str(p):digest(p) for p in sources},
        input_field_sha256={str(f['source']):f['source_sha256'] for f in fields})
    (output/'validation.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False))
    snapshots = output/'sources'
    snapshots.mkdir(exist_ok=True)
    for p in sources:
        if p.suffix == '.py':
            shutil.copyfile(p, snapshots/p.name)
    print(output/'index.html', flush=True)


if __name__ == '__main__':
    main()
