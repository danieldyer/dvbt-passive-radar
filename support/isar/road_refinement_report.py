"""Review plots for raw short-arc geometry, site hypotheses and target transfer."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from fit_road_arc_raw import load_problem
from road_arc_geometry import road_state

NAMES = ['Seven', 'SBS', 'Nine', 'Ten', 'ABC']
COLORS = ['#0072b2', '#009e73', '#e69f00', '#cc79a7', '#d55e00']


def code(row):
    return ''.join(map(str, row['assignment']))


def error(row):
    return sum(band['error'] for target in row['prediction'] for band in target['channels'])


def energy(row):
    return np.array([[band['prediction_fraction'] for band in target['channels']] for target in row['prediction']])


def finish(figure, output, name):
    figure.savefig(output/name, dpi=160, bbox_inches='tight'); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sites', type=Path, default=Path('docs/data/isar_20261017/road-sites-32/fits.json'))
    parser.add_argument('--common', type=Path, default=Path('docs/data/isar_20261017/road-common-phase-multistart/fits.json'))
    parser.add_argument('--transfer', type=Path, default=Path('docs/data/isar_20261017/road-calibration-phase-transfer/validation.json'))
    parser.add_argument('--continuity', type=Path, default=Path('docs/data/isar_20261017/road-continuity-00000/fit.json'))
    parser.add_argument('--fixed-sites', type=Path, default=Path('docs/data/isar_20261017/road-fixed-associations-sites-32/fits.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sites, common, transfer = [json.loads(p.read_text()) for p in (args.sites, args.common, args.transfer)]
    if not all(d['complete'] for d in (sites, common, transfer)) or len(sites['trials']) != 32:
        raise ValueError('The review requires complete site, phase and transfer controls')
    trials = sorted(sites['trials'], key=lambda r: r['cost'])
    fixed = json.loads(args.fixed_sites.read_text())
    if not fixed['complete'] or len(fixed['trials']) != 32:
        raise ValueError('The fixed-road control requires all 32 assignments')
    for i, label in enumerate(fixed['labels']):
        if len({(t['associations'][i]['road'], t['associations'][i]['route']) for t in fixed['trials']}) != 1:
            raise ValueError(f'Road association is not fixed for {label}')
    fixed_trials = sorted(fixed['trials'], key=lambda r:r['cost'])
    best = trials[0]; nominal = next(t for t in trials if code(t) == '02002')
    shared = common['trials'][common['best_training_trial']]
    labels = sites['labels']; x = np.arange(len(trials))
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, layout='constrained')
    for ax, key, label in zip(axes, (lambda r: r['cost'], error),
                              ('Training objective above minimum', 'Held-out complex error above minimum')):
        values = np.array([key(t) for t in trials]); values -= values.min()
        ax.bar(x, values, color=['#0072b2' if t['assignment'][1] == 0 else '#d55e00' for t in trials])
        for i, t in enumerate(trials):
            if t['bound_indices']:
                ax.scatter(i, values[i]+max(values.max()*.025, 1.), marker='x', color='black', s=22)
        ax.set_ylabel(label); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_title('All 32 permitted assignments · identical raw data and training split\nBlue: SBS at Ornata; orange: SBS at Ridge; ×: a fitted parameter reaches a bound')
    axes[1].set_xticks(x, [code(t) for t in trials], rotation=65)
    axes[1].set_xlabel('Seven / SBS / Nine / Ten / ABC site code: 0 Ornata, 1 Eyre, 2 Ridge')
    finish(fig, args.output, '01-sites.png')

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout='constrained')
    for ax, key, title in zip(axes, (lambda r:r['cost'], error), ('Training objective', 'Unused-sample complex error')):
        values = np.array([key(t) for t in fixed_trials])
        ax.bar(np.arange(32), values-values.min(), color=['#0072b2' if t['assignment'][1] == 0 else '#d55e00' for t in fixed_trials])
        ax.set_ylabel('Difference from minimum');ax.set_title(title)
    axes[1].set_xticks(np.arange(32), [code(t) for t in fixed_trials], rotation=65)
    axes[1].set_xlabel('Seven / SBS / Nine / Ten / ABC site code: 0 Ornata, 1 Eyre, 2 Ridge')
    fig.suptitle('Control: the same seven targets on the same road routes in every assignment\nA, B, C, K, O, Q, R · E and S excluded because their road associations change')
    finish(fig, args.output, '08-fixed-roads.png')

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout='constrained')
    for ax, row, title in zip(axes, [nominal, best, shared],
            ['Nominal assignment 02002\nIndependent band phases', 'Best training assignment '+code(best)+'\nIndependent band phases',
             'Common-phase hypothesis 02002\nBest tested phase start; bound contacts retained']):
        values = energy(row)*100
        image = ax.imshow(values, vmin=-5, vmax=50, cmap='viridis', aspect='auto')
        for i in range(len(labels)):
            for j in range(5):
                ax.text(j, i, f'{values[i, j]:.0f}', ha='center', va='center', color='white' if values[i,j] < 28 else 'black', fontsize=9)
        ax.set_xticks(range(5), NAMES, rotation=40); ax.set_yticks(range(len(labels)), labels); ax.set_title(title, fontsize=11)
    fig.colorbar(image, ax=axes, label='Held-out complex energy explained versus stationary-only (%)', shrink=.8)
    finish(fig, args.output, '02-prediction.png')

    model, _, _ = load_problem('docs/data/isar_20261017/road-arc-geometry/fits.json', best['assignment'], labels,
                              free_antennas=True, free_origins=True)
    states = np.array(best['road_states'])
    points = np.array([road_state(road, state)[0] for road, state in zip(model.geometry.roads, states)])/1000
    routes = json.loads(Path('build/calibration-20261017/road-routes.json').read_text())['routes']
    all_points = np.r_[points[:, :2], [[0., 0.]]]
    low, high = all_points.min(axis=0)-.6, all_points.max(axis=0)+.6
    roads = []
    for route in routes:
        xy = np.array(route['enu_m'])[:, :2]/1000
        if np.any(np.all((xy >= low)&(xy <= high), axis=1)):
            roads.append(xy)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
    offsets = {'A': (-23, -15), 'C': (12, -19), 'K': (-10, 17), 'Q': (13, 12), 'O': (10, 4)}
    for panel, ax in enumerate(axes):
        ax.add_collection(LineCollection(roads, colors='#d3d3d3', linewidths=.55))
        for index, (point, association, road, state) in enumerate(zip(points, best['associations'], model.geometry.roads, states)):
            label = labels[index]
            if panel and label in ('E', 'S'):
                continue
            ax.scatter(*point[:2], s=35, color='#0072b2', zorder=4)
            if panel or label in ('E', 'S'):
                ax.annotate(label, point[:2], xytext=offsets.get(label, (5, 5)), textcoords='offset points',
                            arrowprops=dict(arrowstyle='-', color='#555', lw=.5), fontsize=10, zorder=5)
            p, v, _ = road_state(road, state)
            ax.arrow(*point[:2], *(v[:2]*8/1000), width=.002 if panel else .004, color='#0072b2', zorder=3, length_includes_head=True)
        ax.scatter(0, 0, marker='*', s=150, color='black', label='Receiver', zorder=5)
        ax.set(xlim=(-1.3, .25) if panel else (low[0], high[0]),
               ylim=(-.25, .50) if panel else (low[1], high[1]),
               xlabel='East of receiver (km)', ylabel='North of receiver (km)',
               title='Albert Street / Victoria Parade detail' if panel else 'All nine conditional road hypotheses')
        ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=8)
    fig.suptitle('North up · fitted positions are not independently observed vehicle locations\nRoads © OpenStreetMap contributors · arrows show 8 s of fitted velocity', fontsize=11)
    finish(fig, args.output, '03-roads.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for row, label in [(nominal, 'Nominal sites, independent phases'), (best, 'Best sites, independent phases'),
                       (shared, 'Common phase, nominal sites (bounds)')]:
        eig = np.array(row['diagnostics']['data_information_eigenvalues_after_nuisance'])[:16]
        axes[0].semilogy(np.arange(1, 17), np.maximum(eig, 1e-10), '.-', label=label)
    axes[0].axhline(1, color='black', ls=':', label='Unit location-prior information')
    axes[0].set(xlabel='Physical antenna mode (strongest first)', ylabel='Data-only information in prior-standardized coordinates',
                title='Motion, band amplitudes and range origins profiled')
    axes[0].legend(fontsize=8)
    origins = np.array([r['independent_calibration']['range_origins_m'] for r in transfer['trials']])
    for c in range(5):
        axes[1].plot(range(len(labels)), origins[:, c], '.-', color=COLORS[c], label=NAMES[c])
    axes[1].set_xticks(range(len(labels)), [r['held_target'] for r in transfer['trials']])
    axes[1].set(xlabel='Target excluded from calibration', ylabel='Fitted envelope origin (m)',
                title='Shared origins: transfer sensitivity, nominal sites')
    axes[1].legend(fontsize=8)
    finish(fig, args.output, '04-information.png')

    fig, ax = plt.subplots(figsize=(11, 4.8), layout='constrained')
    baseline = np.array([error(row['nominal_held_target']) for row in transfer['trials']])
    for offset, name, color in [(-.18, 'independent', '#0072b2'), (.18, 'common', '#d55e00')]:
        errors = np.array([error(row[name+'_held_target']) for row in transfer['trials']])
        ax.bar(np.arange(len(labels))+offset, 100*(baseline-errors)/baseline, width=.34,
               label=('Independent phases' if name == 'independent' else 'Shared interband phase'), color=color)
    ax.axhline(0, color='black', lw=.8); ax.set_xticks(range(len(labels)), [r['held_target'] for r in transfer['trials']])
    ax.set(xlabel='Entire target excluded from antenna calibration', ylabel='Held-out complex error reduction (%)',
           title='Transfer test · positive is better than nominal geometry\nTarget motion/amplitude fitted on its training rows; calibration frozen from the other eight targets')
    ax.legend(); finish(fig, args.output, '05-transfer.png')

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout='constrained')
    parameters = np.array(best['parameters'])
    for ax, label in zip(axes, ['A', 'K', 'C']):
        index = labels.index(label); arc = model.geometry.arcs[index]
        for ch, likelihood in enumerate(model.likelihoods[index]):
            h, _, _ = model.band_template(parameters, index, ch)
            selected = likelihood.selected; clean_h = likelihood.nuisance.remove(h[selected])
            coefficient = np.vdot(clean_h, likelihood.clean)/np.vdot(clean_h, clean_h).real
            y = likelihood.field.y.ravel(); beta = likelihood.nuisance.solve(y[selected]-coefficient*h[selected])
            clean = (y-likelihood.design@beta).reshape(likelihood.field.y.shape)
            predicted = (coefficient*h).reshape(clean.shape)
            groups = ((likelihood.field.time-arc['first_s'])//.125).astype(int)
            times, phases = [], []
            for group in np.unique(groups[~likelihood.training]):
                keep = (groups == group)&~likelihood.training
                times.append(likelihood.field.time[keep].mean())
                phases.append(np.degrees(np.angle(np.vdot(predicted[keep], clean[keep]))))
            ax.plot(times, phases, 'o-', color=COLORS[ch], label=NAMES[ch], ms=3, lw=.7)
        ax.axhline(0, color='black', lw=.7); ax.set(ylim=(-180, 180), ylabel='Held-out phase error (°)', title=arc['label'])
    axes[0].legend(ncol=5, fontsize=8); axes[-1].set_xlabel('Recording time (s)')
    fig.suptitle('Physical road trajectories: held-out 125 ms coherent blocks\nEach band has its own training-fitted constant phase; weak-band phase is noisy', fontsize=12)
    finish(fig, args.output, '06-phase-prediction.png')

    continuity = json.loads(args.continuity.read_text())
    continuous, _, _ = load_problem('docs/data/isar_20261017/road-arc-geometry/fits.json',
        continuity['assignment'], continuity['labels'], span_arcs=True)
    fit_parameters = np.array(continuity['parameters']); i = continuity['labels'].index('A')
    arc = continuous.geometry.arcs[i]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    for ch, likelihood in enumerate(continuous.likelihoods[i]):
        h, _, _ = continuous.band_template(fit_parameters, i, ch)
        train = likelihood.selected; clean_h = likelihood.nuisance.remove(h[train])
        coefficient = np.vdot(clean_h, likelihood.clean)/np.vdot(clean_h, clean_h).real
        y = likelihood.field.y.ravel(); beta = likelihood.nuisance.solve(y[train]-coefficient*h[train])
        clean = (y-likelihood.design@beta).reshape(likelihood.field.y.shape)
        predicted = (coefficient*h).reshape(clean.shape)
        groups = ((likelihood.field.time-arc['first_s'])//.125).astype(int)
        times, matched = [], []
        for group in np.unique(groups[~likelihood.training]):
            keep = (groups == group)&~likelihood.training
            times.append(likelihood.field.time[keep].mean())
            matched.append(np.vdot(predicted[keep], clean[keep])/np.vdot(predicted[keep], predicted[keep]).real)
        axes[0].plot(times, np.degrees(np.angle(matched)), '.-', color=COLORS[ch], label=NAMES[ch], lw=.8)
        axes[1].plot(times, abs(np.array(matched)), '.-', color=COLORS[ch], lw=.8)
    axes[0].axhline(0, color='black', lw=.6); axes[1].axhline(1, color='black', lw=.6)
    axes[0].set(ylabel='Held-out phase error (°)', ylim=(-180, 180),
        title='Candidate A · one continuous 7 s physical road trajectory\nIndependent constant band phases/amplitudes; nominal all-Ornata coordinates; no fitted antenna correction')
    axes[0].legend(ncol=5, fontsize=8)
    axes[1].set(xlabel='Recording time (s)', ylabel='Matched amplitude /\ntraining-fitted constant amplitude')
    finish(fig, args.output, '07-continuity.png')

    ranking = ''.join(f'<tr><td>{code(t)}</td><td>{t["cost"]:.2f}</td><td>{error(t):.2f}</td><td>{html.escape(str(t["bound_indices"]))}</td></tr>' for t in trials)
    roads_table = ''.join(f'<tr><td>{label}</td><td>{html.escape(a["road"])}</td><td>{html.escape(b["road"])}</td></tr>'
                         for label, a, b in zip(labels, nominal['associations'], best['associations']))
    nominal_loss = sum(b['baseline_error']-b['error'] for t in nominal['prediction'] for b in t['channels'])
    summary = {'accepted_calibration': False, 'new_focused_videos': False,
        'best_training_assignment': best['assignment'], 'best_training_cost': best['cost'], 'nominal_training_cost': nominal['cost'],
        'best_test_error': error(best), 'nominal_test_error': error(nominal),
        'nominal_predictive_energy_above_stationary': nominal_loss,
        'best_data_information_eigenvalues': best['diagnostics']['data_information_eigenvalues_after_nuisance'],
        'fixed_road_control': {'labels': fixed['labels'], 'minimum_cost': fixed_trials[0]['cost'],
            'maximum_cost': fixed_trials[-1]['cost'], 'best_training_assignment': fixed_trials[0]['assignment'],
            'same_route_identities_verified': True},
        'inputs': list(map(str, (args.sites, args.common, args.transfer, args.continuity, args.fixed_sites)))}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Short-arc calibration review</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#18232d}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{padding:6px 12px;border:1px solid #ccd4d9;text-align:left}}.note{{background:#fff1d3;padding:16px}}code{{font-size:.9em}}</style>
<h1>Short vehicle arcs: numerical corrections and calibration checks</h1>
<p class="note">No antenna calibration has been accepted. The raw likelihood now recovers returns that a coarse initial motion could miss. The best training assignment in this conditional screen is <b>{code(best)}</b>, but it is not a conclusive site identification. Road association, weak band returns, phase-model transfer and data-only conditioning remain explicit checks. No new focused video or production metadata correction is claimed.</p>
<p>The nine candidates are A, B, C, E, K, O, Q, R and S. Each contributes one training-selected two-second arc and all five bands. Original complex rows, row epochs, OFDM C0/C1 response and the archived carrier reference are preserved. Fixed-delay stationary clutter and a constant complex moving-target amplitude are fitted together; unused alternating 125 ms blocks are predicted without refitting these coefficients. The receiver horizontal coordinates are fixed to the user’s revised position; its height is bounded to 0–20 m AGL. Site coordinates and heights use the revised user priors and approved one-arcsecond DEM.</p>
<h2>1. The numerical correction</h2>
<p>A coarse road track can initialize a coherent short-arc optimizer on a Doppler sidelobe. Every band’s independently fitted training-only range/rate/curvature now supplies an additional physical motion start. Only the raw training likelihood chooses among them. An independent synthetic regression starts on a wrong sidelobe, recovers the injected state, and verifies that changing held-out samples cannot change the start selection. The full real-data residual Jacobian also agrees with directional finite differences. This fixes an optimizer initialization failure; it does not alter the capture or production processing.</p>
<img src="06-phase-prediction.png" alt="Held-out phase predictions for three targets"><p>These phase plots test temporal prediction within each band. Independent band phases are nuisance parameters fitted on training data; a phase near zero here is not proof of a shared cross-band phase reference.</p>
<h2>2. All permitted site assignments</h2><img src="01-sites.png" alt="Training and held-out errors for all 32 permitted assignments">
<p>Site digits follow Seven/SBS/Nine/Ten/ABC: 0 Ornata, 1 Eyre, 2 Ridge. The nominal hypothesis is 02002. All 32 combinations obey the requested primary/backup restrictions. Lower cost is better; the conditional Gaussian objective retains the original information weights. A best score alone is insufficient: several hypotheses are close, and a location prior can stabilize an antenna estimate even when the observations do not identify it. The coarse road association is reopened for each assignment: E and S change roads between the nominal and best trial. Consequently these scores jointly compare site and road assumptions, and improvements cannot be attributed solely to transmitter locations.</p>
<table><tr><th>Target</th><th>Nominal-assignment road</th><th>Best-trial road</th></tr>{roads_table}</table>
<h3>Control with road association held fixed</h3><img src="08-fixed-roads.png" alt="All 32 assignments using seven targets with identical road routes">
<p>Repeating the entire 32-assignment fit with A, B, C, K, O, Q and R gives training objectives from {fixed_trials[0]['cost']:.3f} to {fixed_trials[-1]['cost']:.3f}: a spread of only {fixed_trials[-1]['cost']-fixed_trials[0]['cost']:.3f}. The route identifiers as well as street names were verified identical across every assignment. The best training assignment is {code(fixed_trials[0])}, and the best unused-sample assignment differs. The larger nine-target ranking therefore cannot be read as a transmitter-site measurement: it was confounded by E/S road associations. These seven nearby street targets alone do not provide a robust assignment decision under this model.</p>
<img src="02-prediction.png" alt="All five bands and nine targets, predictive complex energy"><p>Percentages measure reduction in held-out error relative to a stationary-only prediction, not target detection probability or fraction of intrinsic target SNR recovered. Negative values and weak bands are retained. Common-phase fits are conditional hypothesis tests; bound contacts are reported.</p>
<h2>3. Locations and information</h2><img src="03-roads.png" alt="North-up conditional road positions"><p>The road hypotheses were initialized from the coarse screen, preferring Albert Street or Victoria Parade only where those connected routes were already candidates. The user’s earlier plausibility judgement is not treated as an independent vehicle-position measurement.</p>
<img src="04-information.png" alt="Data-only antenna information and range-origin sensitivity"><p>The information spectrum profiles target motion, complex amplitudes, stationary clutter and shared envelope origins before evaluating antenna information. It excludes location priors. Sixteen antenna coordinates are physically free; the two fixed horizontal receiver coordinates are excluded. Values below 10⁻¹⁰ are shown at the plot floor. These are conditional thermal-model sensitivities, not a guarantee against scene mismatch. Envelope-origin offsets are diagnostic nuisance parameters, not a declaration of instrumental delay.</p>
<h2>4. Transfer to an excluded vehicle</h2><img src="05-transfer.png" alt="Whole-target calibration transfer"><p>Each calibration uses eight vehicles. The ninth receives frozen antenna corrections, envelope origins and, for the shared-phase model, band phases. Only that target’s motion and scattering coefficient are fitted on its training blocks, then tested on its unused blocks. The shared-phase calculation tests distinct circular starts and records their training costs. Neither a failed transfer nor a bounded local fit is presented as a proof that every possible scattering model is impossible.</p>
<h2>5. Relation to prior work and the sea-reflection question</h2>
<p><a href="https://www.mdpi.com/1424-8220/19/15/3365">Zhao et al. (2019)</a> derive calibration-assisted passive-radar localization bounds while accounting for antenna and calibration-target position uncertainty. This supports keeping road/AIS uncertainty separate from measurement noise. Our raw complex likelihood and nuisance-profiled information calculation apply that principle to the present data; their simulated calibration geometry is not a validation of ours.</p>
<p><a href="https://arxiv.org/abs/2005.10528">Dumas and Guérin (2020)</a> use the direct path to calibrate complex receive-array gains in bistatic oceanographic radar. That is distinct from proving that separate DVB-T bands share a target scattering phase; this report tests the latter instead of assuming it.</p>
<p><a href="https://researchspace.csir.co.za/bitstream/handle/10204/1071/lord_2006.pdf">Lord, Nel and Abdul Gaffar (2006)</a> explicitly model sea-reflected ship echoes and the associated Fresnel phase. Their low-grazing HH examples have strong reflected fields, and multipath degrades the simulated image. That supports testing ship–sea paths. It does not identify this VHF echo’s mechanism. The <a href="../../isar_20261016/flash-reflection/index.html">capture-specific sea/facet calculation</a> finds unresolved ship–sea delays and places the sea-only smooth-surface specular centre inland.</p>
<h2>6. First longer continuity control</h2><img src="07-continuity.png" alt="Seven-second vehicle phase and amplitude continuity test">
<p>Candidate A is now fitted continuously from 53 to 60 s, including the interval between its two discovered short arcs. One physical road trajectory drives range and phase; each band retains one constant complex amplitude. Antennas and envelope origins are fixed in this control. SBS and Nine explain about 31% and 29% of held-out residual energy over that span. The amplitude ratio is a diagnostic plotted after fitting, not a normalization applied to the observations. Other tested vehicles have weaker predictions under this deliberately simple longer model; that does not by itself separate amplitude variation, acceleration-model error and loss of phase continuity.</p>
<p>The fixed-road comparison above now separates the road-association changes from antenna changes. The next complementary check is the <a href="../ship-phase-continuity/index.html">ship’s longer phase history with changing brightness</a>, followed by the <a href="../ship-physical-arc-review/index.html">original-data physical geometry test</a>. A stronger long-baseline phase constraint must predict independent data; it cannot be supplied by a location prior or an arbitrary phase correction.</p>
<details><summary>All numerical site scores</summary><table><tr><th>Assignment</th><th>Training objective</th><th>Held-out error</th><th>Bound indices</th></tr>{ranking}</table></details>
<p><a href="summary.json">Summary JSON</a> · <a href="../road-sites-32/fits.json">All nine-target site fits</a> · <a href="../road-fixed-associations-sites-32/fits.json">Seven-target fixed-road control</a> · <a href="../road-calibration-phase-transfer/validation.json">Target transfer fits</a> · <a href="../road-common-phase-multistart/fits.json">Shared-phase starts</a></p></html>'''
    (args.output/'index.html').write_text(report)
    print(args.output/'index.html', flush=True)


if __name__ == '__main__':
    main()
