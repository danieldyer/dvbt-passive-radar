"""Discover short vehicle arcs, then fit and validate each measured band.

Uses training samples for interval selection and fitting. Each fit predicts
unused 125 ms time blocks. Arcs remain candidate observations, not established
point reflectors or accepted antenna calibration constraints.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cluster_phase import crop
from coherent_arc import CoherentArc
from compare_ais_phase import NAMES, COLOURS
from geographic_io import read_profiles
from phase_cluster_report import run_fit
from road_arc import observation, observables


def discover(data, description, duration, count, stationary=False):
    times = np.arange(np.ceil(description['start']), np.floor(description['end']-duration)+1)
    rates = np.linspace(-3., 3., 97)
    origins = np.arange(-2, 3)
    scores = np.zeros((len(times), len(origins), len(rates)))
    for record in data:
        for i, start in enumerate(times):
            for j, centre in enumerate(origins):
                f = crop(record, start, start+duration, np.arange(centre-2, centre+3))
                training = ((f.time-start)//.125).astype(int)%2 == 0
                lag = f.offsets-centre
                h = f.whiten@f.pulse(lag)
                if stationary:
                    likelihood = CoherentArc(f, training)
                    template = (f.root_information[:, None]*h[None, :]).reshape(-1)
                    t = np.repeat(f.time-(start+duration/2), len(f.offsets))
                    columns = likelihood.nuisance.remove((template[:, None]*np.exp(-1j*f.k*t[:, None]*rates))[likelihood.selected])
                    scores[i, j] += abs(likelihood.clean.conj()@columns)**2/np.maximum(np.sum(abs(columns)**2, axis=0), 1e-300)
                    continue
                raw = record['value'][(record['time'] >= start)&(record['time'] < start+duration)]
                columns = [np.flatnonzero(record['offset_samples'] == x)[0] for x in f.offsets]
                product = (raw[:, columns]@f.whiten.T)@h.conj()
                weight = f.root_information**2
                t = f.time[training]-(start+duration/2)
                cross = (weight[training]*product[training])@np.exp(1j*f.k*t[:, None]*rates)
                scores[i, j] += abs(cross)**2/(sum(weight[training])*np.vdot(h, h).real)
    maximum = scores.max(axis=(1, 2))
    selected = []
    for index in np.argsort(-maximum):
        start = times[index]
        if any(abs(start-c['first_s']) < 2*duration for c in selected):
            continue
        origin, rate = np.unravel_index(np.argmax(scores[index]), scores[index].shape)
        selected.append({'first_s': float(start), 'last_s': float(start+duration),
            'centre_sample': int(origins[origin]), 'training_score': float(maximum[index]),
            'discovery_rate_mps': float(rates[rate])})
        if len(selected) == count:
            break
    return {'times_s': times.tolist(), 'maximum_training_score': maximum.tolist(), 'selected': selected}


def publish(results, output):
    arcs = [arc for target in results['targets'] for arc in target['arcs'] if len(arc['channels']) == 5]
    if not arcs:
        return
    phase = np.array([[ch['diagnostics']['test_phase_rms_deg'] for ch in arc['channels']] for arc in arcs])
    fraction = np.array([[ch['diagnostics']['test_complex_prediction_fraction'] for ch in arc['channels']] for arc in arcs])
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, .45*len(arcs)+2)), layout='constrained')
    for ax, matrix, title, cmap, low, high in zip(axes, [phase, 100*fraction],
            ['Withheld phase RMS (degrees)', 'Withheld complex prediction (%)'], ['viridis_r', 'viridis'], [0, 0], [90, 100]):
        im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=low, vmax=high)
        ax.set_xticks(range(5), NAMES); ax.set_yticks(range(len(arcs)), [a['label'] for a in arcs])
        for i in range(len(arcs)):
            for ch in range(5):
                ax.text(ch, i, f'{matrix[i, ch]:.1f}', ha='center', va='center', fontsize=9,
                        color='white' if (matrix[i, ch] < (low+high)/2) == (cmap == 'viridis') else 'black')
        fig.colorbar(im, ax=ax, label=title)
    fig.suptitle('Independent short vehicle arcs · all retained candidate bands shown\nConditional point model; interval selection and fits use training blocks only')
    fig.savefig(output/'arc-quality.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(arcs), 1, figsize=(12, max(5, 2.4*len(arcs))), layout='constrained', squeeze=False)
    for arc, ax in zip(arcs, axes[:, 0]):
        for row, colour in zip(arc['channels'], COLOURS):
            d = row['diagnostics']; t = np.array(d['time_s']); train = np.array(d['training'])
            ax.plot(t[~train], np.array(d['phase_residual_deg'])[~train], '.', color=colour, label=row['channel'])
        ax.axhline(0, color='black', lw=.5); ax.set(title=arc['label'], ylabel='Phase error (°)', ylim=(-180, 180))
    axes[0, 0].legend(ncol=5, fontsize=8); axes[-1, 0].set_xlabel('Recording time (s)')
    fig.savefig(output/'arc-phases.png', dpi=150); plt.close(fig)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Short vehicle arcs</title>
<style>body{font:16px/1.5 system-ui;max-width:1200px;margin:35px auto;padding:0 20px}img{width:100%}</style>
<h1>Short vehicle arcs from the original complex range data</h1>
<p>Each candidate interval is discovered using only alternating 125 ms training blocks. The other blocks test its continuous phase trajectory and interpolated real strengths. A quadratic path drives both delay migration and carrier phase; the original OFDM moments, row epochs, empirical carrier reference and range covariance are retained. Constant band phase is free; phase may not drift independently of the fitted path.</p>
<p>The archived discovery tracks originally contain coarse range/rate observations in one band each. This stage uses the saved complex observations in all five bands. It does not treat copied discovery tracks as independent measurements. Selection used this recording, so these are conditional predictive checks, not independent target discovery on new data.</p>
<p>All arcs and weak channels remain visible. Boundary hits and numerical gradients are recorded in the JSON. No target is accepted as a common multiband point or an antenna calibration yet; association and physical road/antenna geometry must be tested next. B and C, and other overlapping targets, must not be counted twice in a joint likelihood.</p>
<img src="arc-quality.png" alt="Per-band prediction and phase quality for every candidate arc">
<img src="arc-phases.png" alt="Withheld phase errors across short vehicle arcs">
<p><a href="results.json">Fit trajectories, covariance, data provenance and all unused-sample diagnostics</a>.</p></html>'''
    if results.get('model') == 'coherent_with_stationary':
        page = page.replace('interpolated real strengths', 'one constant complex target coefficient plus fixed-delay stationary coefficients')
        page = page.replace('Constant band phase is free; phase may not drift independently of the fitted path.',
            'One constant complex target coefficient and the stationary coefficients are eliminated in the training likelihood and frozen for prediction. Both target templates and observations receive the identical stationary projection. There is no fixed Doppler notch. Per-bin phase RMS includes noise and is not an acceptance gate; coherent prediction statistics are also recorded.')
    (output/'index.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--histories', type=Path, default=Path('build/calibration-20261013/shared-seed-histories'))
    parser.add_argument('--patches', type=Path, default=Path('build/calibration-20261013/shared-seed-patches'))
    parser.add_argument('--templates', type=Path, default=Path('build/calibration-20261013/static-with-psf.gprf'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'E', 'K', 'O', 'Q', 'R', 'S'])
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--arcs-per-target', type=int, default=2)
    parser.add_argument('--coherent', action='store_true', help='One coherent amplitude and explicit stationary nuisance, instead of per-bin real strengths')
    parser.add_argument('--discovery', type=Path, help='Keep exactly the candidate windows from a completed earlier comparison')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'results.json').exists():
        raise ValueError('Use a new output directory; do not silently reuse changed arc inputs')
    header, records = read_profiles(args.templates)
    sources = [args.templates, Path(__file__), *[Path(__file__).with_name(name) for name in
        ['road_arc.py', 'field.py', 'cluster_phase.py', 'coherent_arc.py', 'stationary.py', 'shared_target_geometry.py']]]
    previous = json.loads(args.discovery.read_text()) if args.discovery else None
    if previous is not None:
        if not previous['complete'] or previous['duration_s'] != args.duration:
            raise ValueError('Use completed discovery with the same aperture length')
        sources.append(args.discovery)
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'duration_s': args.duration, 'arcs_per_target': args.arcs_per_target,
        'labels': args.labels, 'targets': [], 'source_sha256': {},
        'model': 'coherent_with_stationary' if args.coherent else 'per_bin_positive_strength'}
    def save():
        (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for label in args.labels:
        source = args.histories/(label+'.json'); sources.append(source)
        description = json.loads(source.read_text()); data = []
        for ch, seed in enumerate(description['channels']):
            path = args.patches/f'{label}-{ch}.npz'; sources.append(path)
            keep = ((records['channel'] == ch)&(records['time'] >= description['start'])&
                    (records['time'] <= description['end'])&(records['information'] > 0))
            pulse = np.average(records[keep]['psf'], axis=0, weights=records[keep]['information'])
            with np.load(path) as raw:
                data.append(observation(raw, seed, pulse))
        discovery = (next(t['discovery'] for t in previous['targets'] if t['label'] == label)
                     if previous is not None else discover(data, description, args.duration, args.arcs_per_target, args.coherent))
        target = {'label': label, 'discovery': discovery, 'arcs': []}; result['targets'].append(target)
        print(label, 'discovery', [(c['first_s'], c['centre_sample']) for c in discovery['selected']], flush=True)
        for index, candidate in enumerate(discovery['selected']):
            arc = {**candidate, 'label': f"{label}{index+1} · {candidate['first_s']:g}–{candidate['last_s']:g} s", 'channels': []}
            target['arcs'].append(arc)
            for ch, (record, seed) in enumerate(zip(data, description['channels'])):
                if args.coherent:
                    centre = candidate['centre_sample']
                    f = crop(record, candidate['first_s'], candidate['last_s'], np.arange(centre-2, centre+3))
                    training = ((f.time-candidate['first_s'])//.125).astype(int)%2 == 0
                    problem = CoherentArc(f, training)
                    fitted = problem.fit(origin_bounds=(centre-1.2, centre+1.2))
                    row = {'interval_s': [candidate['first_s'], candidate['last_s']], 'offset_samples': f.offsets.tolist(),
                        'epoch_s': problem.epoch, 'scale_s': problem.scale, 'degree': 2,
                        'parameters': fitted.parameters.tolist(), 'objective': fitted.objective,
                        'maximum_gradient': fitted.gradient, 'trials': fitted.trials,
                        'conditional_parameter_covariance': fitted.covariance.tolist(),
                        'information_rank': fitted.information_rank, 'diagnostics': problem.diagnostics(fitted)}
                else:
                    problem, fitted, row = run_fit(record, candidate['first_s'], candidate['last_s'],
                        centre=candidate['centre_sample'], rate_bound=4., origin_width=1.2)
                value, covariance = observables(row, seed, (candidate['first_s']+candidate['last_s'])/2)
                row.update(channel=NAMES[ch], rf_hz=seed['rf_hz'], seed=seed,
                    absolute_observables=value.tolist(), absolute_observable_covariance=covariance.tolist(),
                    range_bound_hit=bool(abs(fitted.parameters[0]-candidate['centre_sample']) > 1.1999))
                arc['channels'].append(row); save()
                print(arc['label'], NAMES[ch], 'phase', round(row['diagnostics']['test_phase_rms_deg'], 1),
                      'prediction', round(row['diagnostics']['test_complex_prediction_fraction'], 3),
                      'range bound', row['range_bound_hit'], flush=True)
            publish(result, args.output)
    result['source_sha256'] = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sources}
    result['complete'] = True; save(); publish(result, args.output)


if __name__ == '__main__':
    main()
