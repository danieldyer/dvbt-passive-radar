"""Recheck discovered ship arcs with an explicit stationary-clutter likelihood.

Keep the previously discovered time/range regions frozen. Fit one continuous
quadratic path and either constant complex amplitude or explicitly requested
nonnegative linear amplitude knots, jointly with stationary range coefficients.
Independent band phases and all weak candidates remain
visible. This tests which observations can constrain physical motion; it does
not calibrate antennas or join phase across unrelated flashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from cluster_phase import crop
from coherent_arc import CoherentArc
from amplitude_arc import AmplitudeArc
from compare_ais_phase import NAMES
from flash_siting_report import ais_observables
from motion import C


def archived_observables(data, midpoint):
    """Restore the saved deramp, including across explicitly archived AIS gaps.

    The immutable row reference is authoritative. Re-interpolating AIS with a
    different gap policy must not change the meaning of a measured residual.
    The saved symbol angular frequency gives exact reference path rate; a local
    cubic gives its derivative at the requested epoch (no long extrapolation).
    """
    time=np.asarray(data['row_time'])
    if midpoint<time[0] or midpoint>time[-1]:raise ValueError('Reference epoch lies outside saved rows')
    k=2*np.pi*json.loads(str(data['metadata_json']))['rf_hz']/C
    rate=-np.asarray(data['centre_omega'])/(k*.001024)
    selected=abs(time-midpoint)<=.05
    if np.count_nonzero(selected)<4:raise ValueError('Insufficient saved rows for local reference curvature')
    coefficient=np.polynomial.polynomial.polyfit(time[selected]-midpoint,rate[selected],3)
    return np.array([np.interp(midpoint,time,data['centre_path_m']),np.interp(midpoint,time,rate),coefficient[1]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--discovery', type=Path, default=Path('docs/data/isar_20261016/flash-siting/piecewise-discovery.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amplitude-spacing', type=float, help='Positive linear amplitude knot spacing in seconds; default constant strength')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fits.json').exists():
        raise ValueError('Use a fresh output for this controlled arc comparison')
    discovery = json.loads(args.discovery.read_text())
    candidates = [{'label': 'Flash', 'first_s': 164., 'last_s': 165., 'centre_sample': 0},
                  *discovery['candidates']]
    if args.amplitude_spacing is not None and args.amplitude_spacing <= 0:
        raise ValueError('Amplitude spacing must be positive')
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'amplitude_spacing_s': args.amplitude_spacing,
        'candidates': [dict(c, channels=[]) for c in candidates], 'source_sha256': {}}
    track_path = Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track = read_csv(track_path, reference_height_ellipsoid_m=5., mmsi=563052200)
    sources = [args.discovery, track_path, Path(__file__), Path(__file__).with_name('coherent_arc.py'),
               Path(__file__).with_name('cluster_phase.py'), Path(__file__).with_name('field.py'),
               Path(__file__).with_name('amplitude_arc.py')]
    for ch in range(5):
        source = args.fields/f'channel-{ch}.npz'; sources.append(source)
        with np.load(source) as data:
            for candidate in result['candidates']:
                first, last, centre = candidate['first_s'], candidate['last_s'], candidate['centre_sample']
                field = crop(data, first, last, np.arange(centre-2, centre+3))
                train = ((field.time-first)//.125).astype(int)%2 == 0
                if args.amplitude_spacing is None:
                    model = CoherentArc(field, train)
                else:
                    knots = np.r_[field.data['row_time'][0], np.arange(first+args.amplitude_spacing,last,args.amplitude_spacing),
                                  field.data['row_time'][-1]]
                    model = AmplitudeArc(field, train, knots)
                fit = model.fit((centre-.8, centre+.8), 4.)
                diagnostic = model.diagnostics(fit)
                midpoint = (first+last)/2; u = (midpoint-model.epoch)/model.scale
                transform = np.array([[C/8e6, 0., u, u*u], [0., 0., 1/model.scale, 2*u/model.scale],
                                      [0., 0., 0., 2/model.scale**2]])
                reference = archived_observables(data,midpoint)
                candidate['channels'].append({'channel': NAMES[ch], 'rf_hz': field.metadata['rf_hz'],
                    'epoch_s': model.epoch, 'scale_s': model.scale, 'parameters': fit.parameters.tolist(),
                    'objective': fit.objective, 'maximum_gradient': fit.gradient, 'trials': fit.trials,
                    'conditional_parameter_covariance': fit.covariance.tolist(), 'information_rank': fit.information_rank,
                    'range_bound_hit': bool(abs(fit.parameters[0]-centre) > .7999),
                    'absolute_observables': (reference+transform@fit.parameters).tolist(),
                    'absolute_observable_covariance': (transform@fit.covariance@transform.T).tolist(),
                    'saved_ais_reference_observables': reference.tolist(),
                    'reference_source': 'Immutable saved centre_path_m and centre_omega; no fresh AIS interpolation',
                    'diagnostics': diagnostic})
                print(candidate['label'], NAMES[ch], 'prediction', round(diagnostic['test_complex_prediction_fraction'], 3),
                    'target information retained', round(diagnostic['training_target_information_fraction_after_stationary'], 3), flush=True)
        (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    result.update(complete=True, source_sha256={str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sources})
    (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 8), layout='constrained')
    prediction = np.array([[100*r['diagnostics']['test_complex_prediction_fraction'] for r in c['channels']] for c in result['candidates']])
    information = np.array([[100*r['diagnostics']['training_target_information_fraction_after_stationary'] for r in c['channels']] for c in result['candidates']])
    phase = np.array([[r['diagnostics']['test_coherent_phase_error_deg'] for r in c['channels']] for c in result['candidates']])
    for ax, values, title, lo, hi, cmap in zip(axes, (prediction, information, phase),
            ('Held-out energy explained (%)', 'Target information retained (%)', 'Held-out coherent phase error (°)'),
            (-5, 0, -90), (100, 100, 90), ('viridis', 'viridis', 'coolwarm')):
        image = ax.imshow(values, vmin=lo, vmax=hi, cmap=cmap, aspect='auto')
        ax.set_xticks(range(5), NAMES, rotation=40)
        ax.set_yticks(range(len(candidates)), [f"{c['label']} · {c['first_s']:g} s" for c in candidates])
        for i in range(len(candidates)):
            for j in range(5):
                ax.text(j, i, f'{values[i,j]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if cmap == 'viridis' and values[i,j] < 55 else 'black')
        ax.set_title(title, fontsize=11); fig.colorbar(image, ax=ax, shrink=.8)
    strength = 'constant amplitude' if args.amplitude_spacing is None else f'positive amplitude with {args.amplitude_spacing:g} s knots'
    fig.suptitle(f'Frozen ship candidate regions · one moving point with {strength} + stationary clutter per band\n'
                 'A small phase error does not establish an identifiable moving point when stationary projection removes most information', fontsize=12)
    fig.savefig(args.output/'arc-checks.png', dpi=160); plt.close(fig)
    (args.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ship arc coherence checks</title>
<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}img{max-width:100%}</style>
<h1>Ship arc coherence checks with explicit stationary nuisance</h1>
<p>The time/range regions are frozen from earlier discovery. Every band is fitted independently with one quadratic physical path, a constant phase, and constant-delay clutter. The declared amplitude model is STRENGTH. Training uses alternating 125 ms blocks. Stationary and moving coefficients are frozen before predicting unused blocks. The original range/phase reference and symbol operator are preserved.</p>
<img src="arc-checks.png" alt="All ship arc prediction, information retention and phase errors">
<p>The middle panel is the moving-template energy remaining after eliminating unknown stationary coefficients, divided by its energy before that projection. A small value is an identifiability cost; a small aggregate phase error alone cannot undo it. Weak phases, negative predictions and search-bound contacts remain in the <a href="fits.json">full results</a>. Conditional thermal covariance does not include unknown target-model error. These are candidate checks, not an accepted physical antenna calibration.</p></html>'''.replace('STRENGTH', strength))


if __name__ == '__main__':
    main()
