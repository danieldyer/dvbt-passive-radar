"""Thirty-second secondary covariance, with nested exclusion across ranges.

Unlike the earlier strict range-and-time transfer test, training uses the
same time aperture as the test. Both covariance and output-power selection
exclude the outer test range, and the primary crop is never training data.
This tests local adaptation rather than covariance persistence over time.
The two test apertures overlap and must not be counted as independent trials.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from matrix_filter_benchmark import (EDGES, STRENGTHS, frequency_labels, observation,
    score, signal_block, summarize)
from matrix_spectral_gls import fit_covariance, shrink, statistics
from scene_dictionary import fingerprint
from secondary_filter_benchmark import templates
from spline_motion import SplineMotion


def fit(records, keys):
    covariance, diagnostic = fit_covariance([records[k] for k in keys], iterations=300)
    if not diagnostic['converged']:
        raise ArithmeticError(f'Covariance did not converge: {diagnostic}')
    return covariance, diagnostic


def select(records, observations, group, signal, labels):
    keys = [g for g in records if g != group]
    scores = np.zeros(len(STRENGTHS))
    folds = []
    for holdout in keys:
        learn = [g for g in keys if g != holdout]
        covariance, diagnostic = fit(records, learn)
        values = [float(np.sum(abs(score(signal, *observations[holdout], labels,
            shrink(covariance, strength))['amplitude'])**2)) for strength in STRENGTHS]
        scores += values
        folds.append({'withheld_group': holdout, 'learn_groups': learn,
            'fit': diagnostic, 'output_power_sum': values})
    covariance, diagnostic = fit(records, keys)
    strength = float(STRENGTHS[np.argmin(scores)])
    return covariance, strength, {'fit': diagnostic, 'inner_folds': folds, 'learn_groups': keys,
        'inner_output_power_sum': scores.tolist(), 'selected_strength': strength,
        'strength_candidates': STRENGTHS.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('secondary', 'fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    args = p.parse_args()
    started = wall_time.monotonic()
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Yaw must remain exactly zero')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'benchmark.json').exists():
        raise ValueError('Choose a fresh output directory')
    points = np.array([[-100., -30., 20.], [0., 0., 20.], [100., 30., 20.]])
    report = {'scope': __doc__, 'source': str(args.secondary/'secondary.json'),
        'source_sha256': fingerprint(args.secondary/'secondary.json'),
        'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('local_matrix_filter_benchmark.py', 'matrix_filter_benchmark.py', 'matrix_spectral_gls.py',
             'secondary_filter_benchmark.py', 'field.py', 'spline_motion.py')},
        'point_coordinates_m': points.tolist(),
        'methods': ['unit_covariance', 'secondary_matrix_reml', 'secondary_matrix_reml_inner_cv'],
        'spectral_partition_edges_hz': EDGES, 'channels': [], 'complete': False,
        'production_correction_adopted': False,
        'selection': 'Same aperture throughout; outer heldout range absent from every inner fold. Inner leave-one-secondary-range-out unit-response output power selects C(lambda)=I+lambda(C_REML-I).',
        'conditional_scope': 'Known hypothetical echoes from previously selected no-yaw motion. Other real echoes in secondary observations count as interference. The primary crop never trains covariance. Two apertures overlap; range returns may be correlated. This is not a population sensitivity guarantee.'}
    for c in args.channels:
        channel_started = wall_time.monotonic()
        with np.load(args.secondary/f'channel-{c}.npz') as data:
            time, original, shifts = data['time'], data['original'], data['shifts']
            step = float(np.median(np.diff(time)))
            n = int(round(30./step))
            labels = frequency_labels(n, step)
            starts = [3*512, 4*512]
            point_time, point_values = templates(args.fields/f'channel-{c}.npz', model, c, points)
            secondary = [int(g) for g in np.flatnonzero(shifts != 0)]
            shape = (len(secondary), len(starts), len(points), 3)
            amplitudes = np.empty(shape, complex)
            thermal, nominal = np.empty(shape), np.empty(shape)
            covariances, fits = [], []
            maximum_response_error = maximum_nuisance_error = 0.
            for ti, start in enumerate(starts):
                signal = signal_block(point_time, point_values, time[start:start+n])
                observations, records = {}, {}
                for g in secondary:
                    observations[g] = observation(original[g], data[f'nuisance_q_{g}'], start, n)
                    records[g] = statistics(*observations[g], labels)
                tile_covariances = []
                for gi, g in enumerate(secondary):
                    covariance, strength, diagnostic = select(records, observations, g, signal, labels)
                    fits.append({'aperture': ti, 'withheld_shift_samples': int(shifts[g]), **diagnostic})
                    tile_covariances.append(covariance)
                    for method, trial in enumerate((shrink(covariance, 0.), covariance, shrink(covariance, strength))):
                        result = score(signal, *observations[g], labels, trial)
                        amplitudes[gi, ti, :, method] = result['amplitude']
                        thermal[gi, ti, :, method] = result['thermal_variance']
                        nominal[gi, ti, :, method] = result['nominal_variance']
                        maximum_response_error = max(maximum_response_error, result['response_error'])
                        maximum_nuisance_error = max(maximum_nuisance_error, result['nuisance_error'])
                    print(json.dumps({'channel': c, 'aperture': ti, 'withheld_shift': int(shifts[g]),
                        'strength': strength, 'gain_db': summarize(abs(amplitudes[gi, ti])**2,
                            thermal[gi, ti])['known_echo_snr_gain_db'],
                        'elapsed_seconds': wall_time.monotonic()-channel_started}), flush=True)
                covariances.append(tile_covariances)
        path = args.output/f'channel-{c}.npz'
        np.savez(path, amplitudes=amplitudes, thermal_variance=thermal, nominal_variance=nominal,
            covariance=np.array(covariances), shifts=shifts[secondary],
            strengths=[f['selected_strength'] for f in fits])
        report['channels'].append({'channel': c, 'data': str(path), 'sha256': fingerprint(path),
            'intervals_s': [[float(time[start]), float(time[start+n-1])] for start in starts],
            'aperture_seconds': n*step, 'rows': n, 'outer_fits': fits,
            'band_counts': np.bincount(labels).tolist(),
            **summarize(abs(amplitudes)**2, thermal),
            'maximum_unit_complex_response_error': maximum_response_error,
            'maximum_stationary_response_error': maximum_nuisance_error,
            'runtime_seconds': wall_time.monotonic()-channel_started})
        (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'channel': c, 'stage': 'complete',
            'gain_db': report['channels'][-1]['known_echo_snr_gain_db']}), flush=True)
    report.update(complete=True, runtime_seconds=wall_time.monotonic()-started)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
