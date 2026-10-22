"""Secondary-only range/Doppler covariance with nested output-power selection.

The test echo is an exact, frozen, yaw-free physical template. Every filter
has unit complex response to it and rejects the same stationary nuisance.
The outer test range and times are excluded from all covariance training and
regularization selection. The primary ship-bearing crop never trains weights.
Secondaries may contain real echoes; improvement means less interference to
these specified echoes, not certified sea-clutter or thermal-noise efficiency.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from matrix_spectral_gls import fit_covariance, matched_filters, shrink, statistics
from scene_dictionary import fingerprint
from secondary_filter_benchmark import templates
from spectral_gls import orthogonal_nuisance
from spline_motion import SplineMotion


STRENGTHS = np.array([0., .125, .25, .5, .75, 1.])
EDGES = [.5, 2., 5.]


def frequency_labels(rows, step):
    return np.digitize(abs(np.fft.fftfreq(rows, step)), EDGES)


def observation(original, saved_q, start, rows):
    select = slice(start, start+rows)
    q = orthogonal_nuisance(saved_q[select])
    return (np.fft.fft(original[select], axis=0, norm='ortho'),
        np.fft.fft(q, axis=0, norm='ortho'))


def signal_block(point_time, point_values, times):
    index = int(np.searchsorted(point_time, times[0]))
    np.testing.assert_array_equal(point_time[index:index+len(times)], times)
    a = np.fft.fft(point_values[index:index+len(times)], axis=0, norm='ortho')
    return a/np.sqrt(np.sum(abs(a)**2, axis=(0, 1)))[None, None, :]


def score(a, y, q, labels, covariance):
    h, information = matched_filters(a, q, labels, covariance)
    flat = h.reshape(-1, h.shape[-1])
    response_error = np.max(abs(np.sum(h.conj()*a, axis=(0, 1))-1))
    nuisance_error = np.max(abs(q.reshape(y.size, -1).conj().T@flat))
    return {'amplitude': flat.conj().T@y.ravel(), 'nominal_variance': 1/information,
        'thermal_variance': np.sum(abs(flat)**2, axis=0),
        'response_error': float(response_error), 'nuisance_error': float(nuisance_error)}


def fold_keys(records, withheld_group, withheld_tiles):
    """Keep both outer exclusions in every inner split."""
    return [key for key in records if key[0] != withheld_group and key[1] not in withheld_tiles]


def select_covariance(records, observations, group, train_tiles, hypotheses, labels):
    scores = np.zeros(len(STRENGTHS))
    folds = []
    for split in np.array_split(train_tiles, 3):
        learn = fold_keys(records, group, split)
        check = [key for key in records if key[0] != group and key[1] in split]
        covariance, diagnostic = fit_covariance([records[key] for key in learn], iterations=300)
        if not diagnostic['converged']:
            raise ArithmeticError(f'Inner covariance did not converge: {diagnostic}')
        split_score = np.zeros(len(STRENGTHS))
        for j, strength in enumerate(STRENGTHS):
            trial = shrink(covariance, strength)
            for key in check:
                y, q = observations[key]
                result = score(hypotheses, y, q, labels, trial)
                split_score[j] += float(np.sum(abs(result['amplitude'])**2))
        scores += split_score
        folds.append({'withheld_tiles': split.tolist(), 'fit': diagnostic,
            'output_power_sum': split_score.tolist(), 'learn_keys': learn, 'check_keys': check})
    keys = fold_keys(records, group, [])
    covariance, diagnostic = fit_covariance([records[key] for key in keys], iterations=300)
    if not diagnostic['converged']:
        raise ArithmeticError(f'Outer covariance did not converge: {diagnostic}')
    strength = float(STRENGTHS[np.argmin(scores)])
    return covariance, strength, {'fit': diagnostic, 'inner_folds': folds,
        'strength_candidates': STRENGTHS.tolist(), 'inner_output_power_sum': scores.tolist(),
        'selected_strength': strength, 'learn_keys': keys}


def summarize(power, thermal):
    mean = power.mean(axis=tuple(range(power.ndim-1)))
    return {'output_interference_power': mean.tolist(),
        'known_echo_snr_gain_db': (10*np.log10(mean[0]/mean)).tolist(),
        'mean_white_noise_penalty_db': (10*np.log10((thermal/thermal[..., :1]).mean(
            axis=tuple(range(thermal.ndim-1))))).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('secondary', 'fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--long-seconds', type=float, default=30.)
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    args = p.parse_args()
    started = wall_time.monotonic()
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Keep yaw absent from this experiment')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'benchmark.json').exists():
        raise ValueError('Choose a fresh output directory; preserve previous experiments')
    points = np.array([[-100., -30., 20.], [0., 0., 20.], [100., 30., 20.]])
    report = {'scope': __doc__, 'source': str(args.secondary/'secondary.json'),
        'source_sha256': fingerprint(args.secondary/'secondary.json'),
        'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('matrix_filter_benchmark.py', 'matrix_spectral_gls.py', 'secondary_filter_benchmark.py',
             'field.py', 'spline_motion.py', 'spectral_gls.py')},
        'point_coordinates_m': points.tolist(), 'tile_rows': args.rows,
        'methods': ['unit_covariance', 'secondary_matrix_reml', 'secondary_matrix_reml_inner_cv'],
        'spectral_partition_edges_hz': EDGES, 'channels': [], 'complete': False,
        'production_correction_adopted': False,
        'model': 'Proper-complex Gaussian, block diagonal in unitary Fourier coordinates; full range matrix per absolute-Doppler band, each covariance >= I after existing thermal whitening.',
        'selection': 'C(lambda) = I + lambda (C_REML - I). Inner disjoint-time CV minimizes total unit-response output power on other secondary range groups. Lambda is then frozen for both test durations.',
        'signal_normalization': 'Only known test templates are normalized to unit input energy; original recording data are never power-normalized.',
        'conditional_scope': 'Previously selected fixed N/E motion supplies known hypothetical echoes, not independent physical ground truth. The outer secondary range and all outer test times are excluded from fitting and selection. No primary observations train weights. Long-aperture filters reuse short-aperture covariance and regularization.'}
    for c in args.channels:
        channel_start = wall_time.monotonic()
        with np.load(args.secondary/f'channel-{c}.npz') as data:
            time, original, shifts = data['time'], data['original'], data['shifts']
            n, r = args.rows, original.shape[-1]
            step = float(np.median(np.diff(time)))
            labels = frequency_labels(n, step)
            starts = np.arange(0, len(time)-n+1, n)
            point_time, point_values = templates(args.fields/f'channel-{c}.npz', model, c, points)
            test_tiles = [j for j, start in enumerate(starts) if
                time[start] >= point_time[0] and time[start+n-1] <= point_time[-1]]
            train_tiles = np.array([j for j in range(len(starts)) if j not in test_tiles])
            if len(test_tiles) < 3 or len(train_tiles) < 3:
                raise ValueError('Need separate complete training and test time blocks')
            signals = [signal_block(point_time, point_values, time[starts[t]:starts[t]+n]) for t in test_tiles]
            hypotheses = np.concatenate(signals, axis=2)
            long_rows = int(round(args.long_seconds/step))
            long_start = int(starts[test_tiles[0]])
            long_tiles = np.unique(np.arange(long_start, long_start+long_rows)//n)
            if not set(long_tiles).issubset(test_tiles):
                raise ValueError('Long test aperture must fit entirely within excluded time tiles')
            long_times = time[long_start:long_start+long_rows]
            long_signal = signal_block(point_time, point_values, long_times)
            long_labels = frequency_labels(long_rows, step)
            secondary = np.flatnonzero(shifts != 0)
            records, observations, long_observations = {}, {}, {}
            for g in secondary:
                saved_q = data[f'nuisance_q_{g}']
                for tile, start in enumerate(starts):
                    y, q = observation(original[g], saved_q, start, n)
                    observations[int(g), tile] = (y, q)
                    if tile in train_tiles:
                        records[int(g), tile] = statistics(y, q, labels)
                long_observations[int(g)] = observation(original[g], saved_q, long_start, long_rows)
            print(json.dumps({'channel': c, 'stage': 'statistics',
                'seconds': wall_time.monotonic()-channel_start}), flush=True)
        shape = (len(secondary), len(test_tiles), len(points), 3)
        amplitudes, thermal, nominal = np.empty(shape, complex), np.empty(shape), np.empty(shape)
        long_shape = (len(secondary), len(points), 3)
        long_amplitudes = np.empty(long_shape, complex)
        long_thermal, long_nominal = np.empty(long_shape), np.empty(long_shape)
        fits, covariances = [], []
        maximum_response_error = maximum_nuisance_error = 0.
        for gi, g in enumerate(secondary):
            covariance, strength, diagnostic = select_covariance(records, observations, int(g),
                train_tiles, hypotheses, labels)
            fits.append({'withheld_shift_samples': int(shifts[g]), **diagnostic})
            covariances.append(covariance)
            for method, trial in enumerate((shrink(covariance, 0.), covariance, shrink(covariance, strength))):
                for ti, tile in enumerate(test_tiles):
                    result = score(signals[ti], *observations[int(g), tile], labels, trial)
                    amplitudes[gi, ti, :, method] = result['amplitude']
                    thermal[gi, ti, :, method] = result['thermal_variance']
                    nominal[gi, ti, :, method] = result['nominal_variance']
                    maximum_response_error = max(maximum_response_error, result['response_error'])
                    maximum_nuisance_error = max(maximum_nuisance_error, result['nuisance_error'])
                result = score(long_signal, *long_observations[int(g)], long_labels, trial)
                long_amplitudes[gi, :, method] = result['amplitude']
                long_thermal[gi, :, method] = result['thermal_variance']
                long_nominal[gi, :, method] = result['nominal_variance']
                maximum_response_error = max(maximum_response_error, result['response_error'])
                maximum_nuisance_error = max(maximum_nuisance_error, result['nuisance_error'])
            print(json.dumps({'channel': c, 'withheld_shift': int(shifts[g]), 'cv_strength': strength,
                'short_gain_db': summarize(abs(amplitudes[gi])**2, thermal[gi])['known_echo_snr_gain_db'],
                'long_gain_db': summarize(abs(long_amplitudes[gi])**2, long_thermal[gi])['known_echo_snr_gain_db'],
                'elapsed_seconds': wall_time.monotonic()-channel_start}), flush=True)
        path = args.output/f'channel-{c}.npz'
        np.savez(path, amplitudes=amplitudes, thermal_variance=thermal, nominal_variance=nominal,
            long_amplitudes=long_amplitudes, long_thermal_variance=long_thermal, long_nominal_variance=long_nominal,
            covariance=np.array(covariances), shifts=shifts[secondary], strengths=[f['selected_strength'] for f in fits],
            test_tile_centres_s=[time[starts[t]:starts[t]+n].mean() for t in test_tiles])
        report['channels'].append({'channel': c, 'data': str(path), 'sha256': fingerprint(path),
            'training_tiles': train_tiles.tolist(), 'test_tiles': test_tiles,
            'test_intervals_s': [[float(time[starts[t]]), float(time[starts[t]+n-1])] for t in test_tiles],
            'long_test_interval_s': long_times[[0, -1]].tolist(),
            'short_aperture_seconds': n*step, 'long_aperture_seconds': long_rows*step,
            'short_band_counts': np.bincount(labels).tolist(), 'long_band_counts': np.bincount(long_labels).tolist(),
            'outer_fits': fits, 'short': summarize(abs(amplitudes)**2, thermal),
            'long': summarize(abs(long_amplitudes)**2, long_thermal),
            'maximum_unit_complex_response_error': maximum_response_error,
            'maximum_stationary_response_error': maximum_nuisance_error,
            'runtime_seconds': wall_time.monotonic()-channel_start})
        (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'channel': c, 'stage': 'complete', 'short': report['channels'][-1]['short'],
            'long': report['channels'][-1]['long']}), flush=True)
    report.update(complete=True, runtime_seconds=wall_time.monotonic()-started)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
