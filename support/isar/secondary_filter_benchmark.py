"""Test secondary-only spectral weighting on withheld range/time observations.

The matched filter has unit complex response to each exact yaw-free point
template and eliminates stationary nuisance in its own covariance metric.
Covariance is a piecewise spectral Gaussian model, estimated by REML from
other ranges and times. Regularization is selected by inner time-block CV.
The test compares interference power at unchanged injected signal response;
it does not mistake a lower weighted objective for improved physical SNR.
Secondary observations can include real echoes. No production change follows
from this experiment alone, and no measured CRLB claim is made for the ship.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from field import Field
from imaging import motion_state, sites
from scene_dictionary import fingerprint
from spectral_gls import (fit_variance, matched_filter, orthogonal_nuisance,
    restricted_likelihood, sufficient_statistics)
from spline_motion import SplineMotion


def templates(path, model, channel, points):
    with np.load(path) as d:
        field = Field(dict(d), d['pulse_samples']).windowed(
            model['interval_s'][0]+.03, model['interval_s'][1]-.03)
    tx, rx, _ = sites(Path(model['geometry_directory']))
    prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
    state = SplineMotion(model).apply(field.data['row_time'], prior, tx[model['reference_channel']], rx)
    values = []
    for point in points:
        relative = (prior['position_ecef_m']-field.data['centre_ecef_m']+
            state['translation_offset_m']+state['body_to_ecef']@point)
        position = state['position_ecef_m']+state['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        values.append(field.trajectory(position, velocity, position_offset=relative,
            range_bias_m=model['range_bias_m'][channel]))
    return field.time, np.stack(values, axis=-1)


def training_fit(records, training_tiles):
    """Inner CV uses no observations from the outer withheld range or times."""
    regularization = np.array([0., .25, .5, 1.])
    scores = np.zeros(len(regularization))
    inner = []
    for split in np.array_split(training_tiles, 3):
        learn = [s for t, s in records if t not in split]
        check = [s for t, s in records if t in split]
        fitted, diagnostic = fit_variance(learn)
        if not diagnostic['converged']:
            raise ValueError(f'Inner covariance fit not stationary: {diagnostic}')
        inner.append({'heldout_tiles': split.tolist(), **diagnostic})
        for j, strength in enumerate(regularization):
            scores[j] += sum(restricted_likelihood(fitted*strength, s)[0] for s in check)
    fitted, diagnostic = fit_variance([s for _, s in records])
    if not diagnostic['converged']:
        raise ValueError(f'Outer covariance fit not stationary: {diagnostic}')
    strength = float(regularization[np.argmin(scores)])
    return fitted, strength, {'fit': diagnostic, 'inner_fits': inner,
        'regularization_candidates': regularization.tolist(), 'inner_restricted_nll': scores.tolist(),
        'selected_regularization': strength,
        'regularization_definition': 'Multiply fitted log spectral variances by this factor; zero is the original unit covariance.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('secondary', 'fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--spectral-model', choices=['absolute_bands', 'signed_bands'], default='absolute_bands')
    args = p.parse_args()
    started = wall_time.monotonic()
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Keep yaw absent from this experiment')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'benchmark.json').exists():
        raise ValueError('Choose a fresh output directory to preserve the previous experiment')
    points = np.array([[-100., -30., 20.], [0., 0., 20.], [100., 30., 20.]])
    report = {'scope': __doc__, 'source': str(args.secondary/'secondary.json'),
        'source_sha256': fingerprint(args.secondary/'secondary.json'),
        'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('secondary_filter_benchmark.py', 'spectral_gls.py', 'field.py', 'spline_motion.py')},
        'point_coordinates_m': points.tolist(), 'tile_rows': args.rows,
        'methods': ['unit_covariance', 'secondary_reml', 'secondary_reml_inner_cv'],
        'spectral_model': args.spectral_model, 'channels': [], 'complete': False,
        'production_correction_adopted': False,
        'signal_normalization': 'Only injected templates are normalized to unit input energy. The recording is never power-normalized.',
        'validation_scope': 'Frozen, previously selected yaw-free motion supplies known echo hypotheses. All secondary covariance fitting excludes the outer test range group and every test time tile. The real ship recording is never used to estimate these weights.'}
    for c in range(5):
        with np.load(args.secondary/f'channel-{c}.npz') as data:
            time, original, shifts = data['time'], data['original'], data['shifts']
            n, r = args.rows, original.shape[-1]
            step = float(np.median(np.diff(time)))
            frequency = np.fft.fftfreq(n, step)
            edges = ([.5, 2., 5.] if args.spectral_model == 'absolute_bands' else
                [-5., -2., -1., -.5, -.25, -.1, .1, .25, .5, 1., 2., 5.])
            labels = np.digitize(abs(frequency) if args.spectral_model == 'absolute_bands' else frequency, edges)
            if not np.all(np.bincount(labels) > 0):
                raise ValueError('Every declared spectral band needs samples; increase --rows')
            starts = np.arange(0, len(time)-n+1, n)
            point_time, points_values = templates(args.fields/f'channel-{c}.npz', model, c, points)
            test_tiles = [j for j, start in enumerate(starts) if
                time[start] >= point_time[0] and time[start+n-1] <= point_time[-1]]
            train_tiles = [j for j in range(len(starts)) if j not in test_tiles]
            if len(test_tiles) < 3 or len(train_tiles) < 3:
                raise ValueError('Need separate complete training and test time blocks')
            signals = []
            for tile in test_tiles:
                times = time[starts[tile]:starts[tile]+n]
                index = int(np.searchsorted(point_time, times[0]))
                np.testing.assert_array_equal(point_time[index:index+n], times)
                a = np.fft.fft(points_values[index:index+n], axis=0, norm='ortho')
                a /= np.sqrt(np.sum(abs(a)**2, axis=(0, 1)))[None, None, :]
                signals.append(a)
            secondary = np.flatnonzero(shifts != 0)
            records, test_data = {}, {}
            for g in secondary:
                saved_q = data[f'nuisance_q_{g}']
                for tile, start in enumerate(starts):
                    rows = slice(start, start+n)
                    q = orthogonal_nuisance(saved_q[rows])
                    qhat = np.fft.fft(q, axis=0, norm='ortho')
                    yhat = np.fft.fft(original[g, rows], axis=0, norm='ortho')
                    if tile in test_tiles:
                        test_data[g, tile] = (yhat, qhat)
                    else:
                        records[g, tile] = sufficient_statistics(yhat, qhat, labels)
            amplitudes = np.empty((len(secondary), len(test_tiles), len(points), 3), complex)
            nominal_variance = np.empty(amplitudes.shape)
            thermal_penalty = np.empty(amplitudes.shape)
            fits = []
            maximum_response_error = maximum_nuisance_error = 0.
            for gi, g in enumerate(secondary):
                training = [(tile, s) for (other, tile), s in records.items() if other != g]
                logvar, strength, diagnostic = training_fit(training, np.array(train_tiles))
                fits.append({'heldout_range_shift_samples': int(shifts[g]), **diagnostic})
                for ti, tile in enumerate(test_tiles):
                    y, q = test_data[g, tile]
                    for pi in range(len(points)):
                        a = signals[ti][:, :, pi]
                        for method, values in enumerate((logvar*0., logvar, logvar*strength)):
                            h, information = matched_filter(a, q, labels, values)
                            amplitudes[gi, ti, pi, method] = np.vdot(h, y)
                            nominal_variance[gi, ti, pi, method] = 1/information
                            thermal_penalty[gi, ti, pi, method] = np.sum(abs(h)**2)
                            maximum_response_error = max(maximum_response_error, abs(np.vdot(h, a)-1))
                            maximum_nuisance_error = max(maximum_nuisance_error,
                                float(np.max(abs(q.reshape(n*r, -1).conj().T@h.ravel()))))
                print(json.dumps({'channel': c, 'heldout_shift': int(shifts[g]),
                    'variance': diagnostic['fit']['variance'], 'cv_strength': strength}), flush=True)
        power = abs(amplitudes)**2
        gain = 10*np.log10(power[..., 0].mean()/power.mean(axis=(0, 1, 2)))
        path = args.output/f'channel-{c}.npz'
        np.savez(path, amplitudes=amplitudes, nominal_variance=nominal_variance,
            thermal_variance=thermal_penalty, shifts=shifts[secondary],
            test_tile_centres_s=[time[starts[t]:starts[t]+n].mean() for t in test_tiles])
        report['channels'].append({'channel': c, 'data': str(path), 'sha256': fingerprint(path),
            'spectral_partition_edges_hz': edges,
            'spectral_partition_counts': np.bincount(labels).tolist(),
            'training_tiles': train_tiles, 'test_tiles': test_tiles,
            'test_intervals_s': [[float(time[starts[t]]), float(time[starts[t]+n-1])] for t in test_tiles],
            'outer_fits': fits, 'output_interference_power': power.mean(axis=(0, 1, 2)).tolist(),
            'heldout_injection_snr_gain_db': gain.tolist(),
            'maximum_unit_complex_response_error': float(maximum_response_error),
            'maximum_stationary_response_error': float(maximum_nuisance_error),
            'mean_white_noise_penalty_db': (10*np.log10((thermal_penalty/
                thermal_penalty[..., :1]).mean(axis=(0, 1, 2)))).tolist()})
        (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'channel': c, 'heldout_gain_db': gain.tolist()}), flush=True)
    report.update(complete=True, runtime_seconds=wall_time.monotonic()-started)
    (args.output/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
