"""Nested temporal prediction selects a fixed-N/E distributed-scene penalty.

This controls the overfitting exposed by forcing a diffuse scene to explain
all departures from unit thermal covariance. The outer test blocks never
select the penalty. Geometry, fixed orientation and physical motion remain
unchanged. Grid refinement retains the same per-unit-area scene convention.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from blocked_ridge import RidgePath
from distributed_fixed_frame import fit_scene, prediction_metrics, quadrature
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def inner_masks(time, interval, duration, fold):
    block = np.floor((time-interval[0])/duration).astype(int)
    outer = block%2 == 0
    heldout = outer & ((block//2)%3 == fold)
    return outer & ~heldout, heldout


def choose_penalty(band, target, weights, duration):
    scale = np.sqrt(weights)
    full = band.nuisance.remove(target[band.selected])*scale
    path = RidgePath(full, band.clean)
    relative = np.logspace(-30., 2., 65)
    strengths = path.singular[0]**2*relative
    scores = np.zeros(len(strengths))
    folds = []
    for fold in range(3):
        train_time, test_time = inner_masks(band.field.time, band.model['interval_s'], duration, fold)
        train, test = [np.repeat(v, band.field.y.shape[1]) for v in (train_time, test_time)]
        if np.any(train & test) or np.any((train | test) & ~band.selected):
            raise ValueError('Inner folds read an outer test observation')
        nuisance = Stationary(band.nuisance.design, train)
        a = nuisance.remove(target[train])*scale
        y = nuisance.remove(band.field.y.ravel()[train])
        inner = RidgePath(a, y)
        coefficient = inner.coefficients(strengths)
        nuisance = Stationary(band.nuisance.design, test)
        a = nuisance.remove(target[test])*scale
        y = nuisance.remove(band.field.y.ravel()[test])
        values = np.sum(abs(a@coefficient-y[:, None])**2, axis=0)
        scores += values
        folds.append({'fold': fold, 'training_time_indices': np.flatnonzero(train_time).tolist(),
            'validation_time_indices': np.flatnonzero(test_time).tolist(), 'prediction_error': values.tolist()})
    chosen = int(np.argmin(scores))
    strength = float(strengths[chosen])
    scaled_coefficient = path.coefficients([strength])[:, 0]
    coefficient = scaled_coefficient*scale
    return coefficient, {'regularization_strength': strength, 'relative_strength': float(relative[chosen]),
        'grid_edge': chosen in (0, len(strengths)-1), 'strengths': strengths.tolist(),
        'inner_prediction_errors': scores.tolist(), 'inner_folds': folds,
        'effective_scene_rank': float(np.sum(path.singular**2/(path.singular**2+strength))),
        'coefficient_penalty': float(strength*np.vdot(scaled_coefficient, scaled_coefficient).real),
        'singular_values': path.singular.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--spacing', type=float, nargs='+', default=[20., 10., 5.])
    p.add_argument('--bounds', type=float, nargs=4, default=[-120., 120., -180., 200.])
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--holdout-seconds', type=float, default=.5)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients) or args.holdout_seconds <= 0:
        raise ValueError('Need zero yaw in a fixed ENU frame and positive holdout duration')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'experiment.json').exists():
        raise ValueError('Use a fresh output directory')
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_sha256': fingerprint(args.motion),
        'fields': str(args.fields), 'bounds_enu_m': args.bounds, 'spacing_m': args.spacing,
        'holdout_seconds': args.holdout_seconds, 'yaw_fitted': False, 'motion_fitted': False,
        'point_coordinate_frame': 'east_north_up', 'height_m': 20., 'channels': [], 'complete': False,
        'limitation': 'Motion and prior support decisions remain conditional on this aperture. Independent complex band scenes do not calibrate their relative phase. Inner splits select only the penalty.',
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('distributed_scene_cv.py', 'distributed_fixed_frame.py', 'blocked_ridge.py', 'full_motion.py',
             'field.py', 'spline_motion.py', 'stationary.py')}}
    started = wall_time.monotonic()
    for c in args.channels:
        with np.load(args.fields/f'channel-{c}.npz') as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        selected = np.floor((field.time-curve.interval[0])/args.holdout_seconds).astype(int)%2 == 0
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        band = Band(field, prior, tx[model['reference_channel']], rx, model['selected_points_body_m'][c],
            model, selected, range_bias_m=model['range_bias_m'][c])
        target, _ = band.design(model['parameters'])
        control, prediction = fit_scene(band, target)
        np.savez(args.output/f'channel-{c}-control.npz', time=field.time, selected=selected, prediction=prediction)
        trials = []
        for spacing in args.spacing:
            points, weights = quadrature(args.bounds, spacing)
            band.points = points
            target, _ = band.design(model['parameters'])
            coefficient, tuning = choose_penalty(band, target, weights, args.holdout_seconds)
            prediction = (target@coefficient).reshape(field.y.shape)
            test_nuisance = Stationary(band.nuisance.design, ~band.selected)
            y = test_nuisance.remove(field.y.ravel()[~band.selected]).reshape((~selected).sum(), -1)
            value = test_nuisance.remove(prediction.ravel()[~band.selected]).reshape(y.shape)
            v = prediction_metrics(y, value, field.time[~selected])
            old = control['validation']
            trial = {'spacing_m': spacing, 'points_enu_m': points.tolist(), 'tuning': tuning, 'validation': v,
                'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist(),
                'heldout_matched_energy_gain_db': float(10*np.log10(v['heldout_matched_energy']/old['heldout_matched_energy'])),
                'heldout_prediction_error_change_percent': 100*(v['prediction_error']/old['prediction_error']-1),
                'seconds': wall_time.monotonic()-started}
            path = args.output/f'channel-{c}-{spacing:g}m.npz'
            np.savez(path, time=field.time, selected=selected, prediction=prediction,
                points_enu_m=points, quadrature_weights=weights, coefficients=coefficient)
            trial.update(data=str(path), sha256=fingerprint(path))
            trials.append(trial)
            result['running_channel'] = {'channel': c, 'control': control, 'trials': trials}
            (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'channel': c, 'spacing_m': spacing, 'points': len(points),
                'rank': tuning['effective_scene_rank'], 'relative_strength': tuning['relative_strength'],
                'explained': v['heldout_explained_fraction'], 'gain_db': trial['heldout_matched_energy_gain_db'],
                'seconds': trial['seconds']}), flush=True)
            del target
        result['channels'].append({'channel': c, 'control': control, 'trials': trials,
            'field_sha256': fingerprint(args.fields/f'channel-{c}.npz')})
        result.pop('running_channel', None)
        (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=wall_time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
