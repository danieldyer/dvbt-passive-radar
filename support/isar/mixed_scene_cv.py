"""Select an added diffuse component while retaining the original reflectors.

The core-only prediction is an explicit candidate. The added scene uses the
same per-area quadrature at every spatial refinement. Three inner temporal
folds select a fixed penalty; untouched outer blocks score the result. Physical
motion, geographic orientation, hardware range origins and gains stay fixed.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import fit_scene, prediction_metrics, quadrature
from distributed_scene_cv import inner_masks
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from mixed_scene import MixedPath, mixed_fit
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def choose_penalty(band, target, scale, core, duration):
    full = band.nuisance.remove(target[band.selected])*scale
    path = MixedPath(full, band.clean, core)
    relative = np.logspace(-30., 2., 65)
    strengths = [*(path.path.singular[0]**2*relative), None]
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
        coefficient = MixedPath(a, y, core).coefficients(strengths)
        nuisance = Stationary(band.nuisance.design, test)
        a = nuisance.remove(target[test])*scale
        y = nuisance.remove(band.field.y.ravel()[test])
        values = np.sum(abs(a@coefficient-y[:, None])**2, axis=0)
        scores += values
        folds.append({'fold': fold, 'training_time_indices': np.flatnonzero(train_time).tolist(),
            'validation_time_indices': np.flatnonzero(test_time).tolist(), 'prediction_errors': values.tolist()})
    index = int(np.argmin(scores))
    strength = strengths[index]
    coefficient = path.coefficients([strength])[:, 0]
    a = full if strength is not None else full[:, :core]
    check = mixed_fit(a, band.clean, core, strength)
    difference = float(np.linalg.norm(full@coefficient-a@check['coefficient'])/np.linalg.norm(a@check['coefficient']))
    if difference > 1e-7:
        raise ValueError(f'The independent scene solvers disagree: {difference:g}')
    return coefficient*scale, {'regularization_strength': strength,
        'relative_strength': None if strength is None else float(relative[index]),
        'baseline_selected': strength is None, 'strengths': strengths,
        'inner_prediction_errors': scores.tolist(), 'inner_folds': folds,
        'grid_edge': strength is not None and index in (0, len(strengths)-2),
        'effective_scene_rank': check['effective_rank'], 'augmented_condition': check['condition'],
        'independent_linear_prediction_relative_error': difference,
        'diffuse_singular_values': path.path.singular.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--spacing', type=float, nargs='+', default=[20., 10.])
    p.add_argument('--bounds', type=float, nargs=4, default=[-120., 120., -180., 200.])
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--holdout-seconds', type=float, default=.5)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients) or args.holdout_seconds <= 0:
        raise ValueError('Require zero yaw and positive holdout duration')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output/'experiment.json'
    if manifest.exists():
        raise ValueError('Use a fresh output directory')
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_sha256': fingerprint(args.motion),
        'fields': str(args.fields), 'bounds_enu_m': args.bounds, 'spacing_m': args.spacing,
        'holdout_seconds': args.holdout_seconds, 'yaw_fitted': False, 'motion_fitted': False,
        'point_coordinate_frame': 'east_north_up', 'height_m': 20., 'channels': [], 'complete': False,
        'limitation': 'Motion and point support were selected using this aperture previously. This is conditional transfer, not independent target discovery. Independent complex band responses do not calibrate inter-band phase.',
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('mixed_scene_cv.py', 'mixed_scene.py', 'distributed_fixed_frame.py', 'distributed_scene_cv.py',
             'blocked_ridge.py', 'full_motion.py', 'field.py', 'spline_motion.py', 'stationary.py')}}
    started = wall_time.monotonic()
    for c in args.channels:
        with np.load(args.fields/f'channel-{c}.npz') as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        selected = np.floor((field.time-curve.interval[0])/args.holdout_seconds).astype(int)%2 == 0
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        core_points = np.asarray(model['selected_points_body_m'][c])
        band = Band(field, prior, tx[model['reference_channel']], rx, core_points,
            model, selected, range_bias_m=model['range_bias_m'][c])
        target, _ = band.design(model['parameters'])
        control, _ = fit_scene(band, target)
        trials = []
        for spacing in args.spacing:
            points, weights = quadrature(args.bounds, spacing)
            band.points = np.r_[core_points, points]
            scale = np.sqrt(np.r_[np.ones(len(core_points)), weights])
            target, _ = band.design(model['parameters'])
            coefficient, tuning = choose_penalty(band, target, scale, len(core_points), args.holdout_seconds)
            prediction = (target@coefficient).reshape(field.y.shape)
            nuisance = Stationary(band.nuisance.design, ~band.selected)
            y = nuisance.remove(field.y.ravel()[~band.selected]).reshape((~selected).sum(), -1)
            value = nuisance.remove(prediction.ravel()[~band.selected]).reshape(y.shape)
            validation = prediction_metrics(y, value, field.time[~selected])
            baseline = control['validation']
            path = args.output/f'channel-{c}-{spacing:g}m.npz'
            np.savez(path, time=field.time, selected=selected, prediction=prediction,
                points_enu_m=band.points, quadrature_weights=scale**2, coefficients=coefficient)
            trial = {'spacing_m': spacing, 'core_points_enu_m': core_points.tolist(),
                'diffuse_points_enu_m': points.tolist(), 'tuning': tuning, 'validation': validation,
                'heldout_matched_energy_gain_db': float(10*np.log10(validation['heldout_matched_energy']/baseline['heldout_matched_energy'])),
                'heldout_prediction_error_change_percent': 100*(validation['prediction_error']/baseline['prediction_error']-1),
                'data': str(path), 'sha256': fingerprint(path), 'seconds': wall_time.monotonic()-started}
            trials.append(trial)
            result['running_channel'] = {'channel': c, 'control': control, 'trials': trials}
            manifest.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'channel': c, 'spacing_m': spacing,
                'rank': tuning['effective_scene_rank'], 'relative_strength': tuning['relative_strength'],
                'gain_db': trial['heldout_matched_energy_gain_db'],
                'error_percent': trial['heldout_prediction_error_change_percent'],
                'explained': validation['heldout_explained_fraction'], 'seconds': trial['seconds']}), flush=True)
            del target
        result['channels'].append({'channel': c, 'control': control, 'trials': trials,
            'field_sha256': fingerprint(args.fields/f'channel-{c}.npz')})
        result.pop('running_channel', None)
        manifest.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=wall_time.monotonic()-started)
    manifest.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
