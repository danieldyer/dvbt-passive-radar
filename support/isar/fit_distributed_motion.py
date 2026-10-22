"""Refit physical translation using a regularized distributed fixed-N/E scene.

The penalty is frozen from nested training-only temporal validation. Search
uses information-weighted representative epochs; exact original-row values,
gradients, feasibility and KKT conditions govern final acceptance. No yaw,
antenna coordinate, range-origin or channel calibration change is introduced.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import quadrature
from distributed_scene_cv import inner_masks
from field import Field
from fit_physical_profile import solve
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy, representative
from refine_full_capture_motion import constraints
from ridge_motion import RidgeMotion, profile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def aggregate(checks):
    predicted = sum(c['prediction_energy'] for c in checks)
    cross = sum(complex(*c['cross_real_imag']) for c in checks)
    error = sum(c['prediction_error'] for c in checks)
    total = sum(c['stationary_only_energy'] for c in checks)
    return {'prediction_error': error, 'stationary_only_energy': total,
        'heldout_explained_fraction': 1-error/total,
        'heldout_matched_energy': float(abs(cross)**2/predicted),
        'phase_rms_degrees_by_channel': [c['phase_rms_degrees'] for c in checks]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=20.)
    p.add_argument('--iterations', type=int, default=80)
    args = p.parse_args()
    source = json.loads(args.training.read_text())
    if not source['complete'] or [c['channel'] for c in source['channels']] != list(range(5)):
        raise ValueError('Finish all five training-only penalty selections')
    model_path = Path(source['source_motion'])
    model = json.loads(model_path.read_text())
    if source['source_sha256'] != fingerprint(model_path):
        raise ValueError('Physical motion changed after penalty selection')
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw exactly zero')
    if args.output.exists():
        raise ValueError('Preserve previous motion fits')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    points, weights = quadrature(source['bounds_enu_m'], args.spacing)
    bands, fast, strengths, null_checks = [], [], [], []
    for c in range(5):
        chosen = next(t for t in source['channels'][c]['trials'] if t['spacing_m'] == args.spacing)
        strengths.append(chosen['tuning']['regularization_strength'])
        path = Path(source['fields'])/f'channel-{c}.npz'
        if source['channels'][c]['field_sha256'] != fingerprint(path):
            raise ValueError('Original field changed after penalty selection')
        with np.load(path) as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        selected = np.floor((field.time-curve.interval[0])/source['holdout_seconds']).astype(int)%2 == 0
        local = []
        for f in (field, representative(field)):
            prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+f.data['row_time'], model)
            local.append(Band(f, prior, tx[model['reference_channel']], rx, points, model, selected,
                range_bias_m=model['range_bias_m'][c]))
        local[1].nuisance, local[1].clean, local[1].energy = local[0].nuisance, local[0].clean, local[0].energy
        zero_error = 0.
        for fold in range(3):
            _, check = inner_masks(field.time, model['interval_s'], source['holdout_seconds'], fold)
            check = np.repeat(check, field.y.shape[1])
            nuisance = Stationary(local[0].nuisance.design, check)
            value = nuisance.remove(field.y.ravel()[check])
            zero_error += float(np.vdot(value, value).real)
        finite_error = min(chosen['tuning']['inner_prediction_errors'])
        is_null = zero_error <= finite_error
        null_checks.append({'channel': c, 'zero_prediction_inner_error': zero_error,
            'finite_penalty_inner_error': finite_error, 'zero_prediction_selected': is_null,
            'scope': 'Complete the training-only penalty search with its infinite-penalty, zero-signal limit; no outer test observation is read.'})
        if is_null:
            strengths[-1] = None
        bands.append(local[0]); fast.append(local[1])
    exact = RidgeMotion(bands, model['parameters'], [weights]*5, strengths)
    approximate = RidgeMotion(fast, model['parameters'], [weights]*5, strengths)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    frozen = (exact.active >= curve.sections[1].start)&(exact.active < curve.sections[1].stop)
    lower[frozen] = upper[frozen] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = exact.validation(exact.initial)
    identity = {'training': str(args.training), 'training_sha256': fingerprint(args.training),
        'source_motion': str(model_path), 'source_motion_sha256': fingerprint(model_path),
        'spacing_m': args.spacing, 'fixed_penalties': strengths, 'null_checks': null_checks, 'scope': __doc__}
    started = wall_time.monotonic()
    def checkpoint(stage):
        def save(parameters, trace):
            print(json.dumps({'stage': stage, **trace[-1]}), flush=True)
            args.output.with_suffix('.checkpoint.json').write_text(json.dumps({**identity, 'stage': stage,
                'parameters': exact.expand(parameters).tolist(), 'joint_parameters': parameters.tolist(),
                'trace': trace}, indent=2)+'\n')
        return save
    parameters, search = solve(approximate, matrix, lo, hi, lower, upper, iterations=args.iterations,
        checkpoint=checkpoint('search'))
    print(json.dumps({'stage': 'search_complete', 'converged': search['converged'],
        'objective': search['objective'], 'kkt': search['maximum_kkt_residual']}), flush=True)
    exact.initial = parameters.copy()
    parameters, fit = solve(exact, matrix, lo, hi, lower, upper, iterations=0,
        curvature_gradient=lambda p: approximate.objective(p)[1], checkpoint=checkpoint('exact'))
    validation = exact.validation(parameters)
    errors = []
    for a, b, strength in zip(bands, fast, strengths):
        if strength is None:
            errors.append({'zero_prediction_selected': True})
            continue
        full = profile(a, exact.expand(parameters), np.sqrt(weights), strength)
        short = profile(b, exact.expand(parameters), np.sqrt(weights), strength)
        errors.append({'per_point_maximum_loss_db': max(discrepancy(full['target'][:, j], short['target'][:, j])[
            'matched_snr_loss_db'] for j in range(len(points))),
            'same_coefficient_prediction': discrepancy(full['target']@full['physical_coefficient'],
                short['target']@full['physical_coefficient']),
            'refitted_prediction': discrepancy(full['target']@full['physical_coefficient'],
                short['target']@short['physical_coefficient'])})
    before_summary, after_summary = aggregate(before), aggregate(validation)
    result = {**model, **identity, 'parameters': exact.expand(parameters).tolist(), 'fit': fit,
        'search_fit': search, 'before': before, 'after': validation,
        'before_distributed_summary': before_summary, 'after_distributed_summary': after_summary,
        'heldout_matched_energy_gain_db': float(10*np.log10(after_summary['heldout_matched_energy']/before_summary['heldout_matched_energy'])),
        'heldout_prediction_error_change_percent': 100*(after_summary['prediction_error']/before_summary['prediction_error']-1),
        'distributed_points_enu_m': points.tolist(), 'search_approximation_check': errors,
        'scope': __doc__, 'phase_centres_validated': False, 'calibration_validated': False,
        'selected_points_body_m': [points.tolist()]*5,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('fit_distributed_motion.py', 'ridge_motion.py', 'distributed_scene_cv.py', 'full_motion.py',
             'field.py', 'spline_motion.py', 'midpoint_model.py', 'stationarity.py', 'fit_physical_profile.py')},
        'seconds': wall_time.monotonic()-started, 'complete': True}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': fit['converged'], 'kkt': fit['maximum_kkt_residual'],
        'gain_db': result['heldout_matched_energy_gain_db'], 'error_percent': result['heldout_prediction_error_change_percent']}), flush=True)


if __name__ == '__main__':
    main()
