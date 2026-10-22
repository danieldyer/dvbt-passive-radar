"""Refit physical translation with a nested reflector-plus-diffuse scene.

Yaw is zero. Training-selected per-area diffuse penalties are held fixed;
core-only channels retain their original reflectors and motion information.
The exact original-row operator governs final stationarity and all validation.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import quadrature
from field import Field
from fit_distributed_motion import aggregate
from fit_physical_profile import solve
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy, representative
from mixed_scene import MixedMotion, profile
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def changes(before, after):
    return {'matched_energy_change_db': float(10*np.log10(after['heldout_matched_energy']/before['heldout_matched_energy'])),
        'prediction_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=20.)
    p.add_argument('--iterations', type=int, default=100)
    args = p.parse_args()
    source = json.loads(args.training.read_text())
    if not source['complete'] or [c['channel'] for c in source['channels']] != list(range(5)):
        raise ValueError('Finish all five penalty selections')
    model_path = Path(source['source_motion'])
    model = json.loads(model_path.read_text())
    if source['source_sha256'] != fingerprint(model_path):
        raise ValueError('Source motion changed after penalty selection')
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw exactly zero')
    if args.output.exists():
        raise ValueError('Preserve previous fits')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    diffuse, area = quadrature(source['bounds_enu_m'], args.spacing)
    bands, fast, core_bands, weights, strengths, cores = [], [], [], [], [], []
    for c in range(5):
        trial = next(t for t in source['channels'][c]['trials'] if t['spacing_m'] == args.spacing)
        strength = trial['tuning']['regularization_strength']
        core = np.asarray(trial['core_points_enu_m'])
        np.testing.assert_array_equal(core, model['selected_points_body_m'][c])
        cores.append(len(core)); strengths.append(strength)
        points = core if strength is None else np.r_[core, diffuse]
        weights.append(np.ones(len(core)) if strength is None else np.r_[np.ones(len(core)), area])
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
        bands.append(local[0]); fast.append(local[1])
        core_bands.append(Band(field, local[0].prior, tx[model['reference_channel']], rx, core, model, selected,
            range_bias_m=model['range_bias_m'][c]))
    exact = MixedMotion(bands, model['parameters'], weights, cores, strengths)
    approximate = MixedMotion(fast, model['parameters'], weights, cores, strengths)
    core_only = MixedMotion(core_bands, model['parameters'], [np.ones(n) for n in cores], cores, [None]*5)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    frozen = (exact.active >= curve.sections[1].start)&(exact.active < curve.sections[1].stop)
    lower[frozen] = upper[frozen] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = wall_time.monotonic()
    before = exact.validation(exact.initial)
    baseline = core_only.validation(core_only.initial)
    identity = {'training': str(args.training), 'training_sha256': fingerprint(args.training),
        'source_motion': str(model_path), 'source_motion_sha256': fingerprint(model_path),
        'spacing_m': args.spacing, 'fixed_penalties': strengths, 'core_counts': cores,
        'scene_kind': 'unpenalized_core_plus_area_weighted_diffuse', 'scope': __doc__}
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
    fit['convergence_scope'] = 'First-order constrained stationarity of the declared partially regularized scene model; not global optimality, physical truth or a CRLB certificate.'
    after = exact.validation(parameters)
    sparse_after = core_only.validation(parameters)
    errors = []
    for a, b, weight, core, strength in zip(bands, fast, weights, cores, strengths):
        full = profile(a, exact.expand(parameters), np.sqrt(weight), core, strength)
        short = profile(b, exact.expand(parameters), np.sqrt(weight), core, strength)
        errors.append({'per_point_maximum_loss_db': max(discrepancy(full['target'][:, j], short['target'][:, j])[
            'matched_snr_loss_db'] for j in range(len(a.points))),
            'same_coefficient_prediction': discrepancy(full['target']@full['physical_coefficient'],
                short['target']@full['physical_coefficient']),
            'refitted_prediction': discrepancy(full['target']@full['physical_coefficient'],
                short['target']@short['physical_coefficient'])})
    summaries = {key: aggregate(value) for key, value in
        [('baseline', baseline), ('before', before), ('after', after), ('sparse_after', sparse_after)]}
    output = {**model, **identity, 'parameters': exact.expand(parameters).tolist(), 'fit': fit,
        'search_fit': search, 'baseline': baseline, 'before': before, 'after': after,
        'sparse_at_new_motion': sparse_after, 'summaries': summaries,
        'motion_change': changes(summaries['before'], summaries['after']),
        'baseline_comparison': changes(summaries['baseline'], summaries['after']),
        'sparse_motion_comparison': changes(summaries['baseline'], summaries['sparse_after']),
        'search_approximation_check': errors, 'core_points_enu_m': [b.points.tolist() for b in core_bands],
        'selected_points_body_m': [b.points.tolist() for b in bands],
        'phase_centres_validated': False, 'calibration_validated': False,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('fit_mixed_motion.py', 'mixed_scene.py', 'mixed_scene_cv.py', 'ridge_motion.py',
             'full_motion.py', 'field.py', 'spline_motion.py', 'midpoint_model.py',
             'stationarity.py', 'fit_physical_profile.py')},
        'seconds': wall_time.monotonic()-started, 'complete': True}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': fit['converged'], 'kkt': fit['maximum_kkt_residual'],
        'motion_change': output['motion_change'], 'baseline_comparison': output['baseline_comparison'],
        'sparse_motion_comparison': output['sparse_motion_comparison']}), flush=True)


if __name__ == '__main__':
    main()
