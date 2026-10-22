"""Constrained physical path/yaw and envelope-origin fit with a profiled scene.

Roll, pitch and antenna locations stay fixed. Shared physical reflector
locations have independent complex band responses, re-solved at every trial.
Odd interior one-second blocks are withheld; both aperture edges are observed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, OptimizeResult, minimize

from ais import read_csv, utc_seconds
from field import Field
from fit_joint_motion import information_scaling
from full_motion import Band
from imaging import physical_state, sites
from kinematics import constraints
from physical_profile import PhysicalProfile
from scene_dictionary import fingerprint
from stationarity import check, polish
from validate_motion import summarize


def solve(joint, matrix, lo, hi, lower, upper, *, iterations=100, checkpoint=None, polish_stationarity=True,
          curvature_gradient=None):
    initial = joint.initial.copy()
    first = joint.evaluate(initial)
    initial_kkt, violation = check(initial, first['gradient'], matrix, lo, hi, lower, upper)
    if violation > 1e-7:
        raise ValueError(f'Starting physical motion violates declared bounds by {violation:g}')
    transform, scaling = information_scaling(first['gauss_newton'], 1., lower < upper)
    # Fixed coordinates have already been eliminated by information_scaling.
    # Keeping their zero-row equalities makes SLSQP's constraint Jacobian
    # singular, even though the remaining physical problem is well posed.
    bounded = (lower<upper)&(np.isfinite(lower)|np.isfinite(upper))
    restrictions = [LinearConstraint(transform[bounded], (lower-initial)[bounded], (upper-initial)[bounded])]
    if len(matrix):
        transformed = matrix@transform
        variable = np.any(transformed!=0, axis=1)
        if variable.any():
            restrictions.append(LinearConstraint(transformed[variable],
                (lo-matrix@initial)[variable], (hi-matrix@initial)[variable]))
    trace, started = [], time.perf_counter()
    class StationaryPoint(Exception):
        def __init__(self, coordinates):
            self.coordinates = coordinates.copy()

    def objective(x):
        value, gradient = joint.objective(initial+transform@x)
        return value, transform.T@gradient
    def callback(x):
        physical = initial+transform@x
        evaluation = joint.evaluate(physical)
        kkt, violation = check(physical, evaluation['gradient'], matrix, lo, hi, lower, upper)
        trace.append({'iteration': len(trace)+1, 'objective': evaluation['value'], 'kkt_residual': kkt,
            'constraint_violation': violation, 'evaluations': joint.evaluations, 'seconds': time.perf_counter()-started})
        if checkpoint:
            checkpoint(physical, trace)
        # The callback may be a trial before a line-search acceptance. Require
        # feasibility and improvement over the starting point as well as the
        # same physical-coordinate KKT gate used in the returned diagnostics.
        # Comparing objective changes near geometry round-off after this
        # condition is met can discard an already stationary iterate.
        if kkt <= 1e-7 and violation <= 1e-7 and evaluation['value'] <= first['value']:
            raise StationaryPoint(x)
    if initial_kkt <= 1e-7:
        fitted = OptimizeResult(x=np.zeros(transform.shape[1]), nit=0, success=True,
            message='Initial point satisfies the independently checked KKT gate')
    else:
        try:
            fitted = minimize(objective, np.zeros(transform.shape[1]), jac=True, method='SLSQP',
                constraints=restrictions, callback=callback, options={'maxiter': iterations, 'ftol': 1e-13})
        except StationaryPoint as stop:
            fitted = OptimizeResult(x=stop.coordinates, nit=len(trace), success=True,
                message='Stopped at the independently checked KKT gate')
    parameters = initial+transform@fitted.x
    final = joint.evaluate(parameters)
    kkt, violation = check(parameters, final['gradient'], matrix, lo, hi, lower, upper)
    refinement = None
    # Feasibility and independently measured curvature/stationarity govern
    # refinement; an iteration-limit flag must not suppress a valid correction.
    if polish_stationarity and kkt > 1e-8 and violation <= 1e-7:
        parameters, refinement = polish(joint.objective, parameters, transform, matrix, lo, hi, lower, upper,
            curvature_gradient=curvature_gradient)
        final = joint.evaluate(parameters)
        kkt, violation = check(parameters, final['gradient'], matrix, lo, hi, lower, upper)
    return parameters, {'converged': bool(kkt <= 1e-7 and violation <= 1e-7),
        'termination': str(fitted.message), 'iterations': fitted.nit, 'evaluations': joint.evaluations,
        'initial_objective': first['value'], 'objective': final['value'],
        'initial_kkt_residual': initial_kkt, 'maximum_kkt_residual': kkt,
        'maximum_constraint_violation': violation, 'trace': trace, 'scaling': scaling,
        'scene_conditions': final['scene_conditions'], 'stationarity_polish': refinement,
        'seconds': time.perf_counter()-started,
        'convergence_scope': 'First-order stationarity of the constrained, fixed-support, independent-band likelihood. This alone does not validate physical heading, global optimality or channel phase calibration.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'motion', 'alignment', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--fit-range-origins', action='store_true')
    parser.add_argument('--path-acceleration', type=float, default=.1, metavar='M_S2')
    parser.add_argument('--heading-rate', type=float, default=.5, metavar='DEG_S')
    parser.add_argument('--heading-acceleration', type=float, default=.01, metavar='DEG_S2')
    args = parser.parse_args()
    support, model, alignment = [json.loads(p.read_text()) for p in (args.support, args.motion, args.alignment)]
    if model.get('rock_spacing_seconds') or model.get('antenna_fit') or np.any(model.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('Use the fixed-roll/pitch, nominal-site, zero-channel-rate control')
    if Path(support['source_motion']).resolve() != args.motion.resolve():
        raise ValueError('Point support was generated using different physical motion')
    if 'points_body_m' in support:
        if not support.get('complete') or not support.get('fit', {}).get('converged'):
            raise ValueError('Finish the sub-grid support refinement before motion fitting')
        points = np.array(support['points_body_m'][:args.count])
        if len(points) != args.count:
            raise ValueError('Requested more points than the refined support contains')
    else:
        chosen = next(t for t in support['trials'] if t['count'] == args.count)
        points = np.array(chosen['points_body_m'])
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        block = np.floor(field.time-model['interval_s'][0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        bands.append(Band(field, prior, tx[3], rx, points, model, selected, range_bias_m=model['range_bias_m'][c]))
    joint = PhysicalProfile(bands, model['parameters'], fit_ranges=args.fit_range_origins)
    limits = np.array([args.path_acceleration, args.heading_rate, args.heading_acceleration])
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError('Physical motion search limits must be positive and finite')
    matrix, lo, hi, kinematics = constraints(track, epoch, model, tx[3], rx, joint.active, len(joint.initial),
        path_acceleration_mps2=args.path_acceleration, heading_rate_degrees_ps=args.heading_rate,
        heading_acceleration_degrees_ps2=args.heading_acceleration)
    lower = np.r_[np.full(bands[0].npath-1, -30.), np.full(joint.nmotion-bands[0].npath+1, -20.),
        np.full(len(bands) if joint.fit_ranges else 0, -75.)]
    upper = -lower
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = {'source_support': str(args.support), 'source_motion': str(args.motion),
        'source_sha256': {str(p): fingerprint(p) for p in (args.support, args.motion, args.alignment, args.ais)},
        'support_count': args.count, 'fit_range_origins': args.fit_range_origins,
        'declared_motion_limits': limits.tolist(),
        'temporal_holdout': 'Odd interior one-second blocks; both aperture edges included in training.',
        'limitation': 'The original motion and support coordinates remain conditional on this aperture. This refit tests the physical correction on newly withheld time blocks, not independent target discovery.'}
    before = joint.validation(joint.initial)
    def checkpoint(parameters, trace):
        print(json.dumps(trace[-1]), flush=True)
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({**identity,
            'joint_parameters': parameters.tolist(), 'parameters': joint.expand(parameters).tolist(),
            'range_bias_m': parameters[joint.nmotion:].tolist() if joint.fit_ranges else model['range_bias_m'],
            'trace': trace}, indent=2)+'\n')
    parameters, result = solve(joint, matrix, lo, hi, lower, upper, iterations=args.iterations, checkpoint=checkpoint)
    after = joint.validation(parameters)
    output = {k: model[k] for k in ('interval_s', 'path_spacing_seconds', 'heading_spacing_seconds',
        'nominal_start_utc', 'attitude_prior', 'reference_height_ellipsoid_m', 'reference_channel',
        'yaw_convention', 'geometry_directory', 'channel_gains_real_imag')}
    output.update(identity=identity, parameters=joint.expand(parameters).tolist(), joint_parameters=parameters.tolist(),
        points_body_m=points.tolist(), range_bias_m=parameters[joint.nmotion:].tolist() if joint.fit_ranges else model['range_bias_m'],
        antenna_fit=None, channel_path_rate_bias_mps=[0.]*5,
        physical_profile={**result, 'before': before, 'after': after, 'before_summary': summarize(before),
            'after_summary': summarize(after), 'kinematic_constraints': kinematics},
        phase_centres_validated=False, absolute_heading_validated=False, independent_complex_band_responses=True,
        scope=__doc__)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('converged', 'objective', 'maximum_kkt_residual', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
