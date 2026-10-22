"""Refit physical motion after separating changing reflector strength from phase."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls

from ais import read_csv, utc_seconds
from amplitude_motion import AmplitudeMotion
from coherent_motion import CoherentMotion
from field import Field
from fit_joint_motion import information_scaling
from full_motion import Band
from imaging import physical_state, sites
from kinematics import constraints
from stationarity import polish
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'motion', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--amplitude-block-seconds', type=float, default=3.)
    p.add_argument('--iterations', type=int, default=120)
    p.add_argument('--objective-tolerance', type=float, default=1e-13)
    p.add_argument('--polish', action='store_true', help='Check positive tangent curvature and refine the analytic KKT equations after the line search')
    p.add_argument('--fix-motion', action='store_true')
    p.add_argument('--refine-points', action='store_true', help='Refine shared horizontal reflector locations; the first point and all heights stay fixed')
    p.add_argument('--point-radius-m', type=float, default=10.)
    p.add_argument('--independent-magnitudes', action='store_true', help='Permit nonnegative reflector strengths to differ by frequency, preserving shared point phases')
    p.add_argument('--fit-path-rates', action='store_true', help='Fit four relative channel excess-path rates, with range, carrier phase and symbol Doppler coupled')
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source, model = json.loads(args.alignment.read_text()), json.loads(args.motion.read_text())
    if model.get('antenna_fit'):
        raise ValueError('Use the nominal-location control; unaccepted geometry stays out of this experiment')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, bands = utc_seconds(model['nominal_start_utc']), []
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        bands.append(Band(field, prior, tx[3], rx, model['points_body_m'], model,
            np.arange(len(field.time))%2 == 0, range_bias_m=model['range_bias_m'][c]))
    gains = np.array([complex(*z) for z in model['channel_gains_real_imag']])
    static = CoherentMotion(bands, gains, model['parameters'])
    amplitude = static.project(static.initial)[0]
    if model.get('amplitude_model'):
        amplitude = np.exp(1j*np.asarray(model['amplitude_model']['point_phase_radians']))
    joint = AmplitudeMotion(bands, model['parameters'], gains, amplitude, args.amplitude_block_seconds,
        refine_points=args.refine_points, independent_magnitudes=args.independent_magnitudes, fit_path_rates=args.fit_path_rates)
    initial = joint.initial
    matrix, lo, hi, kinematics = constraints(track, epoch, model, tx[3], rx, joint.active, len(initial))
    lower = np.r_[np.full(bands[0].npath-1, -30.), np.full(joint.nmotion-bands[0].npath+1, -20.),
                  np.full(5, -75.), np.full(joint.ngain, -5.), np.full(joint.ngain+joint.points, -np.inf)]
    upper = -lower
    if args.refine_points:
        lower = np.r_[lower, initial[joint.point_slice]-args.point_radius_m]
        upper = np.r_[upper, initial[joint.point_slice]+args.point_radius_m]
    if args.fit_path_rates:
        lower, upper = np.r_[lower, np.full(joint.ngain, -.1)], np.r_[upper, np.full(joint.ngain, .1)]
    if args.fix_motion:
        lower[:joint.nmotion] = upper[:joint.nmotion] = initial[:joint.nmotion]
    if args.independent_magnitudes:
        # Band gain magnitude and all reflector magnitudes have an exact scale
        # gauge. Keep the declared gains; only their phases remain adjustable.
        s = slice(joint.gain_slice.start, joint.gain_slice.start+joint.ngain)
        lower[s] = upper[s] = initial[s]
    free = lower < upper
    finite = np.isfinite(lower)&free
    expected = joint.information(initial)
    transform, scaling = information_scaling(expected, joint.energy, free)
    varying = np.any(matrix[:, free] != 0, axis=1)
    restrictions = [LinearConstraint(transform[finite], (lower-initial)[finite], (upper-initial)[finite])]
    if np.any(varying):
        restrictions.append(LinearConstraint((matrix@transform)[varying], (lo-matrix@initial)[varying], (hi-matrix@initial)[varying]))
    before = joint.validation(initial)
    def objective(x):
        value, gradient = joint.objective(initial+transform@x)
        return value, transform.T@gradient
    started, trace = time.perf_counter(), []
    def callback(x):
        values = initial+transform@x
        if joint.last is None or not np.array_equal(joint.last[0], values):
            joint.objective(values)
        record = {'iteration': len(trace)+1, 'objective': joint.last[1], 'seconds': time.perf_counter()-started}
        trace.append(record)
        print(json.dumps(record), flush=True)
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({'nonlinear_parameters': values.tolist(), 'trace': trace}, indent=2)+'\n')
    result = minimize(objective, np.zeros(int(free.sum())), jac=True, method='SLSQP', constraints=restrictions,
        callback=callback, options={'maxiter': args.iterations, 'ftol': args.objective_tolerance})
    parameters = initial+transform@result.x
    polishing = None
    if args.polish:
        print(json.dumps({'stage': 'Constrained gradient polish', 'seconds': time.perf_counter()-started}), flush=True)
        parameters, polishing = polish(joint.objective, parameters, transform, matrix, lo, hi, lower, upper)
    value, gradient = joint.objective(parameters)
    v = matrix@parameters
    active = np.r_[matrix[v >= hi-1e-6], -matrix[v <= lo+1e-6],
        np.eye(len(parameters))[(parameters >= upper-1e-6)&finite],
        -np.eye(len(parameters))[(parameters <= lower+1e-6)&finite]][:, free]
    multiplier = nnls(active.T, -gradient[free])[0] if len(active) else np.empty(0)
    residual = gradient[free]+(active.T@multiplier if len(active) else 0)
    violation = float(max(0., np.max(np.r_[lo-v, v-hi, lower-parameters, parameters-upper])))
    motion, origins, gains, phases = joint.unpack(parameters)
    fit = joint.project(parameters)
    after = joint.validation(parameters)
    keep = ('interval_s', 'path_spacing_seconds', 'heading_spacing_seconds', 'nominal_start_utc',
        'attitude_prior', 'reference_height_ellipsoid_m', 'reference_channel', 'yaw_convention', 'geometry_directory', 'points_body_m')
    output = {k:model[k] for k in keep}
    output['points_body_m'] = joint.point_positions(parameters).tolist()
    output['point_refinement'] = {'enabled': args.refine_points, 'radius_m': args.point_radius_m,
        'initial_points_body_m': model['points_body_m'],
        'bound_hit': bool(args.refine_points and np.any(abs(parameters[joint.point_slice]-initial[joint.point_slice]) > .999*args.point_radius_m)),
        'scope': 'Shared horizontal coordinates. First point and heights fixed to specify a local scene frame, not an independently measured hull frame.'}
    if args.fit_path_rates:
        output['channel_path_rate_bias_mps'] = joint.path_rates(parameters).tolist()
        output['path_rate_registration'] = {'epoch_s': float(np.mean(model['interval_s'])),
            'bound_mps': .1, 'reference_channel': 3,
            'scope': 'Effective channel-relative excess-path rate, shared by all body points. Changes envelope, RF phase and symbol Doppler together. Does not independently measure transmitter frequency or antenna position.'}
    output.update(parameters=motion.tolist(), nonlinear_parameters=parameters.tolist(), input_motion=str(args.motion),
        source_alignment=str(args.alignment), fixed_motion=args.fix_motion, range_bias_m=origins.tolist(),
        channel_gains_real_imag=np.c_[gains.real, gains.imag].tolist(), holdout_seconds=0.,
        amplitude_model={'duration_s': args.amplitude_block_seconds, 'magnitudes': fit['magnitude'].tolist(),
            'shared_across_channels': not args.independent_magnitudes,
            'point_phase_radians': np.angle(phases).tolist(), 'linear_optimality': fit['linear_optimality'],
            'scope': 'Nonnegative magnitudes may vary; each point has one constant complex phase shared across time and channels. Independent channel magnitudes, when enabled, do not introduce per-channel point phases.'},
        coherent_motion={'objective': value, 'converged': bool((result.success or (polishing and polishing['accepted'])) and violation < 1e-7 and abs(residual).max() < 1e-7),
            'maximum_kkt_residual': float(abs(residual).max()), 'maximum_constraint_violation': violation,
            'objective_tolerance': args.objective_tolerance,
            'gradient_polishing': polishing,
            'iterations': result.nit, 'termination': str(result.message), 'before': before, 'after': after,
            'before_summary': summarize(before), 'after_summary': summarize(after),
            'trace': trace, 'seconds': time.perf_counter()-started, 'scaling': scaling, 'kinematic_constraints': kinematics},
        scope=__doc__, phase_centres_validated=False, absolute_heading_validated=False)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:output['coherent_motion'][k] for k in ('objective', 'converged', 'maximum_kkt_residual', 'seconds', 'after_summary')}), flush=True)


if __name__ == '__main__':
    main()
