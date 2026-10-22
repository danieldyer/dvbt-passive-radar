"""Shared-scene physical motion and optional antenna-location refinement.

Roll/pitch and scattering support stay fixed. Antenna refinement profiles channel
gains and permits independent envelope origins, so an unknown constant carrier
phase or sample origin is not interpreted as a measured antenna displacement.
Time/spatial transfer must still pass before adopting any calibration.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize, nnls

from ais import read_csv, utc_seconds
from antenna_motion import Antennas
from coherent_motion import CoherentMotion
from fit_channel_alignment import load_trial
from fit_joint_motion import information_scaling
from full_motion import Band
from imaging import physical_state, sites
from kinematics import constraints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'scene', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--profile-channel-gains', action='store_true', help='Refit common-scene channel gains at each motion candidate and eliminate them from the motion information')
    p.add_argument('--fit-antenna-locations', action='store_true', help='Fit correlated physical transmitter/receiver offsets; requires profiled gains and free range origins')
    p.add_argument('--fit-range-origins', action='store_true', help='Fit one compressed-envelope origin per channel, including the reference channel')
    p.add_argument('--fix-motion', action='store_true', help='Conditional control: hold the input physical trajectory fixed')
    p.add_argument('--initial-motion', type=Path, help='Use a previously fitted motion as the starting point, preserving the original observations and prior')
    args = p.parse_args()
    if args.fit_antenna_locations and not (args.profile_channel_gains and args.fit_range_origins):
        p.error('Antenna fitting requires --profile-channel-gains and --fit-range-origins')
    source, curve, alignment = load_trial(args.alignment, args.ais, args.geometry_directory)
    model = curve.description
    if model.get('rock_spacing_seconds'):
        raise ValueError('Roll and pitch must remain fixed for this experiment')
    scene = json.loads(args.scene.read_text())
    if scene['input_alignment'] != str(args.alignment):
        raise ValueError('The scene belongs to a different alignment trial')
    fit = scene['fit']
    points = np.array(fit['points_body_m'])
    gains = np.asarray(fit['gain_amplitude'])*np.exp(1j*np.radians(fit['gain_phase_degrees']))
    tx, rx, configuration = sites(args.geometry_directory)
    antennas = None
    if args.fit_antenna_locations:
        geometry = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
        antennas = Antennas(geometry['ecef_to_enu'], configuration['channel_sites'],
                           receiver_sigma=geometry['receiver']['prior_sigma_enu_m'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    bands = []
    for c, old in enumerate(alignment.bands):
        prior = physical_state(track, epoch+old.field.data['row_time'], model['attitude_prior'])
        bands.append(Band(old.field, prior, tx[3], rx, points, model, old.selected_time,
                          range_bias_m=fit['range_bias_m'][c]))
    resume = None if args.initial_motion is None else json.loads(args.initial_motion.read_text())
    seed = curve.parameters if resume is None else np.array(resume['parameters'])
    joint = CoherentMotion(bands, gains, seed, profile_channel_gains=args.profile_channel_gains,
                          antennas=antennas, fit_range_origins=args.fit_range_origins)
    initial = joint.initial.copy()
    if resume is not None:
        if antennas is not None and resume.get('antenna_fit'):
            initial[joint.antenna_slice] = resume['antenna_fit']['standardised_parameters']
        if args.fit_range_origins and resume.get('range_bias_m'):
            initial[joint.range_slice] = resume['range_bias_m']
    matrix, lo, hi, kinematics = constraints(track, epoch, model, tx[3], rx, joint.active, len(initial))
    lower = np.r_[np.full(bands[0].npath-1, -30.), np.full(joint.nmotion-bands[0].npath+1, -20.),
                  np.full(joint.nantenna, -3.), np.full(len(bands) if args.fit_range_origins else 0, -75.)]
    upper = -lower
    if args.fix_motion:
        lower[:joint.nmotion] = upper[:joint.nmotion] = initial[:joint.nmotion]
    free = lower < upper
    if not np.any(free):
        p.error('No variable motion, range or antenna parameters')
    expected = joint.information(initial)
    expected[joint.antenna_slice, joint.antenna_slice] += np.eye(joint.nantenna)
    transform, scaling = information_scaling(expected, joint.energy, free)
    restrictions = [LinearConstraint(transform[free], (lower-initial)[free], (upper-initial)[free])]
    varying_rows = np.any(matrix[:, free] != 0, axis=1)
    if np.any(varying_rows):
        restrictions.append(LinearConstraint((matrix@transform)[varying_rows],
            (lo-matrix@initial)[varying_rows], (hi-matrix@initial)[varying_rows]))
    constant = (matrix@initial)[~varying_rows]
    if np.any(constant < lo[~varying_rows]-1e-7) or np.any(constant > hi[~varying_rows]+1e-7):
        raise ValueError('The fixed motion violates the declared kinematic constraints')
    before = joint.validation(initial)
    def objective(x):
        value, gradient = joint.objective(initial+transform@x)
        return value, transform.T@gradient
    started, trace = time.perf_counter(), []
    def callback(x):
        parameters = initial+transform@x
        if joint.last is None or not np.array_equal(joint.last[0], parameters):
            joint.objective(parameters)
        record = {'iteration': len(trace)+1, 'objective': joint.last[1], 'seconds': time.perf_counter()-started}
        trace.append(record)
        print(json.dumps(record), flush=True)
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({
            'parameters': joint.expand(parameters).tolist(), 'joint_parameters': parameters.tolist(),
            'trace': trace, 'input_scene': str(args.scene)}, indent=2)+'\n')
    result = minimize(objective, np.zeros(int(free.sum())), jac=True, method='SLSQP', constraints=restrictions,
        callback=callback, options={'maxiter': args.iterations, 'ftol': 1e-13})
    parameters = initial+transform@result.x
    value, gradient = joint.objective(parameters)
    v = matrix@parameters
    active = np.r_[matrix[v >= hi-1e-6], -matrix[v <= lo+1e-6],
                   np.eye(len(parameters))[(parameters >= upper-1e-6)&free],
                   -np.eye(len(parameters))[(parameters <= lower+1e-6)&free]][:, free]
    multiplier = nnls(active.T, -gradient[free])[0] if len(active) else np.empty(0)
    residual = gradient[free]+(active.T@multiplier if len(active) else 0)
    violation = float(max(0., np.max(np.r_[lo-v, v-hi, lower-parameters, parameters-upper])))
    keep = ('interval_s', 'path_spacing_seconds', 'heading_spacing_seconds', 'nominal_start_utc',
            'attitude_prior', 'reference_height_ellipsoid_m', 'reference_channel', 'yaw_convention', 'geometry_directory')
    output = {k:model[k] for k in keep}
    gains = joint.project(parameters)[-1]
    output.update(parameters=joint.expand(parameters).tolist(), joint_parameters=parameters.tolist(),
        input_motion=source['input_motion'], initial_motion=str(args.initial_motion),
        input_scene=str(args.scene), points_body_m=points.tolist(), range_bias_m=joint.range_origins(parameters).tolist(),
        channel_gains_real_imag=np.c_[gains.real, gains.imag].tolist(), holdout_seconds=source['holdout_seconds'],
        channel_gains_profiled=args.profile_channel_gains, inner_gain_convergence=joint.gain_diagnostics,
        fixed_motion=args.fix_motion, fitted_range_origins=args.fit_range_origins,
        antenna_fit=None if antennas is None else antennas.describe(parameters[joint.antenna_slice]),
        antenna_information=None if antennas is None else joint.antenna_information(parameters),
        antenna_information_with_unknown_points=None if antennas is None else joint.antenna_information(parameters, profile_point_coordinates=True),
        coherent_motion={'objective': value, 'converged': bool(result.success and violation < 1e-7 and abs(residual).max() < 1e-7),
            'maximum_kkt_residual': float(abs(residual).max()), 'maximum_constraint_violation': violation,
            'antenna_bound_hit': bool(np.any(abs(parameters[joint.antenna_slice]) >= 3.-1e-5)),
            'range_bound_hit': bool(args.fit_range_origins and np.any(abs(parameters[joint.range_slice]) >= 75.-1e-5)),
            'iterations': result.nit, 'termination': str(result.message), 'before': before,
            'after': joint.validation(parameters), 'trace': trace, 'seconds': time.perf_counter()-started,
            'scaling': scaling, 'kinematic_constraints': kinematics},
        scope=__doc__, phase_centres_validated=False, absolute_heading_validated=False)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({k:output['coherent_motion'][k] for k in ('objective', 'converged', 'maximum_kkt_residual', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
