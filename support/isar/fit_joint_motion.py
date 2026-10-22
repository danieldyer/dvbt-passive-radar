"""Joint ship-motion and antenna-location fit, with held-out complex-data tests."""
import argparse
import json
from pathlib import Path
import time as clock

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize

from ais import read_csv, utc_seconds
from antenna_motion import Antennas
from dense_scatterers import points
from field import Field
from full_motion import Band
from imaging import physical_state, sites
from kinematics import constraints as motion_constraints


def information_scaling(expected, energy, free):
    """Whiten only variable coordinates; fixed locations never become probes."""
    eigen, vectors = np.linalg.eigh(expected[np.ix_(free, free)]/energy)
    floor = max(0., float(eigen.max(initial=0.)))*1e-8
    transform = np.zeros((len(free), int(np.sum(free))))
    transform[free] = ((vectors/np.sqrt(np.maximum(eigen, floor)))@vectors.T
        if floor > 0 else np.eye(np.sum(free)))
    return transform, {'initial_expected_hessian_eigenvalues': eigen.tolist(),
        'eigenvalue_floor': float(floor), 'variable_coordinates': np.flatnonzero(free).tolist(),
        'scope': 'Invertible preconditioning on variable coordinates only. Fixed coordinates are eliminated; the likelihood and physical bounds are unchanged.'}


class Joint:
    def __init__(self, bands, antennas, initial_motion):
        self.bands, self.antennas = bands, antennas
        self.initial_motion = np.asarray(initial_motion).copy()
        npath = bands[0].npath
        self.initial_motion[:npath] -= self.initial_motion[npath-1]
        self.active = np.r_[np.arange(npath-1), np.arange(npath, len(self.initial_motion))]
        self.nmotion = len(self.active)
        self.energy = sum(b.energy for b in bands)
        self.bounds = [(-30., 30.)]*(npath-1)+[(-20., 20.)]*(len(self.initial_motion)-npath)+[(-3., 3.)]*antennas.count
        self.initial = np.r_[self.initial_motion[self.active], np.zeros(antennas.count)]
        self.evaluations = 0
        self.last = None

    def split(self, parameters):
        motion = np.zeros_like(self.initial_motion)
        motion[self.active] = parameters[:self.nmotion]
        return motion, parameters[self.nmotion:]

    def objective(self, parameters):
        motion, locations = self.split(parameters)
        value, gradient = float(locations@locations)/2, np.r_[np.zeros(self.nmotion), locations.copy()]
        for channel, band in enumerate(self.bands):
            v, g = band.evaluate(motion, self.antennas.offsets(locations, channel))
            value += v
            gradient[:self.nmotion] += g[:len(motion)][self.active]
            gradient[self.nmotion:] += self.antennas.mapping(channel).T@g[-6:]
        self.evaluations += 1
        self.last = (parameters.copy(), value/self.energy, gradient/self.energy)
        return self.last[1:]

    def validation(self, parameters):
        motion, locations = self.split(parameters)
        return [b.validation(motion, self.antennas.offsets(locations, c)) for c, b in enumerate(self.bands)]

    def information_matrix(self, parameters):
        motion, locations = self.split(parameters)
        information = np.zeros((len(parameters), len(parameters)))
        for c, band in enumerate(self.bands):
            local = band.information(motion, self.antennas.offsets(locations, c))
            transform = block_diag(np.eye(len(motion))[:, self.active], self.antennas.mapping(c))
            information += transform.T@local@transform
        return information

    def information(self, parameters):
        information = self.information_matrix(parameters)
        n = self.nmotion
        # Eliminate motion nuisance directions before judging the antenna data.
        eigen, vectors = np.linalg.eigh(information[:n, :n])
        keep = eigen > eigen.max()*1e-10
        inverse = (vectors[:, keep]/eigen[keep])@vectors[:, keep].T
        marginal = information[n:, n:]-information[n:, :n]@inverse@information[:n, n:]
        marginal = (marginal+marginal.T)/2
        values, axes = np.linalg.eigh(marginal)
        covariance = (axes/(1+np.maximum(values, 0)))@axes.T
        physical = self.antennas.factor@covariance@self.antennas.factor.T
        return {'antenna_information_eigenvalues_before_prior': values.tolist(),
            'motion_rank_at_relative_tolerance_1e_10': int(keep.sum()),
            'motion_parameter_count': n,
            'conditional_antenna_posterior_sigma_enu_m': np.sqrt(np.maximum(0, np.diag(physical))).reshape(-1, 3).tolist(),
            'antenna_prior_sigma_enu_m': np.sqrt(np.diag(self.antennas.factor@self.antennas.factor.T)).reshape(-1, 3).tolist(),
            'scope': 'Local Gaussian information conditional on fixed scattering-point coordinates, the thermal covariance, fixed AIS midpoint pose, and the licensed tower assignment. Motion and all complex point/stationary responses are eliminated. This optimistic conditional calculation is not a calibrated uncertainty for real antenna locations.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'images', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--points', type=int, default=8)
    p.add_argument('--holdout-seconds', type=float, default=2.)
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--resume', type=Path)
    p.add_argument('--optimizer', choices=['lbfgs', 'scaled-slsqp'], default='lbfgs')
    p.add_argument('--kinematic-bounds', action='store_true', help='Bound actual reference path acceleration and heading rate/acceleration; use scaled-slsqp')
    p.add_argument('--fix-antenna-locations', action='store_true', help='Control fit: keep all locations nominal with the same motion, constraints and validation mask')
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    epoch = utc_seconds(model['nominal_start_utc'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, config = sites(args.geometry_directory)
    geo = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
    antennas = Antennas(geo['ecef_to_enu'], config['channel_sites'], receiver_sigma=geo['receiver']['prior_sigma_enu_m'])
    bands = []
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            locations = points(image, args.points, 30.)
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
            np.floor((field.time-model['interval_s'][0])/args.holdout_seconds).astype(int)%2 == 0)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        bands.append(Band(field, prior, tx[3], rx, locations, model, selected))
    joint = Joint(bands, antennas, model['parameters'])
    if args.fix_antenna_locations:
        joint.bounds[joint.nmotion:] = [(0., 0.)]*antennas.count
    resume = None if args.resume is None else json.loads(args.resume.read_text())
    initial = joint.initial if resume is None else np.asarray(resume['joint_parameters'])
    if resume is not None and (resume['input_motion'] != str(args.motion) or resume['point_source'] != str(args.images)
                               or resume['holdout_seconds'] != args.holdout_seconds
                               or resume.get('fixed_antenna_locations', False) != args.fix_antenna_locations):
        raise ValueError('Resume state uses different input, support or holdout')
    physical_constraint, kinematics = None, None
    if args.kinematic_bounds:
        if args.optimizer != 'scaled-slsqp':
            raise ValueError('Kinematic constraints require the constrained optimizer')
        matrix, lo, hi, kinematics = motion_constraints(track, epoch, model, tx[3], rx, joint.active, len(initial))
        physical_constraint = LinearConstraint(matrix, lo, hi)
        projected = minimize(lambda x: (.5*np.sum((x-initial)**2), x-initial), initial, jac=True,
            constraints=physical_constraint, bounds=joint.bounds, method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-12})
        if not projected.success or np.max(np.r_[lo-matrix@projected.x, matrix@projected.x-hi]) > 1e-7:
            raise ValueError('Could not construct a feasible kinematic starting curve')
        initial = projected.x
    before = joint.validation(initial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started, trace = clock.perf_counter(), []
    def callback(parameters):
        if joint.last is None or not np.array_equal(joint.last[0], parameters):
            joint.objective(parameters)
        _, value, gradient = joint.last
        record = {'iteration': len(trace)+1, 'evaluations': joint.evaluations, 'objective': value,
                  'maximum_gradient': float(abs(gradient).max()), 'seconds': clock.perf_counter()-started}
        trace.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {'joint_parameters': parameters.tolist(), 'trace': trace, 'input_motion': str(args.motion),
                      'point_source': str(args.images), 'holdout_seconds': args.holdout_seconds, 'points': args.points,
                      'fixed_antenna_locations': args.fix_antenna_locations}
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps(checkpoint, indent=2)+'\n')
    lower, upper = np.array(joint.bounds).T
    scaling = None
    if args.optimizer == 'lbfgs':
        result = minimize(joint.objective, initial, jac=True, bounds=joint.bounds, method='L-BFGS-B', callback=callback,
            options={'maxiter': args.iterations, 'gtol': 1e-8, 'ftol': 1e-14, 'maxcor': 40, 'maxls': 30})
    else:
        expected = joint.information_matrix(initial)
        expected[joint.nmotion:, joint.nmotion:] += np.eye(antennas.count)
        free = lower < upper
        transform, scaling = information_scaling(expected, joint.energy, free)
        def scaled(x):
            value, gradient = joint.objective(initial+transform@x)
            return value, transform.T@gradient
        constraint = [LinearConstraint(transform[free], (lower-initial)[free], (upper-initial)[free])]
        if physical_constraint is not None:
            constraint.append(LinearConstraint(matrix@transform, lo-matrix@initial, hi-matrix@initial))
        result = minimize(scaled, np.zeros(int(np.sum(free))), jac=True, constraints=constraint, method='SLSQP',
            callback=lambda x: callback(initial+transform@x), options={'maxiter': args.iterations, 'ftol': 1e-11})
        result.x = initial+transform@result.x
        result.fun, result.jac = joint.objective(result.x)
    gradient = result.jac.copy()
    gradient[((result.x <= lower+1e-7)&(gradient > 0))|((result.x >= upper-1e-7)&(gradient < 0))] = 0
    constraint_stationarity = None
    if physical_constraint is not None:
        # KKT residual accounts for active physical inequalities. A raw
        # unconstrained gradient is not a constrained convergence criterion.
        from scipy.optimize import nnls
        v = matrix@result.x
        active_rows = np.r_[matrix[v >= hi-1e-6], -matrix[v <= lo+1e-6],
                            np.eye(len(result.x))[result.x >= upper-1e-6], -np.eye(len(result.x))[result.x <= lower+1e-6]]
        multipliers = nnls(active_rows.T, -result.jac)[0] if len(active_rows) else np.empty(0)
        residual = result.jac+(active_rows.T@multipliers if len(active_rows) else 0)
        constraint_stationarity = {'maximum_kkt_residual': float(abs(residual).max()),
            'maximum_constraint_violation': float(max(0, np.max(np.r_[lo-v, v-hi]))),
            'active_inequalities': len(active_rows)}
        gradient = residual
    motion, locations = joint.split(result.x)
    feasible = constraint_stationarity is None or constraint_stationarity['maximum_constraint_violation'] <= 1e-7
    record = {'converged': bool(result.success and feasible and abs(gradient).max() <= 1e-7),
        'termination': str(result.message), 'maximum_gradient': float(abs(result.jac).max()),
        'maximum_projected_gradient': float(abs(gradient).max()), 'trace': trace,
        'iterations': result.nit, 'evaluations': joint.evaluations, 'objective': float(result.fun),
        'optimizer': args.optimizer, 'coordinate_scaling': scaling,
        'kinematic_bounds': kinematics, 'constraint_stationarity': constraint_stationarity,
        'location_prior_penalty': float(locations@locations)/2,
        'bound_hit': bool(np.any((result.x <= lower+1e-5)&(lower < upper)) or np.any((result.x >= upper-1e-5)&(lower < upper))),
        'fixed_antenna_locations': args.fix_antenna_locations,
        'before': before, 'after': joint.validation(result.x),
        'information': ({'scope': 'Antenna locations fixed for this control; no location uncertainty inferred.'}
                        if args.fix_antenna_locations else joint.information(result.x)),
        'seconds': clock.perf_counter()-started,
        'scope': 'Shared physical motion and five transmitter plus one receiver phase centres; fixed AIS midpoint and image-selected point coordinates. Independent complex response for every point/channel. No geographic calibration is adopted by optimizer convergence alone.'}
    metadata = {k: model[k] for k in ('interval_s', 'path_spacing_seconds', 'heading_spacing_seconds',
        'nominal_start_utc', 'attitude_prior', 'reference_height_ellipsoid_m', 'reference_channel',
        'yaw_convention', 'geometry_directory')}
    metadata.update(parameters=motion.tolist(), joint_likelihood=record, antenna_fit=antennas.describe(locations),
        joint_parameters=result.x.tolist(), input_motion=str(args.motion), point_source=str(args.images),
        points_body_m=[b.points.tolist() for b in bands], holdout_seconds=args.holdout_seconds,
        phase_centres_validated=False, absolute_heading_validated=False)
    args.output.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: record[k] for k in ('converged', 'iterations', 'seconds', 'maximum_projected_gradient')}), flush=True)


if __name__ == '__main__':
    main()
