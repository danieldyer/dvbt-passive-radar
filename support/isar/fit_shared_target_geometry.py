"""Fit physical antenna coordinates jointly to road targets and ship profiles.

This is a conditional local experiment, not an accepted site calibration.
Road discovery and initialization used the full recording. Alternating blocks
test prediction after fitting; whole-target and road-branch transfer remain
separate acceptance checks. Original noise units and source profiles are kept.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from ais import utc_seconds
from audit_joint_sites import load_joint
from fit_physical_profile import solve
from full_motion import Band
from geometry_profile import GeometryProfile
from imaging import motion_state, sites
from model_quadrature import quadrature
from refine_full_capture_motion import constraints
from road_fit import load_trial
from road_validation import related_intervals, split_problem
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget, SharedGeometry
from spline_motion import SplineMotion
from validate_motion import summarize


def load_experiment(ship_path, road_path, fields, geometry, labels, nodes):
    fitted, saved = [json.loads(p.read_text()) for p in (ship_path, road_path)]
    exact = load_joint(fitted, fields, geometry)
    curve = SplineMotion(exact.bands[0].model)
    if not exact.bands[0].model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Shared-target trials require zero yaw in fixed N/E axes')
    reduced = []
    for band in exact.bands:
        field = quadrature(band.field, nodes)
        prior = motion_state(None, utc_seconds(band.model['nominal_start_utc'])+field.data['row_time'], band.model)
        fast = Band(field, prior, band.reference_transmitter, band.receiver, band.points,
                    band.model, band.selected_time, range_bias_m=band.range_bias)
        fast.nuisance, fast.clean, fast.energy = band.nuisance, band.clean, band.energy
        reduced.append(fast)
    ship = GeometryProfile(reduced, curve.parameters, exact.antennas,
                           base_offsets=exact.base_offsets, range_origins=fitted['initial_range_bias_m'])
    paths = [Path(saved[k]) for k in ('input_routes', 'input_histories', 'input_patches', 'input_templates')]
    original = load_trial(saved, *paths)
    _, receiver, _ = sites(geometry)
    tx = np.asarray([option['transmitter_ecef_m'] for option in fitted['site_assignment']])
    tx = (tx-receiver)@ship.antennas.rotation.T
    p = np.asarray(saved['reduced_state'])
    roads, selected, dropped = [], [], []
    for label in labels:
        index = saved['labels'].index(label)
        problem = original.problems[index]
        overlap = next((old for old in selected if related_intervals(problem, original.problems[saved['labels'].index(old)])), None)
        if overlap is not None:
            dropped.append({'label': label, 'overlaps': overlap,
                'reason': 'Possible fragments in the same time/range/Doppler support; do not count overlapping observations twice.'})
            continue
        initial = p[9+index*original.n:9+(index+1)*original.n]
        train, test = [split_problem(problem, parity).fields for parity in (0, 1)]
        roads.append(RoadTarget(problem, train, test, initial, tx, ship.antennas))
        selected.append(label)
    if not roads:
        raise ValueError('Select at least one distinct vehicle interval')
    joint = SharedGeometry(ship, roads)
    # Same ship constraints as its previous joint-site experiment. Road
    # feasibility and declared lane/motion priors are inherited, not tightened.
    _, matrix, lo, hi, lower, upper = constraints(ship.bands[0].model, common_path=True)
    origins = np.asarray(fitted['initial_range_bias_m'])
    lower = np.r_[lower, np.full(ship.antennas.count, -3.), origins-75.]
    upper = np.r_[upper, np.full(ship.antennas.count, 3.), origins+75.]
    yaw = (ship.active >= curve.sections[1].start)&(ship.active < curve.sections[1].stop)
    lower[np.flatnonzero(yaw)] = upper[np.flatnonzero(yaw)] = 0.
    for road in roads:
        a, b = road.problem.model.bounds()
        lower, upper = np.r_[lower, a], np.r_[upper, b]
    matrix = np.pad(matrix, ((0, 0), (0, len(joint.initial)-matrix.shape[1])))
    return joint, exact, (matrix, lo, hi, lower, upper), selected, dropped


def prediction(joint, original_ship, parameters, labels):
    ship = original_ship.validation(parameters[:joint.ship_count])
    vehicles = {label: road.predict(parameters[section], parameters[joint.antenna_slice])
        for label, road, section in zip(labels, joint.roads, joint.road_slices)}
    return {'ship_channels': ship, 'ship_summary': summarize(ship), 'vehicles': vehicles,
        'scope': 'Coefficients fitted on training blocks predict withheld blocks. Ship uses its declared stationary nuisance projection; vehicles retain their original range/Doppler observation operators.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ship-fit', type=Path, required=True)
    parser.add_argument('--road-fit', type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q', 'B', 'C', 'P', 'R', 'S', 'E'])
    parser.add_argument('--nodes', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--initial', type=Path, help='Resume from this experiment checkpoint or result, retaining its input identity')
    parser.add_argument('--fixed-antennas', action='store_true', help='Matched control: keep nominal licensed antenna locations fixed')
    parser.add_argument('--audit-only', action='store_true', help='Recorded-data value/gradient cross-check; no fit')
    args = parser.parse_args()
    started = time.monotonic()
    joint, original, bounds, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, args.labels, args.nodes)
    matrix, lo, hi, lower, upper = bounds
    if args.fixed_antennas:
        lower[joint.antenna_slice] = upper[joint.antenna_slice] = 0.
    source = json.loads(args.road_fit.read_text())
    files = [args.ship_fit, args.road_fit, Path(source['input_routes']), Path(source['input_templates']),
        *[args.fields/f'channel-{c}.npz' for c in range(5)],
        *[args.geometry/name for name in ('geometry-prior.json', 'sites-main.json', 'acma-transmitters.json')],
        *[Path(source['input_histories'])/(label+'.json') for label in labels],
        *[Path(source['input_patches'])/f'{label}-{c}.npz' for label in labels for c in range(5)]]
    identity = {'sources': {str(p): fingerprint(p) for p in files}, 'labels': labels,
        'dropped_overlapping_intervals': dropped, 'nodes': args.nodes,
        'site_assignment': joint.ship.antennas.channel_sites.tolist(),
        'fixed_antennas': args.fixed_antennas,
        'implementation_sha256': {str(p): fingerprint(p) for p in [Path(__file__),
            Path(__file__).with_name('shared_target_geometry.py'), Path(__file__).with_name('geometry_profile.py'),
            Path(__file__).with_name('physical_profile.py'), Path(__file__).with_name('model_quadrature.py')]}}
    if args.initial:
        saved = json.loads(args.initial.read_text())
        # A fixed-location control may seed the subsequent free-location fit.
        left, right = copy.deepcopy(saved['identity']), copy.deepcopy(identity)
        left.pop('fixed_antennas'); right.pop('fixed_antennas')
        if left != right:
            raise ValueError('Initial parameters were fitted to different sources or observation models')
        joint.initial = np.asarray(saved['parameters'])
        if args.fixed_antennas:
            joint.initial[joint.antenna_slice] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {'identity': identity, 'scope': __doc__, 'complete': False,
        'accepted_calibration': False, 'yaw_fitted': False,
        'road_origin_model': 'The existing vehicle envelope origins are retained at zero; five ship envelope origins remain nuisance parameters. This conditional origin assumption needs sensitivity checks before calibration acceptance.',
        'prior_scope': 'One correlated antenna prior; declared road and kinematic constraints retained. No prior vehicle-calibration posterior is reused as independent evidence.',
        'noise_scope': 'Block-diagonal empirical range/time covariance between separated target crops; cross-target clutter covariance is not yet estimated.',
        'parameter_count': len(joint.initial), 'free_parameters': int(np.sum(lower < upper)),
        'parameter_order': 'Existing ship motion, shared standardized antenna coordinates, five ship origins, then one road-motion vector per selected label.',
        'antenna_slice': [joint.antenna_slice.start, joint.antenna_slice.stop],
        'road_slices': [[s.start, s.stop] for s in joint.road_slices]}
    print(json.dumps({'stage': 'loaded', 'labels': labels, 'dropped': dropped,
                      'parameters': len(joint.initial), 'seconds': time.monotonic()-started}), flush=True)
    if args.audit_only:
        parameters = joint.initial
        result = joint.evaluate(parameters)
        probes = []
        for index in [joint.antenna_slice.start, joint.antenna_slice.stop-2,
                      joint.road_slices[0].start, joint.road_slices[-1].stop-1]:
            delta = np.eye(len(parameters))[index]*1e-4
            numeric = (joint.objective(parameters+delta)[0]-joint.objective(parameters-delta)[0])/2e-4
            probes.append({'index': index, 'analytic': float(result['gradient'][index]),
                           'numeric': numeric, 'absolute_error': abs(numeric-result['gradient'][index])})
        exact = original.evaluate(parameters[:joint.ship_count])
        fast = joint.ship.evaluate(parameters[:joint.ship_count])
        base.update(parameters=parameters.tolist(), probes=probes,
            original_vs_quadrature_objective_delta=(fast['value']-exact['value'])*original.energy,
            original_vs_quadrature_gradient_max=float(np.max(abs(fast['gradient']-exact['gradient']))),
            complete=True, kind='recorded_data_derivative_audit')
        args.output.write_text(json.dumps(base, indent=2)+'\n')
        print(json.dumps({'stage': 'audit_complete', 'probes': probes}), flush=True)
        return
    def checkpoint(parameters, trace):
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({**base,
            'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        print(json.dumps({'stage': 'fit', **trace[-1]}), flush=True)
    parameters, fitted = solve(joint, matrix, lo, hi, lower, upper,
                              iterations=args.iterations, checkpoint=checkpoint)
    print(json.dumps({'stage': 'fit_complete', 'converged': fitted['converged'],
                      'kkt': fitted['maximum_kkt_residual']}), flush=True)
    predicted = prediction(joint, original, parameters, labels)
    # Original-row gradient audit, not just a small relative template error.
    exact, fast = original.evaluate(parameters[:joint.ship_count]), joint.ship.evaluate(parameters[:joint.ship_count])
    original_gradient = joint.evaluate(parameters)['gradient'].copy()
    original_gradient[:joint.ship_count] += (exact['gradient']-fast['gradient'])*original.energy/joint.energy
    from stationarity import check
    kkt, violation = check(parameters, original_gradient, matrix, lo, hi, lower, upper)
    result = {**base, 'complete': True, 'parameters': parameters.tolist(), 'fit': fitted,
        'antenna_fit': joint.ship.antennas.describe(parameters[joint.antenna_slice]),
        'antenna_bound_hits': int(np.sum(abs(parameters[joint.antenna_slice]) > 3-1e-5)),
        'original_row_maximum_kkt_residual': kkt, 'original_row_constraint_violation': violation,
        'validation': predicted, 'seconds': time.monotonic()-started}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'original_kkt': kkt,
        'ship_validation': predicted['ship_summary'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
