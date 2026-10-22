"""Refine the common-origin likelihood with quadrature or original rows.

The physical model, observations, constraints and nuisance parameters stay
unchanged. Preserve the original result, record the numerical implementation,
and check final stationarity against original rows independently of quadrature.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from fit_physical_profile import solve
from fit_shared_origins import predictions
from fit_shared_target_geometry import load_experiment
from scene_dictionary import fingerprint
from shared_origin_geometry import SharedOriginGeometry
from shared_target_geometry import SharedGeometry
from stationarity import check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--nodes', type=int, default=0, help='Zero uses original rows; a positive value uses Gauss quadrature')
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--free-antennas', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve the existing result')
    source = json.loads(args.source.read_text())
    if not source['complete']:
        raise ValueError('Use a terminal result, including its original-row check')
    identity = copy.deepcopy(source['identity'])
    inherited = identity['legacy_identity']
    for group in ('sources', 'implementation_sha256'):
        for file, digest in inherited[group].items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('Legacy input changed: '+file)
    for file, digest in identity['implementation_sha256'].items():
        if fingerprint(Path(file)) != digest:
            raise ValueError('Common-origin input changed: '+file)
    for file in (args.ship_fit, args.road_fit):
        if inherited['sources'].get(str(file)) != fingerprint(file):
            raise ValueError('Use the original source fits')
    if args.nodes < 0:
        raise ValueError('Use zero for original rows or a positive quadrature order')
    legacy, original, bounds, labels, _ = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], args.nodes or inherited['nodes'])
    if args.nodes == 0:
        legacy = SharedGeometry(original, legacy.roads, initial_ship=np.asarray(source['parameters'])[:legacy.ship_count])
    joint = SharedOriginGeometry(legacy)
    joint.initial = np.array(source['parameters'])
    matrix, lo, hi, lower, upper = bounds
    if args.free_antennas:
        identity['fixed_antennas'] = False
    if identity['fixed_antennas']:
        lower[joint.antenna_slice] = upper[joint.antenna_slice] = 0.
        joint.initial[joint.antenna_slice] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = {key: source[key] for key in ('origin_start', 'parameter_count', 'range_origin_bounds_m',
        'legacy_vehicle_origins_m', 'range_slice', 'antenna_slice', 'road_slices')}
    base.update(scope=__doc__, identity=identity, complete=False, accepted_calibration=False,
        yaw_fitted=False, initial_source=str(args.source), initial_source_sha256=fingerprint(args.source),
        numerical_refinement={'quadrature_nodes': args.nodes, 'solver_stationarity_gate': 1e-7,
            'original_rows': args.nodes == 0,
            'extra_curvature_polish': False, 'source_original_row_kkt': source['original_row_maximum_kkt_residual'],
            'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
            'scope': 'Same physical original-row likelihood and observations; only integration accuracy and local numerical stopping change. Original-row gate remains 1e-7.'})
    def checkpoint(parameters, trace):
        path = args.output.with_suffix('.checkpoint.json'); temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({**base, 'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        temporary.replace(path)
        print(json.dumps({'stage': 'fit', **trace[-1]}), flush=True)
    print(json.dumps({'stage': 'loaded', 'nodes': args.nodes, 'fixed_antennas': identity['fixed_antennas'],
        'seconds': time.monotonic()-started}), flush=True)
    parameters, fit = solve(joint, matrix, lo, hi, lower, upper, iterations=args.iterations,
        checkpoint=checkpoint, polish_stationarity=False)
    exact = original.evaluate(parameters[:joint.ship_count])
    fast = joint.ship.evaluate(parameters[:joint.ship_count])
    value = joint.evaluate(parameters)
    gradient = value['gradient'].copy()
    correction = (exact['gradient']-fast['gradient'])*original.energy/joint.energy
    gradient[:joint.ship_count] += correction
    kkt, violation = check(parameters, gradient, matrix, lo, hi, lower, upper)
    result = {**base, 'complete': True, 'parameters': parameters.tolist(), 'fit': fit,
        'range_origins_m': parameters[joint.range_slice].tolist(),
        'antenna_fit': joint.ship.antennas.describe(parameters[joint.antenna_slice]),
        'antenna_bound_hits': int(np.sum(abs(parameters[joint.antenna_slice]) > 3-1e-5)),
        'original_row_maximum_kkt_residual': kkt, 'original_row_constraint_violation': violation,
        'original_row_training_objective': value['value']+(exact['value']-fast['value'])*original.energy/joint.energy,
        'maximum_original_quadrature_gradient_delta': float(abs(correction).max()),
        'validation': predictions(joint, original, parameters, labels), 'seconds': time.monotonic()-started}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': fit['converged'], 'original_row_kkt': kkt,
        'quadrature_gradient_delta': result['maximum_original_quadrature_gradient_delta'],
        'origins_m': result['range_origins_m'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
