"""Polish original-row stationarity with declared quadrature search curvature.

Quadrature supplies a numerical Newton preconditioner only. Every proposed
state is accepted against the unchanged original-row objective, feasibility
and gradient. The original nonstationary result is preserved.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from fit_joint_motion import information_scaling
from fit_shared_origins import predictions
from fit_shared_target_geometry import load_experiment
from scene_dictionary import fingerprint
from shared_origin_geometry import SharedOriginGeometry
from shared_target_geometry import SharedGeometry
from stationarity import check, polish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--nodes', type=int, default=2)
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve the existing result')
    source = json.loads(args.source.read_text()); identity = source['identity']
    if not source['complete']:
        raise ValueError('Use a terminal result, including its failed stationarity check')
    for hashes in [identity['legacy_identity']['sources'], identity['legacy_identity']['implementation_sha256'],
                   identity['implementation_sha256'], source.get('numerical_refinement', {}).get('implementation_sha256', {})]:
        for file, digest in hashes.items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('Source changed: '+file)
    legacy, ship, bounds, labels, _ = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], args.nodes)
    fast = SharedOriginGeometry(legacy)
    original = SharedOriginGeometry(SharedGeometry(ship, legacy.roads))
    parameters = np.asarray(source['parameters'])
    matrix, lo, hi, lower, upper = bounds
    if identity['fixed_antennas']:
        lower[original.antenna_slice] = upper[original.antenna_slice] = 0.
    value = original.evaluate(parameters)
    transform, scaling = information_scaling(value['gauss_newton'], 1., lower < upper)
    print(json.dumps({'stage': 'loaded', 'original_kkt': check(parameters, value['gradient'], *bounds)[0],
        'search_nodes': args.nodes, 'seconds': time.monotonic()-started}), flush=True)
    parameters, refinement = polish(original.objective, parameters, transform, matrix, lo, hi, lower, upper,
        curvature_gradient=lambda q: fast.objective(q)[1])
    value = original.evaluate(parameters)
    kkt, violation = check(parameters, value['gradient'], matrix, lo, hi, lower, upper)
    result = copy.deepcopy(source)
    result.update(scope=__doc__, initial_source=str(args.source), initial_source_sha256=fingerprint(args.source),
        parameters=parameters.tolist(), range_origins_m=parameters[original.range_slice].tolist(),
        antenna_fit=ship.antennas.describe(parameters[original.antenna_slice]),
        antenna_bound_hits=int(np.sum(abs(parameters[original.antenna_slice]) > 3-1e-5)),
        original_row_maximum_kkt_residual=kkt, original_row_constraint_violation=violation,
        original_row_training_objective=value['value'], maximum_original_quadrature_gradient_delta=0.,
        validation=predictions(original, ship, parameters, labels), seconds=time.monotonic()-started)
    result['fit'] = {**source['fit'], 'converged': bool(kkt <= 1e-7 and violation <= 1e-7),
        'objective': value['value'], 'maximum_kkt_residual': kkt, 'maximum_constraint_violation': violation,
        'termination': 'Local Newton refinement checked on original rows', 'stationarity_polish': refinement,
        'refinement_scaling': scaling}
    files = [Path(__file__), Path('support/isar/stationarity.py'), Path('support/isar/fit_joint_motion.py')]
    result['numerical_refinement'] = {'original_rows': True, 'search_curvature_nodes': args.nodes,
        'solver_stationarity_gate': 1e-7, 'source_original_row_kkt': source['original_row_maximum_kkt_residual'],
        'implementation_sha256': {str(file): fingerprint(file) for file in files}, 'scope': __doc__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': result['fit']['converged'], 'original_kkt': kkt,
        'accepted': refinement['accepted'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
