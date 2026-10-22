"""Evaluate a common-origin solution in the unchanged legacy observation model.

This is a feasible-point comparison, not an optimized legacy refit. A lower
legacy objective at this point proves the old solution was not global, without
attributing improvement to the new coupling assumption.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_shared_target_geometry import load_experiment, prediction
from scene_dictionary import fingerprint
from shared_target_geometry import SharedGeometry
from stationarity import check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing feasible-point control')
    source = json.loads(args.source.read_text())
    if not source['complete'] or not source['identity']['fixed_antennas']:
        raise ValueError('Use a terminal fixed-antenna source')
    identity = source['identity']['legacy_identity']
    for group in ('sources', 'implementation_sha256'):
        for file, digest in identity[group].items():
            if fingerprint(Path(file)) != digest:
                raise ValueError('An inherited input changed: '+file)
    legacy_path = Path(source['identity']['legacy_control'])
    if fingerprint(legacy_path) != source['identity']['legacy_control_sha256']:
        raise ValueError('Legacy fit changed')
    old = json.loads(legacy_path.read_text())
    legacy, exact, bounds, labels, _ = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], identity['nodes'])
    joint = SharedGeometry(exact, legacy.roads)
    parameters = np.asarray(source['parameters'])
    matrix, lo, hi, lower, upper = bounds
    lower[joint.antenna_slice] = upper[joint.antenna_slice] = 0.
    value = joint.evaluate(parameters)
    residual, violation = check(parameters, value['gradient'], matrix, lo, hi, lower, upper)
    if violation > 1e-7:
        raise ValueError('The comparison state is not feasible in the legacy model')
    original = joint.evaluate(np.asarray(old['parameters']))
    files = [args.source, legacy_path, Path(__file__), Path('support/isar/shared_target_geometry.py')]
    result = {'scope': __doc__, 'complete': True, 'original_rows': True, 'optimized': False,
        'source': str(args.source), 'legacy_control': str(legacy_path),
        'legacy_original_row_objective': original['value'],
        'new_state_legacy_original_row_objective': value['value'],
        'old_solution_has_feasible_lower_objective': bool(value['value'] < original['value']-1e-10),
        'feasible_state_kkt_residual': residual, 'constraint_violation': violation,
        'validation': prediction(joint, exact, parameters, labels),
        'accepted_calibration': False, 'source_sha256': {str(file): fingerprint(file) for file in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('legacy_original_row_objective', 'new_state_legacy_original_row_objective',
        'old_solution_has_feasible_lower_objective', 'feasible_state_kkt_residual', 'constraint_violation')}), flush=True)


if __name__ == '__main__':
    main()
