"""Transfer common-origin geometry to a completely excluded road target.

The calibration fit never sees that target's likelihood. Both controls retain
the same observations and priors, except that the second frees antenna
coordinates. Transferred origins and antenna coordinates are frozen while
fitting the excluded target's motion and complex responses. Road discovery,
scene support and initialization used this recording, so this is conditional
transfer rather than an independent validation of the road association.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_joint_motion import information_scaling
from fit_physical_profile import solve
from fit_shared_origins import predictions
from fit_shared_target_geometry import load_experiment
from road_validation import related_intervals
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad, SharedOriginGeometry
from shared_target_geometry import SharedGeometry
from stationarity import check, polish
from validate_shared_geometry import fit_target


def subset(legacy, original_ship, bounds, parameters, indices):
    """Remove excluded target parameters, observations and auxiliary priors."""
    columns = np.r_[np.arange(legacy.ship_count), *[
        np.arange(legacy.road_slices[i].start, legacy.road_slices[i].stop) for i in indices]]
    roads = [legacy.roads[i] for i in indices]
    fast = SharedOriginGeometry(SharedGeometry(legacy.ship, roads))
    exact = SharedOriginGeometry(SharedGeometry(original_ship, roads))
    fast.initial = exact.initial = np.array(parameters)[columns].copy()
    matrix, lo, hi, lower, upper = bounds
    return fast, exact, (matrix[:, columns], lo, hi, lower[columns], upper[columns])


class FrozenOriginRoad:
    """Adapt the existing motion-only solver without exposing origin variables."""
    def __init__(self, original, origins):
        self.road = OriginRoad(original)
        self.origins = np.array(origins, copy=True)
        self.problem, self.nmotion, self.energy = original.problem, original.nmotion, original.energy

    def evaluate(self, motion, antenna):
        return self.road.evaluate(motion, antenna, self.origins)

    def predict(self, motion, antenna):
        return self.road.predict(motion, antenna, self.origins)


def fit_control(fast, exact, bounds, initial, fixed, iterations, output, identity, labels):
    if output.exists():
        raise ValueError('Preserve the existing transfer control: '+str(output))
    matrix, lo, hi, lower, upper = [np.array(a, copy=True) for a in bounds]
    if fixed:
        lower[exact.antenna_slice] = upper[exact.antenna_slice] = 0.
    exact.initial = np.array(initial, copy=True)
    exact.initial[exact.antenna_slice] = 0. if fixed else exact.initial[exact.antenna_slice]
    started = time.monotonic()
    def checkpoint(parameters, trace):
        temporary = output.with_suffix('.checkpoint.tmp')
        temporary.write_text(json.dumps({'identity': identity, 'fixed_antennas': fixed,
            'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        temporary.replace(output.with_suffix('.checkpoint.json'))
        print(json.dumps({'stage': 'calibration', 'fixed': fixed, **trace[-1]}), flush=True)
    parameters, fit = solve(exact, matrix, lo, hi, lower, upper, iterations=iterations,
        checkpoint=checkpoint, polish_stationarity=False)
    value = exact.evaluate(parameters)
    kkt, violation = check(parameters, value['gradient'], matrix, lo, hi, lower, upper)
    refinement = None
    # Only refine a nearby numerical stop. A distant failed optimizer needs a
    # new documented start rather than being relabelled as a Newton polish.
    if 1e-7 < kkt < 1e-5 and violation <= 1e-7:
        transform, _ = information_scaling(value['gauss_newton'], 1., lower < upper)
        parameters, refinement = polish(exact.objective, parameters, transform,
            matrix, lo, hi, lower, upper, curvature_gradient=lambda q: fast.objective(q)[1])
        value = exact.evaluate(parameters)
        kkt, violation = check(parameters, value['gradient'], matrix, lo, hi, lower, upper)
    fit.update(converged=bool(kkt <= 1e-7 and violation <= 1e-7), objective=value['value'],
        maximum_kkt_residual=kkt, maximum_constraint_violation=violation)
    result = {'identity': identity, 'complete': True, 'fixed_antennas': fixed,
        'parameters': parameters.tolist(), 'fit': fit, 'stationarity_polish': refinement,
        'original_row_maximum_kkt_residual': kkt, 'original_row_constraint_violation': violation,
        'range_origins_m': parameters[exact.range_slice].tolist(),
        'antenna_fit': exact.ship.antennas.describe(parameters[exact.antenna_slice]),
        'validation': predictions(exact, exact.ship, parameters, labels),
        'accepted_calibration': False, 'seconds': time.monotonic()-started}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'calibration_complete', 'fixed': fixed, 'kkt': kkt,
        'converged': fit['converged'], 'seconds': result['seconds']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'ship-fit', 'road-fit', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Choose a fresh transfer directory')
    source = json.loads(args.source.read_text()); saved = json.loads(args.road_fit.read_text())
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Use a stationary completed common-origin source')
    identity = source['identity']; inherited = identity['legacy_identity']
    for hashes in [inherited['sources'], inherited['implementation_sha256'], identity['implementation_sha256'],
                   source.get('numerical_refinement', {}).get('implementation_sha256', {})]:
        for path, digest in hashes.items():
            if fingerprint(Path(path)) != digest:
                raise ValueError('Source or implementation changed: '+path)
    for path in (args.ship_fit, args.road_fit):
        if inherited['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Use the source fits from the existing experiment')
    legacy, original, bounds, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or labels != identity['labels'] or args.label not in labels:
        raise ValueError('Require the same disjoint target set and a known excluded label')
    excluded = labels.index(args.label); road = legacy.roads[excluded]
    retained = [i for i in range(len(labels)) if i != excluded]
    if any(related_intervals(road.problem, legacy.roads[i].problem) for i in retained):
        raise ValueError('A potentially overlapping target interval remains in calibration')
    fast, exact, bounds = subset(legacy, original, bounds, source['parameters'], retained)
    training_labels = [labels[i] for i in retained]
    identity = {'scope': __doc__, 'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'calibration_labels': training_labels, 'excluded_label': args.label,
        'inherited_identity': source['identity'], 'original_rows': True, 'yaw_fixed_zero': True,
        'implementation_sha256': {str(p): fingerprint(p) for p in [Path(__file__),
            Path('support/isar/fit_physical_profile.py'), Path('support/isar/stationarity.py'),
            Path('support/isar/validate_shared_geometry.py')]}}
    args.output.mkdir(parents=True)
    (args.output/'experiment.json').write_text(json.dumps({'identity': identity, 'complete': False}, indent=2)+'\n')
    fixed = fit_control(fast, exact, bounds, exact.initial, True, args.iterations,
        args.output/'fixed.json', identity, training_labels)
    if not fixed['fit']['converged']:
        raise ValueError('Fixed control did not pass unchanged original-row stationarity')
    fitted = fit_control(fast, exact, bounds, fixed['parameters'], False, args.iterations,
        args.output/'fitted.json', identity, training_labels)
    if not fitted['fit']['converged']:
        raise ValueError('Free-antenna control did not pass unchanged original-row stationarity')
    old_index = saved['labels'].index(args.label)
    starts = [np.asarray(saved['selected_roads'][old_index]['state']), road.initial]
    transferred = []
    for case in (fixed, fitted):
        parameters = np.asarray(case['parameters'])
        target = FrozenOriginRoad(road, parameters[exact.range_slice])
        transferred.append(fit_target(target, parameters[exact.antenna_slice], starts, args.iterations))
    a, b = [case['prediction'] for case in transferred]
    errors = [sum(c['prediction_error'] for c in case) for case in (a, b)]
    energies = [sum(c['matched_energy'] for c in case) for case in (a, b)]
    stationary = [case['starts'][case['selected_start']]['fit']['converged'] for case in transferred]
    result = {'identity': identity, 'complete': bool(all(stationary)), 'accepted_calibration': False,
        'fixed': transferred[0], 'fitted': transferred[1],
        'summary': {'prediction_error_reduction_percent': float(100*(1-errors[1]/errors[0])),
            'matched_energy_change_db': float(10*np.log10(energies[1]/energies[0])),
            'selected_fits_stationary': stationary}, 'seconds': time.monotonic()-started}
    (args.output/'transfer.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.output/'experiment.json').write_text(json.dumps({'identity': identity,
        'complete': result['complete'], 'summary': result['summary']}, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', **result['summary'], 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
